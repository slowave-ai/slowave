"""IngestService: converts a closed session's raw events into episodic memories.

Owns all episode-formation logic previously scattered as private methods on
SlowaveEngine. Extracted so it can be tested and reasoned about independently.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import numpy as np

from slowave.latent.episodic_store import EpisodicStore
from slowave.latent.salience import SalienceEngine
from slowave.latent.transition_model import TransitionModel
from slowave.storage.sqlite_db import SQLiteDB
from slowave.symbolic.episode_text import EpisodeTextStore
from slowave.symbolic.raw_log import RawLog

log = logging.getLogger(__name__)


class IngestService:
    """Converts raw session events into latent episodic memories.

    Produces both micro-episodes (sliding window over adjacent events) and a
    macro-episode (whole-session trace). The transition model contributes
    predictive-surprise salience once consolidation has run at least once.
    """

    def __init__(
        self,
        *,
        raw_log: RawLog,
        episodic: EpisodicStore,
        episode_text: EpisodeTextStore,
        salience: SalienceEngine,
        transition_model: TransitionModel,
        db: SQLiteDB,
    ):
        self.raw_log = raw_log
        self.episodic = episodic
        self.episode_text = episode_text
        self.salience = salience
        self.transition_model = transition_model
        self.db = db

    # ---- public API --------------------------------------------------------

    def form_episodes(self, session_id: str) -> list[int]:
        """Convert a session's raw events into multi-scale episodes.

        Returns a list of the new episode IDs (empty if no embeddable events).

        Changing this method's episode-formation output for existing data?
        Bump ``SlowaveConfig.current_logic_version`` (slowave/core/config.py)
        so customer DBs auto-rebuild via RebuildService.
        """
        events = self.raw_log.list_session(session_id)
        # Exclude context_query events (activate calls) from episode formation.
        # Retrieval queries are cues, not memories — encoding them as episodes
        # causes consolidation to produce episodic summaries whose central text
        # is the query itself ("remember Karpathy Guidelines...") blended with
        # project-specific content from the same session.
        embeddable = [e for e in events if e.embedding is not None and e.type != "context_query"]
        if not embeddable:
            return []

        made: list[int] = []

        # Micro episodes: sliding windows over adjacent events. Window=2 keeps
        # user/assistant exchanges compact; singleton fallback handles 1-turn sessions.
        window = 2 if len(embeddable) >= 2 else 1
        for start in range(0, len(embeddable) - window + 1):
            chunk = embeddable[start : start + window]
            emb = self._mean_embedding([e.embedding for e in chunk if e.embedding is not None])
            if emb is None:
                continue
            text = "\n".join(self._event_text(e) for e in chunk if e.content.strip())
            source = "\n".join(e.content.strip() for e in chunk if e.content.strip())
            salience, surprise = self._episode_salience(emb)
            if any(e.type.startswith("remember:") for e in chunk):
                salience = min(1.5, salience + 0.6)
            ep_id = self.episodic.add(
                event_id=f"micro_{session_id}_{start}",
                ts=chunk[len(chunk) // 2].ts,
                embedding=emb,
                salience=salience,
                metadata={
                    "session_id": session_id,
                    "kind": "micro",
                    "n_events": len(chunk),
                    "prediction_error": surprise,
                },
            )
            self.episode_text.put(
                episode_id=ep_id,
                content_text=text,
                source_content=source,
                event_ids=[e.id for e in chunk],
                session_id=session_id,
            )
            made.append(ep_id)

        # Macro episode: whole-session trace.
        macro_emb = self._mean_embedding(
            [e.embedding for e in embeddable if e.embedding is not None]
        )
        if macro_emb is not None:
            # Use `embeddable`, not `events`, here -- matching macro_emb above
            # and event_ids below. A non-embeddable event (e.g. slowave_commit's
            # task_complete "outcome=X" marker, logged with disable_encoder=True)
            # has no vector to contribute to macro_emb, but its text would still
            # get pulled into macro_text/macro_source if `events` were used here,
            # producing an episode whose text describes MORE than its embedding
            # represents -- when embeddable has exactly one entry, macro_emb is
            # that single event's embedding unchanged, while macro_text would
            # have an unrelated event's content appended to it.
            macro_text = "\n".join(self._event_text(e) for e in embeddable if e.content.strip())
            macro_source = "\n".join(e.content.strip() for e in embeddable if e.content.strip())
            salience, surprise = self._episode_salience(macro_emb)
            salience = max(salience * 0.8, 0.05)  # macro useful but less atomic
            if any(e.type.startswith("remember:") for e in events):
                salience = min(1.5, salience + 0.6)
            ep_id = self.episodic.add(
                event_id=f"macro_{session_id}",
                ts=embeddable[len(embeddable) // 2].ts,
                embedding=macro_emb,
                salience=salience,
                metadata={
                    "session_id": session_id,
                    "kind": "macro",
                    "n_events": len(embeddable),
                    "prediction_error": surprise,
                },
            )
            self.episode_text.put(
                episode_id=ep_id,
                content_text=macro_text,
                source_content=macro_source,
                event_ids=[e.id for e in embeddable],
                session_id=session_id,
            )
            made.append(ep_id)

        return made

    def prototypes_for_episodes(self, episode_ids: list[int]) -> list[int]:
        """Return prototype IDs that have any of the given episodes mapped to them."""
        if not episode_ids:
            return []
        conn = self.db.connect()
        ph = ",".join(["?"] * len(episode_ids))
        rows = conn.execute(
            f"SELECT DISTINCT prototype_id FROM episode_prototype_map "
            f"WHERE episode_id IN ({ph})",
            tuple(int(e) for e in episode_ids),
        ).fetchall()
        return [int(r["prototype_id"]) for r in rows]

    def link_session_episodes(self, session_id: str, episode_ids: list[int]) -> None:
        """Back-link newly-formed episodes to schemas remembered during this session.

        During a live session, ``remember()`` creates schemas with empty
        ``supporting_episode_ids`` (the episodes don't exist yet).  Now that
        they have been formed, we walk every raw event in this session,
        look up which episode(s) carry it, and update the schema row so
        ``support_count`` / stability scores reflect real evidence.
        """
        conn = self.db.connect()
        now = int(time.time())
        try:
            # Build a raw_event_id → [episode_id, …] map from the just-formed
            # episodes.  episode_text.event_ids is a JSON array of ints.
            ep_rows = conn.execute(
                "SELECT episode_id, event_ids FROM episode_text "
                "WHERE session_id = ? AND episode_id IN ({})".format(
                    ",".join("?" * len(episode_ids))
                ),
                [session_id, *episode_ids],
            ).fetchall()

            ev_to_ep: dict[int, list[int]] = {}
            for r in ep_rows:
                ep_id = int(r["episode_id"])
                try:
                    eids = json.loads(r["event_ids"])
                except (json.JSONDecodeError, TypeError):
                    continue
                # event_ids is stored as {"ids": [1, 2, …]} (see episode_text.py:49).
                raw_ids = eids.get("ids", []) if isinstance(eids, dict) else eids
                for eid in raw_ids:
                    try:
                        ev_to_ep.setdefault(int(eid), []).append(ep_id)
                    except (TypeError, ValueError):
                        continue

            if not ev_to_ep:
                return

            # Find schemas whose schema_evidence still has episode_id=NULL
            # (the placeholder left by remember() during the live session)
            # and that reference a raw event from this session.
            key_ids = list(ev_to_ep.keys())
            ph = ",".join("?" * len(key_ids))
            schema_rows = conn.execute(
                f"SELECT DISTINCT se.schema_id, se.raw_event_id "
                f"FROM schema_evidence se "
                f"WHERE se.raw_event_id IN ({ph}) AND se.episode_id IS NULL",
                tuple(key_ids),
            ).fetchall()

            for sr in schema_rows:
                schema_id = int(sr["schema_id"])
                raw_id = sr["raw_event_id"]
                if raw_id is None:
                    continue
                try:
                    raw_id = int(raw_id)
                except (TypeError, ValueError):
                    continue
                ep_ids = ev_to_ep.get(raw_id)
                if not ep_ids:
                    continue

                # Merge into existing supporting_episode_ids.
                cur = conn.execute(
                    "SELECT supporting_episode_ids FROM schemas WHERE id = ?",
                    (schema_id,),
                ).fetchone()
                if cur is None:
                    continue
                try:
                    payload = json.loads(cur["supporting_episode_ids"])
                except (json.JSONDecodeError, TypeError):
                    payload = {}
                existing = set(
                    int(x) for x in (payload.get("ids", []) if isinstance(payload, dict) else [])
                )
                existing.update(ep_ids)
                conn.execute(
                    "UPDATE schemas SET supporting_episode_ids = ?, last_updated_ts = ? "
                    "WHERE id = ?",
                    (
                        json.dumps({"ids": sorted(existing)}),
                        now,
                        schema_id,
                    ),
                )

                # Update the schema_evidence row so future lookups don't
                # re-enter this path for the same schema.
                conn.execute(
                    "UPDATE schema_evidence SET episode_id = ? "
                    "WHERE schema_id = ? AND raw_event_id = ? AND episode_id IS NULL",
                    (ep_ids[0], schema_id, raw_id),
                )

            conn.commit()
        except Exception:
            log.warning(
                "link_session_episodes failed for session %s",
                session_id,
                exc_info=True,
            )
            try:
                conn.rollback()
            except Exception:
                pass

    # ---- private helpers ---------------------------------------------------

    def _event_text(self, e: Any) -> str:
        role = "User" if "user" in e.type.lower() else "Assistant"
        if e.type.startswith("remember:"):
            role = "Remember"
        return f"{role}: {e.content.strip()}"

    def _mean_embedding(self, embeddings: list[np.ndarray]) -> np.ndarray | None:
        if not embeddings:
            return None
        emb = np.stack(embeddings, axis=0).mean(axis=0).astype(np.float32)
        norm = float(np.linalg.norm(emb))
        return emb / norm if norm > 1e-12 else emb

    def _episode_salience(self, embedding: np.ndarray) -> tuple[float, float]:
        if self.episodic.count() >= 1:
            scores, ids = self.episodic.search(embedding, top_k=1)
            nn_sim = float(scores[0]) if scores.size else -1.0
            nearest_id = int(ids[0]) if ids.size and int(ids[0]) != -1 else None
        else:
            nn_sim = -1.0
            nearest_id = None
        novelty = self.salience.compute_novelty_salience(nn_similarity=nn_sim)
        surprise = 0.0
        if nearest_id is not None:
            try:
                prev = self.episodic.get(nearest_id)
                pred = (
                    self.transition_model.predict(prev.embedding.reshape(1, -1))
                    .reshape(-1)
                    .astype(np.float32)
                )
                pred_norm = float(np.linalg.norm(pred))
                if pred_norm > 1e-12:
                    pred = pred / pred_norm
                surprise = max(0.0, min(1.0, 1.0 - float(pred.dot(embedding))))
            except Exception:
                surprise = 0.0
        return max(0.01, novelty + self.salience.cfg.surprise_weight * surprise), surprise
