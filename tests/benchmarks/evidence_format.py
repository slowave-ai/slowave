"""Boundary-preserving evidence formatting for answer-generation benchmarks."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _date(ts: int | float | None) -> str:
    if not ts:
        return "unknown"
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%MZ")


def format_retrieved_evidence(
    result: Any,
    *,
    include_schemas: bool = True,
    include_episodes: bool = True,
    episodes_first: bool = False,
) -> str:
    """Render ranked schemas and episodes without flattening their boundaries.

    The ordinary keyword metrics continue to use their legacy flat hypothesis.
    This representation is for downstream answer models, which need to know
    where one memory ends and another begins and which dates are attached to
    each item.
    """
    schema_section = ""
    episode_section = ""

    if include_schemas and result.schemas:
        schemas = ["=== SCHEMAS (stable extracted knowledge) ==="]
        for rank, schema in enumerate(result.schemas, start=1):
            score = result.schema_rank_scores.get(schema.id)
            score_text = f" | score={score:.4f}" if score is not None else ""
            schemas.append(
                f"[SCHEMA {rank} | source_date=unknown{score_text}]\n"
                f"{schema.content_text.strip()}"
            )
        schema_section = "\n\n".join(schemas)

    if include_episodes and result.episode_texts:
        episodes = ["=== EPISODES (raw conversation excerpts) ==="]
        for rank, episode in enumerate(result.episode_texts, start=1):
            episodes.append(
                f"[EPISODE {rank} | date={_date(episode.get('ts'))}]\n"
                f"{str(episode.get('content_text', '')).strip()}"
            )
        episode_section = "\n\n".join(episodes)

    sections = [schema_section, episode_section]
    if episodes_first:
        sections.reverse()
    sections = [section for section in sections if section]
    return "\n\n".join(sections)
