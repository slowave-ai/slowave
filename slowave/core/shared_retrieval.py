"""Experimental shared discovery request and single-pass relevance assessment."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from slowave.core.applicability_ranking import PairScorer

POLICY_VERSION = "shared-multilingual-v1"

# Shared-pipeline absolute relevance floors. The shipped defaults are the
# measured LIVE operating point (-1.25 activate / -2 recall), validated on a
# read-only production-scale snapshot (408 memories,
# private/audits/20261006_memory_recovery/wp4/windowed_probe.json): negatives
# clean (0-6, only the known dev-journal 'wrote'-collision noise), on-topic
# queries surface their memories. The earlier corpus-v1 calibration (-6.05
# both endpoints, commit a38e6c7) is REJECTED by live evidence: at production
# scale the ~110-candidate discovery pool sits almost entirely in the
# [-6, -1.25] logit band, so a -6.05 floor admitted 43-106 memories per
# unrelated request (RESULTS §12). Standing measured finding: no static floor
# serves both regimes (live scale needs >= -1.25; tiny/cold-start scopes need
# <= -6.05); scale/pool-aware admission is the planned real fix. Env
# overrides remain authoritative.
_ACTIVATE_LOGIT_DEFAULT = -1.25
_RECALL_LOGIT_DEFAULT = -2.0


def enabled() -> bool:
    value = os.environ.get("SLOWAVE_RETRIEVAL_PIPELINE", "current")
    if value not in {"current", "shared-v1"}:
        raise ValueError("SLOWAVE_RETRIEVAL_PIPELINE must be current or shared-v1")
    return value == "shared-v1"


@dataclass(frozen=True)
class SharedRequest:
    task: str
    goal: str | None = None
    semantic_context: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    endpoint: str = "activate"

    @property
    def queries(self) -> list[str]:
        # No inferred splitting, lexical language rules, or extra votes for duplicates.
        values = (self.task, self.goal or "", self.semantic_context or "")
        unique: dict[str, str] = {}
        for value in values:
            if value.strip():
                unique.setdefault(" ".join(value.casefold().split()), value.strip())
        return list(unique.values())

    @property
    def assessment_query(self) -> str:
        fields = [f"Task: {self.task.strip()}"]
        if self.goal and self.goal.strip():
            fields.append(f"Goal: {self.goal.strip()}")
        if self.semantic_context and self.semantic_context.strip():
            fields.append(f"Semantic context: {self.semantic_context.strip()}")
        if self.context:
            fields.append(
                "Context: " + json.dumps(self.context, ensure_ascii=False, sort_keys=True)
            )
        return "\n".join(fields)

    @property
    def threshold(self) -> float:
        # Shared-pipeline absolute relevance floors. The defaults are the
        # measured LIVE operating point: activate -1.25, recall -2. Validated
        # on a read-only production-scale snapshot (408 memories,
        # private/audits/20261006_memory_recovery/wp4/windowed_probe.json):
        # unrelated requests admit 0-6 memories (only the known dev-journal
        # 'wrote'-collision noise) and on-topic queries still surface their
        # memories.
        #
        # History: the original uncalibrated floors (activate 0, recall -2)
        # starved the core proactive scenario on corpus v1 — relevant_total
        # was 0 across the whole WP-5 shakeout matrix because every required
        # first-page pair scores between about -6.2 and -1.6 logits. WP-4's
        # corpus-v1 sweep moved both defaults to -6.05 (commit a38e6c7), a
        # real small-corpus result that was REJECTED the same day by live
        # dogfooding evidence: the pipeline admits on the best
        # sentence-bounded WINDOW per memory, and at production scale the
        # discovery pool sits almost entirely in the [-6, -1.25] logit band,
        # so -6.05 admitted 43-106 memories per unrelated request (RESULTS
        # §12; corpus v1's single-sentence memories made window = whole
        # memory, so the harness sweep could never see this). No static floor
        # serves both regimes — scale/pool-aware admission is the planned
        # follow-up work package.
        #
        # Recall keeps the live -2 floor: the shipped-code recall default
        # must not leak on negatives at live scale, and the enforced
        # invariant is activation never looser than recall. Env overrides
        # remain authoritative.
        activate_default = _ACTIVATE_LOGIT_DEFAULT
        recall_default = _RECALL_LOGIT_DEFAULT
        activate = float(os.environ.get("SLOWAVE_ACTIVATE_RELEVANCE_LOGIT", str(activate_default)))
        recall = float(os.environ.get("SLOWAVE_RECALL_RELEVANCE_LOGIT", str(recall_default)))
        if not np.isfinite([activate, recall]).all() or activate < recall:
            raise ValueError("finite relevance thresholds require activate >= recall")
        if self.endpoint not in {"activate", "recall"}:
            raise ValueError("unknown shared retrieval endpoint")
        return activate if self.endpoint == "activate" else recall

    @property
    def candidate_depth(self) -> int:
        depth = int(os.environ.get("SLOWAVE_RETRIEVAL_CANDIDATE_DEPTH", "64"))
        if not 1 <= depth <= 256:
            raise ValueError("candidate depth must be between 1 and 256")
        return depth


def source_windows(text: str, max_chars: int = 1024) -> list[tuple[int, int]]:
    """Exact sentence-boundary windows; overlap retains preceding qualifications.

    Oversized individual sentences remain whole internally rather than quietly
    severing a negation/condition. Their transport is explicitly excluded later.
    """
    if not text:
        return []
    if len(text) <= max_chars:
        return [(0, len(text))]
    sentences = list(re.finditer(r".+?(?:[.!?](?=\s|$)|\n|$)", text, re.S))
    windows: list[tuple[int, int]] = []
    index = 0
    while index < len(sentences):
        start = sentences[index].start()
        if index and sentences[index].end() - sentences[index - 1].start() <= max_chars:
            start = sentences[index - 1].start()
        end = sentences[index].end()
        following = index + 1
        while following < len(sentences) and sentences[following].end() - start <= max_chars:
            end = sentences[following].end()
            following += 1
        windows.append((start, end))
        index = following
    return windows


def assess(
    scorer: PairScorer, request: SharedRequest, texts: list[str]
) -> tuple[list[float], list[tuple[int, int]]]:
    """Score source windows once; best window is the candidate's evidence.

    This is one model assessment stage, not full-source scoring followed by
    another excerpt assessment. No relative-to-best admission threshold.
    """
    owners: list[tuple[int, tuple[int, int]]] = []
    windows: list[str] = []
    for index, text in enumerate(texts):
        for span in source_windows(text):
            owners.append((index, span))
            windows.append(text[span[0] : span[1]])
    values = [-float("inf")] * len(texts)
    spans = [(0, 0)] * len(texts)
    if not windows:
        return values, spans
    matrix = scorer.score([request.assessment_query], windows)
    if matrix.shape != (len(windows), 1) or not np.isfinite(matrix).all():
        raise ValueError("invalid shared relevance output")
    for (owner, span), score in zip(owners, matrix[:, 0]):
        if float(score) > values[owner]:
            values[owner] = float(score)
            spans[owner] = span
    return values, spans
