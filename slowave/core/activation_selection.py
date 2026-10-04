"""Explainable admission of complementary task context.

Ranking evidence is not applicability. This deterministic assessor deliberately
requires task-specific lexical, explicit applicability, or strong semantic evidence;
it never treats a new ID, a high rank or a constraint label as proof of value.
Discovery evidence is assessed before a candidate is exposed.
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any, Literal

from slowave.symbolic.schema_store import SchemaStore

SelectionMode = Literal["legacy_activation", "deliberate_recall", "complementary_activation"]
_GENERIC = frozenset(
    # Corpus-derived English stopword list (see
    # private/benchmarks/activation_memory_flow for the source corpus); extend
    # from calibration data, not ad hoc.
    "a an the and or to of for in on at by with from is are be been this that it its "
    "use using stored recorded requirements requirement prepare safely every all "
    "before after must should do not when if while already please task goal review "
    "validate check verify get ready honor implement fix update deployment project "
    "ok start meanwhile which who where what stores system platform provider handles settings current does did".split()
)


# Final applicability evidence is stricter than the 0.20 discovery floor.
SEMANTIC_CONTRIBUTION_FLOOR = 0.75
CROSS_LANGUAGE_CONTRIBUTION_FLOOR = 0.55


@lru_cache(maxsize=2048)
def _language(text: str) -> str | None:
    from langdetect import DetectorFactory, detect_langs
    from langdetect.lang_detect_exception import LangDetectException

    DetectorFactory.seed = 0
    try:
        languages = detect_langs(text)
    except LangDetectException:
        return None
    return languages[0].lang if languages and languages[0].prob >= 0.9 else None


def _semantic_floor(task: str, memory: str) -> float:
    task_language = _language(task)
    memory_language = _language(memory)
    if task_language and memory_language and task_language != memory_language:
        return CROSS_LANGUAGE_CONTRIBUTION_FLOOR
    return SEMANTIC_CONTRIBUTION_FLOOR


@dataclass(frozen=True)
class TaskNeed:
    text: str
    provenance: tuple[str, ...]


@dataclass(frozen=True)
class ActivationTask:
    needs: tuple[TaskNeed, ...]
    metadata: dict[str, Any]

    @classmethod
    def build(
        cls,
        task: str,
        goal: str | None = None,
        semantic_context: str | None = None,
        task_context: dict[str, Any] | None = None,
    ) -> ActivationTask:
        """Keep explicit semantic text separate from structured metadata.

        Identical task/goal/context cues get one vote and retain all origins.
        Prose works as a whole need without speculative conjunction splitting.
        """
        texts: dict[str, tuple[str, list[str]]] = {}
        for source, text in (
            ("task", task),
            ("initial_goal", goal),
            ("semantic_context", semantic_context),
        ):
            if not text or not text.strip():
                continue
            parts = re.split(r"\s+and\s+(?=(?:which|who|where|what)\b)", text, flags=re.I)
            for part in parts:
                if not part.strip():
                    continue
                key = " ".join(part.casefold().split())
                if key not in texts:
                    texts[key] = (part.strip(), [])
                texts[key][1].append(source)
        facets = re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+(.+?)\s*$", task)
        for text in facets[:256]:
            key = " ".join(text.casefold().split())
            if key not in texts:
                texts[key] = (text.strip(), [])
            texts[key][1].append("explicit_list_item")
        metadata = deepcopy(task_context or {})
        # String values in structured task context are user-supplied scope cues,
        # so let them support retrieval without merging them into prose metadata.
        for value in metadata.values():
            values = value if isinstance(value, list) else [value]
            for cue in values:
                if not isinstance(cue, str) or not cue.strip():
                    continue
                key = " ".join(cue.casefold().split())
                if key not in texts:
                    texts[key] = (cue.strip(), [])
                texts[key][1].append("task_context")
        return cls(
            tuple(TaskNeed(text, tuple(origins)) for text, origins in texts.values()),
            metadata,
        )

    @property
    def cues(self) -> list[str]:
        return [need.text for need in self.needs]


def _applicability_terms(text: str) -> set[str]:
    # Release and deployment name the same operational context; this mapping
    # applies only to explicit applicability clauses, never general topic rank.
    aliases = {"deployment": "release", "deploy": "release", "deploying": "release"}
    return {
        aliases.get(term.casefold(), term.casefold()) for term in SchemaStore.lexical_tokens(text)
    }


def claim_key(text: str) -> str:
    """Exact normalized content grouping; preserve negation and parameters."""
    return " ".join(text.casefold().split())


def _claim_words(text: str) -> str:
    # Preserve order, negation, conditions and parameters when comparing supplied
    # facts. Only grammatical filler may disappear, never logical qualifiers.
    protected = {"not", "no", "never", "without", "unless", "only", "before", "after", "if", "when"}
    ignored = _GENERIC - protected
    return " ".join(
        term.casefold()
        for term in SchemaStore.lexical_tokens(text)
        if term.casefold() not in ignored
    )


def specific_terms(text: str) -> set[str]:
    return {term.casefold() for term in SchemaStore.lexical_tokens(text)} - _GENERIC


@dataclass(frozen=True)
class ContributionCandidate:
    memory_id: int
    text: str
    score: float
    eligible: bool
    eligibility_reason: str
    need_indexes: tuple[int, ...]
    # Channel evidence remains independent of fused rank.
    channel_evidence: tuple[dict[str, Any], ...] = ()
    facets: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContributionDecision:
    memory_id: int
    selected: bool
    reason: str
    supporting_need: int | None = None
    contribution: str | None = None
    source_spans: tuple[tuple[int, int], ...] = ()
    redundant_with: int | None = None
    selected_position: int | None = None
    channel_evidence: tuple[dict[str, Any], ...] = ()
    # Number of specific task terms shared by the best contribution span.
    # Admission and delivery ordering use this contribution strength.
    shared_terms: int = 0
    # True when a qualifying contribution span was found.
    strong: bool = False


@dataclass(frozen=True)
class ContributionSelection:
    selected_ids: tuple[int, ...]
    decisions: tuple[ContributionDecision, ...]


# Delivery taxonomy for contribution decisions. The selector classifies every
# evaluated candidate; the delivery layer consumes these sets, so every reason
# select_complementary can emit must be classified here (guarded by a unit
# test). A reason in NOT_DELIVERABLE_REASONS keeps the candidate out of the
# catalog; TRANSPORT_EXCLUDED_REASONS names renderability exclusions recorded
# by the service; everything else is deliverable.
DELIVERABLE_REASONS = frozenset(
    {
        "uncovered_need",
        "complementary_fact",
        "decision_constraint",
        "safety_prerequisite",
        "semantic_contribution",
    }
)
NOT_DELIVERABLE_REASONS = frozenset(
    {
        "ineligible",
        "insufficient_evidence",
        "structured_facet_conflict",
        "already_in_input",
        "redundant_with",
        "need_already_covered",
        "wrong_task_facet",
        "insufficient_applicability",
    }
)
TRANSPORT_EXCLUDED_REASONS = frozenset({"oversize_without_contribution_span"})


def _compatible_facets(facets: dict[str, Any], metadata: dict[str, Any]) -> tuple[bool, bool]:
    """Only explicit shared fields may support or conflict; absent fields are unknown."""
    matched = False
    for key in ("task_type", "application", "environment", "component", "operation"):
        if key not in facets or key not in metadata:
            continue
        if facets[key] != metadata[key]:
            return False, False
        matched = True
    return True, matched


def select_complementary(
    task: ActivationTask, candidates: list[ContributionCandidate], *, scope: str | None = None
) -> ContributionSelection:
    """Annotate discovered candidates with contribution evidence.

    Lexical support excludes scope-only overlap and respects explicit facets.
    Applicability clauses and strong dense paraphrases provide independent paths.
    Contribution excerpts are contiguous exact source sentences, never generated claims.
    """
    scope_terms = specific_terms(scope.split(":", 1)[-1]) if scope else set()
    task_terms = [specific_terms(need.text) - scope_terms for need in task.needs]
    # Generated goals aid discovery but cannot independently establish user need.
    user_needs = {
        i
        for i, need in enumerate(task.needs)
        if any(
            source in need.provenance
            for source in ("task", "explicit_list_item", "semantic_context")
        )
    }
    question_coverage = {
        index: max(
            (
                len((specific_terms(candidate.text) - scope_terms) & task_terms[index])
                for candidate in candidates
                if candidate.eligible
            ),
            default=0,
        )
        for index in user_needs
        if re.search(r"\b(which|who|where|what)\b", task.needs[index].text, re.I)
    }
    supplied = "\n".join(task.cues)
    normalized_supplied = claim_key(supplied)
    supplied_claims = tuple(_claim_words(cue) for cue in task.cues)
    seen: dict[str, int] = {}
    covered: set[int] = set()
    selected: list[int] = []
    decisions: list[ContributionDecision] = []
    for candidate in sorted(candidates, key=lambda item: (-item.score, item.memory_id)):
        base = ContributionDecision(
            memory_id=candidate.memory_id,
            selected=False,
            reason="",
            channel_evidence=candidate.channel_evidence,
        )
        if not candidate.eligible:
            decisions.append(replace(base, selected=False, reason="ineligible"))
            continue
        if not candidate.need_indexes and not any(
            evidence.get("relevance_passed") for evidence in candidate.channel_evidence
        ):
            decisions.append(replace(base, reason="insufficient_evidence"))
            continue
        compatible, facet_match = _compatible_facets(candidate.facets, task.metadata)
        if not compatible:
            decisions.append(replace(base, selected=False, reason="structured_facet_conflict"))
            continue
        best: tuple[int, int, int, int] | None = None
        # Sentence boundaries retain condition, negation, units and parameter
        # values. A sentence with multiple clauses is never stitched or shortened.
        for match in re.finditer(r".+?(?:[.!?](?=\s|$)|\n|$)", candidate.text):
            span_text = match.group().strip()
            terms = specific_terms(span_text) - scope_terms
            for index in sorted(user_needs):
                if index not in user_needs or not task_terms[index]:
                    continue
                shared = len(terms & task_terms[index])
                explicit_structured_cue = "task_context" in task.needs[index].provenance
                explicit_list_cue = "explicit_list_item" in task.needs[index].provenance
                # An explicit applicability clause is evidence independent of
                # incidental topical overlap: "For <context>, <requirement>".
                context = re.match(r"\s*For\s+([^,;:.]+)[,;:]", span_text, re.I)
                context_terms = _applicability_terms(context.group(1)) if context else set()
                cue_terms = _applicability_terms(task.needs[index].text)
                operational = not re.search(r"\b(which|who|where)\b", task.needs[index].text, re.I)
                applicable_context = (
                    operational
                    and len(context_terms) >= 2
                    and context_terms <= cue_terms
                    and bool(context_terms - scope_terms)
                )
                if not applicable_context:
                    if shared < (
                        1 if facet_match or explicit_structured_cue or explicit_list_cue else 2
                    ):
                        continue
                    if (
                        not explicit_list_cue
                        and not facet_match
                        and shared / max(1, len(task_terms[index])) <= 2 / 3
                        and not (shared >= 3 and shared >= question_coverage.get(index, 10**9))
                    ):
                        continue
                proposal = (shared, -index, match.start(), match.end())
                if best is None or proposal[:2] > best[:2]:
                    best = proposal
        semantic = False
        if best is None:
            # Dense discovery uses a permissive floor; final semantic admission
            # must clear a separate absolute evidence floor on a user cue.
            supported = [
                e
                for e in candidate.channel_evidence
                if e.get("need_index") in user_needs
                and e.get("need_index") in candidate.need_indexes
                and e.get("dense_cosine") is not None
                and task_terms[e["need_index"]]
                and (
                    e["need_index"] not in question_coverage
                    or not (specific_terms(candidate.text) & task_terms[e["need_index"]])
                    or len(specific_terms(candidate.text) & task_terms[e["need_index"]])
                    >= question_coverage[e["need_index"]]
                )
                and e["dense_cosine"] >= CROSS_LANGUAGE_CONTRIBUTION_FLOOR
                and (
                    e["dense_cosine"] >= SEMANTIC_CONTRIBUTION_FLOOR
                    or e["dense_cosine"]
                    >= _semantic_floor(task.needs[e["need_index"]].text, candidate.text)
                )
            ]
            if supported and len(candidate.text) <= 1024:
                need = max(supported, key=lambda e: e["dense_cosine"])["need_index"]
                best = (0, -need, 0, len(candidate.text))
                semantic = True
            else:
                decisions.append(replace(base, selected=False, reason="wrong_task_facet"))
                continue
        strength, negative_index, start, end = best
        need = -negative_index
        contribution = candidate.text[start:end].strip()
        key = claim_key(contribution)
        normalized_claim = _claim_words(contribution)
        if (key and key in normalized_supplied) or (
            normalized_claim and any(normalized_claim in cue for cue in supplied_claims)
        ):
            decisions.append(
                replace(
                    base,
                    selected=False,
                    reason="already_in_input",
                    supporting_need=need,
                    contribution=contribution,
                    source_spans=((start, end),),
                    shared_terms=strength,
                    strong=True,
                )
            )
            continue
        if key in seen:
            decisions.append(
                replace(
                    base,
                    selected=False,
                    reason="redundant_with",
                    supporting_need=need,
                    contribution=contribution,
                    source_spans=((start, end),),
                    redundant_with=seen[key],
                    shared_terms=strength,
                    strong=True,
                )
            )
            continue
        reason = (
            "semantic_contribution"
            if semantic
            else "uncovered_need" if need not in covered else "complementary_fact"
        )
        kind = candidate.facets.get("schema_class")
        if kind in {"decision", "constraint"}:
            reason = "decision_constraint"
        elif kind == "warning":
            reason = "safety_prerequisite"
        seen[key] = candidate.memory_id
        covered.add(need)
        selected.append(candidate.memory_id)
        decisions.append(
            replace(
                base,
                selected=True,
                reason=reason,
                supporting_need=need,
                contribution=contribution,
                source_spans=((start, end),),
                selected_position=len(selected) - 1,
                shared_terms=strength,
                strong=True,
            )
        )
    return ContributionSelection(tuple(selected), tuple(decisions))
