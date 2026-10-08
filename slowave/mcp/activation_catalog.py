"""Pure preparation and versioned paging shared by activation and recall.

This module does not publish sessions, cursors or exposure. Callers must first
prepare the complete catalog, then atomically publish its first page. Legacy
continuation lists remain the responsibility of their existing reader.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

POLICY_VERSION = "activation-pool-relative-v1"
SUPPORTED_POLICIES = {
    POLICY_VERSION,
    "recall-pool-relative-v1",
    # Historical identities: stored cursors and snapshots must keep loading.
    "activation-complementary-v1",
    "relevant-set-v2",
    "shared-multilingual-v1",
    "activate-hybrid-dogfood-v1",
    "multilingual-retrieval-baseline-v1",
}
# v1 snapshots predate the variable first page and always served five
# memories there; v2 records the strong-run first page size explicitly.
SUPPORTED_PAGING_CONTRACT_VERSIONS = (
    "configurable-memory-budget-pages-v5",
    "ten-memory-budget-pages-v4",
    "payload-budget-pages-v3",
    "strong-run-first-page-v2",
    "five-memory-pages-v1",
)
PAGING_CONTRACT_VERSION = "configurable-memory-budget-pages-v5"
SNAPSHOT_FORMAT_VERSION = 1
# Legacy snapshot record-count bounds, retained only for cursor compatibility.
MEMORY_PAGE_SIZE = 5
# Effective page size for new activation/recall requests when clients omit it.
DEFAULT_MEMORY_PAGE_SIZE = 5
MEMORY_PAGE_TARGET = 10
# Legacy v2/v4 limits. New snapshots freeze a configurable 1–10 memory bound
# (defaulted by DEFAULT_MEMORY_PAGE_SIZE) alongside payload budgets; old
# cursors keep their old bounds.
FIRST_PAGE_MAX_MEMORIES = 10
PROCEDURE_PAGE_SIZE = 3
# Bound serialized content, never the number of qualifying records. Separate
# section budgets keep procedure size from displacing declarative context.
MEMORY_PAGE_CHARS = 12_288
PROCEDURE_PAGE_CHARS = 24_576
MAX_PREVIEW_CHARS = 1024
MAX_MEMORY_RECORD_CHARS = 8192
MAX_PROCEDURE_RECORD_CHARS = 16384
MAX_METADATA_CHARS = 4096
# Worst-case escaped records plus metadata and JSON section framing. Bounds
# apply to serialized JSON, rather than unescaped source character counts.
MAX_PAGE_CHARS = (
    FIRST_PAGE_MAX_MEMORIES * MAX_MEMORY_RECORD_CHARS
    + PROCEDURE_PAGE_SIZE * MAX_PROCEDURE_RECORD_CHARS
    + MAX_METADATA_CHARS
    + 1024
)
MAX_SNAPSHOT_CHARS = 10_000_000


class CatalogPreparationError(ValueError):
    """The catalog cannot be represented honestly inside its declared bounds."""


class CatalogCompatibilityError(ValueError):
    """The persisted format/contract needs a capable reader; retry is safe."""


def _encode(value: Any) -> str:
    # Escaping is the larger envelope, so an ensure_ascii=False transport is
    # also bounded. Reject NaN/Infinity before any caller starts publication.
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _bounded(value: Any, limit: int, label: str) -> None:
    if len(_encode(value)) > limit:
        raise CatalogPreparationError(f"unsupported oversized {label}")


def _budget_end(
    records: list[dict[str, Any]], offset: int, budget: int, limit: int | None = None
) -> int:
    """Pack a stable prefix by serialized size, with guaranteed forward progress."""
    end, size = offset, 2  # JSON list brackets
    while end < len(records):
        if limit is not None and end - offset >= limit:
            break
        record_size = len(_encode(records[end])) + (1 if end > offset else 0)
        if end > offset and size + record_size > budget:
            break
        # Individually bounded records may exceed the normal section budget;
        # carry one intact rather than drop content or strand the cursor.
        size += record_size
        end += 1
    return end


def source_preview(
    source: str, spans: tuple[tuple[int, int], ...]
) -> tuple[str, dict[str, Any] | None]:
    """Retain all admitted contiguous source contributions without new claims.

    Short sources remain verbatim. Longer sources require validated excerpts;
    each excerpt is separated by a newline and carries its original offsets.
    Conditions and negations are the selector's responsibility to include in
    each complete contribution span. No silent prefix truncation is permitted.
    """
    for start, end in spans:
        if not (
            isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(source)
        ):
            raise CatalogPreparationError("invalid contribution source span")
    if len(source) <= MAX_PREVIEW_CHARS:
        return source, None
    if not spans:
        raise CatalogPreparationError("long selected memory requires contribution source spans")
    ordered = sorted(set(spans))
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if not (0 <= start < end <= len(source)):
            raise CatalogPreparationError("invalid contribution source span")
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    preview = "\n".join(source[start:end] for start, end in merged)
    if len(preview) > MAX_PREVIEW_CHARS:
        raise CatalogPreparationError("required source contributions exceed preview bound")
    # This is a versioned provenance indication, not generated claim text.
    return preview, {
        "version": 1,
        "source_chars": len(source),
        "source_spans": [list(span) for span in merged],
    }


@dataclass(frozen=True)
class FrozenActivationCatalog:
    """One immutable JSON snapshot with separate memory/procedure offsets."""

    snapshot_json: str

    @classmethod
    def prepare(
        cls,
        *,
        retrieval_id: str,
        session_id: str,
        scope: str,
        memories: list[dict[str, Any]],
        procedures: list[dict[str, Any]],
        contribution_spans: dict[str, tuple[tuple[int, int], ...]],
        catalog_truncated: bool,
        first_page_metadata: dict[str, Any] | None = None,
        first_page_size: int | None = None,
        page_size: int = DEFAULT_MEMORY_PAGE_SIZE,
        policy_version: str = POLICY_VERSION,
        recall_evidence: dict[str, Any] | None = None,
    ) -> FrozenActivationCatalog:
        if type(page_size) is not int or not 1 <= page_size <= 10:
            raise CatalogPreparationError("page_size must be an integer from 1 to 10")
        if policy_version not in SUPPORTED_POLICIES:
            raise CatalogPreparationError("unsupported retrieval policy")
        metadata = dict(first_page_metadata or {})
        if set(metadata) - {"memory_state", "warnings", "continuity_id", "continuity_state"}:
            raise CatalogPreparationError("activation metadata cannot override catalog fields")
        _bounded(metadata, MAX_METADATA_CHARS, "activation metadata")
        if recall_evidence is not None:
            if set(recall_evidence) != {"evidence", "evidence_mode", "evidence_truncated"}:
                raise CatalogPreparationError("invalid recall evidence fields")
            recall_evidence = dict(recall_evidence)
            recall_evidence["evidence"] = list(recall_evidence["evidence"])
            while len(_encode(recall_evidence)) > MEMORY_PAGE_CHARS and recall_evidence["evidence"]:
                recall_evidence["evidence"].pop()
                recall_evidence["evidence_truncated"] = True
            _bounded(recall_evidence, MEMORY_PAGE_CHARS, "recall evidence")
        page_one = MEMORY_PAGE_SIZE if first_page_size is None else first_page_size
        if not (isinstance(page_one, int) and 1 <= page_one <= FIRST_PAGE_MAX_MEMORIES):
            raise CatalogPreparationError("invalid first page size")
        if (
            not retrieval_id
            or not session_id
            or not scope
            or len(max((retrieval_id, session_id, scope), key=len)) > 256
        ):
            raise CatalogPreparationError("invalid catalog binding")
        frozen_memories = []
        seen: set[str] = set()
        for memory in memories:
            memory_id = memory.get("memory_id")
            if (
                not isinstance(memory_id, str)
                or not memory_id
                or memory_id in seen
                or len(memory_id) > 256
            ):
                raise CatalogPreparationError("invalid or duplicate selected memory ID")
            seen.add(memory_id)
            value = dict(memory)
            source = value.get("content")
            if not isinstance(source, str):
                raise CatalogPreparationError("selected memory content must be a string")
            preview, indication = source_preview(source, contribution_spans.get(memory_id, ()))
            value["content"] = preview
            if indication:
                provenance = dict(value.get("provenance") or {})
                provenance["source_excerpt"] = indication
                value["provenance"] = provenance
            _bounded(value, MAX_MEMORY_RECORD_CHARS, "memory record")
            frozen_memories.append(value)
        frozen_procedures = []
        for procedure in procedures:
            pid = procedure.get("procedure_id")
            if not isinstance(pid, str) or not pid or pid in seen or len(pid) > 256:
                raise CatalogPreparationError("invalid or duplicate selected procedure ID")
            seen.add(pid)
            # Preserve already-selected bounded previews and complete safety
            # caveats. Unsupported content fails the entire preparation.
            _bounded(procedure, MAX_PROCEDURE_RECORD_CHARS, "procedure record")
            frozen_procedures.append(dict(procedure))
        snapshot = {
            "originating_policy_version": policy_version,
            "paging_contract_version": (
                PAGING_CONTRACT_VERSION if first_page_size is None else "strong-run-first-page-v2"
            ),
            "memory_page_chars": MEMORY_PAGE_CHARS,
            "procedure_page_chars": PROCEDURE_PAGE_CHARS,
            "snapshot_format_version": SNAPSHOT_FORMAT_VERSION,
            "memory_page_size": MEMORY_PAGE_SIZE,
            "requested_page_size": page_size,
            "first_page_size": page_one,
            "procedure_page_size": PROCEDURE_PAGE_SIZE,
            "retrieval_id": retrieval_id,
            "session_id": session_id,
            "scope": scope,
            "relevant_total": len(frozen_memories),
            "catalog_truncated": bool(catalog_truncated),
            "section_order": ["memories", "procedures"],
            "first_page_metadata": metadata,
            "memories": frozen_memories,
            "procedures": frozen_procedures,
        }
        if recall_evidence is not None:
            snapshot["recall_evidence"] = recall_evidence
        _bounded(snapshot, MAX_SNAPSHOT_CHARS, "activation snapshot")
        catalog = cls(_encode(snapshot))
        # Preflight every page including the tail; a later oversized record
        # must not strand a session that has already exposed its first page.
        offset = (0, 0)
        while True:
            _, successor = catalog.page(*offset)
            if successor is None:
                break
            offset = successor
        return catalog

    @classmethod
    def read(cls, value: str) -> FrozenActivationCatalog:
        catalog = cls(value)
        catalog._snapshot()
        return catalog

    def _snapshot(self) -> dict[str, Any]:
        snapshot = json.loads(self.snapshot_json)
        if not isinstance(snapshot, dict):
            raise CatalogCompatibilityError("legacy continuation requires legacy reader")
        if snapshot.get("originating_policy_version") not in SUPPORTED_POLICIES:
            raise CatalogCompatibilityError("unsupported retrieval policy")
        expected = {
            "snapshot_format_version": SNAPSHOT_FORMAT_VERSION,
            "memory_page_size": MEMORY_PAGE_SIZE,
            "procedure_page_size": PROCEDURE_PAGE_SIZE,
            "section_order": ["memories", "procedures"],
        }
        if any(snapshot.get(key) != value for key, value in expected.items()):
            raise CatalogCompatibilityError(
                "unsupported activation snapshot or paging contract version"
            )
        expected_paging = snapshot.get("paging_contract_version")
        if expected_paging not in SUPPORTED_PAGING_CONTRACT_VERSIONS:
            raise CatalogCompatibilityError(
                "unsupported activation snapshot or paging contract version"
            )
        if expected_paging == PAGING_CONTRACT_VERSION:
            page_size = snapshot.get("requested_page_size")
            if type(page_size) is not int or not 1 <= page_size <= 10:
                raise CatalogCompatibilityError("unsupported memory page size")
        if expected_paging in {
            PAGING_CONTRACT_VERSION,
            "ten-memory-budget-pages-v4",
            "payload-budget-pages-v3",
        }:
            if (
                snapshot.get("memory_page_chars") != MEMORY_PAGE_CHARS
                or snapshot.get("procedure_page_chars") != PROCEDURE_PAGE_CHARS
            ):
                raise CatalogCompatibilityError("unsupported activation payload budget")
        # Snapshots prepared before the variable first page carry no
        # first_page_size and served five memories on page one.
        first_page_size = snapshot.get("first_page_size", MEMORY_PAGE_SIZE)
        if not (
            isinstance(first_page_size, int) and 1 <= first_page_size <= FIRST_PAGE_MAX_MEMORIES
        ):
            raise CatalogCompatibilityError("unsupported activation first page size")
        snapshot["first_page_size"] = first_page_size
        return snapshot

    def page(
        self, memory_offset: int = 0, procedure_offset: int = 0
    ) -> tuple[dict[str, Any], tuple[int, int] | None]:
        snapshot = self._snapshot()
        memories, procedures = snapshot["memories"], snapshot["procedures"]
        if not (0 <= memory_offset <= len(memories) and 0 <= procedure_offset <= len(procedures)):
            raise ValueError("invalid activation catalog offsets")
        first_page_size = snapshot["first_page_size"]
        memory_span = first_page_size if memory_offset == 0 else MEMORY_PAGE_SIZE
        if snapshot["paging_contract_version"] in {
            PAGING_CONTRACT_VERSION,
            "ten-memory-budget-pages-v4",
            "payload-budget-pages-v3",
        }:
            limit = (
                snapshot["requested_page_size"]
                if snapshot["paging_contract_version"] == PAGING_CONTRACT_VERSION
                else (
                    MEMORY_PAGE_TARGET
                    if snapshot["paging_contract_version"] == "ten-memory-budget-pages-v4"
                    else None
                )
            )
            memory_end = _budget_end(memories, memory_offset, snapshot["memory_page_chars"], limit)
            procedure_end = _budget_end(
                procedures,
                procedure_offset,
                snapshot["procedure_page_chars"],
                PROCEDURE_PAGE_SIZE if limit is not None else None,
            )
        else:
            memory_end = min(memory_offset + memory_span, len(memories))
            procedure_end = min(procedure_offset + PROCEDURE_PAGE_SIZE, len(procedures))
        more = memory_end < len(memories) or procedure_end < len(procedures)
        data = {
            "retrieval_id": snapshot["retrieval_id"],
            "retrieval_policy_version": snapshot["originating_policy_version"],
            "relevant_total": snapshot["relevant_total"],
            "catalog_truncated": snapshot["catalog_truncated"],
            "memories": memories[memory_offset:memory_end],
            "procedures": procedures[procedure_offset:procedure_end],
            "more_available": more,
        }
        if memory_offset == 0 and procedure_offset == 0:
            data.update(snapshot["first_page_metadata"])
            data["session_id"] = snapshot["session_id"]
        if "recall_evidence" in snapshot:
            evidence = snapshot["recall_evidence"]
            data.update(
                {
                    "evidence": (
                        evidence["evidence"] if memory_offset == 0 and procedure_offset == 0 else []
                    ),
                    "evidence_mode": evidence["evidence_mode"],
                    "evidence_truncated": (
                        evidence["evidence_truncated"]
                        if memory_offset == 0 and procedure_offset == 0
                        else False
                    ),
                }
            )
        remaining_memories, remaining_procedures = memories[memory_end:], procedures[procedure_end:]
        remaining = remaining_memories + remaining_procedures
        data["accessible_field"] = (
            {
                "extra_candidates": len(remaining),
                "kinds": sorted(
                    {str(item.get("pathway") or "memory") for item in remaining_memories}
                    | ({"procedure"} if remaining_procedures else set())
                ),
                "approx_extra_tokens": (sum(len(_encode(item)) for item in remaining) + 3) // 4,
            }
            if remaining
            else {}
        )
        # Reserve final cursor metadata even when the caller allocates it only
        # after preparing this page. Public IDs/cursors have bounded lengths.
        _bounded(
            {"ok": True, "data": {**data, "continue_from": "cur_" + "x" * 64}},
            MAX_PAGE_CHARS,
            "activation page",
        )
        return data, (memory_end, procedure_end) if more else None
