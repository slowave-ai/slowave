"""Contribution selection is independently testable and has no page-size cap."""

from dataclasses import asdict

import pytest

from slowave.core.activation_selection import (
    ActivationTask,
    ContributionCandidate,
    select_complementary,
)
from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine


def candidate(sid, text, *, score=1.0, eligible=True, needs=(0,), facets=None):
    return ContributionCandidate(
        sid,
        text,
        score,
        eligible,
        "eligible" if eligible else "inactive",
        needs,
        ({"need_index": 0, "relevance_passed": bool(needs), "dense_cosine": None},),
        facets or {},
    )


def test_task_representation_deduplicates_votes_and_keeps_provenance():
    task = ActivationTask.build(
        "Review migration timeout",
        " review  migration timeout ",
        "Avoid worker overlap",
        {"environment": "production"},
    )
    assert len(task.needs) == 3
    assert task.needs[0].provenance == ("task", "initial_goal")
    assert "production" in " ".join(task.cues)
    assert task.metadata == {"environment": "production"}


def test_one_need_can_keep_distinct_constraints_with_no_catalog_cap():
    task = ActivationTask.build("Validate migration timeout settings")
    items = [
        candidate(i, f"Migration timeout for shard {i} must stay at {i+10} seconds.")
        for i in range(100)
    ]
    result = select_complementary(task, items)
    assert result.selected_ids == tuple(range(100))
    assert result.decisions[0].reason == "uncovered_need"
    assert result.decisions[-1].reason == "complementary_fact"
    assert result.decisions[-1].selected_position == 99


def test_scope_name_or_rank_alone_cannot_prove_task_value():
    task = ActivationTask.build("Prepare Atlas deployment")
    result = select_complementary(
        task,
        [candidate(1, "Atlas deployment announcement uses the orange banner.", score=100)],
        scope="project:atlas",
    )
    assert result.selected_ids == ()
    assert result.decisions[0].reason == "wrong_task_facet"


def test_ineligible_and_failed_evidence_are_not_rescued_by_warning_type():
    task = ActivationTask.build("Validate migration timeout")
    result = select_complementary(
        task,
        [
            candidate(1, "Migration timeout is 12 seconds.", eligible=False),
            candidate(
                2, "Migration timeout is 12 seconds.", needs=(), facets={"schema_class": "warning"}
            ),
        ],
    )
    assert result.selected_ids == ()
    assert [d.reason for d in result.decisions] == ["ineligible", "insufficient_evidence"]


def test_exact_claim_duplicates_keep_strongest_representative_and_parameters_survive():
    task = ActivationTask.build("Validate migration timeout")
    result = select_complementary(
        task,
        [
            candidate(1, "Migration timeout is 12 seconds.", score=0.5),
            candidate(2, " Migration   timeout is 12 seconds.", score=0.9),
            candidate(3, "Migration timeout is 7 seconds.", score=0.7),
        ],
    )
    assert result.selected_ids == (2, 3)
    duplicate = next(d for d in result.decisions if d.memory_id == 1)
    assert duplicate.reason == "redundant_with"
    assert duplicate.redundant_with == 2


def test_structured_conflicts_reject_but_missing_fields_are_not_conflicts():
    task = ActivationTask.build(
        "Validate migration timeout", task_context={"environment": "production"}
    )
    result = select_complementary(
        task,
        [
            candidate(1, "Migration timeout is 12 seconds.", facets={"environment": "test"}),
            candidate(2, "Migration timeout is 7 seconds."),
        ],
    )
    assert result.selected_ids == (2,)
    conflicting = next(d for d in result.decisions if d.memory_id == 1)
    assert conflicting.reason == "structured_facet_conflict"


def test_fact_already_supplied_is_not_reintroduced():
    task = ActivationTask.build(
        "Validate migration timeout", semantic_context="Migration timeout is 12 seconds."
    )
    result = select_complementary(task, [candidate(1, "Migration timeout is 12 seconds.")])
    assert result.selected_ids == ()
    assert result.decisions[0].reason == "already_in_input"


def test_contribution_after_500_characters_preserves_exact_qualifying_context():
    task = ActivationTask.build("Validate migration lock timeout")
    clause = "For Atlas, if the worker is active, do not run migration; the lock timeout must stay at 12 seconds."
    text = ("An unrelated historical note about the office garden. " * 14) + clause
    result = select_complementary(task, [candidate(1, text)])
    assert result.selected_ids == (1,)
    decision = result.decisions[0]
    start, end = decision.source_spans[0]
    assert start > 500
    assert text[start:end].strip() == clause == decision.contribution
    assert "do not" in decision.contribution


def test_service_records_all_evaluated_evidence_and_preserves_legacy_policy(tmp_path):
    eng = SlowaveEngine(
        SlowaveConfig(db_path=str(tmp_path / "contributions.db"), dim=8, disable_encoder=True)
    )
    try:
        ids = [
            eng.schemas.create(
                content_text=f"Migration lock timeout for shard {i} is {i+12} seconds.",
                facets={"schema_class": "constraint"},
                embedding=None,
                scope_id="project:test",
                dedupe=False,
            )
            for i in range(8)
        ]
        task = ActivationTask.build("Validate migration lock timeout")
        legacy = eng.relevant_catalog(task.cues, scope="project:test")
        assert len(legacy.items) == 1
        widened = eng.relevant_catalog(
            task.cues,
            scope="project:test",
            selection_mode="complementary_activation",
            activation_task=task,
        )
        assert {item.schema.id for item in widened.items} == set(ids)
        assert len(widened.decisions) == widened.candidate_count == 8
        assert all(d.channel_evidence[0]["lexical_rank"] is not None for d in widened.decisions)
        assert all(d.channel_evidence[0]["dense_cosine"] is None for d in widened.decisions)
        assert all(d.source_spans for d in widened.decisions)
        assert all(d.reason == "decision_constraint" for d in widened.decisions)
        assert asdict(widened.decisions[0])["contribution"]
        with pytest.raises(ValueError, match="explicit ActivationTask"):
            eng.relevant_catalog(
                task.cues, scope="project:test", selection_mode="complementary_activation"
            )
    finally:
        eng.close()


def test_complementary_delivery_orders_by_contribution_strength(tmp_path):
    """Relevant contributions lead; a topic neighbor is excluded, not appended."""
    eng = SlowaveEngine(
        SlowaveConfig(db_path=str(tmp_path / "delivery_order.db"), dim=8, disable_encoder=True)
    )
    try:
        task = ActivationTask.build("Validate the PostgreSQL migration lock timeout")
        contents = [
            "The migration lock saga retries restarts.",
            "The migration lock timeout is 12 seconds.",
            "The PostgreSQL migration lock timeout is 12 seconds.",
        ]
        ids = [
            eng.schemas.create(
                content_text=content,
                facets={},
                embedding=None,
                scope_id="project:test",
                dedupe=False,
            )
            for content in contents
        ]
        catalog = eng.relevant_catalog(
            task.cues,
            scope="project:test",
            selection_mode="complementary_activation",
            activation_task=task,
        )
        assert [item.schema.id for item in catalog.items] == [ids[2], ids[1]]
        rejected = next(d for d in catalog.decisions if d.memory_id == ids[0])
        assert rejected.selected is False
        assert rejected.reason == "wrong_task_facet"
    finally:
        eng.close()


def test_generated_goal_cannot_independently_admit_unrelated_history():
    task = ActivationTask.build(
        "ok start using slowave meanwhile",
        "Resume using Slowave memory and verify the connected lifecycle",
    )
    item = candidate(1, "The lifecycle history describes memory verification.", needs=(1,))
    result = select_complementary(task, [item], scope="project:slowave-private")
    assert result.selected_ids == ()


def test_scope_only_task_abstains_even_when_dense_similarity_is_high():
    task = ActivationTask.build("ok start using slowave meanwhile")
    item = ContributionCandidate(
        1,
        "Installed the Slowave wheel and verified service health.",
        1.0,
        True,
        "eligible",
        (0,),
        ({"need_index": 0, "dense_cosine": 0.99},),
        {},
    )
    assert select_complementary(task, [item], scope="project:slowave-private").selected_ids == ()


def test_strong_cross_language_evidence_survives_without_lexical_overlap():
    task = ActivationTask.build("Cosa devo fare prima di committare codice?")
    item = ContributionCandidate(
        1,
        "Run the linter before committing code.",
        1.0,
        True,
        "eligible",
        (0,),
        ({"need_index": 0, "dense_cosine": 0.7},),
        {},
    )
    result = select_complementary(task, [item])
    assert result.selected_ids == (1,)
    assert result.decisions[0].reason == "semantic_contribution"
    assert result.decisions[0].contribution == item.text


def test_weak_dense_similarity_does_not_rescue_wrong_facet():
    task = ActivationTask.build("Which platform stores the candidate search index?")
    item = ContributionCandidate(
        1,
        "The candidate search export format is NDJSON.",
        1.0,
        True,
        "eligible",
        (0,),
        ({"need_index": 0, "dense_cosine": 0.59},),
        {},
    )
    assert select_complementary(task, [item]).selected_ids == ()


def test_singular_question_can_retrieve_multiple_independent_constraints():
    task = ActivationTask.build("What migration timeout constraints apply?")
    result = select_complementary(
        task,
        [
            candidate(1, "Migration timeout constraints for shard A require 12 seconds."),
            candidate(2, "Migration timeout constraints for shard B require 30 seconds."),
        ],
    )
    assert result.selected_ids == (1, 2)


def test_dotted_paths_do_not_split_a_relevant_build_requirement():
    task = ActivationTask.build(
        "Which files must be rebuilt so changes to the dashboard React source appear in the running dashboard?"
    )
    text = (
        "Editing dashboard/ui/src/* (React + Vite) has no effect on the running dashboard "
        "until npm run build is run; vite.config.ts sets outDir to ../static, so the build "
        "replaces hashed asset files and index.html must accompany the new assets."
    )
    result = select_complementary(task, [candidate(1, text)])
    assert result.selected_ids == (1,)
    assert result.decisions[0].contribution == text


def test_supplied_context_is_not_reexposed_but_opposite_claim_is_preserved():
    task = ActivationTask.build(
        "Review the blue environment",
        semantic_context="The Atlas deployment already runs in the blue environment.",
    )
    result = select_complementary(
        task,
        [
            candidate(1, "Atlas deployment runs in the blue environment.", needs=(0, 1)),
            candidate(2, "Atlas deployment does not run in the blue environment.", needs=(0, 1)),
        ],
        scope="project:atlas",
    )
    assert result.selected_ids == (2,)
    assert result.decisions[0].reason == "already_in_input"


def test_imperative_prose_has_same_lexical_admission_as_a_question():
    text = "Migration lock timeout must remain 12 seconds."
    command = ActivationTask.build(
        "Investigate migration lock timeout failures and explain the blocking behavior"
    )
    question = ActivationTask.build(
        "What migration lock timeout failures explain the blocking behavior?"
    )
    assert select_complementary(command, [candidate(1, text)]).selected_ids == (1,)
    assert select_complementary(question, [candidate(1, text)]).selected_ids == (1,)


def test_structured_context_supports_a_user_declared_need():
    task = ActivationTask.build("Investigate this failure", task_context={"component": "migration"})
    item = candidate(1, "Migration requires a 12 second lock timeout.")
    assert select_complementary(task, [item]).selected_ids == (1,)


def test_same_language_semantic_fallback_accepts_point_six():
    task = ActivationTask.build("Prevent another accidental commit after switching branches")
    item = ContributionCandidate(
        1,
        "Check the checkout before editing files.",
        1,
        True,
        "eligible",
        (0,),
        ({"need_index": 0, "dense_cosine": 0.60},),
        {},
    )
    assert select_complementary(task, [item]).selected_ids == (1,)


def test_applicability_recovers_real_branch_warning_without_admitting_history(tmp_path):
    import numpy as np

    warning = "After a user merges a feature, check the current checkout before editing; otherwise fixes can land on env/dev and be absent from the feature PR."
    history = "A different feature was pushed to env/prod last month."

    class Scorer:
        def score(self, queries, memories):
            return np.array(
                [[5.0 if text == warning else -8.0 for _ in queries] for text in memories]
            )

    eng = SlowaveEngine(
        SlowaveConfig(db_path=str(tmp_path / "branch.db"), dim=8, disable_encoder=True)
    )
    try:
        ids = [
            eng.schemas.create(
                content_text=text,
                facets={"source_kind": "explicit_remember"},
                embedding=None,
                scope_id="project:test",
                dedupe=False,
            )
            for text in (warning, history)
        ]
        eng._retrieval._applicability_scorer = Scorer()
        task = ActivationTask.build(
            "PR merged; go back to the feature branch, pull latest from env/prod, and push to remote."
        )
        catalog = eng.relevant_catalog(
            task.cues,
            scope="project:test",
            selection_mode="complementary_activation",
            activation_task=task,
        )
        assert [item.schema.id for item in catalog.items] == [ids[0]]
        assert catalog.applicability_status == "applied"
        assert next(d for d in catalog.decisions if d.memory_id == ids[0]).source_spans
    finally:
        eng.close()


def test_model_failure_keeps_semantic_fallback_and_reports_degraded_status(tmp_path):
    class Missing:
        def score(self, queries, memories):
            raise OSError("cached model missing")

    eng = SlowaveEngine(
        SlowaveConfig(db_path=str(tmp_path / "fallback.db"), dim=8, disable_encoder=True)
    )
    try:
        sid = eng.schemas.create(
            content_text="Migration lock timeout must remain 12 seconds.",
            facets={},
            embedding=None,
            scope_id="project:test",
            dedupe=False,
        )
        eng._retrieval._applicability_scorer = Missing()
        task = ActivationTask.build(
            "Investigate migration lock timeout failures and fix the blocking operation"
        )
        catalog = eng.relevant_catalog(
            task.cues,
            scope="project:test",
            selection_mode="complementary_activation",
            activation_task=task,
        )
        assert [item.schema.id for item in catalog.items] == [sid]
        assert catalog.applicability_status == "unavailable"
    finally:
        eng.close()


def test_model_localizes_long_semantic_source_without_prefix_truncation(tmp_path):
    import numpy as np

    clause = "Check the checkout before editing; otherwise fixes can land on env/dev and be absent from the feature PR."
    text = "Unrelated office garden notes. " * 50 + clause

    class Scorer:
        def score(self, queries, memories):
            return np.array(
                [[5.0 if clause in text else -8.0 for _ in queries] for text in memories]
            )

    eng = SlowaveEngine(
        SlowaveConfig(db_path=str(tmp_path / "long.db"), dim=8, disable_encoder=True)
    )
    try:
        sid = eng.schemas.create(
            content_text=text,
            facets={"source_kind": "explicit_remember"},
            embedding=None,
            scope_id="project:test",
            dedupe=False,
        )
        eng._retrieval._applicability_scorer = Scorer()
        task = ActivationTask.build("Return to the feature branch after the user merge and push")
        catalog = eng.relevant_catalog(
            task.cues,
            scope="project:test",
            selection_mode="complementary_activation",
            activation_task=task,
        )
        decision = next(d for d in catalog.decisions if d.memory_id == sid)
        assert decision.selected
        start, end = decision.source_spans[0]
        assert start > 500
        assert clause in text[start:end]
        assert end - start <= 1024
    finally:
        eng.close()


def test_branch_context_excludes_other_feature_history_but_keeps_portable_warning(tmp_path):
    import numpy as np

    class Scorer:
        def score(self, queries, memories):
            return np.full((len(memories), len(queries)), 5.0)

    eng = SlowaveEngine(
        SlowaveConfig(db_path=str(tmp_path / "branches.db"), dim=8, disable_encoder=True)
    )
    try:
        history = eng.schemas.create(
            content_text="The feat/old-widget branch was merged into env/prod.",
            facets={"schema_class": "fact"},
            embedding=None,
            scope_id="project:test",
            dedupe=False,
        )
        warning = eng.schemas.create(
            content_text="After merging feat/old-widget, check the checkout before editing to avoid committing fixes to the shared branch.",
            facets={"schema_class": "lesson", "source_kind": "explicit_remember"},
            embedding=None,
            scope_id="project:test",
            dedupe=False,
        )
        stale = eng.schemas.create(
            content_text="The feat/older branch was merged into env/prod.",
            facets={"schema_class": "fact", "injectable": False},
            embedding=None,
            scope_id="project:test",
            dedupe=False,
        )
        eng._retrieval._applicability_scorer = Scorer()
        task = ActivationTask.build(
            "Return to the feature branch after the merge and push",
            "Merge env/prod into fix/current-widget and push",
        )
        catalog = eng.relevant_catalog(
            task.cues,
            scope="project:test",
            selection_mode="complementary_activation",
            activation_task=task,
        )
        assert [item.schema.id for item in catalog.items] == [warning]
        assert (
            next(d for d in catalog.decisions if d.memory_id == history).reason
            == "structured_facet_conflict"
        )
        assert next(d for d in catalog.decisions if d.memory_id == stale).reason == "ineligible"
    finally:
        eng.close()


def test_bare_read_action_does_not_consume_startup_tokens(tmp_path):
    import numpy as np

    class Scorer:
        def score(self, queries, memories):
            return np.full((len(memories), len(queries)), 5.0)

    eng = SlowaveEngine(
        SlowaveConfig(db_path=str(tmp_path / "narration.db"), dim=8, disable_encoder=True)
    )
    try:
        narration = eng.schemas.create(
            content_text="Read the current dashboard metrics in README.md.",
            facets={},
            embedding=None,
            scope_id="project:test",
            dedupe=False,
        )
        useful = eng.schemas.create(
            content_text="The dashboard metrics exclude episodes because customers need durable memories and procedures.",
            facets={"source_kind": "explicit_remember"},
            embedding=None,
            scope_id="project:test",
            dedupe=False,
        )
        eng._retrieval._applicability_scorer = Scorer()
        task = ActivationTask.build("Assess dashboard metrics and propose user-facing improvements")
        catalog = eng.relevant_catalog(
            task.cues,
            scope="project:test",
            selection_mode="complementary_activation",
            activation_task=task,
        )
        assert [item.schema.id for item in catalog.items] == [useful]
        assert (
            next(d for d in catalog.decisions if d.memory_id == narration).reason
            == "insufficient_evidence"
        )
    finally:
        eng.close()


def test_numbered_task_facet_rejects_observation_of_another_hop(tmp_path):
    import numpy as np

    class Scorer:
        def score(self, queries, memories):
            return np.full((len(memories), len(queries)), 5.0)

    eng = SlowaveEngine(
        SlowaveConfig(db_path=str(tmp_path / "hops.db"), dim=8, disable_encoder=True)
    )
    try:
        ids = [
            eng.schemas.create(
                content_text=text,
                facets={"source_kind": "explicit_remember", "schema_class": "lesson"},
                embedding=None,
                scope_id="project:test",
                dedupe=False,
            )
            for text in (
                "Hop 9 failed because no query fetched the current cash position.",
                "Hop 10 failed because the maximum tool count was exhausted.",
            )
        ]
        eng._retrieval._applicability_scorer = Scorer()
        task = ActivationTask.build(
            "What is the new hop 9 failure about and should it be addressed here?"
        )
        catalog = eng.relevant_catalog(
            task.cues,
            scope="project:test",
            selection_mode="complementary_activation",
            activation_task=task,
        )
        assert [item.schema.id for item in catalog.items] == [ids[0]]
        assert (
            next(d for d in catalog.decisions if d.memory_id == ids[1]).reason
            == "structured_facet_conflict"
        )
    finally:
        eng.close()
