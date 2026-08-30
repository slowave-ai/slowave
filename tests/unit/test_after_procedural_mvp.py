from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "private" / "experiments" / "after_procedural_mvp.py"
)
SPEC = importlib.util.spec_from_file_location("after_procedural_mvp", MODULE_PATH)
assert SPEC and SPEC.loader
mvp = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mvp
SPEC.loader.exec_module(mvp)


class FakeEncoder:
    dim = 4

    VECTORS = {
        "test a": [1.0, 0.0, 0.0, 0.0],
        "test b": [0.98, 0.02, 0.0, 0.0],
        "debug a": [0.0, 1.0, 0.0, 0.0],
        "debug b": [0.02, 0.98, 0.0, 0.0],
        "verify behavior\narrange act assert\ncover edge cases": [1.0, 0.0, 0.0, 0.0],
        "verify variants\narrange act assert\ncover edge cases": [0.98, 0.02, 0.0, 0.0],
        "find cause\nreproduce isolate fix\nverify regression": [0.0, 1.0, 0.0, 0.0],
        "repair cause\nreproduce isolate fix\nverify regression": [0.02, 0.98, 0.0, 0.0],
        "arrange act assert": [1.0, 0.0, 0.0, 0.0],
        "cover edge cases": [0.95, 0.05, 0.0, 0.0],
        "reproduce isolate fix": [0.0, 1.0, 0.0, 0.0],
        "verify regression": [0.05, 0.95, 0.0, 0.0],
        "new testing task": [1.0, 0.0, 0.0, 0.0],
        "new unrelated task": [0.0, 0.0, 1.0, 0.0],
        "verify behavior\nverify variants\narrange act assert": [1.0, 0.0, 0.0, 0.0],
    }

    def encode_many(self, texts):
        return np.asarray([self.VECTORS[text] for text in texts], dtype=np.float32)


def exp(task, instruction, goal, step, lesson):
    return mvp.Experience(task, instruction, goal, [step], [step], [lesson])


def test_build_candidates_is_blind_and_separates_two_procedures():
    experiences = [
        exp("t1", "test a", "verify behavior", "arrange act assert", "cover edge cases"),
        exp("t2", "test b", "verify variants", "arrange act assert", "cover edge cases"),
        exp("d1", "debug a", "find cause", "reproduce isolate fix", "verify regression"),
        exp("d2", "debug b", "repair cause", "reproduce isolate fix", "verify regression"),
    ]
    candidates = mvp.build_candidates(experiences, FakeEncoder(), threshold=0.8)
    assert len(candidates) == 2
    assert {frozenset(c.member_task_ids) for c in candidates} == {
        frozenset({"t1", "t2"}),
        frozenset({"d1", "d2"}),
    }


def test_retrieve_candidate_matches_and_abstains():
    experiences = [
        exp("t1", "test a", "verify behavior", "arrange act assert", "cover edge cases"),
        exp("t2", "test b", "verify variants", "arrange act assert", "cover edge cases"),
    ]
    candidates = mvp.build_candidates(experiences, FakeEncoder(), threshold=0.8)
    hit, score = mvp.retrieve_candidate(
        "new testing task", candidates, FakeEncoder(), threshold=0.5
    )
    assert hit is not None and score > 0.9
    miss, score = mvp.retrieve_candidate(
        "new unrelated task", candidates, FakeEncoder(), threshold=0.5
    )
    assert miss is None and score == 0.0


def test_pairwise_f1_penalizes_merge_and_fragmentation():
    gold = {"a": "x", "b": "x", "c": "y", "d": "y"}
    perfect = mvp.pairwise_f1([["a", "b"], ["c", "d"]], gold)
    merged = mvp.pairwise_f1([["a", "b", "c", "d"]], gold)
    fragmented = mvp.pairwise_f1([["a"], ["b"], ["c", "d"]], gold)
    assert perfect == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert merged["precision"] < 1.0
    assert fragmented["recall"] < 1.0


def test_frozen_selection_is_twenty_unique_tasks():
    selected = mvp.selected_tasks()
    ids = [rel for _, _, rel in selected]
    assert len(selected) == 20
    assert len(set(ids)) == 20
    assert sum(split == "train" for _, split, _ in selected) == 12
    assert sum(split == "eval" for _, split, _ in selected) == 8


def test_agent_prompt_enforces_after_output_boundary():
    prompt = mvp.agent_prompt("build the artifact", "no guidance")
    assert "every deliverable under `output/`" in prompt
    assert "Never place\n  deliverables only at the workspace root" in prompt


def test_agent_tools_enforce_after_output_boundary(tmp_path):
    (tmp_path / "environment").mkdir()
    result = mvp.execute_agent_tool(
        tmp_path, "write_file", {"path": "package/module.py", "content": "value = 1\n"}
    )
    assert result.startswith("WROTE output/package/module.py")
    assert (tmp_path / "output" / "package" / "module.py").read_text() == "value = 1\n"

    rejected = mvp.execute_agent_tool(
        tmp_path, "write_file", {"path": "environment/input.py", "content": "changed\n"}
    )
    assert rejected.startswith("ERROR: benchmark inputs are read-only")

    mvp.execute_agent_tool(
        tmp_path, "write_file", {"path": "solution.py", "content": "print('ok')\n"}
    )
    assert (tmp_path / "solution.py").is_file()
