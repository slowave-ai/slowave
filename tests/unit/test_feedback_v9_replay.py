from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine

_SCRIPT = Path(__file__).parents[2] / "scripts" / "validate_feedback_v9_replay.py"
_SPEC = importlib.util.spec_from_file_location("validate_feedback_v9_replay", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
report = _MODULE.report


def test_replay_recommends_zero_start_for_outcome_coupled_history(tmp_path) -> None:
    db_path = tmp_path / "replay.db"
    eng = SlowaveEngine(SlowaveConfig(db_path=str(db_path), dim=8, disable_encoder=True))
    try:
        eng.record_retrieval(
            retrieval_id="rec_replay",
            response={"schemas": [{"id": "sch_1"}]},
        )
        conn = eng.db.connect()
        conn.execute(
            "INSERT INTO context_feedback_events (context_id, feedback, outcome, "
            "feedback_signal_json, outcome_reward, used_memory_ids_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("rec_replay", "useful", "success", json.dumps({}), 1.0, '["sch_1"]', 1),
        )
        conn.commit()
    finally:
        eng.close()
    result = report(str(db_path))
    assert result["integrity"] == "ok"
    assert result["outcome_coupled_legacy_rows"] == 1
    assert result["safe_for_historical_backfill"] is False
    assert result["recommended_migration"] == "zero_start"
