from __future__ import annotations

import numpy as np

from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine


class _SameVectorEncoder:
    def encode(self, text: str) -> np.ndarray:
        del text
        vector = np.ones(8, dtype=np.float32)
        return vector / np.linalg.norm(vector)


def test_close_same_scope_remembers_do_not_mutate_existing_truth(tmp_path) -> None:
    eng = SlowaveEngine(
        SlowaveConfig(db_path=str(tmp_path / "encoding.db"), dim=8, disable_encoder=True)
    )
    try:
        eng.encoder = _SameVectorEncoder()
        old = eng.remember(content="The refund window is 30 days.", type="decision")
        eng.remember(content="The refund window is 14 days.", type="decision")

        schema = eng.schemas.get(old.schema_id)
        assert schema.status == "active"
        assert schema.is_labile is False
    finally:
        eng.close()


def test_remember_result_has_no_supersession_signal(tmp_path) -> None:
    eng = SlowaveEngine(
        SlowaveConfig(db_path=str(tmp_path / "result.db"), dim=8, disable_encoder=True)
    )
    try:
        result = eng.remember(content="The refund window is 30 days.", type="decision")
        assert "superseded_schema_ids" not in result.as_dict()
        assert not hasattr(result, "superseded_schema_ids")
    finally:
        eng.close()
