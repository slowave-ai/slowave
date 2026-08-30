from __future__ import annotations

import numpy as np

from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine


def test_recurrence_can_clear_decay_lability_without_changing_truth(tmp_path) -> None:
    eng = SlowaveEngine(
        SlowaveConfig(db_path=str(tmp_path / "labile.db"), dim=8, disable_encoder=True)
    )
    try:
        vector = np.ones(8, dtype=np.float32)
        vector /= np.linalg.norm(vector)
        schema_id = eng.schemas.create(
            content_text="recurrence recovery schema",
            facets={},
            tags=[],
            embedding=vector,
            is_labile=True,
            dedupe=False,
        )
        eng.schemas.refresh_utility(schema_id)

        for _ in range(3):
            eng.schemas.reinforce(schema_id, amount=0.05)

        schema = eng.schemas.get(schema_id)
        assert schema.is_labile is False
        assert schema.status == "active"
    finally:
        eng.close()
