import json

import numpy as np
import pytest

from slowave.core.config import SlowaveConfig
from slowave.core.engine import RememberResult, SlowaveEngine


class _StubEncoder:
    def __init__(self, dim: int = 32):
        self._dim = dim

    def encode(self, text: str) -> np.ndarray:
        seed = int(abs(hash(text)) % (2**31))
        v = np.random.default_rng(seed).standard_normal(self._dim).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-12)


def _make_engine(tmp_path, dim: int = 32) -> SlowaveEngine:
    eng = SlowaveEngine(
        SlowaveConfig(db_path=str(tmp_path / "test.db"), dim=dim, disable_encoder=True)
    )
    eng.encoder = _StubEncoder(dim)
    return eng


@pytest.fixture()
def eng(tmp_path):
    engine = _make_engine(tmp_path)
    yield engine
    engine.close()


def test_remember_result_is_backward_compatible_int(eng):
    result = eng.remember(content="In project atlas, use PostgreSQL.", type="decision")

    assert isinstance(result, int)
    assert isinstance(result, RememberResult)
    assert int(result) == result.event_id
    assert result == result.event_id

    assert result.event_id > 0
    assert result.schema_id > 0
    assert result.created_schema is not None
    assert result.disposition == "created"
    assert result.created_schema.id == result.schema_id


def test_remember_result_serializes_as_int_for_existing_payloads(eng):
    result = eng.remember(content="The user prefers concise technical answers.", type="preference")

    payload = {"event_id": result, "type": "preference"}

    assert json.loads(json.dumps(payload)) == {
        "event_id": result.event_id,
        "type": "preference",
    }


def test_remember_result_as_dict_is_json_friendly(eng):
    result = eng.remember(content="Always run tests before release.", type="habit")

    assert result.as_dict() == {
        "event_id": result.event_id,
        "schema_id": result.schema_id,
        "disposition": result.disposition,
    }


def test_exact_duplicate_reports_matched_without_claiming_creation(eng):
    first = eng.remember(content="Use stable identifiers.", type="decision")
    second = eng.remember(content="Use stable identifiers.", type="decision")

    assert first.disposition == "created"
    assert second.disposition == "matched"
    assert second.schema_id == first.schema_id
    assert second.created_schema is None
