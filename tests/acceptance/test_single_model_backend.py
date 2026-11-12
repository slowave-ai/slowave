"""A server loads exactly one model, and it is the pinned encoder.

Skips when the pinned snapshot is not present locally: the assertion is about
which ONNX sessions a running server opens, not about a fake encoder.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from slowave.symbolic.bekko_encoder import MODEL, model_root


def _model_available() -> bool:
    try:
        model_root()
    except (RuntimeError, OSError):
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _model_available(), reason="pinned Bekko snapshot is not available locally"
)


def test_the_server_opens_one_onnx_session_and_it_is_the_pinned_encoder(monkeypatch) -> None:
    ort = pytest.importorskip("onnxruntime")
    recorded: list[str] = []
    original = ort.InferenceSession

    def audited(path, *args, **kwargs):
        recorded.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(ort, "InferenceSession", audited)

    from slowave.core.config import SlowaveConfig
    from slowave.core.engine import SlowaveEngine
    from slowave.core.services.search_preparation import prepare_search

    path = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
    try:
        engine = SlowaveEngine(SlowaveConfig(db_path=path, dim=384))
        prepare_search(engine)
        engine.remember(
            content="Use tabs, never spaces, in Go files.", type="constraint", scope="project:one"
        )
        engine.relevant_catalog(["format the Go sources"], scope="project:one")

        assert recorded, "the encoder did not load at all"
        assert all(MODEL.split("/")[-1] in entry for entry in recorded), recorded
        assert len(set(recorded)) == 1, recorded
    finally:
        for suffix in ("", "-wal", "-shm"):
            if os.path.exists(path + suffix):
                os.remove(path + suffix)
