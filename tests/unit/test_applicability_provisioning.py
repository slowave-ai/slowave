"""Asset provisioning must be pinned, cache-first and independent of query text."""

from unittest.mock import Mock

import pytest

from slowave.symbolic import applicability_encoder as module


@pytest.fixture
def backend(monkeypatch):
    import huggingface_hub
    import onnxruntime
    import transformers

    module._backend.cache_clear()
    cached = Mock(return_value="/cache/model.onnx")
    snapshot = Mock(return_value="/cache/snapshot")
    tokenizer = Mock(return_value=object())
    session = Mock(return_value=object())
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", cached)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer)
    monkeypatch.setattr(onnxruntime, "InferenceSession", session)
    yield cached, snapshot, tokenizer, session
    module._backend.cache_clear()


def test_cached_model_never_requests_online_provisioning(backend):
    cached, snapshot, tokenizer, session = backend
    module._backend()
    assert cached.call_args.kwargs["local_files_only"] is True
    assert snapshot.call_args.kwargs["local_files_only"] is True
    assert tokenizer.call_args.kwargs["local_files_only"] is True
    assert cached.call_args.kwargs["revision"] == module.REVISION
    assert session.call_count == 1
    module._backend()
    assert session.call_count == 1


def test_missing_cache_provisions_only_pinned_required_assets(backend):
    from huggingface_hub.errors import LocalEntryNotFoundError

    cached, snapshot, tokenizer, session = backend
    cached.side_effect = LocalEntryNotFoundError("missing")
    module._backend()
    assert snapshot.call_args.args == (module.MODEL,)
    assert snapshot.call_args.kwargs["revision"] == module.REVISION
    patterns = snapshot.call_args.kwargs["allow_patterns"]
    assert len(patterns) == 5
    assert patterns[0].startswith("onnx/model_")
    assert all("pytorch" not in pattern for pattern in patterns)
    assert session.call_args.args[0].startswith("/cache/snapshot/onnx/")
    assert tokenizer.call_args.kwargs["local_files_only"] is True


def test_offline_or_download_failure_propagates_to_visible_fallback(backend):
    from huggingface_hub.errors import LocalEntryNotFoundError

    cached, snapshot, _, session = backend
    cached.side_effect = LocalEntryNotFoundError("missing")
    snapshot.side_effect = LocalEntryNotFoundError("offline")
    with pytest.raises(OSError):
        module._backend()
    assert not session.called


def test_partial_model_cache_with_missing_tokenizer_is_repaired(backend):
    _, snapshot, tokenizer, session = backend
    tokenizer.side_effect = [OSError("tokenizer not cached"), object()]
    module._backend()
    assert "allow_patterns" in snapshot.call_args.kwargs
    assert tokenizer.call_count == 2
    assert session.call_count == 1
