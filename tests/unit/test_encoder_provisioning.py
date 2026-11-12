"""Provisioning of the pinned encoder: cache-first, revision-pinned, verified."""

from __future__ import annotations

from pathlib import Path

import pytest

from slowave.symbolic import bekko_encoder


def test_publication_is_pinned_to_one_revision_and_model() -> None:
    assert bekko_encoder.REVISION and len(bekko_encoder.REVISION) == 40
    assert bekko_encoder.MODEL == "hotchpotch/bekko-embedding-v1-a8m"
    assert bekko_encoder.DIM == 384


def test_cached_snapshot_is_used_without_touching_the_network(monkeypatch, tmp_path) -> None:
    folder = "models--" + bekko_encoder.MODEL.replace("/", "--")
    snapshot = tmp_path / folder / "snapshots" / bekko_encoder.REVISION
    (snapshot / "onnx").mkdir(parents=True)
    (snapshot / "onnx" / "model.onnx").write_bytes(b"weights")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    bekko_encoder.model_root.cache_clear()

    def fail(*args, **kwargs):  # a network call would mean a cache miss
        raise AssertionError("snapshot_download must not be called when the cache is populated")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fail)
    try:
        assert bekko_encoder.model_root() == snapshot
    finally:
        bekko_encoder.model_root.cache_clear()


def test_explicit_model_directory_is_accepted_and_validated(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(bekko_encoder.MODEL_DIR_ENV, str(tmp_path))
    bekko_encoder.model_root.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="onnx/model.onnx"):
            bekko_encoder.model_root()
        (tmp_path / "onnx").mkdir()
        (tmp_path / "onnx" / "model.onnx").write_bytes(b"weights")
        assert bekko_encoder.model_root() == tmp_path
    finally:
        bekko_encoder.model_root.cache_clear()


def test_substituted_weights_are_rejected(tmp_path) -> None:
    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx" / "model.onnx").write_bytes(b"not the pinned weights")
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "special_tokens_map.json").write_text("{}")

    with pytest.raises(ValueError, match="pinned revision"):
        bekko_encoder.BekkoEncoder(Path(tmp_path))
