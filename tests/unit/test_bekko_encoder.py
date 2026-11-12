import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from slowave.symbolic.bekko_encoder import DIM, MODEL, REVISION, BekkoEncoder


def test_bekko_encoder_masks_padding_and_normalizes(monkeypatch, tmp_path):
    (tmp_path / "onnx").mkdir()
    (tmp_path / "config.json").write_text(json.dumps({"pad_token_id": 0}))
    (tmp_path / "special_tokens_map.json").write_text(json.dumps({"pad_token": "[PAD]"}))
    monkeypatch.setattr(
        "slowave.symbolic.bekko_encoder.CONFIG_SHA256",
        hashlib.sha256((tmp_path / "config.json").read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        "slowave.symbolic.bekko_encoder.SPECIAL_TOKENS_SHA256",
        hashlib.sha256((tmp_path / "special_tokens_map.json").read_bytes()).hexdigest(),
    )
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "onnx" / "model.onnx").write_bytes(b"test weights")
    monkeypatch.setattr(
        "slowave.symbolic.bekko_encoder.WEIGHT_SHA256", hashlib.sha256(b"test weights").hexdigest()
    )
    monkeypatch.setattr(
        "slowave.symbolic.bekko_encoder.TOKENIZER_SHA256", hashlib.sha256(b"{}").hexdigest()
    )

    class Tokenizer:
        @classmethod
        def from_file(cls, path):
            return cls()

        def enable_truncation(self, **kwargs):
            pass

        def encode_batch(self, texts):
            return [SimpleNamespace(ids=[1, 2], attention_mask=[1, 0]) for _ in texts]

    class Session:
        def __init__(self, path, **kwargs):
            self.path = path

        def get_inputs(self):
            return [SimpleNamespace(name="input_ids"), SimpleNamespace(name="attention_mask")]

        def get_outputs(self):
            return [SimpleNamespace(name="last_hidden_state")]

        def run(self, unused, feeds):
            assert feeds["input_ids"].dtype == np.int64
            tokens = np.zeros((1, 2, DIM), dtype=np.float32)
            tokens[0, 0, 0] = 2
            tokens[0, 1, 1] = 9
            return [tokens]

    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=Tokenizer))
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(InferenceSession=Session, SessionOptions=SimpleNamespace),
    )
    encoder = BekkoEncoder(tmp_path)
    result = encoder.encode("a")
    assert result.shape == (DIM,)
    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(0.0)
    assert encoder.embedding_spec()["model"] == MODEL
    assert encoder.embedding_spec()["revision"] == REVISION
    assert encoder.encode_many([]).shape == (0, DIM)


def test_bekko_encoder_rejects_wrong_weights(monkeypatch, tmp_path):
    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx" / "model.onnx").write_bytes(b"wrong weights")
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace())
    with pytest.raises(ValueError, match="weight file"):
        BekkoEncoder(tmp_path)
