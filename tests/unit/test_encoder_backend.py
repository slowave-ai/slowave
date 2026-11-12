"""The encoder layer: one pinned backend, bounded cache, versioned vectors."""

from __future__ import annotations

import numpy as np

from slowave.symbolic.encoder import EncoderConfig, TextEncoder


class Fake:
    """Deterministic stand-in with the same surface as the pinned backend."""

    dim = 4
    model_name = "fake"
    model_revision = "rev"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def encode_many(self, texts):
        self.calls.extend(texts)
        rows = []
        for index, _text in enumerate(texts):
            vector = np.zeros(4, dtype=np.float32)
            vector[index % 4] = 1.0
            rows.append(vector)
        return np.asarray(rows, dtype=np.float32)

    def embedding_spec(self) -> dict:
        return {"model": "fake", "revision": "rev", "version": 7}


def _encoder() -> tuple[TextEncoder, Fake]:
    encoder = TextEncoder(EncoderConfig())
    backend = Fake()
    encoder._backend = backend
    encoder._dim = backend.dim
    return encoder, backend


def test_every_encoder_call_uses_the_one_configured_backend() -> None:
    encoder, backend = _encoder()

    assert encoder.encode("a").shape == (4,)
    assert encoder.encode_many(["a", "b"]).shape == (2, 4)
    assert backend.calls == ["a", "b"], "repeated text is served from the cache"


def test_returned_vectors_are_copies() -> None:
    encoder, _ = _encoder()
    first = encoder.encode("a")
    first[0] = 99.0

    assert encoder.encode("a")[0] != 99.0


def test_cache_is_bounded():
    encoder, backend = _encoder()
    for index in range(4200):
        encoder.encode(str(index))

    assert len(encoder._vectors) == 4096
    assert len(backend.calls) == 4200


def test_embedding_spec_identifies_the_shared_space() -> None:
    encoder, _ = _encoder()

    spec = encoder.embedding_spec()

    assert spec["model"] == "fake"
    assert spec["revision"] == "rev"
    assert spec["shared_search"] is True, "formation and search are the same space"
    assert spec["format"] == "float32-le"
