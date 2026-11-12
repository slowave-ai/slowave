"""Text encoder: the pinned Bekko multilingual encoder.

One model serves memory formation and candidate search, so the same vector
space describes stored schemas, episodes and passages. ``TextEncoder`` is a
thin, thread-safe wrapper that owns the pinned backend, caches exact-text
vectors and reports the identity used to version stored vectors.

The backend verifies the SHA-256 of its weights, tokenizer, config and special
notice before it loads, so a partially downloaded or substituted snapshot fails
loudly instead of silently producing vectors from another space.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock

import numpy as np


@dataclass(frozen=True)
class EncoderConfig:
    """Encoding options; the pinned backend always normalises its output."""

    normalize: bool = True


class TextEncoder:
    """Lazy wrapper around the pinned Bekko ONNX backend."""

    def __init__(self, cfg: EncoderConfig | None = None):
        self.cfg = cfg or EncoderConfig()
        self._backend = None
        self._dim: int | None = None
        self._vectors: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_lock = RLock()

    def _ensure_loaded(self) -> None:
        with self._cache_lock:
            self._load()

    def _load(self) -> None:
        """Initialize the encoding backend on first use."""
        if self._backend is not None:
            return
        from slowave.symbolic.bekko_encoder import BekkoEncoder, model_root

        backend = BekkoEncoder(model_root())
        self._dim = backend.dim
        self._backend = backend

    @property
    def dim(self) -> int:
        """Embedding dimension of the pinned encoder."""
        self._ensure_loaded()
        assert self._dim is not None
        return self._dim

    def encode(self, text: str) -> np.ndarray:
        """Embed one text; return a copy so callers cannot mutate the cache."""
        return self.encode_many([text])[0]

    def encode_many(self, texts: list[str]) -> np.ndarray:
        """Batch cache misses, retaining bounded exact-text inference reuse."""
        with self._cache_lock:
            self._ensure_loaded()
            assert self._backend is not None
            missing = list(dict.fromkeys(text for text in texts if text not in self._vectors))
            current = {text: self._vectors[text] for text in texts if text in self._vectors}
            for offset in range(0, len(missing), 16):
                batch = missing[offset : offset + 16]
                vectors = self._backend.encode_many(batch)
                for text, vector in zip(batch, vectors):
                    current[text] = np.asarray(vector, dtype=np.float32).copy()
            for text in texts:
                self._vectors[text] = current[text]
                self._vectors.move_to_end(text)
            while len(self._vectors) > 4096:
                self._vectors.popitem(last=False)
            return (
                np.asarray([current[text] for text in texts], dtype=np.float32)
                .reshape(len(texts), self.dim)
                .copy()
            )

    def embedding_spec(self) -> dict:
        """Identity of the space stored vectors belong to.

        ``shared_search`` records that formation and search share this encoder,
        so passage vectors are never confused with a separate retrieval model.
        """
        self._ensure_loaded()
        if not hasattr(self, "_embedding_spec"):
            backend = self._backend
            assert backend is not None
            tokenizer = getattr(backend, "_tokenizer", None)
            native = getattr(tokenizer, "backend_tokenizer", None)
            if native is not None:
                tokenizer_spec = json.loads(native.to_str())
                # Tokenizer runtime padding/truncation changes between calls;
                # encoding semantics are explicitly versioned below instead.
                tokenizer_spec.pop("padding", None)
                tokenizer_spec.pop("truncation", None)
                tokenizer_hash = hashlib.sha256(
                    json.dumps(tokenizer_spec, sort_keys=True).encode()
                ).hexdigest()
            else:
                tokenizer_hash = "unavailable"
            spec = {
                "model": getattr(backend, "model_name", "bekko"),
                "revision": getattr(backend, "model_revision", "pinned"),
                "tokenizer": tokenizer_hash,
                "backend": "onnx",
                "pooling": "attention-mean",
                "normalization": True,
                "max_length": 512,
                "format": "float32-le",
                "version": 1,
                "shared_search": True,
            }
            pinned = getattr(backend, "embedding_spec", None)
            if callable(pinned):
                spec.update({k: v for k, v in pinned().items() if k not in {"version", "format"}})
            self._embedding_spec = spec
        return dict(self._embedding_spec)
