"""Text encoder: produces a dense embedding for a piece of text.

Default backend: ONNX Runtime with paraphrase-multilingual-MiniLM-L12-v2
(384-dim, 50+ languages). Swap via EncoderConfig(model_name=...).

The encoder is loaded lazily on first use so that imports stay cheap and
test runs that don't need embeddings (e.g. with synthetic vectors) don't
pay the cost of loading model weights.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class EncoderConfig:
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    normalize: bool = True
    device: str = "cpu"  # Only CPU for ONNX
    use_onnx: bool = True  # Use ONNX Runtime (recommended) vs sentence-transformers
    cache_dir: Optional[str] = None  # Custom cache directory for models


class TextEncoder:
    """Lazy wrapper around text embedding backends (ONNX or sentence-transformers).

    By default uses ONNX Runtime for CPU-optimized inference with no torch.
    If `sentence-transformers` is installed and use_onnx=False, falls back to it.
    If neither is available, `encode` raises a clear error.
    """

    def __init__(self, cfg: EncoderConfig | None = None):
        self.cfg = cfg or EncoderConfig()
        self._backend = None  # Lazy-loaded encoder instance
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

        if self.cfg.use_onnx:
            self._load_onnx_backend()
        else:
            self._load_sentence_transformers_backend()

    def _load_onnx_backend(self) -> None:
        """Load ONNX Runtime backend (recommended, no torch)."""
        try:
            from slowave.symbolic.onnx_encoder import ONNXTextEncoder

            backend = ONNXTextEncoder(
                model_name=self.cfg.model_name,
                cache_dir=self.cfg.cache_dir,
            )
            backend._ensure_loaded()
            self._dim = backend.dim
            self._backend = backend
        except ImportError as e:
            # If ONNX libraries are missing, give a helpful error
            raise ImportError(
                "ONNX Runtime backend is configured but not installed.\n"
                "Install with: pip install onnxruntime transformers\n"
                "Or to use the legacy sentence-transformers backend:\n"
                "  EncoderConfig(use_onnx=False)\n"
                "  pip install torch sentence-transformers"
            ) from e

    def _load_sentence_transformers_backend(self) -> None:
        """Load sentence-transformers backend (legacy, requires torch).

        Only used if explicitly requested via EncoderConfig(use_onnx=False).
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for use_onnx=False.\n"
                "Install with: pip install torch sentence-transformers\n"
                "Or use the default ONNX Runtime backend (recommended for CPU):\n"
                "  EncoderConfig(use_onnx=True)\n"
                "  pip install onnxruntime transformers"
            ) from e

        # Silence HF hub advisory warning and progress bars
        import logging as _logging
        import os as _os

        _os.environ["TQDM_DISABLE"] = "1"
        _hf_log = _logging.getLogger("huggingface_hub")
        _hf_log.setLevel(_logging.ERROR)
        for _h in _hf_log.handlers:
            _h.setLevel(_logging.ERROR)
        _logging.getLogger("huggingface_hub.utils._http").setLevel(_logging.ERROR)

        from slowave.utils.spinner import BrainSpinner

        with BrainSpinner("loading memory"):
            try:
                model = SentenceTransformer(self.cfg.model_name, device=self.cfg.device)
                self._backend = model
                try:
                    self._dim = int(model.get_embedding_dimension())
                except AttributeError:
                    self._dim = int(model.get_sentence_embedding_dimension())
            except (RuntimeError, ModuleNotFoundError, OSError) as e:
                raise RuntimeError(
                    f"sentence-transformers embedding backend failed to load.\n"
                    f"  Error: {type(e).__name__}: {e}\n"
                    f"  Likely cause: torch / torchvision / transformers version mismatch.\n"
                    f"  Fix: run: pip install --upgrade torch torchvision transformers"
                ) from e

    @property
    def dim(self) -> int:
        """Embedding dimension (384 for the default model)."""
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
                if hasattr(self._backend, "encode_many"):
                    vectors = self._backend.encode_many(batch)
                else:
                    vectors = self._backend.encode(
                        batch,
                        normalize_embeddings=self.cfg.normalize,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                    )
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
        """Identity includes resolved weights/tokenizer and encoding semantics."""
        self._ensure_loaded()
        if not hasattr(self, "_embedding_spec"):
            backend = self._backend
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
                tokenizer_hash = self.cfg.model_name
            self._embedding_spec = {
                "model": self.cfg.model_name,
                "revision": getattr(backend, "model_revision", self.cfg.model_name),
                "tokenizer": tokenizer_hash,
                "backend": "onnx" if self.cfg.use_onnx else "sentence-transformers",
                "pooling": "attention-mean",
                "normalization": self.cfg.normalize,
                "max_length": 512,
                "format": "float32-le",
                "version": 1,
            }
        return dict(self._embedding_spec)
