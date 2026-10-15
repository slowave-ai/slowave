"""Pinned local multilingual retrieval embeddings with query/passage roles."""

from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from threading import RLock

import numpy as np

from slowave.symbolic.onnx_encoder import ONNXTextEncoder

MODEL = "Xenova/multilingual-e5-small"
REVISION = "761b726dd34fb83930e26aab4e9ac3899aa1fa78"


@lru_cache(maxsize=1)
def model_root() -> Path:
    from huggingface_hub import snapshot_download

    patterns = ["*.json", "sentencepiece.bpe.model", "onnx/model_quantized.onnx"]
    try:
        root = snapshot_download(
            MODEL, revision=REVISION, allow_patterns=patterns, local_files_only=True
        )
        if not (Path(root) / "onnx/model_quantized.onnx").exists():
            raise FileNotFoundError("retrieval model weights missing")
    except (OSError, FileNotFoundError):
        try:
            root = snapshot_download(MODEL, revision=REVISION, allow_patterns=patterns)
        except (OSError, FileNotFoundError) as net_exc:
            raise RuntimeError(
                "retrieval encoder assets unavailable: cache miss and download "
                f"failed ({net_exc}); retrieval degrades to stored channels"
            ) from net_exc
    return Path(root)


class _RetrievalONNX(ONNXTextEncoder):
    def __init__(self, root: Path) -> None:
        self.root = root
        super().__init__(model_name=str(root))

    def _get_onnx_model_path(self) -> Path:
        return self.root / "onnx/model_quantized.onnx"


class RetrievalEncoder:
    """Cache passage vectors by exact content; stored formation vectors stay intact."""

    def __init__(self) -> None:
        root = model_root()
        self.backend = _RetrievalONNX(root)
        self._passages: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = RLock()

    def passages(self, texts: list[str]) -> np.ndarray:
        with self._lock:
            return self._cached_passages(texts)

    def _cached_passages(self, texts: list[str]) -> np.ndarray:
        missing = list(dict.fromkeys(text for text in texts if text not in self._passages))
        current = {text: self._passages[text] for text in texts if text in self._passages}
        for index in range(0, len(missing), 16):
            batch = missing[index : index + 16]
            vectors = self.backend.encode_many(["passage: " + text for text in batch])
            current.update(zip(batch, vectors))
        for text in texts:
            self._passages[text] = current[text]
            self._passages.move_to_end(text)
        while len(self._passages) > 4096:
            self._passages.popitem(last=False)
        return (
            np.asarray([current[text] for text in texts], dtype=np.float32)
            .reshape(len(texts), 384)
            .copy()
        )

    @property
    def dim(self) -> int:
        return self.backend.dim

    def encode_many(self, texts: list[str]) -> np.ndarray:
        return self.backend.encode_many(["passage: " + text for text in texts])

    def embedding_spec(self) -> dict:
        self.backend._ensure_loaded()
        return {
            "model": MODEL,
            "revision": REVISION,
            "tokenizer_revision": REVISION,
            "pooling": "attention-mean",
            "normalization": True,
            "max_length": 512,
            "prefix": "passage: ",
            "format": "float32-le",
            "version": 1,
        }

    def query(self, text: str) -> np.ndarray:
        return self.backend.encode("query: " + text)


@lru_cache(maxsize=1)
def get_retrieval_encoder() -> RetrievalEncoder:
    return RetrievalEncoder()
