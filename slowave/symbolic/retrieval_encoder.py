"""Pinned local multilingual retrieval embeddings with query/passage roles."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

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
        self._passages: dict[str, np.ndarray] = {}

    def passages(self, texts: list[str]) -> np.ndarray:
        missing = list(dict.fromkeys(text for text in texts if text not in self._passages))
        for index in range(0, len(missing), 16):
            batch = missing[index : index + 16]
            vectors = self.backend.encode_many(["passage: " + text for text in batch])
            self._passages.update(zip(batch, vectors))
        result = np.array([self._passages[text] for text in texts], dtype=np.float32)
        if len(self._passages) > 4096:
            self._passages.clear()
        return result

    def query(self, text: str) -> np.ndarray:
        return self.backend.encode("query: " + text)


@lru_cache(maxsize=1)
def get_retrieval_encoder() -> RetrievalEncoder:
    return RetrievalEncoder()
