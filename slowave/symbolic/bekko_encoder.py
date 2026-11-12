"""The pinned Bekko encoder: Slowave's only embedding model.

The same encoder forms memories and searches them, so stored schemas, episodes
and passages share one vector space. Weights, tokenizer, config and special
notice are verified against pinned SHA-256 digests before loading.
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from threading import RLock

import numpy as np

MODEL = "hotchpotch/bekko-embedding-v1-a8m"
REVISION = "c721113d59a1d91b447450324f51c4b3332c924a"
DIM = 384
WEIGHT_SHA256 = "96d8cc6199e96357b21b2fb12f6d7ffd2d4abc7b182fe94b5468fbd6dc819af7"
TOKENIZER_SHA256 = "8bd47075711f75a143d1b78e01a41cc65c1c591b00d3cfeffc23db07adce1392"
CONFIG_SHA256 = "950f344b034bb96a6ac3da045b1cd5ba037c08d8e4978471d407324622f8878e"
SPECIAL_TOKENS_SHA256 = "6e3204c5a4004719c185007a745d1940471a6fb02521cfa0a00306dbb6bc5903"
MAX_LENGTH = 512


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BekkoEncoder:
    def __init__(self, model_root: str | Path):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        root = Path(model_root)
        if _sha256(root / "onnx" / "model.onnx") != WEIGHT_SHA256:
            raise ValueError("Bekko weight file does not match pinned revision")
        if _sha256(root / "tokenizer.json") != TOKENIZER_SHA256:
            raise ValueError("Bekko tokenizer does not match pinned revision")
        if _sha256(root / "config.json") != CONFIG_SHA256:
            raise ValueError("Bekko config does not match pinned revision")
        if _sha256(root / "special_tokens_map.json") != SPECIAL_TOKENS_SHA256:
            raise ValueError("Bekko special tokens do not match pinned revision")
        config = json.loads((root / "config.json").read_text())
        # tokenizers reads the pinned tokenizer.json directly. It is the same
        # library the transformers wrapper delegates to, so ids, special tokens
        # and truncation are unchanged; it only skips the 120 MB wrapper.
        self._tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=MAX_LENGTH)
        self._pad_token_id = int(config.get("pad_token_id", 0))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.environ.get("SLOWAVE_EMBEDDING_THREADS", "4"))
        opts.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(root / "onnx" / "model.onnx"), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._lock = RLock()

    @property
    def dim(self) -> int:
        return DIM

    @property
    def model_name(self) -> str:
        return MODEL

    @property
    def model_revision(self) -> str:
        """Identity recorded by TextEncoder.embedding_spec for stored vectors."""
        return f"{REVISION}:{Path('onnx/model.onnx').name}"

    def embedding_spec(self) -> dict:
        return {
            "model": MODEL,
            "revision": REVISION,
            "weight_sha256": WEIGHT_SHA256,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "config_sha256": CONFIG_SHA256,
            "special_tokens_sha256": SPECIAL_TOKENS_SHA256,
            "backend": "onnx",
            "pooling": "attention-mean",
            "normalization": True,
            "max_length": MAX_LENGTH,
            "format": "float32-le",
            "version": 1,
        }

    def encode(self, text: str) -> np.ndarray:
        return self.encode_many([text])[0]

    def encode_many(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, DIM), dtype=np.float32)
        with self._lock:
            encodings = self._tokenizer.encode_batch(list(texts))
            length = max(len(encoding.ids) for encoding in encodings)
            ids = np.full((len(encodings), length), self._pad_token_id, dtype=np.int64)
            mask = np.zeros((len(encodings), length), dtype=np.int64)
            for row, encoding in enumerate(encodings):
                size = len(encoding.ids)
                ids[row, :size] = encoding.ids
                mask[row, :size] = encoding.attention_mask
            enc = {"input_ids": ids, "attention_mask": mask}
            feeds = {}
            for inp in self._session.get_inputs():
                if inp.name in enc:
                    feeds[inp.name] = np.asarray(enc[inp.name], dtype=np.int64)
                elif inp.name == "token_type_ids":
                    feeds[inp.name] = np.zeros_like(enc["input_ids"], dtype=np.int64)
                elif inp.name == "position_ids":
                    feeds[inp.name] = np.broadcast_to(
                        np.arange(enc["input_ids"].shape[1], dtype=np.int64), enc["input_ids"].shape
                    )
                else:
                    raise ValueError(f"Unsupported Bekko ONNX input: {inp.name}")
            outputs = self._session.run(None, feeds)
            names = [output.name for output in self._session.get_outputs()]
            vectors = np.asarray(
                (
                    outputs[names.index("sentence_embedding")]
                    if "sentence_embedding" in names
                    else outputs[0]
                ),
                dtype=np.float32,
            )
            if vectors.ndim == 3:
                mask = enc["attention_mask"][..., None]
                vectors = (vectors * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
            if vectors.shape != (len(texts), DIM):
                raise ValueError("Invalid Bekko embedding shape")
            vectors = vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
            if not np.isfinite(vectors).all():
                raise ValueError("Invalid Bekko embeddings")
            return np.asarray(vectors, dtype=np.float32)


MODEL_DIR_ENV = "SLOWAVE_BEKKO_MODEL_DIR"


def _cache_candidates() -> list[Path]:
    """Roots that may hold a huggingface_hub model cache."""
    roots: list[Path] = []
    explicit = os.environ.get("HF_HUB_CACHE")
    if explicit:
        roots.append(Path(explicit))
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.extend([Path(hf_home), Path(hf_home) / "hub"])
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    return roots


def _snapshot_in_cache(model: str, revision: str) -> Path | None:
    folder = "models--" + model.replace("/", "--")
    for root in _cache_candidates():
        snapshot = root / folder / "snapshots" / revision
        if (snapshot / "onnx" / "model.onnx").exists():
            return snapshot
    return None


def model_available_offline() -> bool:
    """Whether the pinned model can be loaded without touching the network."""
    explicit = os.environ.get(MODEL_DIR_ENV)
    if explicit:
        return (Path(explicit) / "onnx" / "model.onnx").exists()
    return _snapshot_in_cache(MODEL, REVISION) is not None


@lru_cache(maxsize=1)
def model_root() -> Path:
    """Local snapshot of the pinned revision; the network is a last resort."""
    explicit = os.environ.get(MODEL_DIR_ENV)
    if explicit:
        root = Path(explicit)
        if not (root / "onnx" / "model.onnx").exists():
            raise RuntimeError(f"{MODEL_DIR_ENV}={explicit} has no onnx/model.onnx")
        return root
    cached = _snapshot_in_cache(MODEL, REVISION)
    if cached is not None:
        return cached
    from huggingface_hub import snapshot_download

    try:
        downloaded: str = snapshot_download(
            MODEL,
            revision=REVISION,
            allow_patterns=["*.json", "onnx/model.onnx"],
            local_files_only=True,
        )
    except OSError:
        try:
            downloaded = snapshot_download(
                MODEL,
                revision=REVISION,
                allow_patterns=["*.json", "onnx/model.onnx"],
            )
        except OSError as net_error:
            raise RuntimeError(
                "Bekko encoder assets unavailable: cache miss and download failed "
                f"({net_error}); retrieval keeps the stored channels"
            ) from net_error
    return Path(downloaded)
