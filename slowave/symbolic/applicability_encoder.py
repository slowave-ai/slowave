"""Pinned multilingual query/passage cross-encoder using local ONNX inference.

Uses existing ONNX/Transformers dependencies, with no generative calls or
private text sent to model hosting. Model downloads are separate from scoring.
"""

from __future__ import annotations

import platform
from collections import OrderedDict
from functools import lru_cache
from threading import Lock

import numpy as np

MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
REVISION = "1427fd652930e4ba29e8149678df786c240d8825"


@lru_cache(maxsize=1)
def _backend():
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import AutoTokenizer

    filename = (
        "onnx/model_qint8_arm64.onnx"
        if platform.machine().lower() in {"arm64", "aarch64"}
        else "onnx/model_quint8_avx2.onnx"
    )
    # Runtime is explicitly offline. Provision the pinned files before enabling
    # the feature; a missing model produces visible degraded-mode diagnostics.
    path = hf_hub_download(MODEL, filename, revision=REVISION, local_files_only=True)
    root = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])
    return tokenizer, session


class ApplicabilityEncoder:
    """Score all source windows so an answer late in a memory is not lost."""

    def __init__(self) -> None:
        self._cache: OrderedDict[tuple[str, str], float] = OrderedDict()
        self._lock = Lock()

    def score(self, queries: list[str], memories: list[str]) -> np.ndarray:
        with self._lock:
            return self._score(queries, memories)

    def _score(self, queries: list[str], memories: list[str]) -> np.ndarray:
        tokenizer, session = _backend()
        scores = np.full((len(memories), len(queries)), -np.inf, dtype=np.float32)
        pairs: list[tuple[int, int, str, str]] = []
        for index, memory in enumerate(memories):
            missing = []
            for column, query in enumerate(queries):
                key = (query, memory)
                if key in self._cache:
                    scores[index, column] = self._cache[key]
                    self._cache.move_to_end(key)
                else:
                    missing.append((column, query))
            if not missing:
                continue
            tokens = tokenizer.encode(memory, add_special_tokens=False)
            windows = [tokens[start : start + 256] for start in range(0, max(1, len(tokens)), 192)]
            for column, query in missing:
                for window in windows:
                    pairs.append((index, column, query, tokenizer.decode(window)))
        # Length grouping avoids padding every short fact to a long document.
        pairs.sort(key=lambda pair: len(pair[2]) + len(pair[3]))
        required = {entry.name for entry in session.get_inputs()}
        for start in range(0, len(pairs), 16):
            batch = pairs[start : start + 16]
            encoded = tokenizer(
                [pair[2] for pair in batch],
                [pair[3] for pair in batch],
                padding=True,
                truncation="only_first",
                max_length=512,
                return_tensors="np",
            )
            inputs = {name: encoded[name].astype(np.int64) for name in required}
            logits = np.asarray(session.run(None, inputs)[0]).reshape(-1)
            if len(logits) != len(batch) or not np.isfinite(logits).all():
                raise ValueError("invalid applicability model output")
            for (index, column, _, _), logit in zip(batch, logits):
                scores[index, column] = max(scores[index, column], float(logit))
        for index, memory in enumerate(memories):
            for column, query in enumerate(queries):
                self._cache[(query, memory)] = float(scores[index, column])
                self._cache.move_to_end((query, memory))
        while len(self._cache) > 8192:
            self._cache.popitem(last=False)
        return scores
