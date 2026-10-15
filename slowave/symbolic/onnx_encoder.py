"""ONNX Runtime-based text encoder for CPU-optimized inference.

Default model: paraphrase-multilingual-MiniLM-L12-v2 (384-dim, 50+ languages).
Any model with a Xenova ONNX conversion on Hugging Face Hub can be used by
passing model_name to ONNXTextEncoder or EncoderConfig.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from threading import RLock
from typing import Optional

import numpy as np

_logger = logging.getLogger(__name__)


class ONNXTextEncoder:
    """CPU-optimized text encoder using ONNX Runtime.

    Features:
    - No torch required (only onnxruntime + transformers tokenizer)
    - ~750MB smaller footprint than torch + sentence-transformers
    - CPU-first inference with operator fusion
    - Works with any model that has a Xenova ONNX conversion on HF Hub
    - Lazy loading to keep import costs low
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        cache_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._session = None
        self._tokenizer = None
        self._dim = 384
        self._load_lock = RLock()

    def _ensure_loaded(self) -> None:
        with self._load_lock:
            self._load()

    def _load(self) -> None:
        """Lazy load ONNX session and tokenizer on first use."""
        if self._session is not None:
            return

        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "ONNX Runtime and transformers required for text encoding.\n"
                "Install with: pip install onnxruntime transformers\n"
                "(No torch required!)"
            ) from e

        # Disable progress bars and verbose logging
        os.environ["TQDM_DISABLE"] = "1"
        _hf_log = logging.getLogger("huggingface_hub")
        _hf_log.setLevel(logging.ERROR)
        for handler in _hf_log.handlers:
            handler.setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)

        from slowave.utils.spinner import BrainSpinner

        with BrainSpinner("loading memory"):
            try:
                # Download and cache ONNX model
                model_path = self._get_onnx_model_path()
                resolved = model_path.resolve()
                if "snapshots" in model_path.parts:
                    self.model_revision = model_path.parts[model_path.parts.index("snapshots") + 1]
                else:
                    digest = hashlib.sha256()
                    with resolved.open("rb") as weights:
                        for chunk in iter(lambda: weights.read(1024 * 1024), b""):
                            digest.update(chunk)
                    self.model_revision = digest.hexdigest()

                # Create ONNX Runtime session with CPU optimization
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess_options.intra_op_num_threads = int(
                    os.environ.get("SLOWAVE_EMBEDDING_THREADS", str(min(4, os.cpu_count() or 4)))
                )
                sess_options.inter_op_num_threads = 1

                session = ort.InferenceSession(
                    str(model_path),
                    sess_options=sess_options,
                    providers=["CPUExecutionProvider"],
                )

                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name, local_files_only=True
                    )
                except OSError:
                    tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._session, self._tokenizer = session, tokenizer

            except (RuntimeError, ModuleNotFoundError, OSError) as e:
                raise RuntimeError(
                    f"ONNX embedding backend failed to load.\n"
                    f"  Error: {type(e).__name__}: {e}\n"
                    f"  Possible causes:\n"
                    f"    - onnxruntime not installed: pip install onnxruntime\n"
                    f"    - transformers not installed: pip install transformers\n"
                    f"    - Network issue downloading model from Hugging Face\n"
                ) from e

    def _get_onnx_model_path(self) -> Path:
        """Download ONNX model from Hugging Face Hub if needed.

        Returns:
            Path to the cached ONNX model file.
        """
        from huggingface_hub import hf_hub_download

        # Determine cache directory
        if self.cache_dir:
            cache_dir = self.cache_dir
        else:
            # Use HF_HOME or default .cache
            cache_dir = Path(
                os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")
            )

        # Xenova hosts ONNX conversions of popular sentence-transformers models.
        # Repo name is derived from the model's short name (last path component).
        xenova_repo = "Xenova/" + self.model_name.split("/")[-1]
        try:
            try:
                model_file = hf_hub_download(
                    repo_id=xenova_repo,
                    filename="onnx/model.onnx",
                    cache_dir=str(cache_dir),
                    local_files_only=True,
                )
                return Path(model_file)
            except (OSError, ValueError):
                pass
            model_file = hf_hub_download(
                repo_id=xenova_repo,
                filename="onnx/model.onnx",
                cache_dir=str(cache_dir),
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download ONNX model from Hugging Face Hub: {e}\n"
                f"\nTroubleshooting steps:\n"
                f"1. Check internet connection\n"
                f"2. Verify you can access: https://huggingface.co/{xenova_repo}\n"
                f"3. Try clearing the cache: rm -rf ~/.cache/huggingface/hub\n"
                f"4. Or set a custom cache dir: export HF_HOME=/path/to/cache\n"
            ) from e

        return Path(model_file)

    @property
    def dim(self) -> int:
        """Embedding dimension (384 for the default model)."""
        return self._dim

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text string.

        Args:
            text: Input text to embed.

        Returns:
            Float32 embedding vector of shape [384].
            L2-normalized to unit norm.
        """
        self._ensure_loaded()
        embeddings = self.encode_many([text])
        return embeddings[0]

    def encode_many(self, texts: list[str]) -> np.ndarray:
        """Encode multiple text strings in batch.

        Args:
            texts: List of input texts to embed.

        Returns:
            Float32 embedding matrix of shape [N, 384].
            Each row is L2-normalized to unit norm.
        """
        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._session is not None

        # Handle empty batch
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        # Tokenize with padding and truncation
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",  # NumPy arrays, not PyTorch tensors
        )

        # Prepare ONNX inputs (must be int64)
        onnx_inputs = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }

        # Some ONNX models require token_type_ids even when the tokenizer
        # (e.g. XLM-RoBERTa-based multilingual models) does not produce them.
        # Check the model's required inputs and zero-fill if needed.
        required_inputs = {inp.name for inp in self._session.get_inputs()}
        if "token_type_ids" in required_inputs:
            if "token_type_ids" in encoded:
                onnx_inputs["token_type_ids"] = encoded["token_type_ids"].astype(np.int64)
            else:
                onnx_inputs["token_type_ids"] = np.zeros_like(onnx_inputs["input_ids"])

        # Run ONNX inference
        # Output is [batch_size, sequence_length, hidden_size] (last hidden state)
        # We need to pool to get sentence embeddings
        outputs = self._session.run(None, onnx_inputs)
        last_hidden_state = outputs[0]  # [batch_size, seq_len, 384]

        # Mean pooling: average over sequence dimension
        attention_mask = encoded["attention_mask"].astype(np.float32)
        attention_mask_expanded = np.expand_dims(attention_mask, axis=2)  # [batch_size, seq_len, 1]

        sum_embeddings = np.sum(
            last_hidden_state * attention_mask_expanded, axis=1
        )  # [batch_size, 384]
        sum_mask = np.sum(attention_mask_expanded, axis=1)  # [batch_size, 1]

        embeddings = sum_embeddings / (sum_mask + 1e-8)  # [batch_size, 384]

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)

        return embeddings.astype(np.float32)
