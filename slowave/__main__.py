"""`python -m slowave` dispatches to the Click CLI."""

import os
import sys

# Under pythonw.exe (Windows no-console launcher, used by the Task Scheduler
# services) sys.stdout/stderr are None. Libraries that probe them — uvicorn's
# isatty() color detection, logging stream handlers — crash the process at
# startup with no trace. Rebind them to a log file so services survive and
# stay diagnosable.
if sys.stdout is None or sys.stderr is None:
    from slowave.core.paths import runtime_paths

    _log_dir = runtime_paths().logs_dir
    _log_dir.mkdir(parents=True, exist_ok=True)
    _cmd = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].isalnum() else "cli"
    _stream = open(
        _log_dir / f"pythonw-{_cmd}.log", "a", buffering=1, encoding="utf-8", errors="replace"
    )
    if sys.stdout is None:
        sys.stdout = _stream
    if sys.stderr is None:
        sys.stderr = _stream

# macOS note: FAISS + ONNX Runtime can sometimes load multiple OpenMP runtimes.
# Pragmatic workaround to avoid a hard crash.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import logging as _logging

_logging.getLogger("huggingface_hub").setLevel(_logging.ERROR)
_logging.getLogger("onnxruntime").setLevel(_logging.ERROR)

from slowave.cli.main import main

if __name__ == "__main__":
    main()
