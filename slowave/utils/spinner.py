"""Minimal terminal spinners — no dependencies beyond stdlib.

Two spinners:

  BrainSpinner   — for model loading / recall.
                   Cycles through brain-state emoji with a pulsing label.
                   e.g.  🧠 ·· slowaving

  SleepSpinner   — for the background worker idle phase.
                   Drifting zzz to evoke slow-wave sleep.
                   e.g.  💤 z z Z  sleeping  (5 min)

Both write to stderr so they don't pollute stdout/JSON output.
Both are no-ops when stderr is not a TTY (CI, pipes, --json mode).
"""

from __future__ import annotations

import os
import sys
import threading
import time


def _is_tty() -> bool:
    if os.environ.get("SLOWAVE_ACCEPTANCE_QUIET") == "1":
        return False
    return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()


class BrainSpinner:
    """Animated brain-pulse spinner for slow operations.

    Usage::

        with BrainSpinner("slowaving"):
            do_slow_thing()

    Produces a line like:
        🧠 ··· slowaving
    that updates in-place until the context exits, then clears.
    """

    # Each frame is (emoji, dot-trail).  Emoji shifts subtly to suggest
    # "thinking" without being distracting.
    _FRAMES = [
        ("🧠", "·  "),
        ("🧠", "·· "),
        ("🧠", "···"),
        ("🧠", " ··"),
        ("🧠", "  ·"),
        ("🧠", "   "),
    ]
    _INTERVAL = 0.12

    def __init__(self, label: str = "slowaving") -> None:
        self._label = label
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _spin(self) -> None:
        frames = self._FRAMES
        i = 0
        while not self._stop.is_set():
            emoji, dots = frames[i % len(frames)]
            sys.stderr.write(f"\r  {emoji} {dots} {self._label}  ")
            sys.stderr.flush()
            i += 1
            self._stop.wait(self._INTERVAL)
        # Clear the line
        sys.stderr.write("\r" + " " * 32 + "\r")
        sys.stderr.flush()

    def start(self) -> "BrainSpinner":
        if not _is_tty():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def __enter__(self) -> "BrainSpinner":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()


class SleepSpinner:
    """Drifting-zzz spinner for the worker idle phase.

    Usage::

        spinner = SleepSpinner(interval_s=300)
        spinner.start()
        # ... sleep loop ...
        spinner.stop()

    Produces a line like:
        💤 z z Z  sleeping  (4m 32s remaining)
    """

    # zzz drifts left-to-right: small-z, medium-z, capital-Z, then fades
    _ZZZ = ["z  ", "zz ", "zzZ", " zZ", "  Z", "   "]
    _INTERVAL = 0.4

    def __init__(self, interval_s: int = 300) -> None:
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time: float = 0.0

    def _fmt_remaining(self) -> str:
        elapsed = time.monotonic() - self._start_time
        remaining = max(0, self._interval_s - int(elapsed))
        if remaining >= 60:
            return f"{remaining // 60}m {remaining % 60:02d}s"
        return f"{remaining}s"

    def _spin(self) -> None:
        frames = self._ZZZ
        i = 0
        while not self._stop.is_set():
            z = frames[i % len(frames)]
            rem = self._fmt_remaining()
            sys.stderr.write(f"\r  💤 {z}  sleeping  ({rem})  ")
            sys.stderr.flush()
            i += 1
            self._stop.wait(self._INTERVAL)
        sys.stderr.write("\r" + " " * 40 + "\r")
        sys.stderr.flush()

    def start(self) -> "SleepSpinner":
        if not _is_tty():
            return self
        self._start_time = time.monotonic()
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
