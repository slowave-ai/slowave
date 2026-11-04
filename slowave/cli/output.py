"""Professional OSS-style console output.

Provides a unified rendering layer for Slowave CLI commands with:
- Deterministic ASCII symbols by default (✓, !, ✗, -) with optional emoji
- Consistent section headers, status labels, and alignment
- NO_COLOR environment variable support
- JSON serialization for all output modes
- Actionable error/warning messages
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any


class Status(Enum):
    """Status indicator for checks and items."""

    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


class Color(Enum):
    """ANSI color codes."""

    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    CYAN = "\033[36m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


@dataclass(frozen=True)
class Caps:
    """What the attached terminal can render.

    Detected once per render from the output stream and environment, so every
    command degrades the same way (piped output, CI, ``NO_COLOR``, legacy
    Windows code pages) instead of each call site guessing.
    """

    tty: bool
    color: bool
    unicode: bool
    width: int


_UNICODE_GLYPHS = {
    "ok": "✓",
    "warn": "▲",
    "fail": "✗",
    "skip": "–",
    "step": "◆",
    "arrow": "→",
    "dot": "·",
    "rule": "─",
}
_ASCII_GLYPHS = {
    "ok": "+",
    "warn": "!",
    "fail": "x",
    "skip": "-",
    "step": ">",
    "arrow": "->",
    "dot": "-",
    "rule": "-",
}


def detect_caps(stream: Any = None) -> Caps:
    """Detect terminal capabilities for *stream* (default: ``sys.stdout``)."""
    stream = stream if stream is not None else sys.stdout
    try:
        tty = bool(stream.isatty())
    except (AttributeError, ValueError, OSError):
        tty = False
    color = tty and not os.environ.get("NO_COLOR") and os.environ.get("TERM") != "dumb"
    if os.environ.get("FORCE_COLOR") and not os.environ.get("NO_COLOR"):
        color = True
    unicode_ok = os.environ.get("SLOWAVE_ASCII") != "1"
    if unicode_ok:
        encoding = getattr(stream, "encoding", None) or "ascii"
        try:
            "".join(_UNICODE_GLYPHS.values()).encode(encoding)
        except (LookupError, UnicodeEncodeError):
            unicode_ok = False
    try:
        width = shutil.get_terminal_size((80, 24)).columns
    except OSError:
        width = 80
    return Caps(tty=tty, color=color, unicode=unicode_ok, width=max(40, min(width, 100)))


def glyph(name: str, caps: Caps | None = None) -> str:
    """Return the glyph *name* for the terminal (Unicode, or ASCII fallback)."""
    caps = caps or detect_caps()
    return (_UNICODE_GLYPHS if caps.unicode else _ASCII_GLYPHS)[name]


def style(
    text: str, caps: Caps, *, fg: str | None = None, bold: bool = False, dim: bool = False
) -> str:
    """Apply ANSI styling only when *caps* allows colour."""
    if not caps.color:
        return text
    codes = []
    if bold:
        codes.append("1")
    if dim:
        codes.append("2")
    if fg:
        codes.append({"green": "32", "yellow": "33", "red": "31", "cyan": "36"}[fg])
    if not codes:
        return text
    return f"\033[{';'.join(codes)}m{text}\033[0m"


@dataclass
class CheckResult:
    """Result of a single diagnostic check."""

    label: str
    status: Status
    detail: str = ""
    remediation: str = ""


class ConsoleRenderer:
    """Unified output renderer for Slowave CLI."""

    def __init__(self, use_emoji: bool = False, use_color: bool = True):
        """Initialize renderer.

        Args:
            use_emoji: Use emoji symbols instead of ASCII. Defaults to False.
            use_color: Use ANSI colors. Respects NO_COLOR environment variable.
        """
        self.use_emoji = use_emoji
        self.caps = detect_caps()
        # NO_COLOR, non-TTY output and TERM=dumb all disable color.
        self.use_color = use_color and self.caps.color

    def _symbol(self, status: Status) -> str:
        """Get status symbol."""
        if self.use_emoji:
            symbols_emoji = {
                Status.OK: "✅",
                Status.WARN: "⚠️ ",
                Status.FAIL: "❌",
                Status.SKIP: "⏭️ ",
            }
            return symbols_emoji[status]
        else:
            names = {Status.OK: "ok", Status.WARN: "warn", Status.FAIL: "fail", Status.SKIP: "skip"}
            return glyph(names[status], self.caps)

    def _colorize(self, text: str, color: Color) -> str:
        """Apply ANSI color if enabled."""
        if not self.use_color:
            return text
        return f"{color.value}{text}{Color.RESET.value}"

    def section(self, title: str) -> None:
        """Print a section header."""
        click().echo()
        click().echo(self._colorize(f"  {title}", Color.BOLD))
        click().echo(f"  {self._colorize(glyph('rule', self.caps) * len(title), Color.DIM)}")

    def check(
        self,
        label: str,
        status: Status,
        detail: str = "",
        remediation: str = "",
    ) -> None:
        """Print a single check result."""
        symbol = self._symbol(status)
        status_color = {
            Status.OK: Color.GREEN,
            Status.WARN: Color.YELLOW,
            Status.FAIL: Color.RED,
            Status.SKIP: Color.DIM,
        }[status]

        status_text = self._colorize(symbol, status_color)
        msg = f"  {status_text}  {label}"

        if detail:
            msg += f"\n      {self._colorize(detail, Color.DIM)}"
        if remediation:
            msg += f"\n      {remediation}"

        click().echo(msg)

    def item(self, label: str, value: Any, dim: bool = False) -> None:
        """Print a key-value item."""
        value_str = str(value)
        if dim:
            value_str = self._colorize(value_str, Color.DIM)
        click().echo(f"  {label:<25} {value_str}")

    def summary(self, ok: bool, message: str) -> None:
        """Print final summary."""
        click().echo()
        status = Status.OK if ok else Status.WARN
        symbol = self._symbol(status)
        status_text = self._colorize(symbol, Color.GREEN if ok else Color.YELLOW)
        click().echo(f"  {status_text}  {message}")
        click().echo()

    def warning(self, message: str, remediation: str = "") -> None:
        """Print a warning with optional remediation."""
        msg = f"  {self._symbol(Status.WARN)}  {message}"
        if remediation:
            msg += f"\n      {remediation}"
        click().echo(msg)

    def error(self, message: str, remediation: str = "") -> None:
        """Print an error with optional remediation."""
        msg = f"  {self._symbol(Status.FAIL)}  {message}"
        if remediation:
            msg += f"\n      {remediation}"
        click().echo(msg)

    def hint(self, message: str) -> None:
        """Print a hint/suggestion."""
        msg = self._colorize(f"{glyph('arrow', self.caps)} {message}", Color.DIM)
        click().echo(f"  {msg}")

    def title(self, title: str, version: str = "") -> None:
        """Print a command title."""
        click().echo()
        msg = title
        if version:
            msg += f"  {self._colorize(version, Color.DIM)}"
        click().echo(self._colorize(f"  {msg}", Color.BOLD))

    def json(self, data: dict[str, Any]) -> None:
        """Output data as JSON."""
        click().echo(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def click():
    """Lazy import click to avoid circular dependency."""
    import click as _click

    return _click


def safe_emoji(emoji: str, fallback: str = "") -> str:
    """Return *emoji* on non-Windows, *fallback* on Windows (unless SLOWAVE_EMOJI=1).

    Windows consoles (cmd.exe, older PowerShell) can't render emoji — they
    show up as "?" or replacement squares.  This helper lets us use emoji
    everywhere in source while keeping Windows output readable.
    """
    if os.environ.get("SLOWAVE_EMOJI") == "1":
        return emoji
    if sys.platform == "win32":
        return fallback
    return emoji


def get_renderer(use_emoji: bool = False) -> ConsoleRenderer:
    """Factory function to create a renderer.

    Args:
        use_emoji: Force emoji mode (default respects env var).

    Returns:
        ConsoleRenderer configured based on environment.
    """
    # Check for emoji preference in env or arg
    use_emoji = use_emoji or os.environ.get("SLOWAVE_EMOJI") == "1"
    return ConsoleRenderer(use_emoji=use_emoji)


def status_symbol(status: Status, use_emoji: bool = False) -> str:
    """Get status symbol for use in inline labels."""
    if use_emoji:
        return {"ok": "✅", "warn": "⚠️", "fail": "❌", "skip": "⏭️"}[status.value]
    return glyph(status.value)
