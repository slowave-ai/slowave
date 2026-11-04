"""Terminal capability detection drives glyphs and colour for setup/doctor."""

from __future__ import annotations

import io

import pytest

from slowave.cli.output import detect_caps, glyph, style


class _Stream(io.StringIO):
    def __init__(self, encoding: str, tty: bool = False) -> None:
        super().__init__()
        self._enc, self._tty = encoding, tty

    @property
    def encoding(self) -> str:  # type: ignore[override]
        return self._enc

    def isatty(self) -> bool:
        return self._tty


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("NO_COLOR", "FORCE_COLOR", "SLOWAVE_ASCII", "TERM"):
        monkeypatch.delenv(name, raising=False)


def test_utf8_tty_gets_unicode_and_color():
    caps = detect_caps(_Stream("utf-8", tty=True))
    assert caps.unicode and caps.color
    assert glyph("ok", caps) == "✓"
    assert style("x", caps, fg="green").startswith("\033[")


@pytest.mark.parametrize("encoding", ["cp437", "cp1252", "ascii"])
def test_legacy_code_pages_fall_back_to_ascii(encoding):
    caps = detect_caps(_Stream(encoding, tty=True))
    assert not caps.unicode
    assert glyph("ok", caps) == "+" and glyph("fail", caps) == "x"


def test_piped_output_and_no_color_emit_no_escape_codes(monkeypatch):
    assert style("x", detect_caps(_Stream("utf-8")), fg="red") == "x"
    monkeypatch.setenv("NO_COLOR", "1")
    assert not detect_caps(_Stream("utf-8", tty=True)).color


def test_ascii_override(monkeypatch):
    monkeypatch.setenv("SLOWAVE_ASCII", "1")
    assert glyph("step", detect_caps(_Stream("utf-8", tty=True))) == ">"
