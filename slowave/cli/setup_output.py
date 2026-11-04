"""Professional output formatting for setup and cleanup commands."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from slowave.cli.output import ConsoleRenderer, Status, get_renderer


class ChangeType(str, Enum):
    """Types of changes that can be applied."""

    MCP_CONFIG = "mcp_config"
    LIFECYCLE_BLOCK = "lifecycle_block"
    HOOKS = "hooks"
    WORKER_SERVICE = "worker_service"
    BACKUP = "backup"
    DATA = "data"


class ChangeStatus(str, Enum):
    """Status of a change."""

    NEW = "new"
    UPDATE = "update"
    SKIP = "skip"
    ERROR = "error"


@dataclass
class Change:
    """Represents a single change to be applied or removed."""

    change_type: ChangeType
    label: str  # Human-readable label
    status: ChangeStatus
    detail: str = ""  # Additional context
    path: str = ""  # File/service path

    def render(self, renderer: ConsoleRenderer) -> None:
        """Render this change using the renderer."""
        # Map change status to output status
        output_status = {
            ChangeStatus.NEW: Status.OK,
            ChangeStatus.UPDATE: Status.OK,
            ChangeStatus.SKIP: Status.SKIP,
            ChangeStatus.ERROR: Status.FAIL,
        }[self.status]

        # Build detail string
        detail_str = self.detail
        if self.path:
            detail_str = f"{self.path} ({self.detail})" if detail_str else self.path

        renderer.check(self.label, output_status, detail_str)


class SetupSession:
    """Tracks setup/cleanup operation state and results."""

    def __init__(self, operation: str = "setup", dry_run: bool = False):
        self.operation = operation  # "setup" or "cleanup"
        self.dry_run = dry_run
        self.renderer = get_renderer(use_emoji=False)
        self.changes: list[Change] = []
        self.binaries: dict[str, str] = {}
        self.manual_steps: list[str] = []
        self.errors: list[str] = []

    def add_change(self, change: Change) -> None:
        """Record a change."""
        self.changes.append(change)

    def add_binary(self, name: str, path: str) -> None:
        """Record a binary location."""
        self.binaries[name] = path

    def add_manual_step(self, step: str) -> None:
        """Record a manual step required after operation."""
        self.manual_steps.append(step)

    def add_error(self, error: str) -> None:
        """Record an error."""
        self.errors.append(error)

    def count_by_status(self, status: ChangeStatus) -> int:
        """Count changes with a specific status."""
        return sum(1 for c in self.changes if c.status == status)

    def render_header(self, title: str) -> None:
        """Render operation title."""
        self.renderer.title(title)
        if self.dry_run:
            self.renderer.warning("DRY RUN — no files will be modified", "")
        self.renderer.section("Operations")

    def render_changes(self) -> None:
        """Render all recorded changes."""
        for change in self.changes:
            change.render(self.renderer)

    def render_summary(self) -> None:
        """Render operation summary."""
        self.renderer.section("Summary")

        new_count = self.count_by_status(ChangeStatus.NEW)
        update_count = self.count_by_status(ChangeStatus.UPDATE)
        skip_count = self.count_by_status(ChangeStatus.SKIP)
        error_count = len(self.errors)

        if new_count:
            self.renderer.item("Created/installed", f"{new_count}")
        if update_count:
            self.renderer.item("Updated", f"{update_count}")
        if skip_count:
            self.renderer.item("Skipped", f"{skip_count}")
        if error_count:
            self.renderer.item("Errors", f"{error_count}", dim=False)

        if self.dry_run:
            self.renderer.section("Next Steps")
            self.renderer.item("Review", "Re-run without --dry-run to apply")

        if self.manual_steps:
            self.renderer.section("Manual Steps")
            for step in self.manual_steps:
                self.renderer.hint(step)

        # Final status
        if self.errors:
            msg = f"Setup failed with {error_count} error(s)."
            self.renderer.summary(False, msg)
        elif self.dry_run:
            msg = f"Dry run complete. {new_count + update_count} change(s) ready."
            self.renderer.summary(True, msg)
        else:
            total = new_count + update_count
            if total == 0:
                msg = "No changes applied."
            elif total == 1:
                msg = "Setup complete. 1 change applied."
            else:
                msg = f"Setup complete. {total} changes applied."
            self.renderer.summary(True, msg)


class SetupView:
    """Streaming, platform-neutral renderer for ``slowave setup``.

    Rows print as each step completes, so a failure midway still leaves the
    completed steps on screen. Glyphs, colour and width come from
    :func:`slowave.cli.output.detect_caps`; nothing here emits emoji or raw
    control sequences when output is piped or the terminal is limited.
    """

    def __init__(self, verbose: bool = False, label_width: int = 18) -> None:
        from slowave.cli.output import detect_caps

        self.caps = detect_caps()
        self.verbose = verbose
        self.LABEL_WIDTH = label_width

    def _echo(self, text: str = "") -> None:
        from slowave.cli.output import click

        click().echo(text)

    def _style(self, text: str, **kw: bool | str) -> str:
        from slowave.cli.output import style

        return style(text, self.caps, **kw)  # type: ignore[arg-type]

    def _g(self, name: str) -> str:
        from slowave.cli.output import glyph

        return glyph(name, self.caps)

    def banner(self, version: str, subtitle: str, link: str = "") -> None:
        self._echo()
        self._echo(f"  {self._style('slowave', bold=True)}  {self._style(version, dim=True)}")
        self._echo(f"  {self._style(subtitle, dim=True)}")
        if link:
            self._echo(f"  {self._style(f'Github: {link}', dim=True)}")

    def notice(self, text: str) -> None:
        """A highlighted, mode-level notice such as the dry-run banner."""
        self._echo()
        self._echo(f"  {self._style(text, fg='yellow', bold=True)}")

    def group(self, title: str) -> None:
        self._echo()
        self._echo(f"  {self._style(self._g('step'), fg='cyan')} {self._style(title, bold=True)}")

    def row(self, label: str, status: str, detail: str = "", *, note: str = "") -> None:
        """One result row. *status* is ok | warn | fail | skip."""
        color = {"ok": "green", "warn": "yellow", "fail": "red"}.get(status)
        mark = (
            self._style(self._g(status), fg=color, dim=status == "skip")
            if color
            else (self._style(self._g(status), dim=True))
        )
        import textwrap

        prefix = f"    {mark} {label:<{self.LABEL_WIDTH}} "
        indent = " " * (4 + 2 + self.LABEL_WIDTH + 1)
        lines = (
            textwrap.wrap(detail, max(20, self.caps.width - len(indent)), break_long_words=False)
            if detail
            else [""]
        ) or [""]
        self._echo(prefix.rstrip() if not detail else prefix + self._style(lines[0], dim=True))
        for extra in lines[1:]:
            self._echo(indent + self._style(extra, dim=True))
        if note and self.verbose:
            self._echo(f"      {self._style(note, dim=True)}")

    def line(self, text: str, *, dim: bool = False, indent: int = 2) -> None:
        import textwrap

        width = max(20, self.caps.width - indent)
        for part in textwrap.wrap(text, width, break_long_words=False) or [""]:
            self._echo(f"{' ' * indent}{self._style(part, dim=dim)}")

    def blank(self) -> None:
        self._echo()

    def success(self, message: str) -> None:
        self._echo()
        self._echo(f"  {self._style(self._g('ok'), fg='green')} {self._style(message, bold=True)}")

    def failure(self, message: str) -> None:
        self._echo()
        self._echo(f"  {self._style(self._g('fail'), fg='red')} {self._style(message, bold=True)}")

    def next_steps(self, steps: list[tuple[str, str]]) -> None:
        if not steps:
            return
        self._echo()
        self._echo(f"  {self._style('Next', bold=True)}")
        width = max(len(cmd) for cmd, _ in steps)
        for cmd, desc in steps:
            self._echo(f"    {cmd:<{width}}  {self._style(desc, dim=True)}")
        self._echo()
