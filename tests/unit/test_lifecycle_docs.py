"""User-facing lifecycle examples must name real CLI commands and options."""

from pathlib import Path

from click.testing import CliRunner

from slowave.cli.main import cli

ROOT = Path(__file__).resolve().parents[2]


def test_documented_lifecycle_command_options_exist():
    for args in (
        ["setup", "--help"],
        ["start", "--help"],
        ["stop", "--help"],
        ["restart", "--help"],
        ["status", "--services", "--help"],
        ["doctor", "--verbose", "--help"],
        ["uninstall", "--dry-run", "--help"],
        ["purge", "--dry-run", "--help"],
        ["docs", "install", "--no-open"],
        ["docs", "troubleshooting", "--no-open"],
        ["backup", "--help"],
        ["restore", "--help"],
        ["dashboard", "--no-allow-actions", "--help"],
    ):
        result = CliRunner().invoke(cli, args)
        assert result.exit_code == 0, (args, result.output)


def test_existing_guides_are_the_single_entry_points():
    assert not (ROOT / "docs/lifecycle.md").exists()
    for name in ("README.md", "docs/cli.md", "docs/install.md", "docs/troubleshooting.md"):
        content = (ROOT / name).read_text(encoding="utf-8")
        assert "lifecycle.md" not in content
    install = (ROOT / "docs/install.md").read_text(encoding="utf-8")
    for heading in ("## Installation", "## Upgrade", "## Manage services", "## Remove Slowave"):
        assert heading in install
    troubleshooting = (ROOT / "docs/troubleshooting.md").read_text(encoding="utf-8")
    for obsolete in ("slowave status --verbose", "pkill -f", "cat /tmp/slowave"):
        assert obsolete not in troubleshooting
    for heading in ("### macOS", "### Linux", "### Windows PowerShell"):
        assert heading in troubleshooting
    assert "account-scoped" in install
    assert "Windows task names include the logged-in account's SID" in install
    assert "machine-wide; independent managed installations" not in troubleshooting
    assert "SlowaveDaemon-*" in troubleshooting
    assert "SlowaveWorker-*" in troubleshooting
    assert "SlowaveBackup-*" in troubleshooting


def test_public_documentation_local_links_resolve():
    import re

    guides = [
        ROOT / "README.md",
        *(ROOT / "docs").glob("*.md"),
        *(ROOT / "integrations").glob("**/README.md"),
    ]
    for file in guides:
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", file.read_text(encoding="utf-8")):
            path = target.split("#")[0]
            if not path or "://" in path or path.startswith("mailto:"):
                continue
            assert (file.parent / path).exists(), (file, target)
