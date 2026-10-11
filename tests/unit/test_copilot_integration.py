"""Copilot CLI wiring uses fake homes and never starts real services or clients."""

from __future__ import annotations

import json

import click
import pytest
from click.testing import CliRunner

import slowave.cli.cleanup as cleanup
import slowave.cli.setup as setup
from slowave.cli.clients import _copilot_status, get_client_statuses, summarize_client_status
from slowave.cli.output import Status
from slowave.lifecycle import LIFECYCLE_VERSION


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, "_home", lambda: tmp_path)
    monkeypatch.setattr(cleanup, "_home", lambda: tmp_path)
    monkeypatch.delenv("COPILOT_HOME", raising=False)
    monkeypatch.delenv("SLOWAVE_DB", raising=False)
    monkeypatch.setenv("SLOWAVE_HOME", str(tmp_path / "runtime"))
    monkeypatch.setenv("SLOWAVE_MCP_HTTP_PORT", "19876")
    monkeypatch.setattr(setup.shutil, "which", lambda name: None)
    monkeypatch.setattr(setup, "_find_slowave_binary", lambda: "/fake/slowave")
    monkeypatch.setattr(setup.subprocess, "run", lambda *a, **kw: None)
    return tmp_path


def write_config(value):
    path = setup._copilot_mcp_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def configured():
    cfg, _ = setup._patch_copilot_mcp({}, "http://127.0.0.1:19876/mcp")
    write_config(cfg)
    setup._inject_block(setup._copilot_instructions_path(), setup._lifecycle_block("copilot"))
    return cfg


@pytest.mark.parametrize("system", ["Darwin", "Linux", "Windows"])
def test_paths_and_detection_before_first_launch(isolated, monkeypatch, system):
    monkeypatch.setattr(setup, "SYSTEM", system)
    assert setup._copilot_mcp_config_path() == isolated / ".copilot/mcp-config.json"
    assert setup._detected_clients("copilot") == []
    monkeypatch.setattr(
        setup.shutil, "which", lambda name: "/bin/copilot" if name == "copilot" else None
    )
    assert "copilot" in [s.key for s in setup._detected_clients("all")]
    assert not setup._copilot_home().exists()


def test_custom_home_is_shared_by_setup_doctor_and_cleanup(isolated, monkeypatch):
    root = isolated / "custom copilot"
    monkeypatch.setenv("COPILOT_HOME", str(root))
    configured()
    assert setup._copilot_mcp_config_path() == root / "mcp-config.json"
    assert setup._copilot_instructions_path() == root / "copilot-instructions.md"
    assert summarize_client_status(_copilot_status())[0] == Status.OK
    assert get_client_statuses()["copilot"].lifecycle_enabled
    assert cleanup._remove_mcp_configs(False) == 1
    assert cleanup._remove_lifecycle_blocks(False) == 1
    assert not (isolated / ".copilot").exists()


def test_patch_preserves_other_servers_and_preferences():
    other = {"command": "other-server"}
    cfg = {
        "metadata": "keep",
        "mcpServers": {
            "other": other,
            "slowave": {
                "type": "local",
                "command": "slowave-mcp",
                "args": [],
                "env": {},
                "cwd": "/tmp",
                "tools": ["slowave_recall"],
                "enabled": False,
                "timeout": 12000,
            },
        },
    }
    patched, changed = setup._patch_copilot_mcp(cfg, "http://127.0.0.1:19876/mcp")
    assert changed
    assert patched["metadata"] == "keep"
    assert patched["mcpServers"]["other"] == other
    assert patched["mcpServers"]["slowave"] == {
        "type": "http",
        "url": "http://127.0.0.1:19876/mcp",
        "tools": ["slowave_recall"],
        "enabled": False,
        "timeout": 12000,
    }
    assert not setup._patch_copilot_mcp(patched, "http://127.0.0.1:19876/mcp")[1]


@pytest.mark.parametrize("value", [[], None, {"mcpServers": []}, {"mcpServers": {"slowave": None}}])
def test_patch_rejects_malformed_shapes(value):
    with pytest.raises(click.ClickException, match="JSON object"):
        setup._patch_copilot_mcp(value)


@pytest.mark.parametrize("selection", [[], ["copilot"]])
def test_setup_auto_discovers_binary_and_configures_lifecycle(isolated, monkeypatch, selection):
    monkeypatch.setattr(
        setup.shutil, "which", lambda name: "/bin/copilot" if name == "copilot" else None
    )
    args = ["--no-worker"] + (["--client", selection[0]] if selection else [])
    result = CliRunner().invoke(setup.setup_cmd, args, input="y\n")
    assert result.exit_code == 0, result.output
    assert "Found: Copilot CLI" in result.output
    assert "Restart Copilot CLI" in result.output
    cfg = json.loads(setup._copilot_mcp_config_path().read_text())
    assert cfg["mcpServers"]["slowave"] == {
        "type": "http",
        "url": "http://127.0.0.1:19876/mcp",
        "tools": ["*"],
    }
    assert (
        setup._copilot_instructions_path().read_text().strip()
        == setup._lifecycle_block("codex").strip()
    )
    assert summarize_client_status(_copilot_status())[0] == Status.OK


def test_setup_uses_existing_directory_without_binary(isolated):
    setup._copilot_home().mkdir()
    result = CliRunner().invoke(
        setup.setup_cmd, ["--client", "copilot", "--no-worker"], input="y\n"
    )
    assert result.exit_code == 0, result.output
    assert setup._copilot_mcp_config_path().is_file()


def test_explicit_missing_client_is_skipped_like_other_clients(isolated):
    result = CliRunner().invoke(
        setup.setup_cmd, ["--client", "copilot", "--dry-run", "--no-worker"]
    )
    assert result.exit_code == 0, result.output
    assert "Found:" not in result.output
    assert not setup._copilot_home().exists()


def test_dry_run_creates_no_files_even_for_binary_only_install(isolated, monkeypatch):
    monkeypatch.setattr(setup.shutil, "which", lambda name: "/bin/copilot")
    result = CliRunner().invoke(
        setup.setup_cmd, ["--client", "copilot", "--dry-run", "--no-worker"]
    )
    assert result.exit_code == 0, result.output
    assert "Copilot CLI" in result.output
    assert "19876/mcp" in result.output
    assert not list(isolated.iterdir())


def test_upgrade_idempotence_backups_and_safe_removal(isolated):
    path = write_config({"mcpServers": {"other": {"command": "keep"}}})
    instructions = setup._copilot_instructions_path()
    instructions.write_text("# Personal preferences\n", encoding="utf-8")
    original_config = path.read_text()
    result = CliRunner().invoke(
        setup.setup_cmd, ["--client", "copilot", "--no-worker"], input="y\n"
    )
    assert result.exit_code == 0, result.output
    assert any(p.read_text() == original_config for p in path.parent.glob("mcp-config.json.bak.*"))
    before = {p.name: p.read_bytes() for p in path.parent.iterdir()}
    second = CliRunner().invoke(setup.setup_cmd, ["--client", "copilot", "--no-worker"])
    assert second.exit_code == 0, second.output
    assert "Nothing to do" in second.output
    assert before == {p.name: p.read_bytes() for p in path.parent.iterdir()}
    cleanup._remove_mcp_configs(True)
    cleanup._remove_lifecycle_blocks(True)
    assert before == {p.name: p.read_bytes() for p in path.parent.iterdir()}
    cleanup._remove_mcp_configs(False)
    cleanup._remove_lifecycle_blocks(False)
    assert json.loads(path.read_text())["mcpServers"] == {"other": {"command": "keep"}}
    assert instructions.read_text().strip() == "# Personal preferences"


@pytest.mark.parametrize(
    "raw", ["{broken", "[]", '{"mcpServers": []}', '{"mcpServers": {"slowave": null}}']
)
def test_invalid_config_fails_before_overwrite(isolated, raw):
    path = setup._copilot_mcp_config_path()
    path.parent.mkdir()
    path.write_text(raw)
    result = CliRunner().invoke(
        setup.setup_cmd, ["--client", "copilot", "--no-worker"], input="y\n"
    )
    assert result.exit_code == 1, result.output
    assert path.read_text() == raw
    assert not setup._copilot_instructions_path().exists()
    assert summarize_client_status(_copilot_status())[0] == Status.WARN


@pytest.mark.parametrize(
    "edit,warning",
    [
        ({"tools": []}, "excludes lifecycle tools"),
        ({"tools": "*"}, "invalid MCP tools selection"),
        ({"tools": [None]}, "invalid MCP tools selection"),
        ({"enabled": False}, "disabled"),
        ({"disabled": True}, "disabled"),
        ({"url": "http://127.0.0.1:8766/mcp"}, "transport/URL"),
        ({"type": "sse"}, "transport/URL"),
    ],
)
def test_doctor_warns_for_unusable_config(isolated, edit, warning):
    cfg = configured()
    cfg["mcpServers"]["slowave"].update(edit)
    write_config(cfg)
    status, detail = summarize_client_status(_copilot_status())
    assert status == Status.WARN
    assert warning in detail


def test_doctor_accepts_explicit_five_tools(isolated):
    cfg = configured()
    cfg["mcpServers"]["slowave"]["tools"] = [
        "slowave_activate",
        "slowave_remember",
        "slowave_recall",
        "slowave_feedback",
        "slowave_commit",
    ]
    write_config(cfg)
    assert summarize_client_status(_copilot_status())[0] == Status.OK


@pytest.mark.parametrize(
    "text", ["", "<!-- slowave-lifecycle-start v1 -->\nstale\n", "# User rules"]
)
def test_doctor_warns_for_missing_or_stale_instructions(isolated, text):
    configured()
    setup._copilot_instructions_path().write_text(text)
    status, detail = summarize_client_status(_copilot_status())
    assert status == Status.WARN and "instructions missing or stale" in detail


def test_healthy_status_reports_current_lifecycle_version(isolated):
    configured()
    status = get_client_statuses()["copilot"]
    assert status.lifecycle_version == LIFECYCLE_VERSION
    assert status.lifecycle_enabled and status.mcp_configured
    assert "verify in Copilot" in summarize_client_status(status)[1]


def test_disabled_server_in_native_copilot_settings_is_preserved_and_reported(isolated):
    configured()
    settings = setup._copilot_settings_path()
    settings.write_text(json.dumps({"disabledMcpServers": ["slowave"], "theme": "keep"}))
    original = settings.read_bytes()
    result = CliRunner().invoke(setup.setup_cmd, ["--client", "copilot", "--no-worker"])
    assert result.exit_code == 0, result.output
    status, detail = summarize_client_status(_copilot_status())
    assert status == Status.WARN
    assert "copilot mcp enable slowave" in detail
    cleanup._remove_mcp_configs(False)
    cleanup._remove_lifecycle_blocks(False)
    assert settings.read_bytes() == original


@pytest.mark.parametrize("raw", ["{broken", "[]", '{"disabledMcpServers": false}'])
def test_doctor_warns_for_unreadable_copilot_settings(isolated, raw):
    configured()
    setup._copilot_settings_path().write_text(raw)
    assert summarize_client_status(_copilot_status())[0] == Status.WARN
