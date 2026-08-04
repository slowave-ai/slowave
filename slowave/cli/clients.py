"""Client status checking for doctor command."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from slowave.cli.output import Status
from slowave.lifecycle import LIFECYCLE_VERSION


@dataclass
class ClientStatus:
    name: str
    mcp_configured: bool = False
    hooks_installed: bool = False
    lifecycle_enabled: bool = False
    lifecycle_version: str | None = None
    running: bool = False


def _cursor_mcp_config_path() -> Path:
    return Path.home() / ".cursor" / "mcp.json"


def _windsurf_mcp_config_paths() -> list[Path]:
    return [
        Path.home() / ".codeium" / "windsurf" / "mcp_config.json",
        Path.home() / ".codeium" / "mcp_config.json",
    ]


def _json_has_slowave_mcp(path: Path) -> bool:
    """Return True if the config file has a slowave MCP entry (HTTP or legacy stdio)."""
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    servers = data.get("mcpServers") or data.get("servers") or {}
    return isinstance(servers, dict) and "slowave" in servers


def _json_has_slowave_http(path: Path) -> bool:
    """Return True if the config file has a Slowave HTTP (url-based) MCP entry."""
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    servers = data.get("mcpServers") or data.get("servers") or {}
    entry = servers.get("slowave") if isinstance(servers, dict) else None
    # HTTP entry: has "url" key (Cline uses url-only; no type field)
    return isinstance(entry, dict) and "url" in entry and "command" not in entry


def _any_json_has_slowave_mcp(paths: list[Path]) -> bool:
    return any(_json_has_slowave_mcp(path) for path in paths)


def get_client_statuses() -> dict[str, ClientStatus]:
    statuses = {}
    try:
        from slowave.cli.setup import (
            _MARKER_START,
            _claude_desktop_config_path,
            _claude_json_path,
            _claude_md_path,
            _claude_settings_path,
            _cline_mcp_settings_path,
            _clinerules_path,
            _detect_lifecycle_version,
            _read_json,
        )

        cc_json = _claude_json_path()
        cc_settings = _claude_settings_path()
        cc_md = _claude_md_path()
        cc_has_mcp = cc_has_hooks = cc_has_lifecycle = False
        cc_lifecycle_version: str | None = None

        if cc_json.exists():
            try:
                cfg_j = _read_json(cc_json)
                cc_has_mcp = "slowave" in cfg_j.get("mcpServers", {})
            except Exception:
                pass
        if cc_settings.exists():
            try:
                cfg = _read_json(cc_settings)
                if not cc_has_mcp:
                    cc_has_mcp = "slowave" in cfg.get("mcpServers", {})
                cc_has_hooks = any(
                    "SLOWAVE MANDATORY" in str(h.get("command", ""))
                    for group in cfg.get("hooks", {}).get("UserPromptSubmit", [])
                    for h in group.get("hooks", [])
                )
            except Exception:
                pass
        if cc_md.exists():
            try:
                cc_md_text = cc_md.read_text(encoding="utf-8", errors="ignore")
                cc_has_lifecycle = _MARKER_START in cc_md_text
                cc_lifecycle_version = _detect_lifecycle_version(cc_md_text)
            except Exception:
                pass

        if cc_json.exists() or cc_settings.exists() or cc_md.exists():
            statuses["claude_code"] = ClientStatus(
                name="Claude Code",
                mcp_configured=cc_has_mcp,
                hooks_installed=cc_has_hooks,
                lifecycle_enabled=cc_has_lifecycle,
                lifecycle_version=cc_lifecycle_version,
            )

        cd_config = _claude_desktop_config_path()
        if cd_config.exists():
            cd_has_mcp = False
            try:
                cfg = _read_json(cd_config)
                cd_has_mcp = "slowave" in cfg.get("mcpServers", {})
            except Exception:
                pass
            statuses["claude_desktop"] = ClientStatus(
                name="Claude Desktop",
                mcp_configured=cd_has_mcp,
            )

        cline_mcp = _cline_mcp_settings_path()
        cline_rules = _clinerules_path()
        cline_has_mcp = cline_has_lifecycle = False
        cline_lifecycle_version: str | None = None
        if cline_mcp.exists():
            try:
                cfg = _read_json(cline_mcp)
                cline_has_mcp = "slowave" in cfg.get("mcpServers", {})
            except Exception:
                pass
        if cline_rules.exists():
            try:
                cline_rules_text = cline_rules.read_text(encoding="utf-8", errors="ignore")
                cline_has_lifecycle = _MARKER_START in cline_rules_text
                cline_lifecycle_version = _detect_lifecycle_version(cline_rules_text)
            except Exception:
                pass
        if cline_mcp.exists() or cline_rules.exists():
            statuses["cline"] = ClientStatus(
                name="Cline",
                mcp_configured=cline_has_mcp,
                lifecycle_enabled=cline_has_lifecycle,
                lifecycle_version=cline_lifecycle_version,
            )

        cursor_mcp = _cursor_mcp_config_path()
        statuses["cursor"] = ClientStatus(
            name="Cursor",
            mcp_configured=_json_has_slowave_mcp(cursor_mcp),
        )

        windsurf_mcp_paths = _windsurf_mcp_config_paths()
        statuses["windsurf"] = ClientStatus(
            name="Windsurf",
            mcp_configured=_any_json_has_slowave_mcp(windsurf_mcp_paths),
        )

        # OpenCode detection — uses `mcp` key (not `mcpServers`) and `instructions` array
        from slowave.cli.setup import _opencode_config_path, _opencode_instructions_path

        oc_config = _opencode_config_path()
        oc_has_mcp = oc_has_lifecycle = False
        oc_lifecycle_version: str | None = None
        if oc_config.exists():
            try:
                cfg = _read_json(oc_config)
                oc_has_mcp = "slowave" in cfg.get("mcp", {})
            except Exception:
                pass
        oc_inst = _opencode_instructions_path()
        if oc_inst.exists():
            try:
                oc_inst_text = oc_inst.read_text(encoding="utf-8", errors="ignore")
                oc_has_lifecycle = _MARKER_START in oc_inst_text
                oc_lifecycle_version = _detect_lifecycle_version(oc_inst_text)
            except Exception:
                pass
        if oc_config.exists() or oc_inst.exists():
            statuses["opencode"] = ClientStatus(
                name="OpenCode",
                mcp_configured=oc_has_mcp,
                lifecycle_enabled=oc_has_lifecycle,
                lifecycle_version=oc_lifecycle_version,
            )

        # Codex detection — TOML config, mcp_servers key, hooks + AGENTS.md
        from slowave.cli.setup import (
            _HOOKS_MARKER,
            _codex_agents_md_path,
            _codex_config_path,
            _codex_home,
            _read_toml,
        )

        codex_cfg_path = _codex_config_path()
        codex_agents_md = _codex_agents_md_path()
        codex_override = _codex_home() / "AGENTS.override.md"
        codex_has_mcp = codex_has_hooks = codex_has_lifecycle = False
        codex_lifecycle_version: str | None = None
        if codex_cfg_path.exists():
            try:
                ccfg = _read_toml(codex_cfg_path)
                codex_has_mcp = "slowave" in ccfg.get("mcp_servers", {})
                codex_has_hooks = any(
                    _HOOKS_MARKER in h.get("command", "")
                    for event in ("UserPromptSubmit", "Stop")
                    for group in ccfg.get("hooks", {}).get(event, [])
                    for h in group.get("hooks", [])
                )
            except Exception:
                pass
        if codex_agents_md.exists():
            try:
                codex_agents_text = codex_agents_md.read_text(encoding="utf-8", errors="ignore")
                codex_has_lifecycle = _MARKER_START in codex_agents_text
                codex_lifecycle_version = _detect_lifecycle_version(codex_agents_text)
            except Exception:
                pass
        if codex_cfg_path.exists() or codex_agents_md.exists():
            statuses["codex"] = ClientStatus(
                name="Codex",
                mcp_configured=codex_has_mcp,
                hooks_installed=codex_has_hooks,
                # AGENTS.override.md shadows AGENTS.md entirely at this scope — if it
                # exists, the injected block in AGENTS.md is never read by Codex.
                lifecycle_enabled=codex_has_lifecycle and not codex_override.exists(),
                lifecycle_version=codex_lifecycle_version,
            )
    except ImportError:
        pass

    import platform

    system = platform.system()
    worker_running = False
    try:
        if system == "Darwin":
            result = subprocess.run(
                ["launchctl", "list"], capture_output=True, text=True, check=False
            )
            worker_running = "com.slowave.worker" in result.stdout
        elif system == "Linux":
            result = subprocess.run(
                ["systemctl", "--user", "is-active", "slowave-worker"],
                capture_output=True,
                text=True,
                check=False,
            )
            worker_running = result.returncode == 0
        elif system == "Windows":
            result = subprocess.run(
                [
                    "powershell",
                    "-Command",
                    "Get-ScheduledTask -TaskName SlowaveWorker -ErrorAction SilentlyContinue",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            worker_running = "SlowaveWorker" in result.stdout
    except Exception:
        pass

    statuses["worker"] = ClientStatus(name="Background worker", running=worker_running)

    # HTTP MCP daemon check
    import urllib.error
    import urllib.request

    http_daemon_running = False
    try:
        with urllib.request.urlopen("http://127.0.0.1:8766/health", timeout=1) as resp:
            http_daemon_running = resp.status == 200
    except Exception:
        pass
    statuses["http_daemon"] = ClientStatus(name="HTTP MCP daemon", running=http_daemon_running)

    return statuses


def summarize_client_status(client: ClientStatus) -> tuple[Status, str]:
    if "http mcp daemon" in client.name.lower():
        return (
            Status.OK if client.running else Status.SKIP,
            (
                "running on :8766"
                if client.running
                else "not running (start with: slowave serve start)"
            ),
        )
    if "worker" in client.name.lower():
        return (
            Status.OK if client.running else Status.SKIP,
            "running" if client.running else "not running",
        )
    if client.name in {"Cursor", "Windsurf"}:
        if client.mcp_configured:
            if not client.lifecycle_enabled:
                return (
                    Status.WARN,
                    "MCP configured (HTTP); lifecycle feedback not configured or not supported",
                )
            return (Status.OK, "MCP (HTTP)")
        return (Status.SKIP, "not configured (run: slowave setup)")
    if "desktop" in client.name.lower():
        if not client.mcp_configured:
            return (Status.SKIP, "not configured (run: slowave setup)")
        return (Status.WARN, "MCP configured (HTTP); custom instructions missing")
    if client.mcp_configured:
        parts = ["MCP (HTTP)"]
        if client.hooks_installed:
            parts.append("hooks")
        if client.lifecycle_enabled:
            if client.lifecycle_version and client.lifecycle_version != LIFECYCLE_VERSION:
                return (
                    Status.WARN,
                    f"MCP (HTTP); lifecycle block is stale "
                    f"({client.lifecycle_version}, current {LIFECYCLE_VERSION}) — "
                    "run: slowave setup",
                )
            parts.append("lifecycle")
        return (Status.OK, ", ".join(parts))
    return (Status.SKIP, "not configured (run: slowave setup)")
