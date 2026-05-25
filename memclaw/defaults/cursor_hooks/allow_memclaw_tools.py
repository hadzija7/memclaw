#!/usr/bin/env python3
"""Cursor hooks: allow only Memclaw MCP tools.

Installed to ~/.memclaw/.cursor/hooks/ by the Cursor backend.
Handles preToolUse, beforeMCPExecution, beforeReadFile, and beforeShellExecution.
memclaw-hooks-version: 9

Tool allowlists are loaded from hook_policy.json (written on install from
memclaw.backends.tool_policy).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

_POLICY_FILE = "hook_policy.json"


def _load_policy() -> dict:
    policy_path = Path(__file__).resolve().parent / _POLICY_FILE
    data = json.loads(policy_path.read_text(encoding="utf-8"))
    return data


_POLICY = _load_policy()

ALLOWED_MCP_PROVIDER = str(_POLICY["allowed_mcp_provider"])
HOOKS_VERSION = int(_POLICY["hooks_version"])
MEMCLAW_MCP_TOOL_NAMES = frozenset(_POLICY["memclaw_tool_names"])
CURSOR_BUILTIN_TOOL_NAMES = frozenset(_POLICY["cursor_builtin_tool_names"])
CALL_MCP_BRIDGE_TOOL_NAMES = frozenset(_POLICY["call_mcp_bridge_tool_names"])

_DENY_RESPONSE = {
    "permission": "deny",
    "agent_message": (
        "Built-in Cursor tools are disabled in Memclaw. "
        "Use Memclaw MCP tools only (memory_save, memory_search, "
        "image_save, image_search, reminder_create, file_read, file_write, ...)."
    ),
}


def _parse_tool_input(raw: object) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _normalize_tool_name(tool_name: str) -> str:
    return tool_name.strip()


def _is_call_mcp_bridge_tool(tool_name: str) -> bool:
    return _normalize_tool_name(tool_name).lower() in CALL_MCP_BRIDGE_TOOL_NAMES


def _is_builtin_tool(tool_name: str) -> bool:
    normalized = _normalize_tool_name(tool_name).lower()
    if normalized in CURSOR_BUILTIN_TOOL_NAMES:
        return True
    if normalized.startswith("mcp:"):
        provider = _mcp_provider_from_tool_name(tool_name)
        return provider != ALLOWED_MCP_PROVIDER
    return False


def _expected_mcp_port() -> int:
    """Port Memclaw's HTTP MCP server uses (policy, then env, then default)."""
    raw = _POLICY.get("mcp_http_port")
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass
    env = os.environ.get("MEMCLAW_MCP_PORT", "").strip()
    if env:
        try:
            return int(env)
        except ValueError:
            pass
    return 17373


def _is_local_memclaw_mcp_url(url: str) -> bool:
    """Return True when *url* points at Memclaw's local HTTP MCP server."""
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return False
    if parsed.scheme != "http":
        return False
    host = (parsed.hostname or "").lower()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        return False
    url_port = parsed.port
    if url_port is None:
        url_port = 80
    if url_port != _expected_mcp_port():
        return False
    path = (parsed.path or "").rstrip("/")
    return path in {"", "/mcp"} or path.startswith("/mcp/")


def _parse_mcp_colon_tool_name(tool_name: str) -> tuple[str | None, str | None]:
    """Parse ``MCP:provider:tool`` or ``MCP:tool`` (single-server shorthand)."""
    if not tool_name.startswith("MCP:"):
        return None, None
    parts = [part.strip() for part in tool_name.split(":", 2)]
    if len(parts) < 2 or not parts[1]:
        return None, None
    if len(parts) >= 3 and parts[2]:
        return parts[1], parts[2]
    name = parts[1]
    if name in MEMCLAW_MCP_TOOL_NAMES:
        return ALLOWED_MCP_PROVIDER, name
    return name, None


def _mcp_provider_from_tool_name(tool_name: str) -> str | None:
    provider, _tool = _parse_mcp_colon_tool_name(tool_name)
    return provider


def _memclaw_tool_from_tool_name(tool_name: str) -> str | None:
    prefix = f"{ALLOWED_MCP_PROVIDER}_"
    if tool_name.startswith(prefix):
        return tool_name[len(prefix) :]
    if tool_name in MEMCLAW_MCP_TOOL_NAMES:
        return tool_name
    provider, tool = _parse_mcp_colon_tool_name(tool_name)
    if provider == ALLOWED_MCP_PROVIDER and tool in MEMCLAW_MCP_TOOL_NAMES:
        return tool
    return None


def _memclaw_tool_from_mcp_input(tool_input: dict) -> str | None:
    for key in ("toolName", "tool_name", "name"):
        value = tool_input.get(key)
        if value and str(value) in MEMCLAW_MCP_TOOL_NAMES:
            return str(value)
    return None


def _mcp_provider(tool_name: str, tool_input: dict) -> str | None:
    lowered = _normalize_tool_name(tool_name).lower()
    if lowered == "mcp" or lowered in CALL_MCP_BRIDGE_TOOL_NAMES:
        for key in ("providerIdentifier", "provider", "server"):
            value = tool_input.get(key)
            if value:
                return str(value).strip()
        return None
    provider = _mcp_provider_from_tool_name(tool_name)
    if provider:
        return provider
    return None


def _is_allowed_mcp_bridge(tool_name: str, tool_input: dict) -> bool:
    """Return True when an MCP bridge call targets Memclaw."""
    provider = _mcp_provider(tool_name, tool_input)
    if provider == ALLOWED_MCP_PROVIDER:
        return _memclaw_tool_from_mcp_input(tool_input) is not None
    if provider is not None:
        return False
    return _memclaw_tool_from_mcp_input(tool_input) is not None


def is_allowed_tool(tool_name: str, tool_input: dict | None = None) -> bool:
    """Return True when the preToolUse call should be allowed."""
    tool_name = _normalize_tool_name(tool_name)
    tool_input = tool_input or {}

    if _is_call_mcp_bridge_tool(tool_name):
        return _is_allowed_mcp_bridge(tool_name, tool_input)

    if _is_builtin_tool(tool_name):
        return False

    if tool_name.lower() == "mcp":
        return _is_allowed_mcp_bridge(tool_name, tool_input)

    resolved = _memclaw_tool_from_tool_name(tool_name)
    return bool(resolved and resolved in MEMCLAW_MCP_TOOL_NAMES)


def is_allowed_mcp_execution(payload: dict) -> bool:
    """Return True when a beforeMCPExecution call targets Memclaw's local MCP."""
    url = str(payload.get("url", "")).strip()
    if url:
        return _is_local_memclaw_mcp_url(url)

    command = str(payload.get("command", "")).strip()
    if command == ALLOWED_MCP_PROVIDER:
        return True

    tool_name = _normalize_tool_name(str(payload.get("tool_name", "")))
    if _is_builtin_tool(tool_name):
        return False
    tool_input = _parse_tool_input(payload.get("tool_input"))
    return is_allowed_tool(tool_name, tool_input)


def is_allowed_file_read(payload: dict) -> bool:
    """Built-in Read is always blocked; use Memclaw file_read MCP instead."""
    return False


def is_allowed_shell_execution(payload: dict) -> bool:
    """Built-in shell is always blocked in Memclaw."""
    return False


def _handle_event(payload: dict) -> bool:
    event = str(payload.get("hook_event_name", "")).strip()

    if event == "beforeMCPExecution":
        return is_allowed_mcp_execution(payload)
    if event == "beforeReadFile":
        return is_allowed_file_read(payload)
    if event == "beforeShellExecution":
        return is_allowed_shell_execution(payload)
    if event == "preToolUse" or not event:
        tool_name = _normalize_tool_name(str(payload.get("tool_name", "")))
        tool_input = _parse_tool_input(payload.get("tool_input"))
        return is_allowed_tool(tool_name, tool_input)

    return False


def _resolve_memory_dir() -> Path:
    """Return the active Memclaw memory directory for audit logs."""
    configured = os.environ.get("MEMCLAW_MEMORY_DIR", "").strip()
    if configured:
        return Path(configured)
    # Installed at <memory_dir>/.cursor/hooks/allow_memclaw_tools.py
    return Path(__file__).resolve().parent.parent.parent


def _audit_deny(payload: dict) -> None:
    """Append denied hook payloads for troubleshooting (best-effort)."""
    log_path = _resolve_memory_dir() / "hook-deny.jsonl"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        print(json.dumps(_DENY_RESPONSE))
        return 0

    if _handle_event(payload):
        print(json.dumps({"permission": "allow"}))
        return 0

    _audit_deny(payload)
    print(json.dumps(_DENY_RESPONSE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
