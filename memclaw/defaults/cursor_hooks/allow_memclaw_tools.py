#!/usr/bin/env python3
"""Cursor preToolUse hook: allow only Memclaw MCP tools.

Installed to ~/.memclaw/.cursor/hooks/ by the Cursor backend.
memclaw-hooks-version: 1
"""

from __future__ import annotations

import json
import sys

ALLOWED_MCP_PROVIDER = "memclaw"
HOOKS_VERSION = 1

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


def _mcp_provider(tool_name: str, tool_input: dict) -> str | None:
    lowered = tool_name.lower()
    if lowered == "mcp":
        provider = tool_input.get("providerIdentifier") or tool_input.get("provider")
        return str(provider).strip() if provider else None
    if tool_name.startswith("MCP:"):
        parts = tool_name.split(":", 2)
        if len(parts) >= 2 and parts[1].strip():
            return parts[1].strip()
    return None


def is_allowed_tool(tool_name: str, tool_input: dict | None = None) -> bool:
    """Return True when the tool call should be allowed."""
    tool_input = tool_input or {}
    provider = _mcp_provider(tool_name, tool_input)
    return provider == ALLOWED_MCP_PROVIDER


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        print(json.dumps(_DENY_RESPONSE))
        return 0

    tool_name = str(payload.get("tool_name", "")).strip()
    tool_input = _parse_tool_input(payload.get("tool_input"))

    if is_allowed_tool(tool_name, tool_input):
        print(json.dumps({"permission": "allow"}))
        return 0

    print(json.dumps(_DENY_RESPONSE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
