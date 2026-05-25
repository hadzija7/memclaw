"""Canonical tool allow/deny policy shared across agent backends and Cursor hooks."""

from __future__ import annotations

from ..tools import TOOL_DEFINITIONS
from .mcp_tools import MCP_SERVER_NAME

HOOKS_VERSION = 7

MEMCLAW_TOOL_NAMES = frozenset(defn["name"] for defn in TOOL_DEFINITIONS)

# PascalCase names for Claude disallowed_tools and Cursor preToolUse matcher.
BUILTIN_TOOLS_DISALLOW: list[str] = [
    "Bash",
    "BashOutput",
    "KillBash",
    "Read",
    "Write",
    "Edit",
    "NotebookEdit",
    "Grep",
    "Glob",
    "Task",
    "WebFetch",
    "WebSearch",
    "TodoWrite",
    "SlashCommand",
    "ExitPlanMode",
    "Shell",
    "Delete",
    "ListDir",
    "ApplyPatch",
]

# Lowercase variants and Cursor-specific aliases for the standalone hook script.
CURSOR_BUILTIN_TOOL_NAMES = frozenset(
    name.lower() for name in BUILTIN_TOOLS_DISALLOW
) | frozenset(
    {
        "list_dir",
        "search",
        "apply_patch",
    }
)

CURSOR_PRETOOLUSE_MATCHER = "|".join(BUILTIN_TOOLS_DISALLOW)

CALL_MCP_BRIDGE_TOOL_NAMES = frozenset({"call_mcp_tool", "callmcptool"})


def hook_policy_payload() -> dict[str, object]:
    """JSON-serializable policy written beside the installed hook script."""
    return {
        "hooks_version": HOOKS_VERSION,
        "allowed_mcp_provider": MCP_SERVER_NAME,
        "memclaw_tool_names": sorted(MEMCLAW_TOOL_NAMES),
        "cursor_builtin_tool_names": sorted(CURSOR_BUILTIN_TOOL_NAMES),
        "call_mcp_bridge_tool_names": sorted(CALL_MCP_BRIDGE_TOOL_NAMES),
    }
