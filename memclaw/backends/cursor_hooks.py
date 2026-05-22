"""Install and verify Cursor SDK hooks that restrict tool use to Memclaw MCP."""

from __future__ import annotations

import importlib.resources
import json
import re
import shutil
from pathlib import Path

from .mcp_tools import MCP_SERVER_NAME

HOOKS_VERSION = 1
_HOOKS_JSON = "hooks.json"
_HOOK_SCRIPT = "allow_memclaw_tools.py"
_VERSION_MARKER = re.compile(r"memclaw-hooks-version:\s*(\d+)")


def cursor_hooks_dir(memory_dir: Path) -> Path:
    return Path(memory_dir) / ".cursor"


def cursor_hook_script_path(memory_dir: Path) -> Path:
    return cursor_hooks_dir(memory_dir) / "hooks" / _HOOK_SCRIPT


def cursor_hooks_json_path(memory_dir: Path) -> Path:
    return cursor_hooks_dir(memory_dir) / _HOOKS_JSON


def _packaged_defaults() -> Path:
    return Path(str(importlib.resources.files("memclaw.defaults") / "cursor_hooks"))


def _installed_hook_version(script_path: Path) -> int | None:
    if not script_path.is_file():
        return None
    match = _VERSION_MARKER.search(script_path.read_text(encoding="utf-8"))
    if not match:
        return None
    return int(match.group(1))


def cursor_hooks_installed(memory_dir: Path) -> bool:
    """Return True when Memclaw's Cursor preToolUse hook is present and current."""
    hooks_json = cursor_hooks_json_path(memory_dir)
    script_path = cursor_hook_script_path(memory_dir)
    if not hooks_json.is_file() or not script_path.is_file():
        return False
    return _installed_hook_version(script_path) == HOOKS_VERSION


def ensure_cursor_hooks(memory_dir: Path) -> bool:
    """Install or refresh Cursor hooks under *memory_dir*.

    Returns True when hooks are ready to use after this call.
    """
    memory_dir = Path(memory_dir)
    packaged = _packaged_defaults()
    target_dir = cursor_hooks_dir(memory_dir)
    target_hooks = target_dir / "hooks"
    target_hooks.mkdir(parents=True, exist_ok=True)

    shutil.copy2(packaged / _HOOKS_JSON, cursor_hooks_json_path(memory_dir))
    shutil.copy2(packaged / _HOOK_SCRIPT, cursor_hook_script_path(memory_dir))
    cursor_hook_script_path(memory_dir).chmod(0o755)

    return cursor_hooks_installed(memory_dir)


def cursor_hooks_status(memory_dir: Path) -> str:
    """Human-readable hook status for logs and configuration help."""
    hooks_json = cursor_hooks_json_path(memory_dir)
    script_path = cursor_hook_script_path(memory_dir)
    if not hooks_json.is_file() or not script_path.is_file():
        return "missing"
    version = _installed_hook_version(script_path)
    if version != HOOKS_VERSION:
        return f"outdated (found v{version}, need v{HOOKS_VERSION})"
    try:
        config = json.loads(hooks_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "invalid hooks.json"
    entries = config.get("hooks", {}).get("preToolUse", [])
    if not entries:
        return "preToolUse hook not configured"
    command = entries[0].get("command", "")
    if _HOOK_SCRIPT not in command:
        return "preToolUse hook points elsewhere"
    return f"ready (memclaw MCP provider={MCP_SERVER_NAME!r})"
