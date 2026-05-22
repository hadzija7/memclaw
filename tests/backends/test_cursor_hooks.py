"""Tests for Cursor SDK hook installation and policy."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from memclaw.backends.cursor_hooks import (
    HOOKS_VERSION,
    cursor_hook_script_path,
    cursor_hooks_installed,
    cursor_hooks_json_path,
    cursor_hooks_status,
    ensure_cursor_hooks,
)
from memclaw.defaults.cursor_hooks.allow_memclaw_tools import is_allowed_tool


class TestHookPolicy:
    @pytest.mark.parametrize(
        ("tool_name", "tool_input", "expected"),
        [
            ("mcp", {"providerIdentifier": "memclaw", "toolName": "memory_save"}, True),
            ("mcp", {"providerIdentifier": "memclaw", "toolName": "mcp_auth"}, True),
            ("MCP:memclaw:memory_search", {}, True),
            ("Read", {"path": "/tmp/x"}, False),
            ("Grep", {"pattern": "foo"}, False),
            ("Glob", {"globPattern": "**/*"}, False),
            ("Shell", {"command": "ls"}, False),
            ("Task", {"description": "explore"}, False),
            ("mcp", {"providerIdentifier": "other", "toolName": "x"}, False),
        ],
    )
    def test_is_allowed_tool(self, tool_name, tool_input, expected):
        assert is_allowed_tool(tool_name, tool_input) is expected

    def test_hook_script_subprocess(self, tmp_path: Path):
        ensure_cursor_hooks(tmp_path)
        script = cursor_hook_script_path(tmp_path)
        payload = {"tool_name": "Read", "tool_input": {"path": "/etc/passwd"}}

        proc = subprocess.run(
            [sys.executable, str(script)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=True,
        )
        result = json.loads(proc.stdout)
        assert result["permission"] == "deny"

        payload = {
            "tool_name": "mcp",
            "tool_input": {
                "providerIdentifier": "memclaw",
                "toolName": "memory_save",
                "args": {"content": "hello"},
            },
        }
        proc = subprocess.run(
            [sys.executable, str(script)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=True,
        )
        result = json.loads(proc.stdout)
        assert result["permission"] == "allow"


class TestHookInstallation:
    def test_ensure_cursor_hooks_installs_files(self, tmp_path: Path):
        assert not cursor_hooks_installed(tmp_path)
        assert ensure_cursor_hooks(tmp_path) is True
        assert cursor_hooks_json_path(tmp_path).is_file()
        assert cursor_hook_script_path(tmp_path).is_file()
        assert cursor_hooks_installed(tmp_path) is True
        assert cursor_hooks_status(tmp_path).startswith("ready")

    def test_ensure_cursor_hooks_refreshes_outdated_script(self, tmp_path: Path):
        ensure_cursor_hooks(tmp_path)
        script = cursor_hook_script_path(tmp_path)
        script.write_text("# memclaw-hooks-version: 0\n", encoding="utf-8")
        assert cursor_hooks_installed(tmp_path) is False
        assert ensure_cursor_hooks(tmp_path) is True
        assert cursor_hooks_installed(tmp_path) is True

    def test_installed_version_matches_package(self, tmp_path: Path):
        ensure_cursor_hooks(tmp_path)
        script = cursor_hook_script_path(tmp_path)
        assert f"memclaw-hooks-version: {HOOKS_VERSION}" in script.read_text(encoding="utf-8")
