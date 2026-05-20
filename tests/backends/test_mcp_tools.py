"""Tests for Memclaw MCP tool server helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.types import CallToolRequest, CallToolRequestParams, ListToolsRequest

from memclaw.backends.mcp_tools import MCP_SERVER_NAME, build_memclaw_mcp_server
from memclaw.tools import TOOL_DEFINITIONS


@pytest.mark.asyncio
async def test_build_memclaw_mcp_server_lists_tools():
    executor = MagicMock()
    server = build_memclaw_mcp_server(executor)
    result = await server.request_handlers[ListToolsRequest](ListToolsRequest())
    assert len(result.root.tools) == len(TOOL_DEFINITIONS)
    assert {tool.name for tool in result.root.tools} == {defn["name"] for defn in TOOL_DEFINITIONS}


@pytest.mark.asyncio
async def test_build_memclaw_mcp_server_routes_tool_calls():
    executor = AsyncMock()
    executor.execute.return_value = "saved"
    server = build_memclaw_mcp_server(executor)

    result = await server.request_handlers[CallToolRequest](
        CallToolRequest(
            params=CallToolRequestParams(
                name="memory_save",
                arguments={"content": "hello"},
            )
        )
    )

    executor.execute.assert_awaited_once_with("memory_save", {"content": "hello"})
    assert result.root.content[0].text == "saved"


def test_mcp_server_name_matches_cursor_registry():
    assert MCP_SERVER_NAME == "memclaw"
