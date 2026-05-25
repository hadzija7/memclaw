"""Tests for the persistent HTTP MCP bridge."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from memclaw.backends.mcp_bridge import HttpMcpServer, mcp_servers_for


@pytest.mark.asyncio
async def test_http_mcp_server_start_stop_lifecycle():
    server = HttpMcpServer()
    executor = AsyncMock()
    executor.execute.return_value = "ok"

    await server.start(executor, port=0)
    assert server.is_running
    config = server.config
    assert config is not None
    assert config.url.endswith("/mcp")
    assert config.type == "http"
    assert mcp_servers_for(config) == {"memclaw": config}

    await server.stop()
    assert not server.is_running
    assert server.config is None


@pytest.mark.asyncio
async def test_http_mcp_server_start_is_idempotent():
    server = HttpMcpServer()
    executor = AsyncMock()

    await server.start(executor, port=0)
    first_url = server.config.url
    await server.start(executor, port=0)
    assert server.config.url == first_url

    await server.stop()


@pytest.mark.asyncio
async def test_http_mcp_server_stop_when_not_started():
    server = HttpMcpServer()
    await server.stop()
    assert not server.is_running


@pytest.mark.asyncio
async def test_http_mcp_server_list_tools_over_http():
    server = HttpMcpServer()
    executor = AsyncMock()

    await server.start(executor, port=0)
    try:
        config = server.config
        assert config is not None
        async with httpx.AsyncClient() as client:
            response = await client.get(config.url, timeout=5.0)
        assert response.status_code in (200, 405, 406)
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_http_mcp_server_port_in_use_raises():
    server_a = HttpMcpServer()
    server_b = HttpMcpServer()
    executor = MagicMock()

    await server_a.start(executor, port=0)
    port = int(server_a.config.url.split(":")[2].split("/")[0])
    try:
        with pytest.raises(OSError, match="Cannot bind MCP HTTP server"):
            await server_b.start(executor, port=port)
    finally:
        await server_a.stop()
