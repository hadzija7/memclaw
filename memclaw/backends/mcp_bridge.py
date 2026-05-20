"""Ephemeral HTTP MCP bridge for Cursor SDK tool wiring."""

from __future__ import annotations

import asyncio
import contextlib
import socket
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import uvicorn
from cursor_sdk import HttpMcpServerConfig
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.routing import Route

from .mcp_tools import MCP_SERVER_NAME, build_memclaw_mcp_server

if TYPE_CHECKING:
    from ..tools import ToolExecutor


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _wait_until_listening(host: str, port: int, *, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        try:
            reader, writer = await asyncio.open_connection(host, port)
            writer.close()
            await writer.wait_closed()
            return
        except OSError:
            await asyncio.sleep(0.05)
    raise TimeoutError(f"MCP bridge did not start on {host}:{port} within {timeout}s")


class _StreamableHTTPASGIApp:
    """Minimal ASGI adapter for StreamableHTTPSessionManager."""

    def __init__(self, session_manager: StreamableHTTPSessionManager) -> None:
        self._session_manager = session_manager

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        await self._session_manager.handle_request(scope, receive, send)


class EphemeralHttpMcpBridge:
    """Run Memclaw tools as a local HTTP MCP server for one agent turn."""

    def __init__(self, executor: "ToolExecutor") -> None:
        self._executor = executor
        self._port = 0
        self._session_manager: StreamableHTTPSessionManager | None = None
        self._uvicorn_server: uvicorn.Server | None = None
        self._uvicorn_task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> HttpMcpServerConfig:
        mcp_server = build_memclaw_mcp_server(self._executor)
        self._session_manager = StreamableHTTPSessionManager(
            app=mcp_server,
            stateless=True,
        )
        asgi_app = _StreamableHTTPASGIApp(self._session_manager)
        session_manager = self._session_manager

        @asynccontextmanager
        async def lifespan(_app: Starlette):
            async with session_manager.run():
                yield

        self._port = _pick_free_port()
        starlette_app = Starlette(
            routes=[Route("/mcp", endpoint=asgi_app)],
            lifespan=lifespan,
        )
        config = uvicorn.Config(
            starlette_app,
            host="127.0.0.1",
            port=self._port,
            log_level="warning",
        )
        self._uvicorn_server = uvicorn.Server(config)
        self._uvicorn_task = asyncio.create_task(self._uvicorn_server.serve())
        try:
            await _wait_until_listening("127.0.0.1", self._port)
        except BaseException:
            await self._shutdown_uvicorn()
            raise
        return HttpMcpServerConfig(
            url=f"http://127.0.0.1:{self._port}/mcp",
            type="http",
        )

    async def _shutdown_uvicorn(self) -> None:
        if self._uvicorn_server is not None:
            self._uvicorn_server.should_exit = True
        if self._uvicorn_task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._uvicorn_task
        self._uvicorn_task = None
        self._uvicorn_server = None
        self._session_manager = None

    async def __aexit__(self, *_exc: object) -> None:
        await self._shutdown_uvicorn()


def mcp_servers_for(config: HttpMcpServerConfig) -> dict[str, HttpMcpServerConfig]:
    return {MCP_SERVER_NAME: config}
