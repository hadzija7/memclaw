"""Cursor Python SDK agent backend."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, ClassVar

from loguru import logger

from .base import TurnResult
from .mcp_bridge import EphemeralHttpMcpBridge, mcp_servers_for
from .mcp_tools import MCP_SERVER_NAME

if TYPE_CHECKING:
    from rich.console import Console

    from ..config import MemclawConfig
    from ..tools import ToolExecutor

_DEFAULT_MODEL = "composer-2.5"

# Scrub Claude credentials when switching to Cursor so they can't shadow selection.
_DROP_KEYS = ["ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"]


def _cursor_api_key(config: "MemclawConfig") -> str:
    return (config.cursor_api_key or os.environ.get("CURSOR_API_KEY", "")).strip()


def _cursor_model(config: "MemclawConfig") -> str:
    model = (config.cursor_model or os.environ.get("CURSOR_MODEL", "")).strip()
    return model or _DEFAULT_MODEL


def _build_combined_prompt(*, system_prompt: str, user_message: str) -> str:
    return "\n".join(
        [
            system_prompt.strip(),
            "",
            "---",
            "",
            user_message.strip(),
        ]
    )


def _build_user_message(
    *,
    system_prompt: str,
    user_message: str,
    image_b64: str | None = None,
    image_media_type: str = "image/jpeg",
) -> str | Any:
    from cursor_sdk import SDKImage, UserMessage

    prompt = _build_combined_prompt(
        system_prompt=system_prompt,
        user_message=user_message,
    )
    if not image_b64:
        return prompt
    return UserMessage(
        text=prompt,
        images=[SDKImage.from_data(image_b64, image_media_type)],
    )


def _extract_run_text(result: Any) -> str:
    if result is None:
        return ""
    for attr in ("result", "text", "output"):
        value = getattr(result, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    return str(result) if result else ""


def _agent_options(
    *,
    api_key: str,
    cwd: str,
    model: str,
    mcp_servers: dict[str, Any] | None = None,
) -> Any:
    from cursor_sdk import AgentOptions, LocalAgentOptions

    return AgentOptions(
        api_key=api_key,
        model=model,
        local=LocalAgentOptions(cwd=cwd),
        mcp_servers=mcp_servers,
    )


def _log_tool_call(name: str, args: Any) -> None:
    args_str = json.dumps(args, ensure_ascii=False) if args is not None else "{}"
    if len(args_str) > 300:
        args_str = args_str[:300] + "..."
    tool_name = name
    prefix = f"{MCP_SERVER_NAME}_"
    if tool_name.startswith(prefix):
        tool_name = tool_name[len(prefix) :]
    logger.info("Tool call: {name}({args})", name=tool_name, args=args_str)


def _assistant_message_text(message: Any) -> str:
    """Concatenate all text blocks from a Cursor SDK assistant stream message."""
    content = getattr(getattr(message, "message", None), "content", ())
    parts: list[str] = []
    for block in content:
        if getattr(block, "type", None) == "text":
            text = getattr(block, "text", "")
            if text:
                parts.append(text)
    return "".join(parts)


async def _collect_run_result(run: Any, *, max_turns: int) -> TurnResult:
    """Drain the run stream for logging and return a normalized TurnResult."""
    last_text = ""
    tool_steps = 0
    cancelled_for_cap = False

    async for message in run.messages():
        msg_type = getattr(message, "type", None)
        if msg_type == "assistant":
            turn_text = _assistant_message_text(message)
            if turn_text:
                last_text = turn_text
        elif msg_type == "tool_call" and getattr(message, "status", "") == "running":
            _log_tool_call(getattr(message, "name", ""), getattr(message, "args", None))
            tool_steps += 1
            if max_turns > 0 and tool_steps >= max_turns:
                logger.debug("Cursor run capped at max_turns={max}", max=max_turns)
                await run.cancel()
                cancelled_for_cap = True
                break

    if cancelled_for_cap:
        num_turns = max(tool_steps, 1)
        if max_turns > 0:
            num_turns = min(num_turns, max_turns)
        return TurnResult(text=last_text, num_turns=num_turns)

    wait_result = await run.wait()
    wait_text = _extract_run_text(wait_result)
    if len(wait_text) > len(last_text):
        last_text = wait_text
    elif not last_text:
        last_text = wait_text

    num_turns = max(getattr(wait_result, "num_turns", 0) or 0, tool_steps, 1)
    if max_turns > 0:
        num_turns = min(num_turns, max_turns)
    return TurnResult(text=last_text, num_turns=num_turns)


class CursorAgentBackend:
    """Cursor Python SDK implementation of the AgentBackend protocol."""

    name: ClassVar[str] = "cursor"
    display_name: ClassVar[str] = "Cursor SDK"

    def __init__(self, config: "MemclawConfig") -> None:
        self.config = config
        self._api_key = _cursor_api_key(config)
        self._model = _cursor_model(config)
        self._cwd = str(config.memory_dir)
        self._client: Any = None
        self.bills_per_token = True

    @classmethod
    def is_configured(cls, config: "MemclawConfig") -> bool:
        return bool(_cursor_api_key(config))

    @classmethod
    def configuration_help(cls) -> str:
        return (
            "Cursor SDK backend requires CURSOR_API_KEY "
            "(Cursor Dashboard → Integrations, or a team service account key).\n"
            "Optional: CURSOR_MODEL (default: composer-2.5).\n"
            "Set AGENT_BACKEND=cursor in ~/.memclaw/.env to use this backend."
        )

    @classmethod
    def wizard_setup(
        cls,
        console: "Console",
        existing: dict[str, str],
    ) -> tuple[dict[str, str], list[str]]:
        from ..setup import _masked_input

        current = existing.get("CURSOR_API_KEY", os.environ.get("CURSOR_API_KEY", ""))
        answer = _masked_input("Cursor API key (required)")
        value = answer or current
        if not value:
            console.print("[red]Error:[/red] Cursor API key is required.")
            raise SystemExit(1)

        values: dict[str, str] = {"CURSOR_API_KEY": value}

        model_current = existing.get("CURSOR_MODEL", os.environ.get("CURSOR_MODEL", ""))
        from rich.prompt import Prompt

        model_answer = Prompt.ask(
            f"Cursor model [{model_current or _DEFAULT_MODEL}]",
            default=model_current or _DEFAULT_MODEL,
            show_default=False,
        )
        if model_answer.strip():
            values["CURSOR_MODEL"] = model_answer.strip()

        return values, list(_DROP_KEYS)

    async def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        from cursor_sdk import AsyncClient

        self._client = await AsyncClient.launch_bridge(workspace=self._cwd)
        return self._client

    async def close(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            await client.aclose()

    async def run_one_shot(self, *, system_prompt: str, user_message: str) -> str:
        from cursor_sdk import AsyncAgent, CursorAgentError

        if not self._api_key:
            raise RuntimeError("CURSOR_API_KEY is not configured")

        prompt = _build_combined_prompt(
            system_prompt=system_prompt,
            user_message=user_message,
        )
        options = _agent_options(
            api_key=self._api_key,
            cwd=self._cwd,
            model=self._model,
        )

        try:
            client = await self._ensure_client()
            result = await AsyncAgent.prompt(prompt, options, client=client)
            text = _extract_run_text(result)
            if not text.strip():
                return "I couldn't generate a response."
            return text
        except CursorAgentError as exc:
            logger.error("Cursor SDK one-shot failed: {msg}", msg=exc.message)
            raise RuntimeError(f"Cursor SDK error: {exc.message}") from exc

    async def run_turn(
        self,
        *,
        system_prompt: str,
        user_message: str,
        tool_executor: "ToolExecutor",
        image_b64: str | None = None,
        image_media_type: str = "image/jpeg",
        max_turns: int = 10,
    ) -> TurnResult:
        from cursor_sdk import CursorAgentError, SendOptions

        if not self._api_key:
            raise RuntimeError("CURSOR_API_KEY is not configured")

        message = _build_user_message(
            system_prompt=system_prompt,
            user_message=user_message,
            image_b64=image_b64,
            image_media_type=image_media_type,
        )

        try:
            client = await self._ensure_client()
            async with EphemeralHttpMcpBridge(tool_executor) as mcp_config:
                mcp_servers = mcp_servers_for(mcp_config)
                agent = await client.agents.create(
                    _agent_options(
                        api_key=self._api_key,
                        cwd=self._cwd,
                        model=self._model,
                        mcp_servers=mcp_servers,
                    )
                )
                try:
                    run = await agent.send(
                        message,
                        SendOptions(mcp_servers=mcp_servers),
                    )
                    result = await _collect_run_result(run, max_turns=max_turns)
                finally:
                    await agent.close()

            if not result.text.strip():
                result.text = "I couldn't generate a response."
            return result
        except CursorAgentError as exc:
            logger.error("Cursor SDK turn failed: {msg}", msg=exc.message)
            raise RuntimeError(f"Cursor SDK error: {exc.message}") from exc
