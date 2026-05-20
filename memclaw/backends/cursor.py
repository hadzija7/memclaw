"""Cursor Python SDK agent backend."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from loguru import logger

from .base import TurnResult

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


def _build_combined_prompt(
    *,
    system_prompt: str,
    user_message: str,
    image_b64: str | None = None,
    image_media_type: str = "image/jpeg",
) -> str:
    sections = [
        system_prompt.strip(),
        "",
        "---",
        "",
        user_message.strip(),
    ]
    if image_b64:
        sections.extend(
            [
                "",
                (
                    f"[User attached an image ({image_media_type}). "
                    "The Cursor SDK backend does not pass raw image bytes; "
                    "use any description in the message above.]"
                ),
            ]
        )
    return "\n".join(sections)


def _extract_run_text(result: Any) -> str:
    if result is None:
        return ""
    for attr in ("result", "text", "output"):
        value = getattr(result, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    return str(result) if result else ""


def _agent_options(*, api_key: str, cwd: str, model: str) -> Any:
    from cursor_sdk import AgentOptions, LocalAgentOptions

    return AgentOptions(
        api_key=api_key,
        model=model,
        local=LocalAgentOptions(cwd=cwd),
    )


class CursorBackend:
    """Memclaw backend powered by the Cursor Python SDK (local runtime)."""

    name = "cursor"
    display_name = "Cursor SDK"

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
            "Set AGENT_BACKEND=cursor in ~/.memclaw/.env to use this backend.\n"
            "Memclaw tools (memory_save, reminders, etc.) are not executed in-process; "
            "the Cursor agent uses its own tool surface unless you add an MCP bridge."
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
        from cursor_sdk import CursorAgentError, LocalAgentOptions

        if tool_executor is not None:
            logger.debug(
                "Cursor backend received tool_executor; Memclaw tools are not "
                "wired through the SDK (max_turns={n} applies to SDK-internal tools only)",
                n=max_turns,
            )

        if not self._api_key:
            raise RuntimeError("CURSOR_API_KEY is not configured")

        prompt = _build_combined_prompt(
            system_prompt=system_prompt,
            user_message=user_message,
            image_b64=image_b64,
            image_media_type=image_media_type,
        )

        try:
            client = await self._ensure_client()
            agent = await client.agents.create(
                api_key=self._api_key,
                model=self._model,
                local=LocalAgentOptions(cwd=self._cwd),
            )
            try:
                run = await agent.send(prompt)
                text = await run.text()
                if not text or not text.strip():
                    text = _extract_run_text(await run.wait())
            finally:
                await agent.aclose()

            if not text.strip():
                text = "I couldn't generate a response."

            return TurnResult(text=text, num_turns=1)
        except CursorAgentError as exc:
            logger.error("Cursor SDK turn failed: {msg}", msg=exc.message)
            raise RuntimeError(f"Cursor SDK error: {exc.message}") from exc
