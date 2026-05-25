"""Adapters for Cursor SDK stream messages and tool-call naming."""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from .base import TurnResult
from .mcp_tools import MCP_SERVER_NAME


def normalize_tool_call(name: str, args: Any) -> tuple[str, Any]:
    """Map Cursor SDK MCP wrapper calls to Memclaw tool names for logging."""
    if isinstance(args, dict):
        if name.lower() == "mcp":
            inner_name = args.get("toolName") or args.get("tool_name")
            if inner_name:
                inner_args = args.get("args", args)
                return str(inner_name), inner_args
    prefix = f"{MCP_SERVER_NAME}_"
    if name.startswith(prefix):
        return name[len(prefix) :], args
    return name, args


def extract_run_text(result: Any) -> str:
    if result is None:
        return ""
    for attr in ("result", "text", "output"):
        value = getattr(result, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def assistant_message_text(message: Any) -> str:
    """Concatenate all text blocks from a Cursor SDK assistant stream message."""
    content = getattr(getattr(message, "message", None), "content", ())
    parts: list[str] = []
    for block in content:
        if getattr(block, "type", None) == "text":
            text = getattr(block, "text", "")
            if text:
                parts.append(text)
    return "".join(parts)


def _log_tool_call(name: str, args: Any) -> None:
    tool_name, tool_args = normalize_tool_call(name, args)
    if tool_args is None:
        args_str = "{}"
    else:
        try:
            args_str = json.dumps(tool_args, ensure_ascii=False)
        except (TypeError, ValueError):
            args_str = str(tool_args)
    if len(args_str) > 300:
        args_str = args_str[:300] + "..."
    logger.info("Tool call: {name}({args})", name=tool_name, args=args_str)


def _log_tool_result(name: str, args: Any, *, status: str, result: Any) -> None:
    tool_name, _ = normalize_tool_call(name, args)
    if result is None:
        result_str = ""
    elif isinstance(result, str):
        result_str = result
    else:
        try:
            result_str = json.dumps(result, ensure_ascii=False)
        except (TypeError, ValueError):
            result_str = str(result)
    if len(result_str) > 300:
        result_str = result_str[:300] + "..."
    if status == "error":
        logger.warning(
            "Tool {name} failed: {result}",
            name=tool_name,
            result=result_str or "(no details)",
        )
    else:
        logger.info(
            "Tool {name} completed: {result}",
            name=tool_name,
            result=result_str or "(ok)",
        )


async def collect_run_result(run: Any, *, max_turns: int) -> TurnResult:
    """Drain the run stream for logging and return a normalized TurnResult."""
    last_text = ""
    pending_tools: dict[str, str] = {}

    async for message in run.messages():
        msg_type = getattr(message, "type", None)
        if msg_type == "assistant":
            turn_text = assistant_message_text(message)
            if turn_text:
                last_text = turn_text
        elif msg_type == "tool_call":
            status = str(getattr(message, "status", ""))
            name = getattr(message, "name", "")
            args = getattr(message, "args", None)
            call_id = str(getattr(message, "call_id", "") or name)
            if status == "running":
                _log_tool_call(name, args)
                pending_tools[call_id] = name
            elif status in {"completed", "error"}:
                pending_tools.pop(call_id, None)
                _log_tool_result(
                    name,
                    args,
                    status=status,
                    result=getattr(message, "result", None),
                )
            elif status:
                logger.warning(
                    "Tool {name} unexpected status {status!r}",
                    name=normalize_tool_call(name, args)[0],
                    status=status,
                )

    for call_id, name in pending_tools.items():
        tool_name, _ = normalize_tool_call(name, {})
        logger.warning(
            "Tool {name} started but never completed (call_id={call_id}). "
            "It may have been blocked by a Cursor hook or failed before MCP execution.",
            name=tool_name,
            call_id=call_id or "(unknown)",
        )

    wait_result = await run.wait()
    wait_text = extract_run_text(wait_result)
    if len(wait_text) > len(last_text):
        last_text = wait_text
    elif not last_text:
        last_text = wait_text

    num_turns = max(getattr(wait_result, "num_turns", 0) or 0, 1)
    if num_turns > max_turns:
        logger.warning(
            "Cursor SDK reported {reported} turns (Memclaw max_turns={max}); "
            "the SDK does not expose a turn cap — tighten prompts or hooks if needed.",
            reported=num_turns,
            max=max_turns,
        )

    return TurnResult(text=last_text, num_turns=num_turns)
