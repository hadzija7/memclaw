"""Tests for the Cursor SDK backend."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memclaw.backends import REGISTRY, get_backend_class
from memclaw.backends.cursor import CursorBackend, _build_combined_prompt
from memclaw.config import MemclawConfig


@pytest.fixture(autouse=True)
def _isolate_credentials(monkeypatch):
    """Prevent shell env from leaking into MemclawConfig."""
    for name in ("CURSOR_API_KEY", "CURSOR_MODEL"):
        monkeypatch.delenv(name, raising=False)


def _make_config(
    tmp_path: Path,
    *,
    cursor_api_key: str = "",
    cursor_model: str = "",
) -> MemclawConfig:
    return MemclawConfig(
        memory_dir=tmp_path / "m",
        openai_api_key="test-openai-key",
        anthropic_api_key="test-anthropic-key",
        cursor_api_key=cursor_api_key,
        cursor_model=cursor_model,
    )


class TestRegistry:
    def test_both_backends_registered(self):
        assert "cursor" in REGISTRY
        assert "claude" in REGISTRY
        assert REGISTRY["cursor"] is CursorBackend

    def test_get_backend_class_by_name(self):
        assert get_backend_class("cursor") is CursorBackend

    def test_get_backend_class_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown agent backend"):
            get_backend_class("nonexistent")


class TestCursorBackendConfig:
    def test_is_configured_false_without_key(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert CursorBackend.is_configured(cfg) is False

    def test_is_configured_true_with_key(self, tmp_path):
        cfg = _make_config(tmp_path, cursor_api_key="crsr_test_key")
        assert CursorBackend.is_configured(cfg) is True

    def test_configuration_help_mentions_env(self):
        help_text = CursorBackend.configuration_help()
        assert "CURSOR_API_KEY" in help_text
        assert "AGENT_BACKEND=cursor" in help_text

    def test_init_reads_config(self, tmp_path):
        cfg = _make_config(
            tmp_path,
            cursor_api_key="crsr_test_key",
            cursor_model="composer-2.5",
        )
        backend = CursorBackend(cfg)
        assert backend._api_key == "crsr_test_key"
        assert backend._model == "composer-2.5"
        assert backend._cwd == str(cfg.memory_dir)
        assert backend.bills_per_token is True


class TestPromptBuilding:
    def test_combined_prompt_includes_system_and_user(self):
        prompt = _build_combined_prompt(
            system_prompt="You are Memclaw.",
            user_message="Hello",
        )
        assert "You are Memclaw." in prompt
        assert "Hello" in prompt
        assert "---" in prompt

    def test_image_note_when_image_present(self):
        prompt = _build_combined_prompt(
            system_prompt="Sys",
            user_message="Look at this",
            image_b64="abc123",
            image_media_type="image/png",
        )
        assert "image/png" in prompt
        assert "does not pass raw image bytes" in prompt


class TestCursorBackendRuns:
    @pytest.mark.asyncio
    async def test_run_one_shot(self, tmp_path):
        cfg = _make_config(tmp_path, cursor_api_key="crsr_test_key")
        backend = CursorBackend(cfg)

        mock_result = SimpleNamespace(result="Hello from Cursor")

        with patch("cursor_sdk.AsyncClient") as mock_client_cls, patch(
            "cursor_sdk.AsyncAgent.prompt", new_callable=AsyncMock
        ) as mock_prompt:
            mock_client = AsyncMock()
            mock_client_cls.launch_bridge = AsyncMock(return_value=mock_client)
            mock_prompt.return_value = mock_result

            text = await backend.run_one_shot(
                system_prompt="System",
                user_message="User",
            )

        assert text == "Hello from Cursor"
        mock_prompt.assert_awaited_once()
        call_prompt = mock_prompt.await_args.args[0]
        assert "System" in call_prompt
        assert "User" in call_prompt

    @pytest.mark.asyncio
    async def test_run_one_shot_missing_key_raises(self, tmp_path):
        cfg = _make_config(tmp_path)
        backend = CursorBackend(cfg)

        with pytest.raises(RuntimeError, match="CURSOR_API_KEY"):
            await backend.run_one_shot(system_prompt="S", user_message="U")

    @pytest.mark.asyncio
    async def test_run_turn(self, tmp_path):
        cfg = _make_config(tmp_path, cursor_api_key="crsr_test_key")
        backend = CursorBackend(cfg)

        mock_run = AsyncMock()
        mock_run.text = AsyncMock(return_value="Turn response")
        mock_agent = AsyncMock()
        mock_agent.send = AsyncMock(return_value=mock_run)
        mock_agent.aclose = AsyncMock()
        mock_client = AsyncMock()
        mock_client.agents.create = AsyncMock(return_value=mock_agent)

        with patch("cursor_sdk.AsyncClient") as mock_client_cls:
            mock_client_cls.launch_bridge = AsyncMock(return_value=mock_client)

            result = await backend.run_turn(
                system_prompt="System",
                user_message="User",
                tool_executor=MagicMock(),
                max_turns=5,
            )

        assert result.text == "Turn response"
        assert result.num_turns == 1
        mock_agent.send.assert_awaited_once()
        mock_agent.aclose.assert_awaited_once()
