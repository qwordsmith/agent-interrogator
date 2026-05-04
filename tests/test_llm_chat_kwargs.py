"""Regression tests for `_chat` keyword-argument handling in OpenAI providers.

The OpenAI and OpenAI-compatible ``_chat`` methods used to hardcode
``temperature=0.1`` ahead of ``**model_kwargs``, which:

1. Made it impossible for callers to override the temperature, and
2. Raised ``TypeError: got multiple values for keyword argument 'temperature'``
   if a caller did supply one anyway.

This blocked the newer OpenAI reasoning-class models (e.g. gpt-5.x, o1, o3),
which only accept the default ``temperature=1.0`` and reject ``0.1`` outright.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_interrogator import (
    InterrogationConfig,
    LLMConfig,
    ModelProvider,
    OpenAICompatibleConfig,
    OutputMode,
)
from agent_interrogator.llm import LLMInterface, OpenAICompatibleLLM, OpenAILLM
from agent_interrogator.output import OutputManager


def _fake_completion(content: str = "ok") -> MagicMock:
    """Build a fake OpenAI ChatCompletion response object."""
    completion = MagicMock()
    completion.choices = [MagicMock()]
    completion.choices[0].message.content = content
    return completion


class TestDefaultOpenAIChatKwargsHelper:
    """Auto-detection helper that decides whether to send a temperature default."""

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4o",
            "gpt-4.1",
            "gpt-4-turbo",
            "text-davinci-003",
        ],
    )
    def test_legacy_models_get_temperature_default(self, model):
        assert LLMInterface._default_openai_chat_kwargs(model) == {"temperature": 0.1}

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-5",
            "gpt-5o",
            "gpt-5.5",
            "gpt-5-mini",
            "gpt-5-turbo",
            "o1",
            "o1-mini",
            "o1-preview",
            "o3",
            "o3-mini",
            "o4-mini",
            # Case-insensitive
            "GPT-5",
            "O1-MINI",
        ],
    )
    def test_reasoning_models_omit_temperature(self, model):
        assert LLMInterface._default_openai_chat_kwargs(model) == {}

    def test_empty_or_none_model_falls_through_to_default(self):
        assert LLMInterface._default_openai_chat_kwargs("") == {"temperature": 0.1}
        assert LLMInterface._default_openai_chat_kwargs(None) == {"temperature": 0.1}


@pytest.mark.asyncio
class TestOpenAILLMChatKwargs:
    async def test_default_temperature_used_when_not_overridden(self):
        config = InterrogationConfig(
            llm=LLMConfig(
                provider=ModelProvider.OPENAI,
                model_name="gpt-4.1",
                api_key="test-key",
            ),
            output_mode=OutputMode.QUIET,
        )
        with patch("agent_interrogator.llm.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=_fake_completion()
            )
            mock_client_cls.return_value = mock_client

            llm = OpenAILLM(config, OutputManager(OutputMode.QUIET))
            await llm._chat([{"role": "user", "content": "hi"}])

            kwargs = mock_client.chat.completions.create.await_args.kwargs
            assert kwargs["temperature"] == 0.1

    async def test_model_kwargs_temperature_overrides_default(self):
        """Passing temperature in model_kwargs must override the 0.1 default
        rather than raise TypeError on duplicate kwarg."""
        config = InterrogationConfig(
            llm=LLMConfig(
                provider=ModelProvider.OPENAI,
                model_name="gpt-5.5",
                api_key="test-key",
                model_kwargs={"temperature": 1.0},
            ),
            output_mode=OutputMode.QUIET,
        )
        with patch("agent_interrogator.llm.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=_fake_completion()
            )
            mock_client_cls.return_value = mock_client

            llm = OpenAILLM(config, OutputManager(OutputMode.QUIET))
            await llm._chat([{"role": "user", "content": "hi"}])

            kwargs = mock_client.chat.completions.create.await_args.kwargs
            assert kwargs["temperature"] == 1.0

    async def test_reasoning_model_omits_temperature_default(self):
        """Regression: gpt-5.x must not send temperature=0.1 (would 400)."""
        config = InterrogationConfig(
            llm=LLMConfig(
                provider=ModelProvider.OPENAI,
                model_name="gpt-5.5",
                api_key="test-key",
            ),
            output_mode=OutputMode.QUIET,
        )
        with patch("agent_interrogator.llm.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=_fake_completion()
            )
            mock_client_cls.return_value = mock_client

            llm = OpenAILLM(config, OutputManager(OutputMode.QUIET))
            await llm._chat([{"role": "user", "content": "hi"}])

            kwargs = mock_client.chat.completions.create.await_args.kwargs
            assert "temperature" not in kwargs

    async def test_reasoning_model_explicit_override_wins(self):
        """User can still set temperature on a reasoning model if they want."""
        config = InterrogationConfig(
            llm=LLMConfig(
                provider=ModelProvider.OPENAI,
                model_name="gpt-5.5",
                api_key="test-key",
                model_kwargs={"temperature": 1.0},
            ),
            output_mode=OutputMode.QUIET,
        )
        with patch("agent_interrogator.llm.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=_fake_completion()
            )
            mock_client_cls.return_value = mock_client

            llm = OpenAILLM(config, OutputManager(OutputMode.QUIET))
            await llm._chat([{"role": "user", "content": "hi"}])

            kwargs = mock_client.chat.completions.create.await_args.kwargs
            assert kwargs["temperature"] == 1.0

    async def test_other_model_kwargs_are_passed_through(self):
        config = InterrogationConfig(
            llm=LLMConfig(
                provider=ModelProvider.OPENAI,
                model_name="gpt-4.1",
                api_key="test-key",
                model_kwargs={"max_tokens": 1024, "top_p": 0.9},
            ),
            output_mode=OutputMode.QUIET,
        )
        with patch("agent_interrogator.llm.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=_fake_completion()
            )
            mock_client_cls.return_value = mock_client

            llm = OpenAILLM(config, OutputManager(OutputMode.QUIET))
            await llm._chat([{"role": "user", "content": "hi"}])

            kwargs = mock_client.chat.completions.create.await_args.kwargs
            assert kwargs["temperature"] == 0.1
            assert kwargs["max_tokens"] == 1024
            assert kwargs["top_p"] == 0.9


@pytest.mark.asyncio
class TestOpenAICompatibleLLMChatKwargs:
    async def test_model_kwargs_temperature_overrides_default(self):
        config = InterrogationConfig(
            llm=LLMConfig(
                provider=ModelProvider.OPENAI_COMPATIBLE,
                model_name="some-local-model",
                model_kwargs={"temperature": 0.7},
                openai_compatible=OpenAICompatibleConfig(
                    base_url="http://localhost:8000/v1",
                ),
            ),
            output_mode=OutputMode.QUIET,
        )
        with patch("agent_interrogator.llm.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=_fake_completion()
            )
            mock_client_cls.return_value = mock_client

            llm = OpenAICompatibleLLM(config, OutputManager(OutputMode.QUIET))
            await llm._chat([{"role": "user", "content": "hi"}])

            kwargs = mock_client.chat.completions.create.await_args.kwargs
            assert kwargs["temperature"] == 0.7
