"""Tests for the main interrogator module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_interrogator import (
    AgentInterrogator,
    InterrogationConfig,
    LLMConfig,
    ModelProvider,
    OpenAICompatibleConfig,
    OutputMode,
)
from agent_interrogator.models import AgentProfile, Capability, Function, Parameter


@pytest.fixture
def basic_config():
    """Create a basic test configuration."""
    return InterrogationConfig(
        llm=LLMConfig(
            provider=ModelProvider.OPENAI, model_name="test-model", api_key="test-key"
        ),
        output_mode=OutputMode.QUIET,
        max_iterations=3,
    )


@pytest.fixture
def mock_callback():
    """Create a mock async callback."""
    return AsyncMock(return_value="Test response")


class TestAgentInterrogator:
    """Test AgentInterrogator class."""

    @pytest.mark.asyncio
    async def test_initialization(self, basic_config, mock_callback):
        """Test interrogator initialization."""
        with patch("agent_interrogator.interrogator.OutputManager"):
            interrogator = AgentInterrogator(basic_config, mock_callback)

            assert interrogator.config == basic_config
            assert interrogator.agent_callback == mock_callback
            assert isinstance(interrogator.profile, AgentProfile)
            assert interrogator.profile.capabilities == []

    @pytest.mark.asyncio
    async def test_llm_initialization_openai(self, basic_config, mock_callback):
        """Test OpenAI LLM initialization."""
        with patch("agent_interrogator.interrogator.OutputManager"):
            with patch("agent_interrogator.interrogator.OpenAILLM") as mock_openai:
                interrogator = AgentInterrogator(basic_config, mock_callback)
                mock_openai.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_initialization_ollama(self, mock_callback):
        """Test Ollama LLM initialization."""
        config = InterrogationConfig(
            llm=LLMConfig(provider=ModelProvider.OLLAMA, model_name="llama3.2:latest"),
            output_mode=OutputMode.QUIET,
        )

        with patch("agent_interrogator.interrogator.OutputManager"):
            with patch("agent_interrogator.interrogator.OllamaLLM") as mock_ollama:
                interrogator = AgentInterrogator(config, mock_callback)
                mock_ollama.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_initialization_openai_compatible(self, mock_callback):
        """Test OpenAI-compatible LLM initialization."""
        config = InterrogationConfig(
            llm=LLMConfig(
                provider=ModelProvider.OPENAI_COMPATIBLE,
                model_name="local-model",
                openai_compatible=OpenAICompatibleConfig(
                    base_url="http://localhost:8000/v1",
                ),
            ),
            output_mode=OutputMode.QUIET,
        )

        with patch("agent_interrogator.interrogator.OutputManager"):
            with patch(
                "agent_interrogator.interrogator.OpenAICompatibleLLM"
            ) as mock_compat:
                interrogator = AgentInterrogator(config, mock_callback)
                mock_compat.assert_called_once()

    @pytest.mark.asyncio
    async def test_display_profile(self, basic_config, mock_callback):
        """Test profile display functionality."""
        with patch("agent_interrogator.interrogator.OutputManager") as mock_output_cls:
            mock_output = MagicMock()
            mock_output_cls.return_value = mock_output

            interrogator = AgentInterrogator(basic_config, mock_callback)

            # Add some test data to profile
            param = Parameter(name="test_param", type="string", required=True)
            func = Function(name="test_func", description="Test", parameters=[param])
            cap = Capability(
                name="test_cap", description="Test capability", functions=[func]
            )
            interrogator.profile.capabilities.append(cap)

            # Call display profile
            interrogator._display_profile()

            # Should have called print and display_table
            assert mock_output.print.called
            assert mock_output.display_table.called


class TestCallbackIntegration:
    """Test callback integration patterns."""

    @pytest.mark.asyncio
    async def test_callback_without_cleanup(self, basic_config):
        """Test callback without cleanup method works fine."""

        # Simple callback without cleanup
        async def simple_callback(prompt: str) -> str:
            return f"Response to: {prompt}"

        with patch("agent_interrogator.interrogator.OutputManager"):
            with patch("agent_interrogator.interrogator.OpenAILLM"):
                interrogator = AgentInterrogator(basic_config, simple_callback)

                # Should not raise error even without cleanup
                with patch.object(
                    interrogator, "_discover_capabilities", new_callable=AsyncMock
                ) as mock_discover:
                    with patch.object(
                        interrogator, "_analyze_capability", new_callable=AsyncMock
                    ):
                        mock_discover.return_value = []
                        profile = await interrogator.interrogate()
                        assert isinstance(profile, AgentProfile)

    @pytest.mark.asyncio
    async def test_callback_error_handling(self, basic_config):
        """Test handling of callback errors."""

        # Callback that raises error
        async def error_callback(prompt: str) -> str:
            raise ValueError("Test error")

        with patch("agent_interrogator.interrogator.OutputManager"):
            with patch("agent_interrogator.interrogator.OpenAILLM"):
                interrogator = AgentInterrogator(basic_config, error_callback)

                # The interrogator should handle errors gracefully
                with patch.object(
                    interrogator,
                    "_discover_capabilities",
                    side_effect=ValueError("Test error"),
                ):
                    with pytest.raises(ValueError):
                        await interrogator.interrogate()
