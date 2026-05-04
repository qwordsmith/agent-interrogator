"""Tests for configuration module."""

import pytest
from pydantic import ValidationError

from agent_interrogator.config import (
    InterrogationConfig,
    LLMConfig,
    ModelProvider,
    OllamaConfig,
    OpenAICompatibleConfig,
    OutputMode,
)


class TestLLMConfig:
    """Test LLMConfig validation and behavior."""

    def test_openai_config_valid(self):
        """Test valid OpenAI configuration."""
        config = LLMConfig(
            provider=ModelProvider.OPENAI, model_name="gpt-4", api_key="test-key"
        )
        assert config.provider == ModelProvider.OPENAI
        assert config.model_name == "gpt-4"
        assert config.api_key == "test-key"

    def test_openai_config_missing_api_key(self):
        """Test that OpenAI config works without API key (optional field)."""
        # API key is optional, so this should work
        config = LLMConfig(
            provider=ModelProvider.OPENAI, model_name="gpt-4", api_key=None
        )
        assert config.api_key is None

    def test_ollama_config_valid(self):
        """Test valid Ollama configuration."""
        config = LLMConfig(
            provider=ModelProvider.OLLAMA, model_name="llama3.2:latest"
        )
        assert config.provider == ModelProvider.OLLAMA
        assert config.model_name == "llama3.2:latest"
        assert config.api_key is None  # Not required for Ollama

    def test_ollama_with_options(self):
        """Test Ollama configuration with additional options."""
        ollama_config = OllamaConfig(
            host="http://localhost:11434",
            timeout=180.0,
            options={"temperature": 0.7, "top_p": 0.9},
            keep_alive="10m",
        )
        config = LLMConfig(
            provider=ModelProvider.OLLAMA,
            model_name="test-model",
            ollama=ollama_config,
        )
        assert config.ollama.host == "http://localhost:11434"
        assert config.ollama.timeout == 180.0
        assert config.ollama.options == {"temperature": 0.7, "top_p": 0.9}
        assert config.ollama.keep_alive == "10m"

    def test_model_kwargs(self):
        """Test model kwargs are properly stored."""
        config = LLMConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4",
            api_key="test-key",
            model_kwargs={"temperature": 0.7, "max_tokens": 2000},
        )
        assert config.model_kwargs["temperature"] == 0.7
        assert config.model_kwargs["max_tokens"] == 2000

    def test_openai_compatible_config_valid(self):
        """Test valid OpenAI-compatible configuration."""
        compat_config = OpenAICompatibleConfig(
            base_url="http://localhost:8000/v1",
            api_key="test-key",
            timeout=180.0,
        )
        config = LLMConfig(
            provider=ModelProvider.OPENAI_COMPATIBLE,
            model_name="local-model",
            openai_compatible=compat_config,
        )
        assert config.provider == ModelProvider.OPENAI_COMPATIBLE
        assert config.model_name == "local-model"
        assert config.openai_compatible.base_url == "http://localhost:8000/v1"
        assert config.openai_compatible.api_key == "test-key"
        assert config.openai_compatible.timeout == 180.0

    def test_openai_compatible_config_defaults(self):
        """Test OpenAI-compatible config with default values."""
        compat_config = OpenAICompatibleConfig(
            base_url="http://localhost:1234/v1",
        )
        assert compat_config.api_key == "not-required"
        assert compat_config.timeout == 120.0

    def test_openai_compatible_from_dict(self):
        """Test creating OpenAI-compatible config from dictionary."""
        config_dict = {
            "llm": {
                "provider": "openai_compatible",
                "model_name": "my-model",
                "openai_compatible": {
                    "base_url": "https://my-endpoint.com/v1",
                    "api_key": "my-api-key",
                },
            },
        }
        config = InterrogationConfig.model_validate(config_dict)
        assert config.llm.provider == ModelProvider.OPENAI_COMPATIBLE
        assert config.llm.openai_compatible.base_url == "https://my-endpoint.com/v1"
        assert config.llm.openai_compatible.api_key == "my-api-key"


class TestInterrogationConfig:
    """Test InterrogationConfig validation and behavior."""

    def test_default_values(self):
        """Test default configuration values."""
        llm_config = LLMConfig(
            provider=ModelProvider.OPENAI, model_name="gpt-4", api_key="test-key"
        )
        config = InterrogationConfig(llm=llm_config)

        assert config.max_iterations == 5
        assert config.output_mode == OutputMode.STANDARD

    def test_custom_values(self):
        """Test custom configuration values."""
        llm_config = LLMConfig(
            provider=ModelProvider.OPENAI, model_name="gpt-4", api_key="test-key"
        )
        config = InterrogationConfig(
            llm=llm_config, max_iterations=10, output_mode=OutputMode.VERBOSE
        )

        assert config.max_iterations == 10
        assert config.output_mode == OutputMode.VERBOSE

    def test_parse_obj_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "llm": {
                "provider": "openai",
                "model_name": "gpt-4",
                "api_key": "test-key",
                "model_kwargs": {"temperature": 0.5},
            },
            "max_iterations": 3,
            "output_mode": "quiet",
        }

        config = InterrogationConfig.model_validate(config_dict)
        assert config.llm.provider == ModelProvider.OPENAI
        assert config.llm.model_name == "gpt-4"
        assert config.llm.model_kwargs["temperature"] == 0.5
        assert config.max_iterations == 3
        assert config.output_mode == OutputMode.QUIET


class TestOutputMode:
    """Test OutputMode enum."""

    def test_enum_values(self):
        """Test all output mode values exist."""
        assert OutputMode.QUIET.value == "quiet"
        assert OutputMode.STANDARD.value == "standard"
        assert OutputMode.VERBOSE.value == "verbose"

    def test_from_string(self):
        """Test creating output mode from string."""
        assert OutputMode("quiet") == OutputMode.QUIET
        assert OutputMode("standard") == OutputMode.STANDARD
        assert OutputMode("verbose") == OutputMode.VERBOSE
