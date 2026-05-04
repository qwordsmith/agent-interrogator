"""Configuration schema for the agent interrogator."""

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ModelProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    OLLAMA = "ollama"
    OPENAI_COMPATIBLE = "openai_compatible"


class OutputMode(str, Enum):
    """Terminal output modes for the interrogator."""

    QUIET = "quiet"  # No terminal output
    STANDARD = "standard"  # Basic info and status indicators
    VERBOSE = "verbose"  # Detailed prompts and responses


class OpenAIConfig(BaseModel):
    """OpenAI-specific configuration options."""

    timeout: Optional[float] = Field(
        None,
        description="Request timeout in seconds (defaults to the openai SDK default)",
    )


class OllamaConfig(BaseModel):
    """Ollama-specific configuration options."""

    host: str = Field(
        "http://localhost:11434",
        description="Ollama server URL",
    )
    timeout: float = Field(
        120.0,
        description="Request timeout in seconds",
    )
    options: Dict[str, Any] = Field(
        default_factory=dict,
        description="Generation options (temperature, top_p, etc.)",
    )
    keep_alive: str = Field(
        "5m",
        description="How long to keep model loaded",
    )


class OpenAICompatibleConfig(BaseModel):
    """Configuration for custom OpenAI-compatible API endpoints."""

    base_url: str = Field(
        ...,
        description="Base URL for the OpenAI-compatible API endpoint (e.g., http://localhost:8000/v1)",
    )
    api_key: str = Field(
        "not-required",
        description="API key for the endpoint (some endpoints ignore this)",
    )
    timeout: float = Field(
        120.0,
        description="Request timeout in seconds",
    )


class LLMConfig(BaseModel):
    """Configuration for the LLM to be used for interrogation."""

    provider: ModelProvider = Field(
        ...,
        description="The provider of the LLM (OpenAI, Ollama, or OpenAI-compatible)",
    )
    model_name: str = Field(..., description="Name of the model to use")
    api_key: Optional[str] = Field(
        None, description="API key for the provider (if required)"
    )
    model_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional model-specific parameters"
    )
    openai: Optional[OpenAIConfig] = Field(
        None, description="OpenAI-specific configuration options"
    )
    ollama: Optional[OllamaConfig] = Field(
        None, description="Ollama-specific configuration options"
    )
    openai_compatible: Optional[OpenAICompatibleConfig] = Field(
        None, description="Configuration for custom OpenAI-compatible endpoints"
    )


class InterrogationConfig(BaseModel):
    """Main configuration for the agent interrogator."""

    llm: LLMConfig = Field(
        ..., description="Configuration for the LLM to use for interrogation"
    )
    max_iterations: int = Field(
        default=5, description="Maximum number of iterations for capability discovery"
    )
    # TODO: Implement support for different output formats (json/yaml)
    # This will allow users to control how the agent profile and capabilities
    # are serialized in the final output
    output_format: str = Field(
        default="json", description="Format for the output (json/yaml)"
    )
    output_mode: OutputMode = Field(
        default=OutputMode.STANDARD,
        description="Controls the level of terminal output during interrogation",
    )
