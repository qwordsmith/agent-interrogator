"""AI Agent Interrogation Framework."""

from typing import List

from .config import (
    InterrogationConfig,
    LLMConfig,
    ModelProvider,
    OllamaConfig,
    OpenAICompatibleConfig,
    OpenAIConfig,
    OutputMode,
)
from .interrogator import AgentInterrogator
from .models import AgentProfile, Capability, Function, Parameter

__version__: str = "0.2.0"

__all__: List[str] = [
    "AgentInterrogator",
    "InterrogationConfig",
    "LLMConfig",
    "ModelProvider",
    "AgentProfile",
    "Capability",
    "Function",
    "Parameter",
    "OutputMode",
    "OllamaConfig",
    "OpenAICompatibleConfig",
    "OpenAIConfig",
]
