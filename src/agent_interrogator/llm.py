"""LLM interface implementations."""

import json
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional

# OpenAI reasoning-class models (gpt-5.x, o1, o3, o4, future o-series) only
# accept the default sampling temperature (1.0). Sending the library's usual
# 0.1 default produces a 400 BadRequestError. We detect them by name prefix
# and let the model use its own default unless the caller overrides via
# LLMConfig.model_kwargs.
_OPENAI_FIXED_TEMPERATURE_PATTERN = re.compile(r"^(gpt-5|o\d)", re.IGNORECASE)

if TYPE_CHECKING:
    from .output import OutputManager

from ollama import AsyncClient, ResponseError
from openai import AsyncOpenAI

from .config import (
    InterrogationConfig,
    LLMConfig,
    ModelProvider,
    OllamaConfig,
    OpenAICompatibleConfig,
    OpenAIConfig,
)
from .models import Function, Parameter
from .prompt_templates import (  # Discovery templates; Analysis templates; Processing templates; Schema templates
    ANALYSIS_JSON_SCHEMA,
    ANALYSIS_PROCESSING_PROMPT_TEMPLATE,
    ANALYSIS_PROCESSING_SYSTEM_PROMPT,
    ANALYSIS_PROMPT_TEMPLATE,
    DISCOVERY_JSON_SCHEMA,
    DISCOVERY_PROCESSING_PROMPT_TEMPLATE,
    DISCOVERY_PROCESSING_SYSTEM_PROMPT,
    DISCOVERY_PROMPT_TEMPLATE,
    INITIAL_ANALYSIS_PROMPT_TEMPLATE,
    INITIAL_DISCOVERY_PROMPT,
)


class LLMInterface(ABC):
    """Abstract base class for LLM implementations."""

    def __init__(self, config: InterrogationConfig, output_manager: "OutputManager"):
        """Initialize the LLM interface.

        Args:
            config: Full interrogation configuration
            output_manager: OutputManager instance for controlled output
        """
        self.config = config
        self.output = output_manager

    @abstractmethod
    async def _chat(self, messages: List[Dict[str, str]]) -> str:
        """Send messages to the LLM and return the response content.

        Args:
            messages: List of message dicts with 'role' and 'content' keys

        Returns:
            The LLM's response content as a string
        """
        pass

    def _extract_json(self, text: str, required_key: Optional[str] = None) -> Dict[str, Any]:
        """Extract and parse JSON from text, handling various formats.

        Args:
            text: Text potentially containing JSON
            required_key: If provided, only accept JSON objects containing this key

        Returns:
            Parsed JSON dictionary

        Raises:
            ValueError: If no valid JSON found
        """
        import re

        text = text.strip()

        # First try direct JSON parsing
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                if required_key is None or required_key in result:
                    return result
        except json.JSONDecodeError:
            pass

        # Try to find JSON object patterns
        patterns = [
            r"```(?:json)?\s*(\{.*?\})\s*```",  # JSON in markdown code block
            r"\{[^{}]*\}",  # Simple JSON object
            r"\{.*\}",  # Any {...} content (greedy)
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:
                json_str = match.group(1) if match.lastindex else match.group(0)
                try:
                    result = json.loads(json_str)
                    if isinstance(result, dict):
                        if required_key is None or required_key in result:
                            return result
                except json.JSONDecodeError:
                    continue

        raise ValueError("No valid JSON found in response")

    async def generate_prompt(self, context: Dict[str, Any]) -> str:
        """Generate an interrogation prompt based on context.

        Args:
            context: Dictionary containing:
                - phase: str, one of 'discovery', 'analysis'
                - cycle: int, current iteration number
                - previous_responses: list of previous responses
                - discovered_capabilities: list of capabilities found so far
                - capability: dict, current capability being analyzed (for analysis phase)
                - discovered_functions: list of functions found so far (for analysis phase)
        """
        phase = context.get("phase", "discovery")
        cycle = context.get("cycle", 0)

        if cycle == 0:
            if phase == "discovery":
                return INITIAL_DISCOVERY_PROMPT
            else:
                return INITIAL_ANALYSIS_PROMPT_TEMPLATE.format(
                    capability_name=context["capability"].name
                )

        # Build follow-up prompts
        if phase == "discovery":
            system_prompt = DISCOVERY_PROCESSING_SYSTEM_PROMPT
            discovered = context.get("discovered_capabilities", [])
            next_focus = context.get("next_cycle_focus")

            # ``discovered`` may contain either Pydantic Capability instances
            # (the v0.2.0 path from interrogator._discover_capabilities) or
            # plain dicts (legacy/test callers). Handle both.
            def _name(cap: Any) -> str:
                return getattr(cap, "name", None) or (
                    cap.get("name", "") if isinstance(cap, dict) else ""
                )

            def _desc(cap: Any) -> str:
                return getattr(cap, "description", None) or (
                    cap.get("description", "") if isinstance(cap, dict) else ""
                )

            capabilities_str = "\n".join(
                f"- {_name(cap)}: {_desc(cap)}" for cap in discovered
            )

            interrogation_prompt_request = DISCOVERY_PROMPT_TEMPLATE.format(
                capabilities_str=capabilities_str,
                focus_guidance=next_focus,
                context=context.get("previous_responses", []),
            )
        else:
            system_prompt = ANALYSIS_PROCESSING_SYSTEM_PROMPT
            capability = context["capability"]
            discovered_functions = context.get("discovered_functions", [])
            next_focus = context.get("next_cycle_focus")

            functions_str = "\n".join(
                f"- {func.name}: {func.description or ''}"
                for func in discovered_functions
            )

            interrogation_prompt_request = ANALYSIS_PROMPT_TEMPLATE.format(
                capability=capability,
                functions_str=functions_str,
                focus_guidance=next_focus,
                context=context.get("previous_responses", []),
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": interrogation_prompt_request},
        ]
        return await self._chat(messages)

    @staticmethod
    def _format_known_capabilities(capabilities: List[Any]) -> str:
        if not capabilities:
            return "(none yet — every extracted capability should be op=\"add\")"
        lines = []
        for cap in capabilities:
            node_id = getattr(cap, "node_id", "") or ""
            name = getattr(cap, "name", "") or ""
            description = getattr(cap, "description", "") or ""
            lines.append(f"- id={node_id}  name={name}  desc={description}")
        return "\n".join(lines)

    @staticmethod
    def _format_known_functions(functions: List[Any]) -> str:
        if not functions:
            return "(none yet — every extracted function should be op=\"add\")"
        lines = []
        for fn in functions:
            node_id = getattr(fn, "node_id", "") or ""
            name = getattr(fn, "name", "") or ""
            description = getattr(fn, "description", "") or ""
            params = ", ".join(
                f"{p.name}: {p.type}" for p in getattr(fn, "parameters", [])
            )
            return_type = getattr(fn, "return_type", None) or "void"
            lines.append(
                f"- id={node_id}  {name}({params}) -> {return_type}  // {description}"
            )
        return "\n".join(lines)

    @staticmethod
    def _coerce_to_operations(
        result: Dict[str, Any], legacy_key: str
    ) -> Dict[str, Any]:
        """Translate a legacy ``{capabilities|functions: [...]}`` payload into the
        operations format. Tolerates older models that revert to the pre-0.2.0 schema.
        """
        if isinstance(result.get("operations"), list):
            return result
        legacy = result.get(legacy_key)
        if isinstance(legacy, list):
            ops = []
            for item in legacy:
                if isinstance(item, dict):
                    ops.append({"op": "add", **item})
            result = dict(result)
            result["operations"] = ops
        else:
            result = dict(result)
            result.setdefault("operations", [])
        return result

    async def process_discovery_response(
        self, response: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process the agent's response during capability discovery phase.

        Returns:
            Dict containing:
            - operations: List of {op, ...} dicts (add/update against the catalog)
            - next_cycle_focus: Optional guidance for next discovery cycle
            - is_complete: bool
        """
        self.output.print_verbose("[bold cyan]Discovery Response:[/bold cyan]")
        self.output.print_verbose(response)

        known = self._format_known_capabilities(
            context.get("discovered_capabilities", [])
        )
        discovery_prompt = DISCOVERY_PROCESSING_PROMPT_TEMPLATE.format(
            json_format=DISCOVERY_JSON_SCHEMA,
            response=response,
            known_capabilities=known,
        )

        messages = [
            {"role": "system", "content": DISCOVERY_PROCESSING_SYSTEM_PROMPT},
            {"role": "user", "content": discovery_prompt},
        ]
        content = await self._chat(messages)

        try:
            return self._coerce_to_operations(self._extract_json(content), "capabilities")
        except ValueError:
            self.output.print_verbose(
                "[yellow]First attempt failed, trying again with explicit JSON formatting[/yellow]"
            )

            explicit_prompt = (
                "You MUST respond with valid JSON. No other text is allowed.\n"
                + discovery_prompt
            )
            messages = [
                {"role": "system", "content": DISCOVERY_PROCESSING_SYSTEM_PROMPT},
                {"role": "user", "content": explicit_prompt},
            ]
            content = await self._chat(messages)

            try:
                return self._coerce_to_operations(
                    self._extract_json(content), "capabilities"
                )
            except ValueError as e:
                self.output.print_verbose(
                    f"[red]Failed to extract JSON from response: {str(e)}[/red]"
                )
                raise ValueError(
                    "Failed to extract valid JSON from model response after multiple attempts"
                )

    async def process_analysis_response(
        self, response: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process the agent's response during capability analysis phase.

        Returns:
            Dict containing:
            - operations: List of {op, ...} dicts (add/update against the function catalog)
            - next_cycle_focus: Optional guidance for next analysis cycle
            - is_complete: bool
        """
        self.output.print_verbose("[bold cyan]Analysis Response:[/bold cyan]")
        self.output.print_verbose(response)

        capability = context["capability"]
        known = self._format_known_functions(context.get("discovered_functions", []))
        analysis_prompt = ANALYSIS_PROCESSING_PROMPT_TEMPLATE.format(
            capability_name=capability.name,
            json_format=ANALYSIS_JSON_SCHEMA,
            response=response,
            known_functions=known,
        )

        messages = [
            {"role": "system", "content": ANALYSIS_PROCESSING_SYSTEM_PROMPT},
            {"role": "user", "content": analysis_prompt},
        ]
        content = await self._chat(messages)

        try:
            return self._coerce_to_operations(self._extract_json(content), "functions")
        except ValueError:
            self.output.print_verbose(
                "[yellow]First attempt failed, trying again with explicit JSON formatting[/yellow]"
            )

            messages = [
                {
                    "role": "system",
                    "content": "You MUST respond with ONLY a valid JSON object following the schema exactly. No other text or explanation.",
                },
                {"role": "user", "content": analysis_prompt},
            ]
            content = await self._chat(messages)

            try:
                return self._coerce_to_operations(
                    self._extract_json(content), "functions"
                )
            except ValueError as e:
                self.output.print_verbose(
                    "[red]Failed to parse JSON response. Raw content:[/red]"
                )
                self.output.print_verbose(content)
                raise ValueError(
                    f"Failed to process analysis response after multiple attempts: {str(e)}"
                )

    async def process_response(
        self, response: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process the agent's response based on the current phase.

        Args:
            response: Raw response from the agent
            context: Current conversation context

        Returns:
            Structured data extracted from the response
        """
        phase = context.get("phase", "discovery")
        if phase == "discovery":
            return await self.process_discovery_response(response, context)
        else:
            return await self.process_analysis_response(response, context)

    async def should_continue_cycle(self, results: Dict[str, Any]) -> bool:
        """Determine if another cycle should be run based on results."""
        # Stop if marked complete
        if results.get("is_complete", False):
            return False

        return True

    @staticmethod
    def _default_openai_chat_kwargs(model_name: Optional[str]) -> Dict[str, Any]:
        """Pick the right baseline chat kwargs for an OpenAI-shaped client.

        Returns ``{"temperature": 0.1}`` for legacy chat models (deterministic
        extraction), and ``{}`` for reasoning-class models (gpt-5.x, o1, o3,
        o4) that reject any non-default temperature with a 400 error. Callers
        always retain final say through ``LLMConfig.model_kwargs``.
        """
        if model_name and _OPENAI_FIXED_TEMPERATURE_PATTERN.match(model_name):
            return {}
        return {"temperature": 0.1}


class OpenAILLM(LLMInterface):
    """OpenAI-based LLM implementation."""

    def __init__(self, config: InterrogationConfig, output_manager: "OutputManager"):
        super().__init__(config, output_manager)
        if config.llm.provider != ModelProvider.OPENAI:
            raise ValueError("OpenAILLM requires provider to be OPENAI")

        openai_config = config.llm.openai or OpenAIConfig()
        client_kwargs: Dict[str, Any] = {"api_key": config.llm.api_key}
        if openai_config.timeout is not None:
            client_kwargs["timeout"] = openai_config.timeout

        self.client = AsyncOpenAI(**client_kwargs)
        self.model_kwargs = config.llm.model_kwargs

    async def _chat(self, messages: List[Dict[str, str]]) -> str:
        """Send messages to OpenAI and return the response content.

        Default sampling is ``temperature=0.1`` for deterministic extraction
        on legacy chat models. Reasoning-class models (gpt-5.x, o1, o3, o4)
        are auto-detected by name and the temperature default is omitted so
        they fall back to their required default of 1.0. ``LLMConfig.model_kwargs``
        always overrides whatever the library picks.
        """
        chat_kwargs: Dict[str, Any] = {
            **self._default_openai_chat_kwargs(self.config.llm.model_name),
            **self.model_kwargs,
        }
        response = await self.client.chat.completions.create(
            model=self.config.llm.model_name,
            messages=messages,
            **chat_kwargs,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("OpenAI API returned empty content")
        return str(content)


class OpenAICompatibleLLM(LLMInterface):
    """LLM implementation for custom OpenAI-compatible endpoints."""

    def __init__(self, config: InterrogationConfig, output_manager: "OutputManager"):
        super().__init__(config, output_manager)
        if config.llm.provider != ModelProvider.OPENAI_COMPATIBLE:
            raise ValueError(
                "OpenAICompatibleLLM requires provider to be OPENAI_COMPATIBLE"
            )

        # Get endpoint-specific config (required for this provider)
        compat_config = config.llm.openai_compatible
        if compat_config is None:
            raise ValueError(
                "OpenAI-compatible provider requires 'openai_compatible' configuration "
                "with at least 'base_url' specified"
            )

        self.client = AsyncOpenAI(
            base_url=compat_config.base_url,
            api_key=compat_config.api_key,
            timeout=compat_config.timeout,
        )
        self.model_kwargs = config.llm.model_kwargs

        # Log connection details in verbose mode
        self.output.print_verbose(
            "[bold cyan]Connecting to OpenAI-compatible endpoint:[/bold cyan]"
        )
        self.output.print_verbose(f"Base URL: {compat_config.base_url}")
        self.output.print_verbose(f"Model: {config.llm.model_name}")
        self.output.print_verbose(f"Timeout: {compat_config.timeout}s")

    async def _chat(self, messages: List[Dict[str, str]]) -> str:
        """Send messages to the OpenAI-compatible endpoint and return the response content.

        Same auto-detection as ``OpenAILLM`` — reasoning-class model names get
        the temperature default omitted. ``LLMConfig.model_kwargs`` always
        overrides whatever the library picks.
        """
        chat_kwargs: Dict[str, Any] = {
            **self._default_openai_chat_kwargs(self.config.llm.model_name),
            **self.model_kwargs,
        }
        response = await self.client.chat.completions.create(
            model=self.config.llm.model_name,
            messages=messages,
            **chat_kwargs,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("API returned empty content")
        return str(content)


class OllamaLLM(LLMInterface):
    """Ollama-based LLM implementation."""

    def __init__(self, config: InterrogationConfig, output_manager: "OutputManager"):
        super().__init__(config, output_manager)
        if config.llm.provider != ModelProvider.OLLAMA:
            raise ValueError("OllamaLLM requires provider to be OLLAMA")

        # Get Ollama-specific config or use defaults
        ollama_config = config.llm.ollama or OllamaConfig()

        # Initialize the async client
        self.client = AsyncClient(host=ollama_config.host, timeout=ollama_config.timeout)
        self.model_name = config.llm.model_name
        self.options = {"temperature": 0.1, **ollama_config.options}
        self.keep_alive = ollama_config.keep_alive

        # Log model loading details in verbose mode
        self.output.print_verbose(f"[bold cyan]Connecting to Ollama:[/bold cyan]")
        self.output.print_verbose(f"Host: {ollama_config.host}")
        self.output.print_verbose(f"Model: {self.model_name}")
        self.output.print_verbose(f"Timeout: {ollama_config.timeout}s")

    async def _chat(self, messages: List[Dict[str, str]]) -> str:
        """Send messages to Ollama and return the response content."""
        try:
            response = await self.client.chat(
                model=self.model_name,
                messages=messages,
                options=self.options,
                keep_alive=self.keep_alive,
            )
            return response["message"]["content"]
        except ResponseError as e:
            raise ValueError(f"Ollama API error: {str(e)}")
