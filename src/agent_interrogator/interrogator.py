"""Main interrogator implementation."""

from typing import Any, Awaitable, Callable, Dict, List

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .config import InterrogationConfig, ModelProvider
from .llm import LLMInterface, OllamaLLM, OpenAICompatibleLLM, OpenAILLM
from .merge import merge_capability, merge_function
from .models import AgentProfile, Capability, Function, Parameter
from .output import OutputManager

# Type alias for the agent interaction callback
AgentCallback = Callable[[str], Awaitable[str]]

LOGO = r"""[bold blue]
    ___                    __       ____      __                                   __            
   /   | ____ ____  ____  / /_     /  _/___  / /____  ______________  ____ _____ _/ /_____  _____
  / /| |/ __ `/ _ \/ __ \/ __/     / // __ \/ __/ _ \/ ___/ ___/ __ \/ __ `/ __ `/ __/ __ \/ ___/
 / ___ / /_/ /  __/ / / / /_     _/ // / / / /_/  __/ /  / /  / /_/ / /_/ / /_/ / /_/ /_/ / /    
/_/  |_\__, /\___/_/ /_/\__/    /___/_/ /_/\__/\___/_/  /_/   \____/\__, /\__,_/\__/\____/_/     
      /____/                                                       /____/                        
[/bold blue]"""


class AgentInterrogator:
    """Main class for interrogating AI agents.

    Args:
        config (InterrogationConfig): Configuration for the interrogator
        agent_callback (AgentCallback): Async callback function that takes a prompt
            string and returns the agent's response string
    """

    def __init__(self, config: InterrogationConfig, agent_callback: AgentCallback):
        self.config = config
        self.agent_callback = agent_callback
        self.profile = AgentProfile()
        self.output = OutputManager(self.config.output_mode)
        self.llm = self._initialize_llm()

        # Display logo and configuration
        self._display_startup_info()

    def _display_startup_info(self) -> None:
        """Display the ASCII art logo and configuration information."""
        # Print logo
        self.output.print(LOGO)

        # Create configuration table
        config_table = Table(
            title="[bold cyan]Agent Interrogator Configuration[/bold cyan]"
        )
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="green")

        # Add configuration rows
        config_table.add_row("LLM Provider", str(self.config.llm.provider))
        config_table.add_row("Model Name", self.config.llm.model_name)
        config_table.add_row(
            "API Key", "********" if self.config.llm.api_key else "Not provided"
        )
        config_table.add_row("Max Iterations", str(self.config.max_iterations))
        config_table.add_row("Output Mode", str(self.config.output_mode))

        # Add any model-specific kwargs as a list
        kwargs_str = "\n".join(
            f"{k}: {v}" for k, v in self.config.llm.model_kwargs.items()
        )
        if kwargs_str:
            config_table.add_row("Model Settings", kwargs_str)

        self.output.display_table(config_table)
        self.output.print(
            "\n[bold green]Ready to begin interrogation...\n[/bold green]"
        )

    def _initialize_llm(self) -> LLMInterface:
        """Initialize the appropriate LLM based on configuration."""
        if self.config.llm.provider == ModelProvider.OPENAI:
            return OpenAILLM(self.config, self.output)
        elif self.config.llm.provider == ModelProvider.OLLAMA:
            return OllamaLLM(self.config, self.output)
        elif self.config.llm.provider == ModelProvider.OPENAI_COMPATIBLE:
            return OpenAICompatibleLLM(self.config, self.output)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm.provider}")

    def _display_profile(self) -> None:
        """Display the complete agent profile in a structured format."""
        self.output.print(
            "\n[bold magenta]═══ Agent Profile Summary ═══[/bold magenta]\n"
        )

        for capability in self.profile.capabilities:
            # Create a table for each capability
            cap_table = Table(
                title=f"[bold cyan]{capability.name}[/bold cyan]",
                caption=capability.description or "No description provided",
                show_header=True,
                header_style="bold green",
            )
            cap_table.add_column("Function", style="cyan")
            cap_table.add_column("Description", style="white")
            cap_table.add_column("Parameters", style="yellow")
            cap_table.add_column("Return Type", style="magenta")

            # Add rows for each function
            for func in capability.functions:
                params = (
                    "\n".join([f"{p.name}: {p.type}" for p in func.parameters])
                    if func.parameters
                    else "None"
                )
                cap_table.add_row(
                    func.name,
                    func.description or "No description",
                    params,
                    func.return_type or "void",
                )

            self.output.print(cap_table)
            self.output.print("\n")

        # Print summary statistics
        self.output.print(
            f"[bold green]Total Capabilities:[/bold green] {len(self.profile.capabilities)}"
        )
        total_functions = sum(len(cap.functions) for cap in self.profile.capabilities)
        self.output.print(
            f"[bold green]Total Functions:[/bold green] {total_functions}\n"
        )

    async def interrogate(self) -> AgentProfile:
        """Perform the full interrogation process with iterative discovery and analysis."""
        self.output.print("\n[bold cyan]Starting Agent Interrogation...[/bold cyan]\n")

        # Iterative capability discovery
        capabilities = await self._discover_capabilities()
        self.profile.capabilities.extend(capabilities)

        # Iterative analysis of each capability
        for capability in self.profile.capabilities:
            # _analyze_capability modifies the capability in-place and returns discovered functions
            discovered_functions = await self._analyze_capability(capability)
            self.output.print(
                f"[bold cyan]Completed analysis of {capability.name}. Found {len(discovered_functions)} functions.[/bold cyan]"
            )

        # Display the final profile
        self._display_profile()

        return self.profile

    async def _discover_capabilities(self) -> list[Capability]:
        """Discover high-level capabilities of the agent through multiple cycles.

        Uses UPSERT semantics keyed on ``Capability.node_id`` so that re-discovery
        across cycles merges into the existing entry rather than producing duplicates.
        """
        capabilities_by_id: Dict[str, Capability] = {}
        cycle = 0
        previous_responses: List[Dict[str, Any]] = []
        result: Dict[str, Any] = {}

        while cycle < self.config.max_iterations:
            context = {
                "phase": "discovery",
                "cycle": cycle,
                "previous_responses": previous_responses,
                "discovered_capabilities": list(capabilities_by_id.values()),
                "next_cycle_focus": (
                    result.get("next_cycle_focus") if cycle > 0 else None
                ),
            }

            # Generate and send prompt
            prompt = await self.llm.generate_prompt(context)

            # Display the prompt being sent
            self.output.print_verbose(
                Panel(
                    Text(prompt, style="cyan", overflow="fold"),
                    title=f"[bold cyan]Discovery Cycle {cycle + 1}[/bold cyan] - Prompt",
                    border_style="cyan",
                    expand=False,
                )
            )

            # Get response from agent
            response = await self.agent_callback(prompt)

            # Display the response received
            self.output.print_verbose(
                Panel(
                    Text(response, style="cyan", overflow="fold"),
                    title=f"[bold cyan]Discovery Cycle {cycle + 1}[/bold cyan] - Agent Response",
                    border_style="cyan",
                    expand=False,
                )
            )

            # Store the prompt/response pair in conversation history
            previous_responses.append(
                {"prompt": prompt, "response": response, "cycle": cycle}
            )

            # Process response
            result = await self.llm.process_response(response, context)

            # Apply add/update operations against the running catalog
            added_ids, updated_ids = self._apply_capability_operations(
                capabilities_by_id, result.get("operations", [])
            )
            made_progress = bool(added_ids or updated_ids)

            if added_ids:
                self.output.display_status(
                    f"Discovery cycle {cycle + 1}: added {len(added_ids)} new capabilities"
                )
            if updated_ids:
                self.output.display_status(
                    f"Discovery cycle {cycle + 1}: refined {len(updated_ids)} existing capabilities"
                )

            # Check if we should continue discovery
            should_continue = await self.llm.should_continue_cycle(result)
            if cycle > 0 and not made_progress:
                # Convergence: cycle yielded no adds or updates -> nothing more to learn
                self.output.display_status(
                    f"Discovery cycle {cycle + 1}: no new or updated capabilities — converged",
                    "bold yellow",
                )
                should_continue = False

            self.output.print(
                f"[yellow]Discovery cycle {cycle + 1} complete. is_complete={result.get('is_complete', False)}[/yellow]"
            )
            if not should_continue:
                self.output.print(
                    "[bold yellow]Discovery phase complete![/bold yellow]"
                )
                break

            cycle += 1

        return list(capabilities_by_id.values())

    def _build_capability(self, op_data: Dict[str, Any]) -> Capability:
        """Construct a Capability from an op dict, omitting unknown keys."""
        allowed = {"name", "description", "metadata"}
        payload = {k: v for k, v in op_data.items() if k in allowed and v is not None}
        return Capability(**payload)

    def _apply_capability_operations(
        self,
        capabilities_by_id: Dict[str, Capability],
        operations: List[Any],
    ) -> "tuple[List[str], List[str]]":
        """Apply add/update operations to the running capability catalog.

        Returns ``(added_ids, updated_ids)`` for progress reporting and convergence.
        An ``update`` for an unknown id degrades gracefully to an ``add``.
        """
        added_ids: List[str] = []
        updated_ids: List[str] = []

        for op_data in operations:
            if not isinstance(op_data, dict):
                self.output.print(f"[red]Invalid operation entry: {op_data}[/red]")
                continue

            op = (op_data.get("op") or "add").lower()
            try:
                if op == "update":
                    target_id = op_data.get("id")
                    if target_id and target_id in capabilities_by_id:
                        existing = capabilities_by_id[target_id]
                        # Build a patch capability inheriting the existing identity
                        patch = Capability(
                            node_id=target_id,
                            name=op_data.get("name") or existing.name,
                            description=op_data.get("description"),
                            metadata=op_data.get("metadata") or {},
                        )
                        capabilities_by_id[target_id] = merge_capability(
                            existing, patch
                        )
                        updated_ids.append(target_id)
                        continue
                    # Unknown id: fall through to add-by-content semantics
                    self.output.print_verbose(
                        f"[yellow]Update for unknown id {target_id!r}; treating as add[/yellow]"
                    )

                if "name" not in op_data:
                    self.output.print(f"[red]Skipping op without name: {op_data}[/red]")
                    continue
                new_cap = self._build_capability(op_data)
                if new_cap.node_id in capabilities_by_id:
                    capabilities_by_id[new_cap.node_id] = merge_capability(
                        capabilities_by_id[new_cap.node_id], new_cap
                    )
                    updated_ids.append(new_cap.node_id)
                else:
                    capabilities_by_id[new_cap.node_id] = new_cap
                    added_ids.append(new_cap.node_id)
            except Exception as e:  # noqa: BLE001 - surface to user, keep loop alive
                self.output.print(f"[red]Error applying operation {op}: {e}[/red]")

        return added_ids, updated_ids

    async def _analyze_capability(self, capability: Capability) -> List[Function]:
        """Analyze a specific capability in detail through multiple cycles.

        Uses UPSERT semantics keyed on ``Function.node_id`` so that re-discovery
        across cycles merges into the existing function record. The ``capability``
        argument's ``functions`` list is rebuilt from the merged catalog at the end.
        """
        cycle = 0
        previous_responses: List[Dict[str, Any]] = []
        # Seed the catalog with anything already on the capability
        functions_by_id: Dict[str, Function] = {
            f.node_id: f for f in capability.functions
        }
        result: Dict[str, Any] = {}

        while cycle < self.config.max_iterations:
            context = {
                "phase": "analysis",
                "cycle": cycle,
                "previous_responses": previous_responses,
                "capability": capability,
                "discovered_functions": list(functions_by_id.values()),
                "next_cycle_focus": (
                    result.get("next_cycle_focus") if cycle > 0 else None
                ),
            }

            prompt = await self.llm.generate_prompt(context)
            self.output.display_prompt(prompt, cycle + 1, capability.name)

            response = await self.agent_callback(prompt)
            self.output.display_response(response, cycle + 1, capability.name)

            previous_responses.append(
                {"prompt": prompt, "response": response, "cycle": cycle}
            )

            result = await self.llm.process_response(response, context)

            added_ids, updated_ids = self._apply_function_operations(
                functions_by_id, result.get("operations", [])
            )
            made_progress = bool(added_ids or updated_ids)

            if added_ids:
                self.output.display_status(
                    f"Found {len(added_ids)} new functions in {capability.name}"
                )
            if updated_ids:
                self.output.display_status(
                    f"Refined {len(updated_ids)} existing functions in {capability.name}"
                )

            self.output.display_process_result(result, cycle + 1, capability.name)

            should_continue = await self.llm.should_continue_cycle(result)
            if cycle > 0 and not made_progress:
                self.output.display_status(
                    f"Analysis of {capability.name}: no new or updated functions — converged",
                    "bold yellow",
                )
                should_continue = False

            self.output.display_status(
                f"Analysis cycle {cycle + 1} complete. is_complete={result.get('is_complete', False)}"
            )
            if not should_continue:
                self.output.display_status(
                    f"Analysis of {capability.name} complete!", "bold yellow"
                )
                break

            cycle += 1

        # Rewrite the capability's function list from the merged catalog
        capability.functions = list(functions_by_id.values())
        return capability.functions

    @staticmethod
    def _normalize_parameters(raw_params: Any) -> List[Parameter]:
        """Coerce LLM-emitted parameter entries into ``Parameter`` instances."""
        if not isinstance(raw_params, list):
            return []
        normalized: List[Parameter] = []
        for param in raw_params:
            if isinstance(param, Parameter):
                normalized.append(param)
            elif isinstance(param, dict):
                normalized.append(Parameter(**param))
            elif isinstance(param, str):
                if ":" in param:
                    name, ptype = param.split(":", 1)
                    normalized.append(Parameter(name=name.strip(), type=ptype.strip()))
                else:
                    normalized.append(Parameter(name=param.strip(), type="string"))
        return normalized

    def _build_function(self, op_data: Dict[str, Any]) -> Function:
        """Construct a Function from an op dict, omitting unknown keys."""
        payload: Dict[str, Any] = {}
        if op_data.get("name"):
            payload["name"] = op_data["name"]
        if op_data.get("description") is not None:
            payload["description"] = op_data["description"]
        if op_data.get("return_type") is not None:
            payload["return_type"] = op_data["return_type"]
        if "parameters" in op_data:
            payload["parameters"] = self._normalize_parameters(op_data["parameters"])
        return Function(**payload)

    def _apply_function_operations(
        self,
        functions_by_id: Dict[str, Function],
        operations: List[Any],
    ) -> "tuple[List[str], List[str]]":
        """Apply add/update operations to the running function catalog.

        Returns ``(added_ids, updated_ids)``. An update for an unknown id falls
        back to add-by-content semantics so we never drop information.
        """
        added_ids: List[str] = []
        updated_ids: List[str] = []

        for op_data in operations:
            if not isinstance(op_data, dict):
                self.output.print(f"[red]Invalid operation entry: {op_data}[/red]")
                continue

            op = (op_data.get("op") or "add").lower()
            try:
                if op == "update":
                    target_id = op_data.get("id")
                    if target_id and target_id in functions_by_id:
                        existing = functions_by_id[target_id]
                        patch = Function(
                            node_id=target_id,
                            name=op_data.get("name") or existing.name,
                            description=op_data.get("description"),
                            parameters=self._normalize_parameters(
                                op_data.get("parameters", [])
                            ),
                            return_type=op_data.get("return_type"),
                        )
                        functions_by_id[target_id] = merge_function(existing, patch)
                        updated_ids.append(target_id)
                        continue
                    self.output.print_verbose(
                        f"[yellow]Update for unknown id {target_id!r}; treating as add[/yellow]"
                    )

                if not op_data.get("name"):
                    self.output.print(
                        f"[red]Skipping function op without name: {op_data}[/red]"
                    )
                    continue
                new_fn = self._build_function(op_data)
                if new_fn.node_id in functions_by_id:
                    functions_by_id[new_fn.node_id] = merge_function(
                        functions_by_id[new_fn.node_id], new_fn
                    )
                    updated_ids.append(new_fn.node_id)
                else:
                    functions_by_id[new_fn.node_id] = new_fn
                    added_ids.append(new_fn.node_id)
            except Exception as e:  # noqa: BLE001 - surface to user, keep loop alive
                self.output.print(f"[red]Error applying operation {op}: {e}[/red]")

        return added_ids, updated_ids
