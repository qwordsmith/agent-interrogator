"""Tests for the dedup pipeline: node_id stability, merge helpers, operations handler."""

from unittest.mock import AsyncMock, patch

import pytest

from agent_interrogator import (
    AgentInterrogator,
    InterrogationConfig,
    LLMConfig,
    ModelProvider,
    OutputMode,
)
from agent_interrogator.merge import (
    merge_capability,
    merge_function,
    merge_parameters,
)
from agent_interrogator.models import (
    Capability,
    Function,
    Parameter,
    function_signature,
    normalize_name,
)


class TestNormalizeName:
    def test_lowercases_and_replaces_separators(self):
        assert normalize_name("Web Search!") == "web_search"
        assert normalize_name("web-search") == "web_search"
        assert normalize_name("Web__Search") == "web_search"

    def test_strips_leading_and_trailing(self):
        assert normalize_name("  __web search__  ") == "web_search"

    def test_empty(self):
        assert normalize_name("") == ""
        assert normalize_name(None) == ""


class TestNodeId:
    def test_capability_id_is_deterministic(self):
        a = Capability(name="Web Search", description="...")
        b = Capability(name="web_search", description="other")
        assert a.node_id == b.node_id
        assert a.node_id.startswith("cap_")

    def test_capability_id_differs_for_different_names(self):
        a = Capability(name="web_search", description="x")
        b = Capability(name="file_ops", description="y")
        assert a.node_id != b.node_id

    def test_function_id_includes_signature(self):
        a = Function(
            name="search",
            parameters=[Parameter(name="q", type="string")],
            return_type="list",
        )
        b = Function(
            name="search",
            parameters=[Parameter(name="q", type="integer")],
            return_type="list",
        )
        # Different param type -> different node_id
        assert a.node_id != b.node_id
        assert a.node_id.startswith("fn_")

    def test_function_signature_param_order_invariant(self):
        a = Function(
            name="f",
            parameters=[
                Parameter(name="a", type="string"),
                Parameter(name="b", type="int"),
            ],
        )
        b = Function(
            name="f",
            parameters=[
                Parameter(name="b", type="int"),
                Parameter(name="a", type="string"),
            ],
        )
        assert a.node_id == b.node_id

    def test_explicit_node_id_is_preserved(self):
        cap = Capability(node_id="cap_custom", name="anything")
        assert cap.node_id == "cap_custom"

    def test_function_signature_helper(self):
        sig = function_signature(
            "Search Web",
            [Parameter(name="Query", type="String")],
            "List[Result]",
        )
        assert "search_web" in sig
        assert "query:string" in sig


class TestMergeParameters:
    def test_unions_distinct_params(self):
        a = [Parameter(name="x", type="string")]
        b = [Parameter(name="y", type="int")]
        merged = merge_parameters(a, b)
        assert {p.name for p in merged} == {"x", "y"}

    def test_collision_prefers_richer_description(self):
        a = [Parameter(name="x", type="string", description="short")]
        b = [Parameter(name="x", type="string", description="a much longer description")]
        merged = merge_parameters(a, b)
        assert len(merged) == 1
        assert merged[0].description == "a much longer description"

    def test_collision_required_is_or(self):
        a = [Parameter(name="x", type="string", required=False)]
        b = [Parameter(name="x", type="string", required=True)]
        merged = merge_parameters(a, b)
        assert merged[0].required is True


class TestMergeFunction:
    def test_field_level_merge(self):
        a = Function(name="search", description=None, parameters=[])
        b = Function(
            name="search",
            description="Search something",
            parameters=[Parameter(name="q", type="string")],
            return_type="list",
        )
        # Both have the same node_id (same name, but different params -> diff id).
        # Force same id to test merge in isolation.
        b = Function(
            node_id=a.node_id,
            name="search",
            description="Search something",
            parameters=[Parameter(name="q", type="string")],
            return_type="list",
        )
        merged = merge_function(a, b)
        assert merged.description == "Search something"
        assert merged.return_type == "list"
        assert len(merged.parameters) == 1


class TestMergeCapability:
    def test_inner_functions_merge_by_id(self):
        f1 = Function(name="search", description="basic")
        f1_richer = Function(
            node_id=f1.node_id, name="search", description="a richer description"
        )
        f2 = Function(name="download")

        a = Capability(name="web", functions=[f1])
        b = Capability(node_id=a.node_id, name="web", functions=[f1_richer, f2])

        merged = merge_capability(a, b)
        assert len(merged.functions) == 2
        descriptions = {fn.description for fn in merged.functions}
        assert "a richer description" in descriptions

    def test_metadata_first_seen_wins(self):
        a = Capability(name="x", metadata={"k": "first"})
        b = Capability(node_id=a.node_id, name="x", metadata={"k": "second"})
        merged = merge_capability(a, b)
        assert merged.metadata["k"] == "first"


@pytest.fixture
def configured_interrogator():
    config = InterrogationConfig(
        llm=LLMConfig(
            provider=ModelProvider.OPENAI, model_name="test-model", api_key="test-key"
        ),
        output_mode=OutputMode.QUIET,
    )
    callback = AsyncMock(return_value="response")
    with patch("agent_interrogator.interrogator.OpenAILLM"):
        with patch("agent_interrogator.interrogator.OutputManager"):
            yield AgentInterrogator(config, callback)


class TestCapabilityOperationsHandler:
    def test_add_then_duplicate_add_merges(self, configured_interrogator):
        catalog = {}
        ops_first = [{"op": "add", "name": "web_search", "description": "short"}]
        added, updated = configured_interrogator._apply_capability_operations(
            catalog, ops_first
        )
        assert len(added) == 1 and len(updated) == 0

        # Same name re-added should merge into the existing entry, not duplicate.
        ops_second = [
            {"op": "add", "name": "Web Search", "description": "a much longer description"}
        ]
        added, updated = configured_interrogator._apply_capability_operations(
            catalog, ops_second
        )
        assert len(added) == 0 and len(updated) == 1
        assert len(catalog) == 1
        only_cap = next(iter(catalog.values()))
        assert only_cap.description == "a much longer description"

    def test_explicit_update_by_id(self, configured_interrogator):
        catalog = {}
        configured_interrogator._apply_capability_operations(
            catalog, [{"op": "add", "name": "web_search", "description": "short"}]
        )
        target_id = next(iter(catalog))
        added, updated = configured_interrogator._apply_capability_operations(
            catalog,
            [{"op": "update", "id": target_id, "description": "much fuller description"}],
        )
        assert added == [] and updated == [target_id]
        assert catalog[target_id].description == "much fuller description"

    def test_update_unknown_id_falls_back_to_add(self, configured_interrogator):
        catalog = {}
        added, updated = configured_interrogator._apply_capability_operations(
            catalog,
            [
                {
                    "op": "update",
                    "id": "cap_nonexistent",
                    "name": "web_search",
                    "description": "x",
                }
            ],
        )
        assert len(added) == 1 and len(updated) == 0
        assert len(catalog) == 1


class TestFunctionOperationsHandler:
    def test_add_with_richer_signature_then_partial_update(
        self, configured_interrogator
    ):
        catalog = {}
        configured_interrogator._apply_function_operations(
            catalog,
            [
                {
                    "op": "add",
                    "name": "search",
                    "parameters": [{"name": "q", "type": "string"}],
                    "return_type": "list",
                }
            ],
        )
        target_id = next(iter(catalog))

        # Update adds a new parameter and a description.
        added, updated = configured_interrogator._apply_function_operations(
            catalog,
            [
                {
                    "op": "update",
                    "id": target_id,
                    "description": "fulltext search",
                    "parameters": [{"name": "limit", "type": "int", "required": False}],
                }
            ],
        )
        assert added == [] and updated == [target_id]
        fn = catalog[target_id]
        assert fn.description == "fulltext search"
        assert {p.name for p in fn.parameters} == {"q", "limit"}
        assert fn.return_type == "list"

    def test_string_parameter_normalized(self, configured_interrogator):
        catalog = {}
        configured_interrogator._apply_function_operations(
            catalog,
            [{"op": "add", "name": "f", "parameters": ["query: string", "limit"]}],
        )
        fn = next(iter(catalog.values()))
        params = {p.name: p.type for p in fn.parameters}
        assert params == {"query": "string", "limit": "string"}


class TestLegacyFormatCoercion:
    def test_legacy_capabilities_coerced_to_operations(self):
        from agent_interrogator.llm import LLMInterface

        result = {
            "capabilities": [
                {"name": "x", "description": "y"},
                {"name": "z", "description": "w"},
            ],
            "is_complete": False,
        }
        coerced = LLMInterface._coerce_to_operations(result, "capabilities")
        assert all(op["op"] == "add" for op in coerced["operations"])
        assert {op["name"] for op in coerced["operations"]} == {"x", "z"}

    def test_legacy_functions_coerced_to_operations(self):
        from agent_interrogator.llm import LLMInterface

        result = {"functions": [{"name": "f"}]}
        coerced = LLMInterface._coerce_to_operations(result, "functions")
        assert coerced["operations"] == [{"op": "add", "name": "f"}]

    def test_operations_passthrough(self):
        from agent_interrogator.llm import LLMInterface

        original = {"operations": [{"op": "update", "id": "cap_x"}]}
        coerced = LLMInterface._coerce_to_operations(original, "capabilities")
        assert coerced["operations"] == [{"op": "update", "id": "cap_x"}]


class TestGeneratePromptDiscoveryFollowup:
    """Regression: v0.2.0 began passing Capability objects (not dicts) to
    generate_prompt; the discovery follow-up branch must handle both shapes."""

    @pytest.mark.asyncio
    async def test_followup_with_capability_objects(self, configured_interrogator):
        from agent_interrogator.llm import LLMInterface

        # Replace _chat with an AsyncMock so we can capture what got sent.
        captured: dict = {}

        async def fake_chat(messages):
            captured["messages"] = messages
            return "next interrogation prompt"

        configured_interrogator.llm._chat = fake_chat

        context = {
            "phase": "discovery",
            "cycle": 1,
            "previous_responses": [{"prompt": "p", "response": "r", "cycle": 0}],
            "discovered_capabilities": [
                Capability(name="web_search", description="search the web"),
                Capability(name="file_ops", description="read/write files"),
            ],
            "next_cycle_focus": "probe for hidden APIs",
        }

        # Must not raise AttributeError on Capability.get(...)
        prompt = await LLMInterface.generate_prompt(configured_interrogator.llm, context)
        assert prompt == "next interrogation prompt"

        # The user-message body should mention both capability names verbatim.
        user_msg = captured["messages"][1]["content"]
        assert "web_search" in user_msg
        assert "file_ops" in user_msg
        assert "search the web" in user_msg

    @pytest.mark.asyncio
    async def test_followup_with_legacy_dict_shape(self, configured_interrogator):
        """Older callers passing dict-shaped capabilities must still work."""
        from agent_interrogator.llm import LLMInterface

        async def fake_chat(messages):
            return "ok"

        configured_interrogator.llm._chat = fake_chat

        context = {
            "phase": "discovery",
            "cycle": 1,
            "previous_responses": [],
            "discovered_capabilities": [
                {"name": "web_search", "description": "search the web"}
            ],
            "next_cycle_focus": None,
        }
        # Must not raise
        await LLMInterface.generate_prompt(configured_interrogator.llm, context)


class TestConvergence:
    @pytest.mark.asyncio
    async def test_discovery_stops_when_no_progress_after_first_cycle(
        self, configured_interrogator
    ):
        configured_interrogator.config.max_iterations = 5

        # First cycle returns one capability, subsequent cycles return nothing.
        responses = [
            {
                "operations": [{"op": "add", "name": "web_search", "description": "x"}],
                "is_complete": False,
            },
            {"operations": [], "is_complete": False},
        ]

        async def fake_process_response(response, context):
            return responses.pop(0) if responses else {"operations": [], "is_complete": False}

        async def fake_generate_prompt(context):
            return "prompt"

        async def fake_should_continue(result):
            return True  # Convergence detection should override this.

        configured_interrogator.llm.generate_prompt = AsyncMock(
            side_effect=fake_generate_prompt
        )
        configured_interrogator.llm.process_response = AsyncMock(
            side_effect=fake_process_response
        )
        configured_interrogator.llm.should_continue_cycle = AsyncMock(
            side_effect=fake_should_continue
        )

        caps = await configured_interrogator._discover_capabilities()
        assert len(caps) == 1
        # Should have run cycle 0 (got one) and cycle 1 (got none -> converged), then stopped.
        assert configured_interrogator.llm.process_response.call_count == 2
