# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-05-03

### Added
- `ModelProvider.OLLAMA` provider with `OllamaConfig` (host, timeout, generation options, keep-alive) for running interrogation against a local Ollama daemon.
- `ModelProvider.OPENAI_COMPATIBLE` provider with `OpenAICompatibleConfig` (base_url, api_key, timeout) for any OpenAI-shaped Chat Completions endpoint (vLLM, LM Studio, LocalAI, custom gateways, etc.).
- `OpenAIConfig` with optional `timeout`, exposed on `LLMConfig.openai`, for parity with the other providers.
- Centralized JSON extraction in `LLMInterface._extract_json` with markdown code-block and regex fallbacks.
- Stable `node_id` on `Capability` and `Function` (sha1 of normalized name + signature) — the foundation for forthcoming graph-backed storage analogous to BloodHound's `ObjectIdentifier`.
- `merge.py` with field-level UPSERT helpers (`merge_capability`, `merge_function`, `merge_parameters`) — prefer-longer for descriptions, union parameters by `(name, type)`, first-seen-wins for metadata conflicts.

### Changed
- **Breaking:** `LLMInterface` refactored. Providers now implement only a small `_chat(messages) -> str` primitive; `generate_prompt`, `process_discovery_response`, and `process_analysis_response` are owned by the base class. Custom provider implementations from 0.1.x will need to be migrated.
- **Breaking:** Discovery and analysis JSON schemas changed from `{capabilities|functions: [...]}` to `{operations: [{op:"add"|"update", id?, ...}]}`. The interrogator LLM can now refine prior entries explicitly instead of re-emitting them. Legacy `{capabilities|functions: [...]}` payloads are still accepted and auto-coerced to add-only operations for backward compatibility.
- Discovery and analysis loops switched from append-only `.extend()` to UPSERT keyed on `node_id`, with field-level merge on collision.
- Convergence stop condition added: cycles after the first that produce zero adds and zero updates short-circuit the loop.

### Fixed
- Duplicate capability and function entries that previously appeared when the agent re-described the same tool across cycles. The interrogator now merges into the existing record (richer description + unioned parameters) instead of inserting a second entry.

## [0.1.1] - 2025-07-27
- Requirements update

## [0.1.0] - 2025-07-27

### Added
- Initial implementation of Agent Interrogator framework
- Support for OpenAI and HuggingFace language models
- Async-first design with customizable callbacks
- Playwright browser automation support
- Multiple output modes (quiet, standard, verbose)
- Rich terminal interface with colored output
- Structured output format for security tool integration
- Basic test suite with output mode testing
- Example callback implementations

### Features
- Automated discovery of agent capabilities and functions
- Iterative analysis with smart prompt adaptation
- Support for HTTP, WebSocket, and browser-based agent interactions
- Configurable via YAML or programmatic interface
- Type-safe configuration with Pydantic models

[0.1.0]: https://github.com/qwordsmith/agent-interrogator/releases/tag/v0.1.0