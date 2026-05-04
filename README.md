# Agent Interrogator

<p align="center">
  <img src="https://raw.githubusercontent.com/qwordsmith/agent-interrogator/refs/heads/main/assets/logo.webp" alt="Agent Interrogator Logo" width="400" />
</p>

<p align="center">
  <strong>Systematically discover and map AI agent attack surface for security research</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/agent-interrogator/">
    <img src="https://badge.fury.io/py/agent-interrogator.svg" alt="PyPI version">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+">
  </a>
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache%202.0-yellow.svg" alt="License: Apache 2.0">
  </a>
</p>

---

## What is Agent Interrogator?

Agent Interrogator is a Python library designed for **security researchers** to systematically discover and analyze AI agent attack surface through automated interrogation. It uses iterative discovery cycles to map an agent's available tools (functions).

### Why Use Agent Interrogator?

- **🔍 Attack Surface Discovery**: Automatically discovers agent capabilities and supporting tools without requiring documentation
- **🛡️ Security Research**: Purpose-built for vulnerability assessment and prompt injection testing
- **📊 Structured Output**: Generates structured profiles perfect for integration with other security tools
- **🔄 Iterative Analysis**: Uses smart prompt adaptation to uncover hidden or complex capabilities
- **🚀 Flexible Integrations**: Works with any agent via customizable callback functions

### Perfect For:
- Security researchers testing AI agents for vulnerabilities
- Red teams conducting agent penetration testing
- Security teams auditing agent functionality

---

## Quick Start

### Installation

```bash
pip install agent-interrogator
```

### Basic Usage

Here's a minimal example that interrogates an agent:

```python
import asyncio
from agent_interrogator import AgentInterrogator, InterrogationConfig, LLMConfig, ModelProvider

# Configure the interrogator
config = InterrogationConfig(
    llm=LLMConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4",
        api_key="your-openai-api-key"
    ),
    max_iterations=5
)

# Define how to interact with your target agent
async def my_agent_callback(prompt: str) -> str:
    """
    This function defines how to send prompts to your target agent.
    Replace this with your actual agent interaction logic.
    """
    # Example: HTTP API call to your agent
    # response = await call_your_agent_api(prompt)
    # return response.text
    
    # For demo purposes, return a mock response
    return "I can help with web searches, file operations, and calculations."

# Run the interrogation
async def main():
    interrogator = AgentInterrogator(config, my_agent_callback)
    profile = await interrogator.interrogate()
    
    # View discovered capabilities
    print(f"Discovered {len(profile.capabilities)} capabilities:")
    for capability in profile.capabilities:
        print(f"  - {capability.name}: {capability.description}")
        for f in capability.functions:
            print(f"    Function Name: {f.name}")
            print(f"    Function Parameters: {f.parameters}")
            print(f"    Function Return Type: {f.return_type}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Expected Output

```
Discovered 3 capabilities:
  - web_search: Search the internet for information
    Function Name: search_web
    Function Parameters: [ { "name": "query", "type": "string", "description": "The search query", "required": true }, { "name": "max_results", "type": "integer", "description": "Maximum number of results", "required": false, "default": 5 } ]
    Function Return Types: list[SearchResult]
...
```

---

## Installation

### Standard Installation

```bash
pip install agent-interrogator
```

### Development Installation

For contributors or advanced users who want to modify the code:

```bash
git clone https://github.com/qwordsmith/agent-interrogator.git
cd agent-interrogator
pip install -e .[dev]
```

### Requirements

- **Python**: 3.9 or higher
- **OpenAI API Key**: For using GPT models (optional, can use Ollama or any OpenAI-compatible endpoint instead)
- **Dependencies**: Automatically installed with pip

---

## Configuration

Agent Interrogator supports OpenAI, a local Ollama daemon, or any OpenAI-compatible endpoint (vLLM, LM Studio, LocalAI, etc.) for analyzing agent responses:

### OpenAI Configuration

```python
from agent_interrogator import InterrogationConfig, LLMConfig, ModelProvider, OutputMode

config = InterrogationConfig(
    llm=LLMConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4",
        api_key="your-openai-api-key"
    ),
    max_iterations=5,  # Maximum discovery cycles
    output_mode=OutputMode.STANDARD  # QUIET, STANDARD, or VERBOSE
)
```

### Ollama Configuration (local models)

```python
from agent_interrogator import OllamaConfig

config = InterrogationConfig(
    llm=LLMConfig(
        provider=ModelProvider.OLLAMA,
        model_name="llama3.2:latest",  # Any model pulled into your Ollama daemon
        ollama=OllamaConfig(
            host="http://localhost:11434",
            timeout=120.0,
            options={"temperature": 0.1, "top_p": 0.9},
            keep_alive="5m",
        )
    ),
    max_iterations=5,
    output_mode=OutputMode.VERBOSE
)
```

### OpenAI-Compatible Endpoint Configuration

Use this for any server that exposes an OpenAI-shaped Chat Completions API
(vLLM, LM Studio, LocalAI, custom gateways, etc.).

```python
from agent_interrogator import OpenAICompatibleConfig

config = InterrogationConfig(
    llm=LLMConfig(
        provider=ModelProvider.OPENAI_COMPATIBLE,
        model_name="local-model",
        openai_compatible=OpenAICompatibleConfig(
            base_url="http://localhost:8000/v1",
            api_key="not-required",  # Some endpoints ignore this
            timeout=120.0,
        )
    ),
    max_iterations=5,
    output_mode=OutputMode.STANDARD
)
```

### Output Modes

- **`QUIET`**: No terminal output (ideal for automated scripts)
- **`STANDARD`**: Shows progress and results (default)
- **`VERBOSE`**: Detailed logging including prompts and responses (useful for debugging)

---

## Implementing Callbacks

The callback function is how Agent Interrogator communicates with your target agent. It must be an async function that takes a prompt string and returns the agent's response.

### Callback Interface

```python
from typing import Awaitable, Callable

# Your callback must match this signature
AgentCallback = Callable[[str], Awaitable[str]]
```

### HTTP API Example

Here's an example for an agent exposed via an HTTP API:

```python
import aiohttp
from typing import Optional

class HTTPAgentCallback:
    def __init__(self, endpoint: str, api_key: Optional[str] = None):
        self.endpoint = endpoint
        self.headers = {}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __call__(self, prompt: str) -> str:
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        async with self.session.post(
            self.endpoint,
            json={"message": prompt, "stream": False},
            headers=self.headers
        ) as response:
            if response.status != 200:
                raise Exception(f"Agent API error: {response.status}")
            
            result = await response.json()
            return result["response"]
    
    async def cleanup(self):
        """Optional cleanup method"""
        if self.session:
            await self.session.close()
            self.session = None

# Usage
callback = HTTPAgentCallback(
    endpoint="https://your-agent-api.com/chat",
    api_key="your-agent-api-key"
)

interrogator = AgentInterrogator(config, callback)
profile = await interrogator.interrogate()
```

### More Examples

For additional callback implementations (WebSocket, Playwright browser automation, process-based agents, etc.), see the [`examples/callbacks.py`](examples/callbacks.py) file.

---

## Understanding Results

Agent Interrogator produces a structured `AgentProfile` containing all discovered capabilities and functions. This data is specifically designed for security research and tool integration.

### Profile Structure

```python
# Access the profile data
profile = await interrogator.interrogate()

# Iterate through capabilities
for capability in profile.capabilities:
    print(f"Capability: {capability.name}")
    print(f"Description: {capability.description}")
    
    for function in capability.functions:
        print(f"  Function: {function.name}")
        print(f"  Description: {function.description}")
        print(f"  Return Type: {function.return_type}")
        
        for param in function.parameters:
            print(f"    Parameter: {param.name}")
            print(f"    Type: {param.type}")
            print(f"    Required: {param.required}")
            print(f"    Default: {param.default}")
```

### Security Research Applications

The structured data enables:

- **Attack Surface Mapping**: Complete inventory of agent capabilities
- **Fuzzing Target Generation**: Automated payload creation for each function
- **Prompt Injection Testing**: Parameter-aware injection attempts
- **Capability Monitoring**: Track changes between agent versions
- **Agent Auditing**: Verify agents operate within expected bounds

---

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e .[dev]

# Run the test suite
pytest tests/

# Run with coverage
pytest tests/ --cov=agent_interrogator
```

### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/agent_interrogator/

# Linting
flake8 src/ tests/
```

### Project Structure

```
agent-interrogator/
├── src/agent_interrogator/    # Main package
│   ├── __init__.py           # Public API
│   ├── interrogator.py       # Core interrogation logic
│   ├── config.py             # Configuration models
│   ├── llm.py                # LLM provider interfaces
│   ├── models.py             # Data models (AgentProfile, etc.)
│   ├── output.py             # Terminal output management
│   └── prompt_templates.py   # LLM prompts
├── examples/                 # Usage examples
├── tests/                    # Test suite
```

---

## Contributing

Contributions are welcome!

### How to Contribute

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Add tests** for your changes
4. **Ensure** all tests pass (`pytest tests/`)
5. **Format** your code (`black src/ tests/`)
6. **Submit** a pull request

### Areas for Contribution

- Callback implementations for different agent types
- Recursive interrogation of agents to agent communication
    - Agents made available to target agent via A2A
    - Agents made available to target agent via MCP
- Performance optimizations for large-scale agent scanning
- Guardrail bypass capabilities
- Integration examples with security tools
- Additional LLM provider support
- Mechanisms to improve agent profile output quality
- Documentation improvements

---

## License

This project is licensed under the **Apache License, Version 2.0** - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [README.md](README.md) and inline code documentation
- **Related Research**: [Research-Paper-Resources](https://github.com/qwordsmith/Research-Paper-Resources/)
- **Issues**: [GitHub Issues](https://github.com/qwordsmith/agent-interrogator/issues)
- **Examples**: See [`examples/`](examples/) directory
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)
