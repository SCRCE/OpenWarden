<p align="center">
  <img src="https://raw.githubusercontent.com/SCRCE/OpenWarden/main/assets/brand/openwarden-logo.svg?v=transparent" alt="OpenWarden" width="600">
</p>

<p align="center">
  Composable guardrails for OpenAI-compatible clients and native PyTorch generation.
</p>

<p align="center">
  <a href="https://github.com/SCRCE/OpenWarden/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/SCRCE/OpenWarden/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://pypi.org/project/openwarden/"><img alt="PyPI" src="https://img.shields.io/pypi/v/openwarden.svg"></a>
  <a href="https://pypi.org/project/openwarden/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/openwarden.svg"></a>
  <img alt="Typed" src="https://img.shields.io/badge/typing-typed-006F66">
  <a href="https://github.com/SCRCE/OpenWarden/blob/main/LICENSE"><img alt="MIT license" src="https://img.shields.io/badge/license-MIT-B72B5B.svg"></a>
</p>

OpenWarden wraps clients and models you already use. It intercepts supported
generation calls, evaluates their input and output, and either returns the
original response object or raises an explicit guard exception.

```bash
pip install openwarden
```

## Quick Start

```python
from openai import OpenAI

from openwarden import GuardMode, GuardedOpenAI, WardenConfig
from openwarden.guards.regex import RegexSecretGuard

client = GuardedOpenAI(
    OpenAI(),
    guard=RegexSecretGuard(),
    config=WardenConfig(
        mode=GuardMode.BOTH,
        fail_open=False,
    ),
)

response = client.responses.create(
    model="your-generation-model",
    instructions="You are concise and helpful.",
    input="Explain Python decorators with a simple example.",
)

print(response.output_text)
```

Allowed calls return the original OpenAI SDK object. Blocked calls raise
`GuardViolation`.

## Supported Surfaces

- Synchronous `OpenAI` and asynchronous `AsyncOpenAI` clients.
- `responses.create(...)` and `chat.completions.create(...)`.
- Native PyTorch or Transformers `model.generate(...)`.
- Input-only, output-only, both, and disabled modes.
- Block, redact, rewrite, annotate, and log-only enforcement actions.
- Deterministic guards, guard pipelines, and prompted guard models.
- Per-call overrides and context-managed overrides.
- RAG retrieval, context injection, and grounding extension points.
- Trace IDs, decisions, latencies, and optional `GuardedResult` metadata.
- Fail-open and fail-closed operation.

Unsupported SDK resources pass through to the wrapped client unchanged.

## Guard Models

`ModelPromptGuard` can call a dedicated policy model through any compatible
OpenAI endpoint:

```python
import os

from openai import OpenAI

from openwarden import GuardMode, GuardedOpenAI, WardenConfig
from openwarden.guards.base import ModelPromptGuard

raw_client = OpenAI()

guard = ModelPromptGuard(
    raw_client,
    model=os.environ["GUARD_MODEL"],
    api="chat_completions",
    require_json=True,
)

client = GuardedOpenAI(
    raw_client,
    guard=guard,
    config=WardenConfig(mode=GuardMode.BOTH, fail_open=False),
)
```

Provider-specific generation parameters can be passed through
`create_kwargs`.

## Direct PyTorch Generation

OpenWarden does not import or pin PyTorch or Transformers. It composes around
compatible objects from the versions selected by your application:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

from openwarden import GuardMode, GuardedPyTorch, WardenConfig
from openwarden.guards.regex import RegexSecretGuard

model_id = "your-org/your-generation-model"
tokenizer = AutoTokenizer.from_pretrained(model_id)
raw_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

model = GuardedPyTorch(
    raw_model,
    tokenizer=tokenizer,
    guard=RegexSecretGuard(),
    config=WardenConfig(mode=GuardMode.BOTH, fail_open=False),
)

inputs = tokenizer("Explain decorators.", return_tensors="pt").to(raw_model.device)
output = model.generate(**inputs, max_new_tokens=200)
```

Allowed calls return the original tensor or Transformers generation object.
`PyTorchPromptGuard` can run a separate policy model locally.

## Guard Decisions

Custom guards subclass `BaseGuard` and return structured decisions:

```python
from openwarden import BaseGuard, GuardDecision


class CompanyPolicyGuard(BaseGuard):
    def check_input(self, context):
        if "confidential" in context.request.text().lower():
            return GuardDecision.block(
                "Confidential content cannot leave this boundary",
                categories=("data_policy",),
            )
        return GuardDecision.allow()
```

## Configuration

```python
from openwarden import GuardAction, GuardMode, WardenConfig

config = WardenConfig(
    mode=GuardMode.BOTH,
    input_action=GuardAction.BLOCK,
    output_action=GuardAction.BLOCK,
    fail_open=False,
    timeout_seconds=10,
)
```

Streaming is conservative by default. When output guarding is enabled,
`stream=True` raises `UnsupportedStreamingModeError` rather than exposing
tokens before the complete response can be evaluated.

## Documentation

- [User guide](https://github.com/SCRCE/OpenWarden/blob/main/docs/user-guide.md)
- [API reference](https://github.com/SCRCE/OpenWarden/blob/main/docs/api-reference.md)
- [Publishing guide](https://github.com/SCRCE/OpenWarden/blob/main/docs/publishing.md)
- [Security policy](https://github.com/SCRCE/OpenWarden/blob/main/SECURITY.md)

## Security

OpenWarden is an application-layer policy boundary. It does not replace
authorization, provider safety systems, deterministic access controls, or
secure secret storage. Keep retrieval authorization in the data layer and use
`fail_open=False` where enforcement is required.

Please report vulnerabilities according to the [security policy](https://github.com/SCRCE/OpenWarden/blob/main/SECURITY.md). Do not
open public issues containing credentials, private prompts, customer data, or
model output.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check openwarden tests examples scripts
python -m pytest -q
python -m build
python -m twine check dist/*
```

See the [contribution guide](https://github.com/SCRCE/OpenWarden/blob/main/CONTRIBUTING.md) before submitting changes.

## License

OpenWarden is available under the [MIT License](https://github.com/SCRCE/OpenWarden/blob/main/LICENSE).
