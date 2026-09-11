# User Guide

This guide shows how to use OpenWarden with OpenAI-compatible Python clients.

## Install

```bash
pip install openwarden
```

OpenWarden depends on the official `openai` Python package and wraps a real `OpenAI` or `AsyncOpenAI` instance.

## Configure a Client

Use the OpenAI SDK environment variables:

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://your-compatible-endpoint/v1"
```

Then wrap the configured client:

```python
from openai import OpenAI
from openwarden import GuardedOpenAI, GuardMode, WardenConfig
from openwarden.guards.regex import RegexSecretGuard

client = GuardedOpenAI(
    OpenAI(),
    guard=RegexSecretGuard(),
    config=WardenConfig(mode=GuardMode.BOTH),
)
```

## Chat Completions

```python
response = client.chat.completions.create(
    model="your-generation-model",
    messages=[
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Explain Python decorators."},
    ],
)

print(response.choices[0].message.content)
```

## Responses

```python
response = client.responses.create(
    model="your-generation-model",
    instructions="You are concise.",
    input="Explain Python decorators.",
)

print(response.output_text)
```

Use the endpoint your model provider supports. Some OpenAI-compatible providers support Chat Completions but not Responses for a given model.

## Guard Model Prompting

For a dedicated safeguard model:

```python
from openwarden.guards.base import ModelPromptGuard

guard = ModelPromptGuard(
    OpenAI(),
    model="your-guard-model",
    api="chat_completions",
    temperature=1,
    max_tokens=512,
    chat_max_tokens_parameter="max_completion_tokens",
    create_kwargs={"top_p": 1},
)
```

Then use it in the wrapper:

```python
client = GuardedOpenAI(
    OpenAI(),
    guard=guard,
    config=WardenConfig(mode=GuardMode.BOTH, fail_open=False),
)
```

## Custom Guards

```python
from openwarden import BaseGuard, GuardDecision

class NoSecretsGuard(BaseGuard):
    def check_input(self, context):
        text = context.request.text().lower()
        if "api key" in text:
            return GuardDecision.block("API keys cannot be sent to the model")
        return GuardDecision.allow()
```

## Direct PyTorch Inference

Wrap a local text-generation model without creating an OpenAI-compatible server:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

from openwarden import GuardMode, GuardedPyTorch, WardenConfig
from openwarden.guards.regex import RegexSecretGuard

tokenizer = AutoTokenizer.from_pretrained("your-org/your-generation-model")
raw_model = AutoModelForCausalLM.from_pretrained(
    "your-org/your-generation-model",
    device_map="auto",
)

model = GuardedPyTorch(
    raw_model,
    tokenizer=tokenizer,
    guard=RegexSecretGuard(),
    config=WardenConfig(mode=GuardMode.BOTH, fail_open=False),
)

inputs = tokenizer("Explain decorators.", return_tensors="pt").to(raw_model.device)
output_ids = model.generate(**inputs, max_new_tokens=200)
```

The wrapper accepts positional or keyword `input_ids`, guards decoded input and
new completion tokens, forwards generation arguments unchanged, and returns the
native model output. Input modification and RAG augmentation require keyword
`input_ids` so OpenWarden can safely re-tokenize the effective prompt.

To run the policy guard locally as well:

```python
from openwarden import PyTorchPromptGuard

guard = PyTorchPromptGuard(
    guard_model,
    tokenizer=guard_tokenizer,
    system_prompt="Apply the company safety policy and return JSON.",
    generation_kwargs={
        "max_new_tokens": 256,
        "do_sample": False,
    },
)
```

`PyTorchPromptGuard` uses `apply_chat_template(...)` when the tokenizer provides
it and otherwise uses a plain system-prompt format. PyTorch and Transformers are
optional runtime dependencies controlled by the host application.

## Pipelines

```python
from openwarden.guards import GuardPipeline
from openwarden.guards.regex import RegexSecretGuard

guard = GuardPipeline([
    RegexSecretGuard(),
    NoSecretsGuard(),
])
```

The first non-allow decision stops the pipeline.

## Failure Behavior

`fail_open=False` blocks when guard infrastructure fails. This is the recommended default for enforcement.

```python
WardenConfig(fail_open=False)
```

`fail_open=True` records the error in the trace and lets the model call continue or returns the already-created response when possible.

```python
WardenConfig(fail_open=True)
```

## Tracing

```python
client = GuardedOpenAI(
    OpenAI(),
    guard=guard,
    config=WardenConfig(return_guarded_result=True),
)

result = client.chat.completions.create(...)

print(result.response)
print(result.guard_trace.trace_id)
print(result.guard_trace.final_outcome)
```

OpenWarden does not log raw prompt or response content by default.

## RAG

Implement a retriever:

```python
from openwarden import RetrievedDocument

class MyRetriever:
    def retrieve(self, query):
        return [
            RetrievedDocument(
                id="policy-1",
                text="Employees receive 20 annual leave days.",
                score=0.91,
                source="hr-policy",
            )
        ]
```

Configure RAG:

```python
from openwarden import RAGConfig, RAGMode, WardenConfig

config = WardenConfig(
    rag=RAGConfig(
        provider=MyRetriever(),
        mode=RAGMode.AUGMENT_AND_VERIFY,
        top_k=5,
        minimum_score=0.70,
    )
)
```

Authorization belongs in your retriever or data layer. The guard model should not decide which documents a user is allowed to retrieve.

## Streaming

OpenWarden blocks streaming by default when output guarding is enabled:

```python
client.chat.completions.create(..., stream=True)
```

This raises `UnsupportedStreamingModeError` unless output guarding is disabled or the stream policy is changed.
