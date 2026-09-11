# API Reference

This is a concise reference for the public API exported from `openwarden`.

## Clients

### `GuardedOpenAI(client, *, guard, config=None)`

Synchronous wrapper around a real OpenAI-compatible client.

Guarded resources:

- `client.responses.create(...)`
- `client.chat.completions.create(...)`

Unsupported resources pass through to the original client.

### `AsyncGuardedOpenAI(client, *, guard, config=None)`

Asynchronous wrapper around an `AsyncOpenAI`-compatible client.

### `GuardedPyTorch(model, *, tokenizer, guard, config=None, model_name=None)`

Synchronous composition wrapper around a local model's `generate(...)` method.
It guards decoded input and generated completion text while preserving the native
PyTorch or Transformers return value. Unknown attributes pass through to the
wrapped model.

## Configuration

### `WardenConfig`

```python
WardenConfig(
    mode=GuardMode.BOTH,
    input_action=GuardAction.BLOCK,
    output_action=GuardAction.BLOCK,
    input_policy=None,
    output_policy=None,
    fail_open=False,
    timeout_seconds=10.0,
    max_rewrites=1,
    allow_input_modification=False,
    return_guarded_result=False,
    rag=None,
)
```

`GuardConfig` is also available as a compatibility name.

### `GuardMode`

- `DISABLED`
- `INPUT`
- `OUTPUT`
- `BOTH`

### `GuardAction`

- `BLOCK`
- `REDACT`
- `REWRITE`
- `ANNOTATE`
- `LOG_ONLY`

## Guard Interface

```python
class Guard:
    def check_input(self, context): ...
    def check_output(self, context): ...
    def check_grounding(self, *, request, response, documents): ...
```

Most users should subclass `BaseGuard`.

## Built-In Guards

### `BaseGuard`

Allows all input, output, and grounding checks by default.

### `RegexSecretGuard`

Blocks text matching secret-like regex patterns.

### `GuardPipeline`

Runs guards in order and returns the first non-allow decision.

### `ModelPromptGuard`

Uses an OpenAI-compatible model to classify input and output.

Important constructor arguments:

- `client`: OpenAI-compatible client.
- `model`: guard model name.
- `api`: `"auto"`, `"responses"`, or `"chat_completions"`.
- `temperature`: guard model temperature.
- `max_tokens`: output budget for guard decision.
- `chat_max_tokens_parameter`: token parameter name for chat guard calls.
- `create_kwargs`: extra provider-specific parameters.
- `require_json`: if true, non-JSON guard output raises.

### `PyTorchPromptGuard`

Uses a local PyTorch-compatible generation model for guard decisions.

- `model`: object with a `generate(...)` method.
- `tokenizer`: callable tokenizer with `decode(...)` support.
- `model_name`: optional name recorded in decision metadata.
- `system_prompt`: optional guard policy prompt.
- `generation_kwargs`: arguments passed to the guard model's `generate(...)`.
- `device`: optional explicit input device; defaults to `model.device`.
- `require_json`: if true, non-JSON guard output raises.

## Models

### `GuardRequest`

Canonical request shape passed to guards.

Fields:

- `endpoint`
- `model`
- `instructions`
- `user_content`
- `conversation`
- `tools`
- `tool_choice`
- `metadata`

### `GuardResponse`

Canonical response shape passed to output guards.

Fields:

- `text`
- `tool_calls`
- `structured_output`
- `refusal`
- `raw_response`
- `metadata`

### `GuardDecision`

```python
GuardDecision(
    verdict=GuardVerdict.ALLOW,
    reason="allowed",
    categories=(),
    replacement=None,
    confidence=None,
    metadata={},
)
```

Helpers:

- `GuardDecision.allow()`
- `GuardDecision.block(reason, categories=...)`

### `GuardVerdict`

- `ALLOW`
- `BLOCK`
- `REDACT`
- `MODIFY`
- `REVIEW`

## Exceptions

- `GuardError`
- `GuardViolation`
- `GuardEvaluationError`
- `RetrievalError`
- `GroundingViolation`
- `UnsupportedEndpointError`
- `UnsupportedStreamingModeError`

## RAG

### `RAGConfig`

```python
RAGConfig(
    provider=my_retriever,
    mode=RAGMode.AUGMENT_AND_VERIFY,
    top_k=5,
    minimum_score=0.70,
    maximum_context_characters=30000,
    require_results=False,
)
```

### `RAGMode`

- `DISABLED`
- `AUGMENT`
- `VERIFY_ONLY`
- `AUGMENT_AND_VERIFY`

### `RetrievedDocument`

```python
RetrievedDocument(
    id="policy-17",
    text="...",
    score=0.92,
    source="...",
    metadata={},
)
```

## Results and Traces

When `return_guarded_result=True`, guarded calls return:

```python
GuardedResult(
    response=original_sdk_response,
    guard_trace=trace,
    output_text=extracted_text,
)
```

`GuardTrace` includes endpoint, model, decisions, retrieval counts, latencies, and final outcome.
