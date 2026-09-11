from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from openwarden import (
    BaseGuard,
    GuardDecision,
    GuardMode,
    GuardViolation,
    GuardedPyTorch,
    PyTorchPromptGuard,
    WardenConfig,
)


class FakeTensor:
    def __init__(self, values, *, device="cpu"):
        self.values = values
        self.device = device

    @property
    def shape(self):
        width = len(self.values[0]) if self.values else 0
        return (len(self.values), width)

    def tolist(self):
        return self.values

    def to(self, device):
        return FakeTensor(self.values, device=device)

    def __getitem__(self, index):
        return self.values[index]


class FakeTokenizer:
    decoded = {
        (1, 2): "safe prompt",
        (3, 4): "allowed output",
        (9,): '{"verdict":"allow","reason":"local guard allowed it"}',
    }

    def __init__(self):
        self.encoded_texts = []

    def __call__(self, text, *, return_tensors):
        assert return_tensors == "pt"
        self.encoded_texts.append(text)
        return {"input_ids": FakeTensor([[7, 8]]), "attention_mask": FakeTensor([[1, 1]])}

    def batch_decode(self, rows, *, skip_special_tokens):
        assert skip_special_tokens is True
        return [self.decoded.get(tuple(row), "unknown") for row in rows]

    def decode(self, row, *, skip_special_tokens):
        assert skip_special_tokens is True
        return self.decoded[tuple(row)]


class FakeModel:
    def __init__(self, output=None):
        self.config = SimpleNamespace(name_or_path="local/model", is_encoder_decoder=False)
        self.device = "cpu"
        self.output = output or FakeTensor([[1, 2, 3, 4]])
        self.calls = []
        self.dtype = "float16"

    def generate(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.output


class RecordingGuard(BaseGuard):
    def __init__(self, *, input_decision=None, output_decision=None):
        self.input_decision = input_decision or GuardDecision.allow()
        self.output_decision = output_decision or GuardDecision.allow()
        self.inputs = []
        self.outputs = []

    def check_input(self, context):
        self.inputs.append(context)
        return self.input_decision

    def check_output(self, context):
        self.outputs.append(context)
        return self.output_decision


def test_guarded_pytorch_guards_generate_and_preserves_native_output():
    model = FakeModel()
    guard = RecordingGuard()
    inference = GuardedPyTorch(model, tokenizer=FakeTokenizer(), guard=guard)
    input_ids = FakeTensor([[1, 2]])

    output = inference.generate(input_ids=input_ids, max_new_tokens=20)

    assert output is model.output
    assert model.calls == [((), {"input_ids": input_ids, "max_new_tokens": 20})]
    assert guard.inputs[0].request.endpoint == "pytorch.generate"
    assert guard.inputs[0].request.model == "local/model"
    assert guard.inputs[0].request.text() == "safe prompt"
    assert guard.outputs[0].response.text == "allowed output"
    assert inference.dtype == "float16"


def test_guarded_pytorch_blocks_before_running_model():
    model = FakeModel()
    guard = RecordingGuard(input_decision=GuardDecision.block("unsafe input"))
    inference = GuardedPyTorch(model, tokenizer=FakeTokenizer(), guard=guard)

    with pytest.raises(GuardViolation) as caught:
        inference.generate(input_ids=FakeTensor([[1, 2]]))

    assert caught.value.phase == "input"
    assert model.calls == []


def test_guarded_pytorch_supports_per_call_options_and_guarded_results():
    model = FakeModel()
    guard = RecordingGuard()
    inference = GuardedPyTorch(
        model,
        tokenizer=FakeTokenizer(),
        guard=guard,
        config=WardenConfig(return_guarded_result=True),
    )

    result = inference.generate(
        input_ids=FakeTensor([[1, 2]]),
        guard_options={"mode": GuardMode.INPUT},
    )

    assert result.response is model.output
    assert result.output_text == "allowed output"
    assert result.guard_trace.mode == GuardMode.INPUT
    assert "guard_options" not in model.calls[0][1]
    assert guard.outputs == []


def test_guarded_pytorch_retokenizes_an_explicit_input_modification():
    class ModifyingGuard(BaseGuard):
        def check_input(self, context):
            replacement = replace(context.request, instructions="Use approved context only.")
            return GuardDecision("modify", reason="add policy", replacement=replacement)

    model = FakeModel()
    tokenizer = FakeTokenizer()
    inference = GuardedPyTorch(
        model,
        tokenizer=tokenizer,
        guard=ModifyingGuard(),
        config=WardenConfig(mode=GuardMode.INPUT, allow_input_modification=True),
    )

    inference.generate(input_ids=FakeTensor([[1, 2]]))

    assert tokenizer.encoded_texts == ["Use approved context only.\nsafe prompt"]
    assert model.calls[0][1]["input_ids"].tolist() == [[7, 8]]


def test_pytorch_prompt_guard_uses_local_guard_model():
    tokenizer = FakeTokenizer()
    model = FakeModel(output=FakeTensor([[7, 8, 9]]))
    guard = PyTorchPromptGuard(
        model,
        tokenizer=tokenizer,
        generation_kwargs={"max_new_tokens": 12, "do_sample": False},
    )
    request = SimpleNamespace(
        request=SimpleNamespace(
            endpoint="pytorch.generate",
            model="main-model",
            instructions=None,
            text=lambda: "safe prompt",
            tools=(),
            tool_choice=None,
            metadata={},
        ),
        policy="company policy",
    )

    decision = guard.check_input(request)

    assert decision.verdict.value == "allow"
    assert decision.reason == "local guard allowed it"
    assert model.calls[0][1]["max_new_tokens"] == 12
    assert model.calls[0][1]["input_ids"].device == "cpu"


def test_guarded_pytorch_context_override_is_scoped():
    model = FakeModel()
    guard = RecordingGuard()
    inference = GuardedPyTorch(model, tokenizer=FakeTokenizer(), guard=guard)

    with inference.guard.override(mode=GuardMode.DISABLED):
        inference.generate(input_ids=FakeTensor([[1, 2]]))

    inference.generate(input_ids=FakeTensor([[1, 2]]))

    assert len(guard.inputs) == 1
    assert len(guard.outputs) == 1
