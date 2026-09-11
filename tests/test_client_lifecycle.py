from __future__ import annotations

from openwarden import (
    CallGuardConfig,
    GuardMode,
    GuardedOpenAI,
    RAGConfig,
    TextContent,
    WardenConfig,
)
from tests._fakes import FakeClient, FakeResponse, RecordingGuard
from openwarden import GuardDecision


def test_disabled_mode_skips_guards_and_still_calls_model():
    client = FakeClient(FakeResponse("ok"))
    guard = RecordingGuard(
        input_decision=GuardDecision.block("would block"),
        output_decision=GuardDecision.block("would block"),
    )
    guarded = GuardedOpenAI(client, guard=guard, config=WardenConfig(mode=GuardMode.DISABLED))

    response = guarded.responses.create(model="gpt-test", input="blocked if checked")

    assert response.output_text == "ok"
    assert guard.inputs == []
    assert guard.outputs == []
    assert len(client.responses.calls) == 1


def test_allowed_response_preserves_original_sdk_object_and_passes_through_attributes():
    raw = FakeResponse("hello")
    client = FakeClient(raw)
    client.files = object()
    client.responses.extra_attr = "passthrough"
    guard = RecordingGuard()

    guarded = GuardedOpenAI(client, guard=guard, config=WardenConfig(mode=GuardMode.BOTH))
    response = guarded.responses.create(model="gpt-test", instructions="Be brief", input="Hi")

    assert response is raw
    assert client.responses.calls[0][1]["input"] == "Hi"
    assert guard.inputs[0].request.user_content == (TextContent("Hi"),)
    assert guard.outputs[0].response.text == "hello"
    assert guarded.files is client.files
    assert guarded.responses.extra_attr == "passthrough"


def test_context_manager_override_temporarily_changes_mode():
    client = FakeClient(FakeResponse("ok"))
    guard = RecordingGuard()
    guarded = GuardedOpenAI(client, guard=guard, config=WardenConfig(mode=GuardMode.BOTH))

    with guarded.guard.override(mode=GuardMode.INPUT):
        guarded.responses.create(model="gpt-test", input="inside")

    guarded.responses.create(model="gpt-test", input="outside")

    assert len(guard.inputs) == 2
    assert len(guard.outputs) == 1


def test_call_guard_options_can_disable_rag():
    class Retriever:
        def retrieve(self, query):
            raise AssertionError("retriever should be disabled")

    client = FakeClient(FakeResponse("ok"))
    guarded = GuardedOpenAI(
        client,
        guard=RecordingGuard(),
        config=WardenConfig(rag=RAGConfig(provider=Retriever())),
    )

    guarded.responses.create(
        model="gpt-test",
        input="hello",
        guard_options=CallGuardConfig(rag=False),
    )

    assert len(client.responses.calls) == 1
