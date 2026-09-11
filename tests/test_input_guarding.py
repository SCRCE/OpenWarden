from __future__ import annotations

import pytest

from openwarden import (
    GuardDecision,
    GuardEvaluationError,
    GuardMode,
    GuardViolation,
    GuardedOpenAI,
    TextContent,
    WardenConfig,
)
from openwarden.models import GuardRequest, GuardVerdict
from tests._fakes import FakeClient, FakeResponse, RecordingGuard


def test_allow_input_modification_applies_guard_request_replacement():
    replacement_request = GuardRequest(
        endpoint="responses.create",
        model="gpt-test",
        instructions=None,
        user_content=(TextContent("sanitized"),),
    )
    decision = GuardDecision(
        GuardVerdict.MODIFY,
        reason="sanitize",
        replacement=replacement_request,
    )
    client = FakeClient(FakeResponse("ok"))
    guarded = GuardedOpenAI(
        client,
        guard=RecordingGuard(input_decision=decision),
        config=WardenConfig(mode=GuardMode.INPUT, allow_input_modification=True),
    )

    guarded.responses.create(model="gpt-test", input="raw")

    assert client.responses.calls[0][1]["input"] == "sanitized"


def test_input_modification_without_permission_blocks():
    decision = GuardDecision(
        GuardVerdict.MODIFY,
        reason="sanitize",
        replacement=GuardRequest(endpoint="responses.create", model=None, instructions=None),
    )
    guarded = GuardedOpenAI(
        FakeClient(),
        guard=RecordingGuard(input_decision=decision),
        config=WardenConfig(mode=GuardMode.INPUT, allow_input_modification=False),
    )

    with pytest.raises(GuardViolation):
        guarded.responses.create(model="gpt-test", input="raw")


def test_input_block_raises_before_openai_call():
    client = FakeClient()
    guard = RecordingGuard(input_decision=GuardDecision.block("blocked", categories=("policy",)))
    guarded = GuardedOpenAI(client, guard=guard, config=WardenConfig(mode=GuardMode.INPUT))

    with pytest.raises(GuardViolation) as exc:
        guarded.responses.create(model="gpt-test", input="bad")

    assert exc.value.phase == "input"
    assert client.responses.calls == []


def test_fail_closed_guard_error_raises_evaluation_error_and_fail_open_continues():
    class BrokenInputGuard(RecordingGuard):
        def check_input(self, context):
            raise RuntimeError("guard crashed")

    strict = GuardedOpenAI(
        FakeClient(),
        guard=BrokenInputGuard(),
        config=WardenConfig(mode=GuardMode.INPUT, fail_open=False),
    )

    with pytest.raises(GuardEvaluationError):
        strict.responses.create(model="gpt-test", input="hello")

    permissive_client = FakeClient(FakeResponse("ok"))
    permissive = GuardedOpenAI(
        permissive_client,
        guard=BrokenInputGuard(),
        config=WardenConfig(mode=GuardMode.INPUT, fail_open=True),
    )

    assert permissive.responses.create(model="gpt-test", input="hello").output_text == "ok"
