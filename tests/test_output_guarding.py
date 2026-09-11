from __future__ import annotations

import pytest

from openwarden import (
    GuardAction,
    GuardDecision,
    GuardMode,
    GuardViolation,
    GuardedOpenAI,
    UnsupportedStreamingModeError,
    WardenConfig,
)
from openwarden.models import GuardVerdict
from openwarden.result import GuardedResult
from tests._fakes import FakeClient, FakeResponse, RecordingGuard


def test_output_block_raises_and_redact_returns_guarded_result():
    blocked = GuardedOpenAI(
        FakeClient(FakeResponse("bad")),
        guard=RecordingGuard(output_decision=GuardDecision.block("bad output")),
        config=WardenConfig(mode=GuardMode.OUTPUT),
    )

    with pytest.raises(GuardViolation) as exc:
        blocked.responses.create(model="gpt-test", input="ok")

    assert exc.value.phase == "output"

    redacted_decision = GuardDecision(
        GuardVerdict.REDACT,
        reason="secret",
        replacement="[REDACTED]",
    )
    redacted = GuardedOpenAI(
        FakeClient(FakeResponse("secret")),
        guard=RecordingGuard(output_decision=redacted_decision),
        config=WardenConfig(mode=GuardMode.OUTPUT, output_action=GuardAction.REDACT),
    )

    result = redacted.responses.create(model="gpt-test", input="ok")

    assert isinstance(result, GuardedResult)
    assert result.output_text == "[REDACTED]"


def test_output_log_only_returns_original_response():
    raw = FakeResponse("bad output")
    client = FakeClient(raw)
    guard = RecordingGuard(output_decision=GuardDecision.block("blocked output"))
    guarded = GuardedOpenAI(
        client,
        guard=guard,
        config=WardenConfig(mode=GuardMode.OUTPUT, output_action=GuardAction.LOG_ONLY),
    )

    assert guarded.responses.create(model="gpt-test", input="ok") is raw


def test_streaming_with_output_guarding_is_disallowed_by_default():
    guarded = GuardedOpenAI(FakeClient(), guard=RecordingGuard(), config=WardenConfig(mode=GuardMode.BOTH))

    with pytest.raises(UnsupportedStreamingModeError):
        guarded.responses.create(model="gpt-test", input="Hi", stream=True)


def test_fail_open_output_guard_error_returns_existing_response_once():
    class BrokenOutputGuard(RecordingGuard):
        def check_output(self, context):
            raise RuntimeError("guard unavailable")

    raw = FakeResponse("already generated")
    client = FakeClient(raw)
    guarded = GuardedOpenAI(
        client,
        guard=BrokenOutputGuard(),
        config=WardenConfig(mode=GuardMode.OUTPUT, fail_open=True),
    )

    assert guarded.responses.create(model="gpt-test", input="Hi") is raw
    assert len(client.responses.calls) == 1
