from __future__ import annotations

from openwarden import GuardMode, GuardedOpenAI, WardenConfig, get_current_guard_trace
from openwarden.models import GuardVerdict
from openwarden.result import GuardedResult
from tests._fakes import FakeClient, FakeResponse, RecordingGuard


def test_return_guarded_result_exposes_trace_and_current_trace():
    client = FakeClient(FakeResponse("ok"))
    guarded = GuardedOpenAI(
        client,
        guard=RecordingGuard(),
        config=WardenConfig(mode=GuardMode.BOTH, return_guarded_result=True),
    )

    result = guarded.responses.create(model="gpt-test", input="hello")

    assert isinstance(result, GuardedResult)
    assert result.response is client.responses.response
    assert result.guard_trace.trace_id
    assert result.guard_trace.input_decision.verdict is GuardVerdict.ALLOW
    assert get_current_guard_trace() is result.guard_trace
