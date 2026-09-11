from __future__ import annotations

from types import SimpleNamespace

from openwarden import BaseGuard, GuardDecision, GroundingDecision


class FakeResponse:
    def __init__(self, output_text: str = "ok", *, output=None):
        self.output_text = output_text
        self.output = output or []
        self.id = "resp_1"
        self.model = "gpt-test"


class FakeResource:
    def __init__(self, response=None):
        self.response = response or FakeResponse()
        self.calls = []

    def create(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.response


class FakeClient:
    def __init__(self, response=None):
        self.responses = FakeResource(response)
        self.chat = SimpleNamespace(completions=FakeResource(response))


class RecordingGuard(BaseGuard):
    def __init__(self, input_decision=None, output_decision=None, grounding=None):
        self.input_decision = input_decision or GuardDecision.allow()
        self.output_decision = output_decision or GuardDecision.allow()
        self.grounding = grounding or GroundingDecision.not_evaluated()
        self.inputs = []
        self.outputs = []
        self.groundings = []

    def check_input(self, context):
        self.inputs.append(context)
        return self.input_decision

    def check_output(self, context):
        self.outputs.append(context)
        return self.output_decision

    def check_grounding(self, *, request, response, documents):
        self.groundings.append((request, response, documents))
        return self.grounding
