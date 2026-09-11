from __future__ import annotations

from types import SimpleNamespace

from openwarden import BaseGuard, GuardDecision, GroundingDecision, ModelPromptGuard, TextContent
from openwarden.guards.pipeline import GuardPipeline
from openwarden.guards.regex import RegexSecretGuard
from openwarden.models import GuardRequest, GuardResponse, GuardVerdict
from tests._fakes import FakeClient, FakeResponse, RecordingGuard


def test_guard_pipeline_first_non_allow_and_grounding_min_score():
    allow = RecordingGuard()
    block = RecordingGuard(input_decision=GuardDecision.block("blocked"))
    pipeline = GuardPipeline([allow, block, RecordingGuard()])

    assert pipeline.check_input(SimpleNamespace()).verdict is GuardVerdict.BLOCK

    class GroundingGuard(BaseGuard):
        def __init__(self, score):
            self.score = score

        def check_grounding(self, *, request, response, documents):
            return GroundingDecision(True, self.score, reason=str(self.score))

    grounding = GuardPipeline([GroundingGuard(0.9), GroundingGuard(0.4)]).check_grounding(
        request=GuardRequest(endpoint="responses.create", model=None, instructions=None),
        response=GuardResponse(text="ok"),
        documents=[],
    )

    assert grounding.score == 0.4


def test_regex_secret_guard_checks_input_and_output():
    guard = RegexSecretGuard()
    fake_secret = "sk-" + "abc123456789"
    request = GuardRequest(
        endpoint="responses.create",
        model=None,
        instructions=None,
        user_content=(TextContent(f"token {fake_secret}"),),
    )

    input_decision = guard.check_input(SimpleNamespace(request=request))
    output_decision = guard.check_output(SimpleNamespace(response=GuardResponse(text=fake_secret)))

    assert input_decision.verdict is GuardVerdict.BLOCK
    assert output_decision.verdict is GuardVerdict.BLOCK


def test_model_prompt_guard_uses_openai_compatible_responses_client():
    model_client = FakeClient(
        FakeResponse('{"verdict":"block","reason":"policy violation","categories":["test"],"confidence":0.8}')
    )
    guard = ModelPromptGuard(model_client, model="guard-model")
    request = GuardRequest(
        endpoint="responses.create",
        model="gpt-test",
        instructions=None,
        user_content=(TextContent("bad"),),
    )

    decision = guard.check_input(SimpleNamespace(request=request, policy="test policy"))

    assert decision.verdict == GuardVerdict.BLOCK
    assert decision.reason == "policy violation"
    call = model_client.responses.calls[0][1]
    assert call["model"] == "guard-model"
    assert call["instructions"]


def test_model_prompt_guard_can_use_chat_completions():
    class ChatCompletions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            message = SimpleNamespace(content='{"verdict":"allow","reason":"ok"}')
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    model_client = SimpleNamespace(chat=SimpleNamespace(completions=ChatCompletions()))
    guard = ModelPromptGuard(model_client, model="guard-model", api="chat_completions")
    request = GuardRequest(
        endpoint="responses.create",
        model="gpt-test",
        instructions=None,
        user_content=(TextContent("safe"),),
    )

    decision = guard.check_input(SimpleNamespace(request=request, policy=None))

    assert decision.verdict == GuardVerdict.ALLOW
    assert model_client.chat.completions.calls[0]["model"] == "guard-model"
    assert model_client.chat.completions.calls[0]["max_completion_tokens"] == 256


def test_model_prompt_guard_parses_safeguard_prose_and_reasoning_field():
    class ChatCompletions:
        def create(self, **kwargs):
            message = SimpleNamespace(
                content=None,
                reasoning="The request does not violate any safety policy. It is legitimate.",
            )
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    model_client = SimpleNamespace(chat=SimpleNamespace(completions=ChatCompletions()))
    guard = ModelPromptGuard(model_client, model="guard-model", api="chat_completions")
    request = GuardRequest(
        endpoint="chat.completions.create",
        model="gpt-test",
        instructions=None,
        user_content=(TextContent("How do I reset my own Linux password?"),),
    )

    decision = guard.check_input(SimpleNamespace(request=request, policy=None))

    assert decision.verdict == GuardVerdict.ALLOW
