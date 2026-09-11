from __future__ import annotations

from types import SimpleNamespace

import pytest

from openwarden import AsyncGuardedOpenAI, AsyncModelPromptGuard, BaseGuard, GuardDecision, GuardMode, GroundingDecision, TextContent, WardenConfig
from openwarden.models import GuardRequest, GuardVerdict
from tests._fakes import FakeResource, FakeResponse


@pytest.mark.asyncio
async def test_async_guarded_responses():
    class AsyncResource(FakeResource):
        async def create(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return self.response

    class AsyncGuard(BaseGuard):
        def __init__(self):
            self.inputs = []

        async def check_input(self, context):
            self.inputs.append(context)
            return GuardDecision.allow()

        async def check_output(self, context):
            return GuardDecision.allow()

        async def check_grounding(self, *, request, response, documents):
            return GroundingDecision.not_evaluated()

    client = SimpleNamespace(responses=AsyncResource(FakeResponse("async ok")))
    guard = AsyncGuard()
    guarded = AsyncGuardedOpenAI(client, guard=guard, config=WardenConfig(mode=GuardMode.INPUT))

    response = await guarded.responses.create(model="gpt-test", input="hello")

    assert response.output_text == "async ok"
    assert guard.inputs[0].request.user_content == (TextContent("hello"),)


@pytest.mark.asyncio
async def test_async_model_prompt_guard():
    class AsyncResource(FakeResource):
        async def create(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return self.response

    model_client = SimpleNamespace(
        responses=AsyncResource(FakeResponse('{"verdict":"allow","reason":"ok"}'))
    )
    guard = AsyncModelPromptGuard(model_client, model="guard-model")
    request = GuardRequest(
        endpoint="responses.create",
        model="gpt-test",
        instructions=None,
        user_content=(TextContent("ok"),),
    )

    decision = await guard.check_input(SimpleNamespace(request=request, policy=None))

    assert decision.verdict == GuardVerdict.ALLOW
    assert model_client.responses.calls[0][1]["model"] == "guard-model"
