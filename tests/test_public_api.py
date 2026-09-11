from __future__ import annotations

from openwarden import GuardMode, GuardedOpenAI, WardenConfig
from openwarden.guards.regex import RegexSecretGuard
from tests._fakes import FakeClient, FakeResponse


def test_openwarden_public_api_example_imports_work():
    client = FakeClient(FakeResponse("decorators wrap functions"))
    guarded = GuardedOpenAI(
        client,
        guard=RegexSecretGuard(),
        config=WardenConfig(mode=GuardMode.BOTH, fail_open=False),
    )

    response = guarded.responses.create(
        model="generation-model",
        instructions="You are a concise and helpful assistant.",
        input="Explain Python decorators with a simple example.",
    )

    assert response.output_text == "decorators wrap functions"
