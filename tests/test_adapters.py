from __future__ import annotations

from types import SimpleNamespace

from openwarden.adapters.chat_completions import ChatCompletionsAdapter
from openwarden.adapters.responses import ResponsesAdapter
from openwarden.models import TextContent
from openwarden import GuardMode, GuardedOpenAI, WardenConfig
from tests._fakes import FakeClient, FakeResponse, RecordingGuard


def test_responses_adapter_extracts_messages_tools_refusal_and_function_calls():
    adapter = ResponsesAdapter()
    request = adapter.extract_request(
        (),
        {
            "model": "gpt-test",
            "instructions": "Be brief",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "tool_choice": "auto",
            "metadata": {"trace": "1"},
        },
    )

    assert request.conversation[0].content == (TextContent("hello"),)
    assert request.tools[0].name == "lookup"

    response = SimpleNamespace(
        output_text="visible",
        output=[
            {"type": "function_call", "name": "lookup", "arguments": {"q": "x"}},
            {"content": [{"type": "refusal", "refusal": "no"}]},
        ],
        id="resp",
        model="gpt-test",
    )
    guarded_response = adapter.extract_response(response)

    assert guarded_response.text == "visible"
    assert guarded_response.tool_calls[0].name == "lookup"
    assert guarded_response.refusal == "no"


def test_chat_adapter_extracts_system_user_and_tool_calls():
    adapter = ChatCompletionsAdapter()
    request = adapter.extract_request(
        (),
        {
            "model": "gpt-test",
            "messages": [
                {"role": "system", "content": "Policy"},
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            ],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
        },
    )

    assert request.instructions == "Policy"
    assert request.user_content == (TextContent("hello"),)
    assert request.tools[0].name == "lookup"

    message = SimpleNamespace(
        content="answer",
        tool_calls=[{"function": {"name": "lookup", "arguments": "{}"}}],
    )
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)], id="chat", model="gpt-test")
    guarded_response = adapter.extract_response(response)

    assert guarded_response.text == "answer"
    assert guarded_response.tool_calls[0].name == "lookup"


def test_chat_completions_create_is_guarded():
    client = FakeClient(FakeResponse("chat ok"))
    guard = RecordingGuard()
    guarded = GuardedOpenAI(client, guard=guard, config=WardenConfig(mode=GuardMode.INPUT))

    guarded.chat.completions.create(
        model="gpt-test",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert guard.inputs[0].request.endpoint == "chat.completions.create"
    assert guard.inputs[0].request.user_content == (TextContent("hello"),)
