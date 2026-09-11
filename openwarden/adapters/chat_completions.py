from __future__ import annotations

from typing import Any

from openwarden.adapters.base import extract_tool_calls_from_items, messages_from_raw, tool_definitions
from openwarden.models import GuardRequest, GuardResponse, Message, TextContent


class ChatCompletionsAdapter:
    endpoint = "chat.completions.create"

    def extract_request(self, args: tuple[object, ...], kwargs: dict[str, object]) -> GuardRequest:
        messages = messages_from_raw(kwargs.get("messages"))
        instructions = "\n".join(
            part.text
            for message in messages
            if message.role in {"system", "developer"}
            for part in message.content
            if isinstance(part, TextContent)
        ) or None
        user_content = tuple(
            part
            for message in messages
            if message.role == "user"
            for part in message.content
        )
        return GuardRequest(
            endpoint=self.endpoint,
            model=kwargs.get("model") if isinstance(kwargs.get("model"), str) else None,
            instructions=instructions,
            user_content=user_content,
            conversation=messages,
            tools=tool_definitions(kwargs.get("tools")),
            tool_choice=kwargs.get("tool_choice"),
            metadata=dict(kwargs.get("metadata") or {}),
        )

    def apply_request(self, original_kwargs: dict[str, object], request: GuardRequest) -> dict[str, object]:
        kwargs = dict(original_kwargs)
        kwargs.pop("guard_options", None)
        if request.model is not None:
            kwargs["model"] = request.model
        if request.conversation:
            kwargs["messages"] = [_message_to_chat_dict(message) for message in request.conversation]
        elif request.user_content:
            text = "\n".join(part.text for part in request.user_content if isinstance(part, TextContent))
            kwargs["messages"] = [{"role": "user", "content": text}]
        kwargs["tools"] = [tool.raw for tool in request.tools] if request.tools else kwargs.get("tools")
        kwargs["tool_choice"] = request.tool_choice if request.tool_choice is not None else kwargs.get("tool_choice")
        return {key: value for key, value in kwargs.items() if value is not None}

    def extract_response(self, response: object) -> GuardResponse:
        choices = getattr(response, "choices", None) or []
        text = None
        tool_calls = ()
        refusal = None
        if choices:
            message = getattr(choices[0], "message", None)
            if isinstance(choices[0], dict):
                message = choices[0].get("message")
            text = _get(message, "content")
            refusal = _get(message, "refusal")
            raw_tool_calls = _get(message, "tool_calls") or []
            tool_calls = extract_tool_calls_from_items(raw_tool_calls)
        return GuardResponse(
            text=text if isinstance(text, str) else None,
            tool_calls=tool_calls,
            refusal=refusal if isinstance(refusal, str) else None,
            raw_response=response,
            metadata={
                key: value
                for key, value in {
                    "id": getattr(response, "id", None),
                    "model": getattr(response, "model", None),
                }.items()
                if value is not None
            },
        )


def _message_to_chat_dict(message: Message) -> dict[str, object]:
    texts = [part.text for part in message.content if isinstance(part, TextContent)]
    return {"role": message.role, "content": "\n".join(texts)}


def _get(value: object, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)
