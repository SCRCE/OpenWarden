from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from openwarden.models import ContentPart, Message, TextContent, ToolCall, ToolDefinition


def content_parts(value: object) -> tuple[ContentPart, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (TextContent(value),)
    if isinstance(value, list):
        parts: list[ContentPart] = []
        for item in value:
            parts.extend(content_parts(item))
        return tuple(parts)
    if isinstance(value, dict):
        item_type = value.get("type")
        if item_type in {"input_text", "output_text", "text"} and isinstance(value.get("text"), str):
            return (TextContent(value["text"]),)
        if "content" in value:
            return content_parts(value["content"])
    text = getattr(value, "text", None)
    if isinstance(text, str):
        return (TextContent(text),)
    return ()


def messages_from_raw(raw_messages: object) -> tuple[Message, ...]:
    if not isinstance(raw_messages, list):
        return ()
    messages: list[Message] = []
    for item in raw_messages:
        if isinstance(item, dict):
            messages.append(
                Message(
                    role=str(item.get("role", "")),
                    content=content_parts(item.get("content")),
                    raw=item,
                )
            )
    return tuple(messages)


def tool_definitions(raw_tools: object) -> tuple[ToolDefinition, ...]:
    if not isinstance(raw_tools, list):
        return ()
    tools = []
    for item in raw_tools:
        name = None
        if isinstance(item, dict):
            function = item.get("function")
            if isinstance(function, dict):
                name = function.get("name")
            name = name or item.get("name")
        tools.append(ToolDefinition(name=name, raw=item))
    return tuple(tools)


def extract_tool_calls_from_items(items: Iterable[object]) -> tuple[ToolCall, ...]:
    calls: list[ToolCall] = []
    for item in items:
        item_type = _get(item, "type")
        if item_type not in {"function_call", "tool_call"} and not _get(item, "function"):
            continue
        function = _get(item, "function")
        name = _get(function, "name") if function is not None else _get(item, "name")
        arguments = _get(function, "arguments") if function is not None else _get(item, "arguments")
        calls.append(ToolCall(name=name, arguments=arguments, raw=item))
    return tuple(calls)


def replace_user_text(request: Any, text: str) -> Any:
    parts = tuple(TextContent(text) if isinstance(part, TextContent) else part for part in request.user_content)
    return replace(request, user_content=parts)


def _get(value: object, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)
