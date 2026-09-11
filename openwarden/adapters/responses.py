from __future__ import annotations

from openwarden.adapters.base import content_parts, extract_tool_calls_from_items, messages_from_raw, tool_definitions
from openwarden.models import GuardRequest, GuardResponse, Message, TextContent


class ResponsesAdapter:
    endpoint = "responses.create"

    def extract_request(self, args: tuple[object, ...], kwargs: dict[str, object]) -> GuardRequest:
        request_input = kwargs.get("input")
        conversation = messages_from_raw(request_input)
        user_content = content_parts(request_input) if not conversation else ()
        metadata = dict(kwargs.get("metadata") or {})
        return GuardRequest(
            endpoint=self.endpoint,
            model=_as_optional_str(kwargs.get("model")),
            instructions=_as_optional_str(kwargs.get("instructions")),
            user_content=user_content,
            conversation=conversation,
            tools=tool_definitions(kwargs.get("tools")),
            tool_choice=kwargs.get("tool_choice"),
            metadata=metadata,
        )

    def apply_request(self, original_kwargs: dict[str, object], request: GuardRequest) -> dict[str, object]:
        kwargs = dict(original_kwargs)
        kwargs.pop("guard_options", None)
        if request.model is not None:
            kwargs["model"] = request.model
        kwargs["instructions"] = request.instructions
        if request.conversation:
            kwargs["input"] = [_message_to_responses_dict(message) for message in request.conversation]
        elif request.user_content:
            texts = [part.text for part in request.user_content if isinstance(part, TextContent)]
            kwargs["input"] = "\n".join(texts)
        kwargs["tools"] = [tool.raw for tool in request.tools] if request.tools else kwargs.get("tools")
        kwargs["tool_choice"] = request.tool_choice if request.tool_choice is not None else kwargs.get("tool_choice")
        kwargs["metadata"] = request.metadata or kwargs.get("metadata")
        return {key: value for key, value in kwargs.items() if value is not None}

    def extract_response(self, response: object) -> GuardResponse:
        output_text = getattr(response, "output_text", None)
        output = getattr(response, "output", None) or []
        refusal = _extract_refusal(output)
        structured = getattr(response, "output_parsed", None)
        metadata = {
            "id": getattr(response, "id", None),
            "model": getattr(response, "model", None),
        }
        return GuardResponse(
            text=output_text if isinstance(output_text, str) else None,
            tool_calls=extract_tool_calls_from_items(output),
            structured_output=structured,
            refusal=refusal,
            raw_response=response,
            metadata={key: value for key, value in metadata.items() if value is not None},
        )


def _message_to_responses_dict(message: Message) -> dict[str, object]:
    texts = [part.text for part in message.content if isinstance(part, TextContent)]
    return {"role": message.role, "content": "\n".join(texts)}


def _extract_refusal(output: object) -> str | None:
    if not isinstance(output, list):
        return None
    for item in output:
        content = getattr(item, "content", None)
        if isinstance(item, dict):
            content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            part_type = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
            if part_type == "refusal":
                text = part.get("refusal") if isinstance(part, dict) else getattr(part, "refusal", None)
                return text if isinstance(text, str) else None
    return None


def _as_optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None
