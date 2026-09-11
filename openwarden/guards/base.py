from __future__ import annotations

import json
import re
from typing import Any

from openwarden.models import (
    GroundingDecision,
    GuardDecision,
    GuardInputContext,
    GuardOutputContext,
    GuardRequest,
    GuardResponse,
    RetrievedDocument,
)


class BaseGuard:
    def check_input(self, context: GuardInputContext) -> GuardDecision:
        return GuardDecision.allow()

    def check_output(self, context: GuardOutputContext) -> GuardDecision:
        return GuardDecision.allow()

    def check_grounding(
        self,
        *,
        request: GuardRequest,
        response: GuardResponse,
        documents: list[RetrievedDocument],
    ) -> GroundingDecision:
        return GroundingDecision.not_evaluated()


class ModelPromptGuard(BaseGuard):
    """Guard that asks an OpenAI-compatible model for structured policy decisions."""

    def __init__(
        self,
        client: Any,
        *,
        model: str,
        system_prompt: str | None = None,
        temperature: float = 0,
        api: str = "auto",
        max_tokens: int = 256,
        chat_max_tokens_parameter: str = "max_completion_tokens",
        create_kwargs: dict[str, Any] | None = None,
        require_json: bool = False,
    ):
        self._client = client
        self._model = model
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
        self._temperature = temperature
        self._api = api
        self._max_tokens = max_tokens
        self._chat_max_tokens_parameter = chat_max_tokens_parameter
        self._create_kwargs = create_kwargs or {}
        self._require_json = require_json

    def check_input(self, context: GuardInputContext) -> GuardDecision:
        return self._classify("input", context.policy, _request_payload(context.request))

    def check_output(self, context: GuardOutputContext) -> GuardDecision:
        payload = {
            "request": _request_payload(context.effective_request),
            "response": _response_payload(context.response),
            "document_ids": [document.id for document in context.retrieved_documents],
        }
        return self._classify("output", context.policy, payload)

    def _classify(self, phase: str, policy: str | None, payload: dict[str, Any]) -> GuardDecision:
        prompt = {
            "phase": phase,
            "policy": policy,
            "payload": payload,
            "schema": {
                "verdict": ["allow", "block", "redact", "modify", "review"],
                "reason": "short human-readable reason",
                "categories": ["optional category strings"],
                "replacement": "optional replacement payload",
                "confidence": "optional number from 0 to 1",
            },
        }
        text = self._create_guard_text(json.dumps(prompt, sort_keys=True))
        if not isinstance(text, str):
            raise ValueError("guard model response did not include text")
        data = _parse_guard_decision_text(text, require_json=self._require_json)
        return GuardDecision(
            verdict=data["verdict"],
            reason=data.get("reason", ""),
            categories=tuple(data.get("categories") or ()),
            replacement=data.get("replacement"),
            confidence=data.get("confidence"),
            metadata={"guard_model": self._model, "phase": phase},
        )

    def _create_guard_text(self, prompt: str) -> str | None:
        if self._api == "responses":
            return self._create_responses_text(prompt)
        if self._api == "chat_completions":
            return self._create_chat_text(prompt)
        try:
            return self._create_responses_text(prompt)
        except Exception:
            if getattr(getattr(self._client, "chat", None), "completions", None) is None:
                raise
            return self._create_chat_text(prompt)

    def _create_responses_text(self, prompt: str) -> str | None:
        response = self._client.responses.create(
            model=self._model,
            instructions=self._system_prompt,
            input=prompt,
            temperature=self._temperature,
            max_output_tokens=self._max_tokens,
            **self._create_kwargs,
        )
        return getattr(response, "output_text", None)

    def _create_chat_text(self, prompt: str) -> str | None:
        kwargs = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": self._temperature,
            self._chat_max_tokens_parameter: self._max_tokens,
            **self._create_kwargs,
        }
        response = self._client.chat.completions.create(**kwargs)
        choices = getattr(response, "choices", None) or []
        if not choices:
            return None
        message = choices[0].get("message") if isinstance(choices[0], dict) else getattr(choices[0], "message", None)
        if isinstance(message, dict):
            return message.get("content") or message.get("reasoning")
        return getattr(message, "content", None) or getattr(message, "reasoning", None)


class AsyncModelPromptGuard(BaseGuard):
    """Async version of ModelPromptGuard for AsyncOpenAI-compatible clients."""

    def __init__(
        self,
        client: Any,
        *,
        model: str,
        system_prompt: str | None = None,
        temperature: float = 0,
        api: str = "auto",
        max_tokens: int = 256,
        chat_max_tokens_parameter: str = "max_completion_tokens",
        create_kwargs: dict[str, Any] | None = None,
        require_json: bool = False,
    ):
        self._client = client
        self._model = model
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
        self._temperature = temperature
        self._api = api
        self._max_tokens = max_tokens
        self._chat_max_tokens_parameter = chat_max_tokens_parameter
        self._create_kwargs = create_kwargs or {}
        self._require_json = require_json

    async def check_input(self, context: GuardInputContext) -> GuardDecision:
        return await self._classify("input", context.policy, _request_payload(context.request))

    async def check_output(self, context: GuardOutputContext) -> GuardDecision:
        payload = {
            "request": _request_payload(context.effective_request),
            "response": _response_payload(context.response),
            "document_ids": [document.id for document in context.retrieved_documents],
        }
        return await self._classify("output", context.policy, payload)

    async def check_grounding(
        self,
        *,
        request: GuardRequest,
        response: GuardResponse,
        documents: list[RetrievedDocument],
    ) -> GroundingDecision:
        return GroundingDecision.not_evaluated()

    async def _classify(self, phase: str, policy: str | None, payload: dict[str, Any]) -> GuardDecision:
        prompt = {
            "phase": phase,
            "policy": policy,
            "payload": payload,
            "schema": {
                "verdict": ["allow", "block", "redact", "modify", "review"],
                "reason": "short human-readable reason",
                "categories": ["optional category strings"],
                "replacement": "optional replacement payload",
                "confidence": "optional number from 0 to 1",
            },
        }
        text = await self._create_guard_text(json.dumps(prompt, sort_keys=True))
        if not isinstance(text, str):
            raise ValueError("guard model response did not include text")
        data = _parse_guard_decision_text(text, require_json=self._require_json)
        return GuardDecision(
            verdict=data["verdict"],
            reason=data.get("reason", ""),
            categories=tuple(data.get("categories") or ()),
            replacement=data.get("replacement"),
            confidence=data.get("confidence"),
            metadata={"guard_model": self._model, "phase": phase},
        )

    async def _create_guard_text(self, prompt: str) -> str | None:
        if self._api == "responses":
            return await self._create_responses_text(prompt)
        if self._api == "chat_completions":
            return await self._create_chat_text(prompt)
        try:
            return await self._create_responses_text(prompt)
        except Exception:
            if getattr(getattr(self._client, "chat", None), "completions", None) is None:
                raise
            return await self._create_chat_text(prompt)

    async def _create_responses_text(self, prompt: str) -> str | None:
        response = await self._client.responses.create(
            model=self._model,
            instructions=self._system_prompt,
            input=prompt,
            temperature=self._temperature,
            max_output_tokens=self._max_tokens,
            **self._create_kwargs,
        )
        return getattr(response, "output_text", None)

    async def _create_chat_text(self, prompt: str) -> str | None:
        kwargs = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": self._temperature,
            self._chat_max_tokens_parameter: self._max_tokens,
            **self._create_kwargs,
        }
        response = await self._client.chat.completions.create(**kwargs)
        choices = getattr(response, "choices", None) or []
        if not choices:
            return None
        message = choices[0].get("message") if isinstance(choices[0], dict) else getattr(choices[0], "message", None)
        if isinstance(message, dict):
            return message.get("content") or message.get("reasoning")
        return getattr(message, "content", None) or getattr(message, "reasoning", None)


def _request_payload(request: GuardRequest) -> dict[str, Any]:
    return {
        "endpoint": request.endpoint,
        "model": request.model,
        "instructions": request.instructions,
        "text": request.text(),
        "tool_names": [tool.name for tool in request.tools],
        "tool_choice": request.tool_choice,
        "metadata": request.metadata,
    }


def _response_payload(response: GuardResponse) -> dict[str, Any]:
    return {
        "text": response.text,
        "tool_calls": [{"name": call.name, "arguments": call.arguments} for call in response.tool_calls],
        "refusal": response.refusal,
        "metadata": response.metadata,
    }


_DEFAULT_SYSTEM_PROMPT = (
    "You are a policy guard. Return only JSON matching the requested schema. "
    "Classify whether the payload complies with the policy. Use block for clear violations, "
    "review for uncertain cases, and allow for compliant content."
)


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)
    bracketed = re.search(r"(\{.*\})", stripped, flags=re.DOTALL)
    if bracketed:
        return bracketed.group(1)
    return stripped


def _parse_guard_decision_text(text: str, *, require_json: bool) -> dict[str, Any]:
    try:
        return json.loads(_extract_json_object(text))
    except json.JSONDecodeError:
        if require_json:
            raise
    lowered = " ".join(text.lower().split())
    allow_phrases = (
        "does not violate",
        "doesn't violate",
        "do not violate",
        "not violate",
        "no safety policy violation",
        "not a safety policy violation",
        "legitimate",
        "permissible",
        "allowed",
        "benign",
    )
    block_phrases = (
        "violates",
        "violate a safety policy",
        "not allowed",
        "disallowed",
        "malicious",
        "unsafe",
        "harmful",
    )
    if any(phrase in lowered for phrase in allow_phrases):
        verdict = "allow"
    elif any(phrase in lowered for phrase in block_phrases):
        verdict = "block"
    else:
        verdict = "review"
    return {
        "verdict": verdict,
        "reason": text.strip(),
        "categories": (),
        "confidence": None,
        "replacement": None,
    }
