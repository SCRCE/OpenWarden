from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from openwarden.exceptions import GuardEvaluationError
from openwarden.models import GuardRequest, GuardResponse, TextContent


@dataclass(frozen=True)
class _GenerationState:
    input_lengths: tuple[int, ...]
    input_texts: tuple[str, ...]
    canonical_text: str
    input_location: str


class PyTorchGenerateAdapter:
    """Translate ``model.generate`` inputs and outputs to OpenWarden models."""

    endpoint = "pytorch.generate"

    def __init__(self, model: Any, tokenizer: Any, *, model_name: str | None = None):
        self._model = model
        self._tokenizer = tokenizer
        self._model_name = model_name or _model_name(model)
        self._state: ContextVar[_GenerationState | None] = ContextVar(
            "openwarden_pytorch_generation_state",
            default=None,
        )

    def extract_request(self, args: tuple[object, ...], kwargs: dict[str, object]) -> GuardRequest:
        input_ids, location = _find_input_ids(args, kwargs)
        rows = _rows(input_ids)
        texts = tuple(self._tokenizer.batch_decode(rows, skip_special_tokens=True))
        self._state.set(
            _GenerationState(
                input_lengths=tuple(len(row) for row in rows),
                input_texts=texts,
                canonical_text="\n".join(texts),
                input_location=location,
            )
        )
        return GuardRequest(
            endpoint=self.endpoint,
            model=self._model_name,
            instructions=None,
            user_content=tuple(TextContent(text) for text in texts),
            metadata={"batch_size": len(texts)},
        )

    def apply_request(self, original_kwargs: dict[str, object], request: GuardRequest) -> dict[str, object]:
        state = self._required_state()
        current_texts = tuple(part.text for part in request.user_content if isinstance(part, TextContent))
        effective_text = request.text()
        if effective_text == state.canonical_text:
            return dict(original_kwargs)
        if state.input_location != "keyword":
            raise GuardEvaluationError(
                "modified PyTorch input requires input_ids to be passed as a keyword argument"
            )
        if len(current_texts) != 1:
            raise GuardEvaluationError("modified batched PyTorch input is not supported")

        encoded = self._tokenizer(effective_text, return_tensors="pt")
        final_kwargs = dict(original_kwargs)
        original_ids = original_kwargs.get("input_ids")
        device = getattr(original_ids, "device", None)
        for name, value in dict(encoded).items():
            if device is not None and hasattr(value, "to"):
                value = value.to(device)
            final_kwargs[name] = value
        return final_kwargs

    def extract_response(self, response: object) -> GuardResponse:
        state = self._required_state()
        sequences = getattr(response, "sequences", response)
        rows = _rows(sequences)
        is_encoder_decoder = bool(
            getattr(getattr(self._model, "config", None), "is_encoder_decoder", False)
        )
        completion_rows = []
        for index, row in enumerate(rows):
            prompt_length = 0 if is_encoder_decoder else state.input_lengths[min(index, len(state.input_lengths) - 1)]
            completion_rows.append(row[prompt_length:])
        texts = tuple(self._tokenizer.batch_decode(completion_rows, skip_special_tokens=True))
        return GuardResponse(
            text="\n".join(texts),
            structured_output=texts,
            raw_response=response,
            metadata={"batch_size": len(texts), "model": self._model_name},
        )

    def _required_state(self) -> _GenerationState:
        state = self._state.get()
        if state is None:
            raise GuardEvaluationError("PyTorch generation state is unavailable")
        return state


def _find_input_ids(args: tuple[object, ...], kwargs: dict[str, object]) -> tuple[object, str]:
    if kwargs.get("input_ids") is not None:
        return kwargs["input_ids"], "keyword"
    if args:
        return args[0], "positional"
    raise GuardEvaluationError("model.generate requires input_ids for input guarding")


def _rows(value: object) -> list[list[Any]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise GuardEvaluationError("token IDs must be a tensor or a list")
    if not value:
        return []
    if not isinstance(value[0], (list, tuple)):
        return [list(value)]
    return [list(row) for row in value]


def _model_name(model: Any) -> str | None:
    config = getattr(model, "config", None)
    return getattr(config, "name_or_path", None) or getattr(model, "name_or_path", None)
