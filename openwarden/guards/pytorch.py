from __future__ import annotations

from typing import Any

from openwarden.guards.base import ModelPromptGuard


class PyTorchPromptGuard(ModelPromptGuard):
    """Run guard-model prompting locally through a PyTorch ``generate`` model."""

    def __init__(
        self,
        model: Any,
        *,
        tokenizer: Any,
        model_name: str | None = None,
        system_prompt: str | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        device: Any | None = None,
        require_json: bool = False,
    ):
        resolved_name = (
            model_name
            or getattr(getattr(model, "config", None), "name_or_path", None)
            or "pytorch-guard"
        )
        super().__init__(
            None,
            model=resolved_name,
            system_prompt=system_prompt,
            require_json=require_json,
        )
        self._torch_model = model
        self._tokenizer = tokenizer
        self._generation_kwargs = (
            {"max_new_tokens": 256, "do_sample": False}
            if generation_kwargs is None
            else dict(generation_kwargs)
        )
        self._device = device

    def _create_guard_text(self, prompt: str) -> str | None:
        rendered = self._render_prompt(prompt)
        encoded = dict(self._tokenizer(rendered, return_tensors="pt"))
        device = self._device or getattr(self._torch_model, "device", None)
        if device is not None:
            encoded = {
                name: value.to(device) if hasattr(value, "to") else value
                for name, value in encoded.items()
            }
        input_ids = encoded.get("input_ids")
        prompt_length = _sequence_length(input_ids)
        output = self._torch_model.generate(**encoded, **self._generation_kwargs)
        sequences = getattr(output, "sequences", output)
        first = _first_sequence(sequences)
        if not bool(
            getattr(getattr(self._torch_model, "config", None), "is_encoder_decoder", False)
        ):
            first = first[prompt_length:]
        return self._tokenizer.decode(first, skip_special_tokens=True)

    def _render_prompt(self, prompt: str) -> str:
        apply_template = getattr(self._tokenizer, "apply_chat_template", None)
        if callable(apply_template):
            return apply_template(
                [
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"{self._system_prompt}\n\n{prompt}"


def _sequence_length(value: object) -> int:
    if value is None:
        return 0
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[-1])
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        return len(value[0])
    if isinstance(value, list):
        return len(value)
    return 0


def _first_sequence(value: object) -> object:
    if hasattr(value, "__getitem__"):
        return value[0]
    raise ValueError("guard model generate response did not include token sequences")
