from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from openwarden.adapters.pytorch import PyTorchGenerateAdapter
from openwarden.config import GuardConfig
from openwarden.resources.responses import GuardedResponses


class _GenerateResource:
    def __init__(self, model: Any):
        self._model = model

    def create(self, *args: object, **kwargs: object) -> object:
        return self._model.generate(*args, **kwargs)


class PyTorchGuardController:
    def __init__(self, inference: "GuardedPyTorch"):
        self._inference = inference

    @contextmanager
    def override(self, **kwargs: object) -> Iterator[None]:
        previous = self._inference._config
        self._inference._config = previous.with_overrides(kwargs)
        self._inference._rebuild_resource()
        try:
            yield
        finally:
            self._inference._config = previous
            self._inference._rebuild_resource()


class GuardedPyTorch:
    """Composition wrapper that guards text generation through ``model.generate``."""

    def __init__(
        self,
        model: Any,
        *,
        tokenizer: Any,
        guard: Any,
        config: GuardConfig | None = None,
        model_name: str | None = None,
    ):
        if not callable(getattr(model, "generate", None)):
            raise TypeError("model must provide a callable generate method")
        if not callable(getattr(tokenizer, "batch_decode", None)):
            raise TypeError("tokenizer must provide batch_decode")
        self._model = model
        self._guard = guard
        self._config = config or GuardConfig()
        self._adapter = PyTorchGenerateAdapter(model, tokenizer, model_name=model_name)
        self.guard = PyTorchGuardController(self)
        self._rebuild_resource()

    def _rebuild_resource(self) -> None:
        self._generate = GuardedResponses(
            _GenerateResource(self._model),
            guard=self._guard,
            config=self._config,
            adapter=self._adapter,
        )

    def generate(self, *args: object, **kwargs: object) -> object:
        return self._generate.create(*args, **kwargs)

    def __getattr__(self, name: str) -> object:
        return getattr(self._model, name)
