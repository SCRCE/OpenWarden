from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from openwarden.config import GuardConfig
from openwarden.resources.chat_completions import GuardedChatCompletions
from openwarden.resources.responses import GuardedResponses


class GuardController:
    def __init__(self, client: "GuardedOpenAI"):
        self._client = client

    @contextmanager
    def override(self, **kwargs: object) -> Iterator[None]:
        previous = self._client._config
        self._client._config = previous.with_overrides(kwargs)
        self._client._rebuild_resources()
        try:
            yield
        finally:
            self._client._config = previous
            self._client._rebuild_resources()


class GuardedOpenAI:
    def __init__(self, client: Any, *, guard: Any, config: GuardConfig | None = None):
        self._client = client
        self._guard = guard
        self._config = config or GuardConfig()
        self.guard = GuardController(self)
        self._rebuild_resources()

    def _rebuild_resources(self) -> None:
        self.responses = GuardedResponses(self._client.responses, guard=self._guard, config=self._config)
        chat = getattr(self._client, "chat", None)
        if chat is not None and getattr(chat, "completions", None) is not None:
            self.chat = _GuardedChat(chat, guard=self._guard, config=self._config)

    def __getattr__(self, name: str) -> object:
        return getattr(self._client, name)


class _GuardedChat:
    def __init__(self, chat: Any, *, guard: Any, config: GuardConfig):
        self._chat = chat
        self.completions = GuardedChatCompletions(chat.completions, guard=guard, config=config)

    def __getattr__(self, name: str) -> object:
        return getattr(self._chat, name)
