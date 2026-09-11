from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from openwarden.config import GuardConfig
from openwarden.resources.async_chat_completions import AsyncGuardedChatCompletions
from openwarden.resources.async_responses import AsyncGuardedResponses


class AsyncGuardController:
    def __init__(self, client: "AsyncGuardedOpenAI"):
        self._client = client

    @asynccontextmanager
    async def override(self, **kwargs: object) -> AsyncIterator[None]:
        previous = self._client._config
        self._client._config = previous.with_overrides(kwargs)
        self._client._rebuild_resources()
        try:
            yield
        finally:
            self._client._config = previous
            self._client._rebuild_resources()


class AsyncGuardedOpenAI:
    def __init__(self, client: Any, *, guard: Any, config: GuardConfig | None = None):
        self._client = client
        self._guard = guard
        self._config = config or GuardConfig()
        self.guard = AsyncGuardController(self)
        self._rebuild_resources()

    def _rebuild_resources(self) -> None:
        self.responses = AsyncGuardedResponses(self._client.responses, guard=self._guard, config=self._config)
        chat = getattr(self._client, "chat", None)
        if chat is not None and getattr(chat, "completions", None) is not None:
            self.chat = _AsyncGuardedChat(chat, guard=self._guard, config=self._config)

    def __getattr__(self, name: str) -> object:
        return getattr(self._client, name)


class _AsyncGuardedChat:
    def __init__(self, chat: Any, *, guard: Any, config: GuardConfig):
        self._chat = chat
        self.completions = AsyncGuardedChatCompletions(chat.completions, guard=guard, config=config)

    def __getattr__(self, name: str) -> object:
        return getattr(self._chat, name)
