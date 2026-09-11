from __future__ import annotations

from typing import Any

from openwarden.adapters.chat_completions import ChatCompletionsAdapter
from openwarden.config import GuardConfig
from openwarden.resources.async_responses import AsyncGuardedResponses


class AsyncGuardedChatCompletions(AsyncGuardedResponses):
    def __init__(self, resource: Any, *, guard: Any, config: GuardConfig, adapter: Any | None = None):
        super().__init__(resource, guard=guard, config=config, adapter=adapter or ChatCompletionsAdapter())
