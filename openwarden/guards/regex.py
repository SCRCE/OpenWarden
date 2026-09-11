from __future__ import annotations

import re

from openwarden.guards.base import BaseGuard
from openwarden.models import GuardDecision, GuardInputContext, GuardOutputContext, GuardVerdict


class RegexSecretGuard(BaseGuard):
    def __init__(self, patterns: list[str] | None = None):
        self._patterns = [re.compile(pattern) for pattern in (patterns or [r"sk-[A-Za-z0-9_\-]{8,}"])]

    def check_input(self, context: GuardInputContext) -> GuardDecision:
        return self._check_text(context.request.text())

    def check_output(self, context: GuardOutputContext) -> GuardDecision:
        return self._check_text(context.response.text or "")

    def _check_text(self, text: str) -> GuardDecision:
        if any(pattern.search(text) for pattern in self._patterns):
            return GuardDecision(
                GuardVerdict.BLOCK,
                reason="secret-like value detected",
                categories=("secret",),
            )
        return GuardDecision.allow()
