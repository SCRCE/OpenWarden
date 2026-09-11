from __future__ import annotations

from openwarden.guards.base import BaseGuard
from openwarden.models import (
    GroundingDecision,
    GuardDecision,
    GuardInputContext,
    GuardOutputContext,
    GuardRequest,
    GuardResponse,
    GuardVerdict,
    RetrievedDocument,
)


class GuardPipeline(BaseGuard):
    def __init__(self, guards: list[BaseGuard]):
        self._guards = list(guards)

    def check_input(self, context: GuardInputContext) -> GuardDecision:
        return self._first_non_allow("check_input", context)

    def check_output(self, context: GuardOutputContext) -> GuardDecision:
        return self._first_non_allow("check_output", context)

    def check_grounding(
        self,
        *,
        request: GuardRequest,
        response: GuardResponse,
        documents: list[RetrievedDocument],
    ) -> GroundingDecision:
        decisions = [
            guard.check_grounding(request=request, response=response, documents=documents)
            for guard in self._guards
        ]
        evaluated = [decision for decision in decisions if not decision.metadata.get("not_evaluated")]
        if not evaluated:
            return GroundingDecision.not_evaluated()
        return min(evaluated, key=lambda decision: decision.score)

    def _first_non_allow(self, method_name: str, context: object) -> GuardDecision:
        for guard in self._guards:
            decision = getattr(guard, method_name)(context)
            if decision.verdict != GuardVerdict.ALLOW:
                return decision
        return GuardDecision.allow()
