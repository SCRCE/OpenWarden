from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from openwarden.config import GuardMode
from openwarden.models import GroundingDecision, GuardDecision, GuardRequest


_current_trace: ContextVar["GuardTrace | None"] = ContextVar("openwarden_current_trace", default=None)


@dataclass
class GuardTrace:
    trace_id: str
    endpoint: str
    mode: GuardMode
    model: str | None = None
    input_decision: GuardDecision | None = None
    retrieval_count: int = 0
    retrieved_document_ids: tuple[str, ...] = ()
    retrieval_latency_ms: float | None = None
    main_model_latency_ms: float | None = None
    output_decision: GuardDecision | None = None
    grounding_decision: GroundingDecision | None = None
    final_outcome: str = "started"
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def start(cls, request: GuardRequest, mode: GuardMode) -> "GuardTrace":
        trace = cls(str(uuid4()), endpoint=request.endpoint, mode=mode, model=request.model)
        _current_trace.set(trace)
        return trace

    def complete(self, outcome: str = "allowed") -> None:
        self.final_outcome = outcome
        self.completed_at = datetime.now(timezone.utc)


def get_current_guard_trace() -> GuardTrace | None:
    return _current_trace.get()
