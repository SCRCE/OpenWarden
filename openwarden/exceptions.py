from __future__ import annotations


class GuardError(Exception):
    """Base exception for guard wrapper errors."""


class GuardEvaluationError(GuardError):
    """Raised when guard infrastructure fails and fail-open is disabled."""


class RetrievalError(GuardError):
    """Raised when retrieval fails and fail-open is disabled."""


class UnsupportedEndpointError(GuardError):
    """Raised for endpoints that cannot be guarded."""


class UnsupportedStreamingModeError(GuardError):
    """Raised when stream policy cannot safely handle a request."""


class GuardViolation(GuardError):
    def __init__(self, *, phase: str, decision: object, trace_id: str):
        self.phase = phase
        self.decision = decision
        self.trace_id = trace_id
        reason = getattr(decision, "reason", "guard violation")
        super().__init__(f"{phase} blocked: {reason}")


class GroundingViolation(GuardViolation):
    """Raised when grounding verification blocks a response."""
