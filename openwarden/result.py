from __future__ import annotations

from dataclasses import dataclass

from openwarden.telemetry.trace import GuardTrace


@dataclass(frozen=True)
class GuardedResult:
    response: object
    guard_trace: GuardTrace
    output_text: str | None = None
