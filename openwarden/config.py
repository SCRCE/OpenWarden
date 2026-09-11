from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any


class GuardMode(str, Enum):
    DISABLED = "disabled"
    INPUT = "input"
    OUTPUT = "output"
    BOTH = "both"

    @property
    def checks_input(self) -> bool:
        return self in {GuardMode.INPUT, GuardMode.BOTH}

    @property
    def checks_output(self) -> bool:
        return self in {GuardMode.OUTPUT, GuardMode.BOTH}


class GuardAction(str, Enum):
    BLOCK = "block"
    REDACT = "redact"
    REWRITE = "rewrite"
    ANNOTATE = "annotate"
    LOG_ONLY = "log_only"


class RAGMode(str, Enum):
    DISABLED = "disabled"
    AUGMENT = "augment"
    VERIFY_ONLY = "verify_only"
    AUGMENT_AND_VERIFY = "augment_and_verify"

    @property
    def augments(self) -> bool:
        return self in {RAGMode.AUGMENT, RAGMode.AUGMENT_AND_VERIFY}

    @property
    def verifies(self) -> bool:
        return self in {RAGMode.VERIFY_ONLY, RAGMode.AUGMENT_AND_VERIFY}


class StreamGuardMode(str, Enum):
    INPUT_ONLY = "input_only"
    BUFFER = "buffer"
    CHUNKED = "chunked"
    DISALLOW = "disallow"


@dataclass(frozen=True)
class RAGConfig:
    provider: Any
    mode: RAGMode | str = RAGMode.AUGMENT_AND_VERIFY
    top_k: int = 5
    minimum_score: float = 0.70
    maximum_context_characters: int = 30_000
    require_results: bool = False
    query_builder: Any | None = None
    injector: Any | None = None
    retrieval_context: Any | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", RAGMode(self.mode))
        if self.top_k <= 0:
            raise ValueError("RAGConfig.top_k must be positive")
        if not 0 <= self.minimum_score <= 1:
            raise ValueError("RAGConfig.minimum_score must be between 0 and 1")
        if self.maximum_context_characters <= 0:
            raise ValueError("RAGConfig.maximum_context_characters must be positive")


@dataclass(frozen=True)
class GuardConfig:
    mode: GuardMode | str = GuardMode.BOTH
    input_action: GuardAction | str = GuardAction.BLOCK
    output_action: GuardAction | str = GuardAction.BLOCK
    input_policy: str | None = None
    output_policy: str | None = None
    fail_open: bool = False
    timeout_seconds: float = 10.0
    max_rewrites: int = 1
    allow_input_modification: bool = False
    return_guarded_result: bool = False
    rag: RAGConfig | None = None
    stream_guard_mode: StreamGuardMode | str = StreamGuardMode.DISALLOW

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", GuardMode(self.mode))
        object.__setattr__(self, "input_action", GuardAction(self.input_action))
        object.__setattr__(self, "output_action", GuardAction(self.output_action))
        object.__setattr__(self, "stream_guard_mode", StreamGuardMode(self.stream_guard_mode))
        if self.timeout_seconds <= 0:
            raise ValueError("GuardConfig.timeout_seconds must be positive")
        if self.max_rewrites < 0:
            raise ValueError("GuardConfig.max_rewrites must be zero or positive")

    def with_overrides(self, overrides: "CallGuardConfig | dict[str, Any] | None") -> "GuardConfig":
        if overrides is None:
            return self
        if isinstance(overrides, CallGuardConfig):
            data = overrides.to_update_dict()
        elif isinstance(overrides, dict):
            data = dict(overrides)
        else:
            raise TypeError("guard_options must be CallGuardConfig, dict, or None")
        if "rag" in data and data["rag"] is False:
            data["rag"] = None
        return replace(self, **data)


class WardenConfig(GuardConfig):
    """OpenWarden public name for guard configuration."""


@dataclass(frozen=True)
class CallGuardConfig:
    mode: GuardMode | str | None = None
    input_action: GuardAction | str | None = None
    output_action: GuardAction | str | None = None
    input_policy: str | None = None
    output_policy: str | None = None
    fail_open: bool | None = None
    rag: RAGConfig | bool | None = None
    return_guarded_result: bool | None = None

    def to_update_dict(self) -> dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if value is not None}
