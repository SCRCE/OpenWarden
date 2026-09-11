from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4


@dataclass(frozen=True)
class ContentPart:
    type: str


@dataclass(frozen=True)
class TextContent(ContentPart):
    text: str

    def __init__(self, text: str):
        object.__setattr__(self, "type", "text")
        object.__setattr__(self, "text", text)


@dataclass(frozen=True)
class Message:
    role: str
    content: tuple[ContentPart, ...] = ()
    raw: object | None = None


@dataclass(frozen=True)
class ToolDefinition:
    name: str | None
    raw: object


@dataclass(frozen=True)
class ToolCall:
    name: str | None
    arguments: object | None
    raw: object


@dataclass(frozen=True)
class GuardRequest:
    endpoint: str
    model: str | None
    instructions: str | None
    user_content: tuple[ContentPart, ...] = ()
    conversation: tuple[Message, ...] = ()
    tools: tuple[ToolDefinition, ...] = ()
    tool_choice: object | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def text(self) -> str:
        parts = []
        if self.instructions:
            parts.append(self.instructions)
        parts.extend(part.text for part in self.user_content if isinstance(part, TextContent))
        for message in self.conversation:
            parts.extend(part.text for part in message.content if isinstance(part, TextContent))
        return "\n".join(part for part in parts if part)


@dataclass(frozen=True)
class GuardResponse:
    text: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    structured_output: object | None = None
    refusal: str | None = None
    raw_response: object | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class GuardVerdict(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    REDACT = "redact"
    MODIFY = "modify"
    REVIEW = "review"


@dataclass(frozen=True)
class GuardDecision:
    verdict: GuardVerdict | str
    reason: str = ""
    categories: tuple[str, ...] = ()
    replacement: object | None = None
    confidence: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "verdict", GuardVerdict(self.verdict))

    @classmethod
    def allow(cls, reason: str = "allowed", **metadata: object) -> "GuardDecision":
        return cls(GuardVerdict.ALLOW, reason=reason, metadata=dict(metadata))

    @classmethod
    def block(
        cls,
        reason: str,
        *,
        categories: tuple[str, ...] = (),
        confidence: float | None = None,
        **metadata: object,
    ) -> "GuardDecision":
        return cls(
            GuardVerdict.BLOCK,
            reason=reason,
            categories=categories,
            confidence=confidence,
            metadata=dict(metadata),
        )


@dataclass(frozen=True)
class RetrievedDocument:
    id: str
    text: str
    score: float
    source: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalContext:
    tenant_id: str | None = None
    user_id: str | None = None
    roles: tuple[str, ...] = ()
    allowed_collections: tuple[str, ...] = ()
    metadata_filters: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalQuery:
    text: str
    top_k: int = 5
    context: RetrievalContext = field(default_factory=RetrievalContext)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class GuardInputContext:
    request: GuardRequest
    policy: str | None = None
    retrieved_documents: tuple[RetrievedDocument, ...] = ()


@dataclass(frozen=True)
class GuardOutputContext:
    original_request: GuardRequest
    effective_request: GuardRequest
    response: GuardResponse
    retrieved_documents: tuple[RetrievedDocument, ...] = ()
    policy: str | None = None


@dataclass(frozen=True)
class GroundingDecision:
    grounded: bool
    score: float
    unsupported_claims: tuple[str, ...] = ()
    supporting_document_ids: tuple[str, ...] = ()
    reason: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def not_evaluated(cls, reason: str = "not evaluated") -> "GroundingDecision":
        return cls(True, 1.0, reason=reason, metadata={"not_evaluated": True})


@dataclass
class ExecutionContext:
    request_id: str
    original_request: GuardRequest
    effective_request: GuardRequest
    input_decision: GuardDecision | None = None
    output_decision: GuardDecision | None = None
    grounding_decision: GroundingDecision | None = None
    retrieved_documents: list[RetrievedDocument] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def start(cls, request: GuardRequest) -> "ExecutionContext":
        return cls(str(uuid4()), original_request=request, effective_request=request)
