from __future__ import annotations

from typing import Any, Protocol

from openwarden.models import (
    GroundingDecision,
    GuardDecision,
    GuardInputContext,
    GuardOutputContext,
    GuardRequest,
    GuardResponse,
    RetrievedDocument,
    RetrievalQuery,
)


class Guard(Protocol):
    def check_input(self, context: GuardInputContext) -> GuardDecision: ...

    def check_output(self, context: GuardOutputContext) -> GuardDecision: ...

    def check_grounding(
        self,
        *,
        request: GuardRequest,
        response: GuardResponse,
        documents: list[RetrievedDocument],
    ) -> GroundingDecision: ...


class AsyncGuard(Protocol):
    async def check_input(self, context: GuardInputContext) -> GuardDecision: ...

    async def check_output(self, context: GuardOutputContext) -> GuardDecision: ...

    async def check_grounding(
        self,
        *,
        request: GuardRequest,
        response: GuardResponse,
        documents: list[RetrievedDocument],
    ) -> GroundingDecision: ...


class EndpointAdapter(Protocol):
    endpoint: str

    def extract_request(self, args: tuple[object, ...], kwargs: dict[str, object]) -> GuardRequest: ...

    def apply_request(self, original_kwargs: dict[str, object], request: GuardRequest) -> dict[str, object]: ...

    def extract_response(self, response: object) -> GuardResponse: ...


class Retriever(Protocol):
    def retrieve(self, query: RetrievalQuery) -> list[RetrievedDocument]: ...


class AsyncRetriever(Protocol):
    async def retrieve(self, query: RetrievalQuery) -> list[RetrievedDocument]: ...


class RetrievalQueryBuilder(Protocol):
    def build(self, request: GuardRequest) -> RetrievalQuery: ...


class ContextInjector(Protocol):
    def inject(self, request: GuardRequest, documents: list[RetrievedDocument]) -> GuardRequest: ...


class GroundingEvaluator(Protocol):
    def evaluate(
        self,
        *,
        answer: GuardResponse,
        documents: list[RetrievedDocument],
        request: GuardRequest,
    ) -> GroundingDecision: ...


class GuardModelClient(Protocol):
    def create(self, **kwargs: Any) -> Any: ...
