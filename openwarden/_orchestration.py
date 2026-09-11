from __future__ import annotations

from time import perf_counter

from openwarden.config import GuardAction, GuardConfig, RAGConfig, StreamGuardMode
from openwarden.exceptions import (
    GuardEvaluationError,
    GuardViolation,
    GroundingViolation,
    RetrievalError,
    UnsupportedStreamingModeError,
)
from openwarden.models import (
    GroundingDecision,
    GuardDecision,
    GuardRequest,
    GuardResponse,
    GuardVerdict,
    RetrievedDocument,
)
from openwarden.rag.injection import DefaultContextInjector
from openwarden.rag.query import DefaultRetrievalQueryBuilder
from openwarden.result import GuardedResult
from openwarden.telemetry.trace import GuardTrace


def strip_guard_options(kwargs: dict[str, object]) -> tuple[dict[str, object], object | None]:
    clean = dict(kwargs)
    return clean, clean.pop("guard_options", None)


def ensure_streaming_supported(kwargs: dict[str, object], config: GuardConfig) -> None:
    if not kwargs.get("stream"):
        return
    if config.mode.checks_output and config.stream_guard_mode == StreamGuardMode.DISALLOW:
        raise UnsupportedStreamingModeError("stream=True is not supported when output guarding is enabled")


def apply_input_decision(
    *,
    request: GuardRequest,
    decision: GuardDecision,
    config: GuardConfig,
    trace: GuardTrace,
) -> GuardRequest:
    if decision.verdict == GuardVerdict.ALLOW:
        return request
    if decision.verdict in {GuardVerdict.BLOCK, GuardVerdict.REVIEW}:
        if config.input_action == GuardAction.LOG_ONLY:
            return request
        raise GuardViolation(phase="input", decision=decision, trace_id=trace.trace_id)
    if decision.verdict in {GuardVerdict.REDACT, GuardVerdict.MODIFY}:
        if not config.allow_input_modification:
            raise GuardViolation(phase="input", decision=decision, trace_id=trace.trace_id)
        if isinstance(decision.replacement, GuardRequest):
            return decision.replacement
        raise GuardEvaluationError("input modification requires GuardDecision.replacement to be a GuardRequest")
    return request


def apply_output_decision(
    *,
    response: object,
    guarded_response: GuardResponse,
    decision: GuardDecision,
    config: GuardConfig,
    trace: GuardTrace,
) -> object:
    if decision.verdict == GuardVerdict.ALLOW:
        return response
    if config.output_action == GuardAction.LOG_ONLY:
        return response
    if decision.verdict in {GuardVerdict.BLOCK, GuardVerdict.REVIEW} or config.output_action == GuardAction.BLOCK:
        raise GuardViolation(phase="output", decision=decision, trace_id=trace.trace_id)
    if config.output_action in {GuardAction.REDACT, GuardAction.REWRITE, GuardAction.ANNOTATE}:
        return GuardedResult(response=response, guard_trace=trace, output_text=_replacement_text(decision, guarded_response))
    return response


def apply_grounding_decision(*, response: object, decision: GroundingDecision, trace: GuardTrace) -> object:
    if decision.grounded:
        return response
    guard_decision = GuardDecision.block(
        decision.reason or "response was not grounded",
        categories=("grounding",),
        grounding_score=decision.score,
    )
    raise GroundingViolation(phase="grounding", decision=guard_decision, trace_id=trace.trace_id)


def retrieve_documents(request: GuardRequest, rag_config: RAGConfig, trace: GuardTrace) -> list[RetrievedDocument]:
    query_builder = rag_config.query_builder or DefaultRetrievalQueryBuilder(
        top_k=rag_config.top_k,
        context=rag_config.retrieval_context,
    )
    query = query_builder.build(request)
    started = perf_counter()
    try:
        documents = rag_config.provider.retrieve(query)
    except Exception as exc:
        raise RetrievalError(str(exc)) from exc
    trace.retrieval_latency_ms = (perf_counter() - started) * 1000
    filtered = _filter_documents(documents, rag_config)
    if rag_config.require_results and not filtered:
        raise RetrievalError("retrieval returned no documents above minimum score")
    trace.retrieval_count = len(filtered)
    trace.retrieved_document_ids = tuple(document.id for document in filtered)
    return filtered


async def async_retrieve_documents(request: GuardRequest, rag_config: RAGConfig, trace: GuardTrace) -> list[RetrievedDocument]:
    query_builder = rag_config.query_builder or DefaultRetrievalQueryBuilder(
        top_k=rag_config.top_k,
        context=rag_config.retrieval_context,
    )
    query = query_builder.build(request)
    started = perf_counter()
    try:
        documents = await rag_config.provider.retrieve(query)
    except Exception as exc:
        raise RetrievalError(str(exc)) from exc
    trace.retrieval_latency_ms = (perf_counter() - started) * 1000
    filtered = _filter_documents(documents, rag_config)
    if rag_config.require_results and not filtered:
        raise RetrievalError("retrieval returned no documents above minimum score")
    trace.retrieval_count = len(filtered)
    trace.retrieved_document_ids = tuple(document.id for document in filtered)
    return filtered


def inject_context(request: GuardRequest, documents: list[RetrievedDocument], rag_config: RAGConfig) -> GuardRequest:
    injector = rag_config.injector or DefaultContextInjector(
        maximum_context_characters=rag_config.maximum_context_characters
    )
    return injector.inject(request, documents)


def maybe_guarded_result(response: object, guarded_response: GuardResponse, config: GuardConfig, trace: GuardTrace) -> object:
    if isinstance(response, GuardedResult):
        return response
    if not config.return_guarded_result:
        return response
    return GuardedResult(response=response, guard_trace=trace, output_text=guarded_response.text)


def handle_infrastructure_error(exc: Exception, config: GuardConfig, trace: GuardTrace, phase: str) -> None:
    if config.fail_open:
        trace.metadata[f"{phase}_error"] = str(exc)
        return
    if isinstance(exc, (GuardViolation, GroundingViolation, UnsupportedStreamingModeError)):
        raise exc
    if isinstance(exc, RetrievalError):
        raise exc
    raise GuardEvaluationError(str(exc)) from exc


def _filter_documents(documents: list[RetrievedDocument], rag_config: RAGConfig) -> list[RetrievedDocument]:
    filtered = [document for document in documents if document.score >= rag_config.minimum_score]
    return filtered[: rag_config.top_k]


def _replacement_text(decision: GuardDecision, guarded_response: GuardResponse) -> str | None:
    if isinstance(decision.replacement, str):
        return decision.replacement
    return guarded_response.text
