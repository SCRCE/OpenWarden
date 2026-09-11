from __future__ import annotations

from time import perf_counter
from typing import Any

from openwarden._orchestration import (
    apply_grounding_decision,
    apply_input_decision,
    apply_output_decision,
    async_retrieve_documents,
    ensure_streaming_supported,
    handle_infrastructure_error,
    inject_context,
    maybe_guarded_result,
    strip_guard_options,
)
from openwarden.adapters.responses import ResponsesAdapter
from openwarden.config import GuardConfig
from openwarden.exceptions import GuardViolation, GroundingViolation, UnsupportedStreamingModeError
from openwarden.models import GuardInputContext, GuardOutputContext
from openwarden.telemetry.trace import GuardTrace


class AsyncGuardedResponses:
    def __init__(self, resource: Any, *, guard: Any, config: GuardConfig, adapter: Any | None = None):
        self._resource = resource
        self._guard = guard
        self._config = config
        self._adapter = adapter or ResponsesAdapter()

    async def create(self, *args: object, **kwargs: object) -> object:
        clean_kwargs, guard_options = strip_guard_options(kwargs)
        config = self._config.with_overrides(guard_options)
        ensure_streaming_supported(clean_kwargs, config)
        request = self._adapter.extract_request(args, clean_kwargs)
        original_request = request
        trace = GuardTrace.start(request, config.mode)
        documents = []
        response = None
        guarded_response = None
        try:
            if config.mode.checks_input:
                decision = await self._guard.check_input(
                    GuardInputContext(request=request, policy=config.input_policy)
                )
                trace.input_decision = decision
                request = apply_input_decision(request=request, decision=decision, config=config, trace=trace)

            if config.rag is not None and config.rag.mode.value != "disabled":
                documents = await async_retrieve_documents(request, config.rag, trace)
                if config.rag.mode.augments:
                    request = inject_context(request, documents, config.rag)

            final_kwargs = self._adapter.apply_request(clean_kwargs, request)
            started = perf_counter()
            response = await self._resource.create(*args, **final_kwargs)
            trace.main_model_latency_ms = (perf_counter() - started) * 1000
            guarded_response = self._adapter.extract_response(response)

            if config.mode.checks_output:
                decision = await self._guard.check_output(
                    GuardOutputContext(
                        original_request=original_request,
                        effective_request=request,
                        response=guarded_response,
                        retrieved_documents=tuple(documents),
                        policy=config.output_policy,
                    )
                )
                trace.output_decision = decision
                response = apply_output_decision(
                    response=response,
                    guarded_response=guarded_response,
                    decision=decision,
                    config=config,
                    trace=trace,
                )

            if config.rag is not None and config.rag.mode.verifies:
                grounding = await self._guard.check_grounding(
                    request=request,
                    response=guarded_response,
                    documents=documents,
                )
                trace.grounding_decision = grounding
                response = apply_grounding_decision(response=response, decision=grounding, trace=trace)

            trace.complete("allowed")
            return maybe_guarded_result(response, guarded_response, config, trace)
        except Exception as exc:
            trace.complete("failed")
            if (
                config.fail_open
                and response is not None
                and not isinstance(exc, (GuardViolation, GroundingViolation, UnsupportedStreamingModeError))
            ):
                trace.metadata["responses_error"] = str(exc)
                if guarded_response is not None:
                    return maybe_guarded_result(response, guarded_response, config, trace)
                return response
            handle_infrastructure_error(exc, config, trace, "responses")
            final_kwargs = self._adapter.apply_request(clean_kwargs, request)
            return await self._resource.create(*args, **final_kwargs)

    def __getattr__(self, name: str) -> object:
        return getattr(self._resource, name)
