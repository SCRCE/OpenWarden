from __future__ import annotations

import pytest

from openwarden import (
    GuardMode,
    GuardedOpenAI,
    GroundingDecision,
    GroundingViolation,
    RAGConfig,
    RAGMode,
    RetrievedDocument,
    RetrievalError,
    WardenConfig,
)
from tests._fakes import FakeClient, FakeResponse, RecordingGuard


def test_grounding_failure_raises_grounding_violation_with_score_metadata():
    guarded = GuardedOpenAI(
        FakeClient(FakeResponse("unsupported claim")),
        guard=RecordingGuard(grounding=GroundingDecision(False, 0.2, reason="unsupported")),
        config=WardenConfig(
            mode=GuardMode.OUTPUT,
            rag=RAGConfig(provider=_StaticRetriever(), mode=RAGMode.VERIFY_ONLY, minimum_score=0.1),
        ),
    )

    with pytest.raises(GroundingViolation) as exc:
        guarded.responses.create(model="gpt-test", input="question")

    assert exc.value.phase == "grounding"
    assert exc.value.decision.metadata["grounding_score"] == 0.2


def test_retrieval_require_results_raises_and_fail_open_allows_call():
    class EmptyRetriever:
        def retrieve(self, query):
            return []

    strict = GuardedOpenAI(
        FakeClient(),
        guard=RecordingGuard(),
        config=WardenConfig(rag=RAGConfig(provider=EmptyRetriever(), require_results=True)),
    )

    with pytest.raises(RetrievalError):
        strict.responses.create(model="gpt-test", input="hello")

    permissive_client = FakeClient(FakeResponse("ok"))
    permissive = GuardedOpenAI(
        permissive_client,
        guard=RecordingGuard(),
        config=WardenConfig(
            fail_open=True,
            rag=RAGConfig(provider=EmptyRetriever(), require_results=True),
        ),
    )

    assert permissive.responses.create(model="gpt-test", input="hello").output_text == "ok"
    assert len(permissive_client.responses.calls) == 1


class _StaticRetriever:
    def retrieve(self, query):
        return [RetrievedDocument(id="doc", text="evidence", score=1.0)]
