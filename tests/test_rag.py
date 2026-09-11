from __future__ import annotations

from openwarden import GuardMode, GuardedOpenAI, GroundingDecision, RAGConfig, RAGMode, RetrievedDocument, TextContent, WardenConfig
from openwarden.models import GuardRequest, Message
from openwarden.rag.injection import DefaultContextInjector
from openwarden.rag.query import DefaultRetrievalQueryBuilder
from tests._fakes import FakeClient, FakeResponse, RecordingGuard


def test_rag_query_builder_and_context_injector_limits_context():
    request = GuardRequest(
        endpoint="responses.create",
        model="gpt-test",
        instructions="Use docs",
        user_content=(TextContent("Question"),),
        conversation=(Message(role="user", content=(TextContent("Conversation question"),)),),
    )

    query = DefaultRetrievalQueryBuilder(top_k=3).build(request)

    assert query.top_k == 3
    assert query.text == "Use docs\nQuestion\nConversation question"

    injected = DefaultContextInjector(maximum_context_characters=30).inject(
        request,
        [RetrievedDocument(id="doc-1", text="A" * 100, score=1.0)],
    )

    assert "Reference material follows" in injected.instructions
    assert "[document_id=doc-1]" in injected.instructions
    assert len(injected.instructions) < len(request.instructions or "") + 300


def test_rag_augment_and_verify_injects_documents_and_checks_grounding():
    class Retriever:
        def __init__(self):
            self.queries = []

        def retrieve(self, query):
            self.queries.append(query)
            return [
                RetrievedDocument(id="doc-low", text="ignore", score=0.1),
                RetrievedDocument(id="doc-1", text="Annual leave is 20 days.", score=0.9),
            ]

    retriever = Retriever()
    client = FakeClient(FakeResponse("20 days"))
    guard = RecordingGuard(grounding=GroundingDecision(True, 0.95, supporting_document_ids=("doc-1",)))
    guarded = GuardedOpenAI(
        client,
        guard=guard,
        config=WardenConfig(
            mode=GuardMode.OUTPUT,
            rag=RAGConfig(provider=retriever, mode=RAGMode.AUGMENT_AND_VERIFY, minimum_score=0.7),
        ),
    )

    guarded.responses.create(model="gpt-test", instructions="Use policy", input="Leave?")

    forwarded = client.responses.calls[0][1]
    assert "document_id=doc-1" in forwarded["instructions"]
    assert "doc-low" not in forwarded["instructions"]
    assert retriever.queries[0].text == "Use policy\nLeave?"
