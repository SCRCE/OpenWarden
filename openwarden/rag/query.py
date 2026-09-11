from __future__ import annotations

from openwarden.models import GuardRequest, RetrievalContext, RetrievalQuery, TextContent


class DefaultRetrievalQueryBuilder:
    def __init__(self, *, top_k: int = 5, context: RetrievalContext | None = None):
        self._top_k = top_k
        self._context = context or RetrievalContext()

    def build(self, request: GuardRequest) -> RetrievalQuery:
        text = "\n".join(
            [
                request.instructions or "",
                *[part.text for part in request.user_content if isinstance(part, TextContent)],
                *[
                    part.text
                    for message in request.conversation
                    if message.role == "user"
                    for part in message.content
                    if isinstance(part, TextContent)
                ],
            ]
        ).strip()
        return RetrievalQuery(text=text, top_k=self._top_k, context=self._context)
