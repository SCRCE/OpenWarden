# Changelog

All notable changes to OpenWarden are documented here.

## 0.1.0 - 2026-09-10

Initial alpha release.

- Added synchronous and asynchronous composition wrappers for OpenAI-compatible clients.
- Added guarded Responses and Chat Completions generation resources.
- Added direct PyTorch and Transformers `model.generate(...)` guarding.
- Added deterministic, prompted, pipeline, and local PyTorch guard implementations.
- Added input, output, both, and disabled modes with explicit enforcement actions.
- Added per-call overrides, trace metadata, guarded results, and fail-open/fail-closed behavior.
- Added RAG retrieval, context injection, and grounding verification extension points.
- Added conservative streaming behavior that prevents unguarded output leakage by default.
