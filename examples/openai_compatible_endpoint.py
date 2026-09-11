"""Guard an OpenAI-compatible endpoint with a dedicated policy model."""

from __future__ import annotations

import os

from openai import OpenAI

from openwarden import GuardMode, GuardViolation, GuardedOpenAI, WardenConfig
from openwarden.guards.base import ModelPromptGuard


def main() -> None:
    generation_model = _required_env("OPENWARDEN_GENERATION_MODEL")
    guard_model = _required_env("OPENWARDEN_GUARD_MODEL")

    raw_client = OpenAI(timeout=float(os.getenv("OPENWARDEN_TIMEOUT_SECONDS", "45")))
    guard = ModelPromptGuard(
        raw_client,
        model=guard_model,
        api="chat_completions",
        require_json=True,
        system_prompt=(
            "Evaluate the supplied request or response against the configured policy. "
            "Return only JSON with verdict, reason, categories, replacement, and confidence."
        ),
    )
    client = GuardedOpenAI(
        raw_client,
        guard=guard,
        config=WardenConfig(
            mode=GuardMode.BOTH,
            fail_open=False,
            return_guarded_result=True,
        ),
    )

    try:
        result = client.chat.completions.create(
            model=generation_model,
            messages=[
                {"role": "system", "content": "You are concise and helpful."},
                {"role": "user", "content": "Explain Python decorators."},
            ],
        )
    except GuardViolation as exc:
        print(f"blocked phase={exc.phase} trace_id={exc.trace_id}")
        print(exc.decision.reason)
        return

    message = result.response.choices[0].message
    print(message.content or getattr(message, "reasoning", ""))
    print(f"trace_id={result.guard_trace.trace_id}")
    print(f"input={_verdict(result.guard_trace.input_decision)}")
    print(f"output={_verdict(result.guard_trace.output_decision)}")


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip().strip("\"'")
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value


def _verdict(decision: object | None) -> str:
    return getattr(getattr(decision, "verdict", None), "value", "not-run")


if __name__ == "__main__":
    main()
