"""Guard local Hugging Face/PyTorch generation without an inference API."""

from __future__ import annotations

import os

from transformers import AutoModelForCausalLM, AutoTokenizer

from openwarden import GuardMode, GuardedPyTorch, PyTorchPromptGuard, WardenConfig


def main() -> None:
    model_id = _required_env("OPENWARDEN_MAIN_MODEL")
    guard_model_id = _required_env("OPENWARDEN_LOCAL_GUARD_MODEL")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    guard_tokenizer = AutoTokenizer.from_pretrained(guard_model_id)
    guard_model = AutoModelForCausalLM.from_pretrained(guard_model_id, device_map="auto")

    guarded_model = GuardedPyTorch(
        model,
        tokenizer=tokenizer,
        guard=PyTorchPromptGuard(
            guard_model,
            tokenizer=guard_tokenizer,
            generation_kwargs={"max_new_tokens": 256, "do_sample": False},
        ),
        config=WardenConfig(mode=GuardMode.BOTH, fail_open=False),
    )

    inputs = tokenizer("Explain Python decorators.", return_tensors="pt").to(model.device)
    output_ids = guarded_model.generate(**inputs, max_new_tokens=200)
    completion_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
    print(tokenizer.decode(completion_ids, skip_special_tokens=True))


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value


if __name__ == "__main__":
    main()
