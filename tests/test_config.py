from __future__ import annotations

import pytest

from openwarden import GuardAction, GuardMode, RAGConfig, WardenConfig


def test_config_validation_and_string_coercion():
    config = WardenConfig(
        mode="both",
        input_action="block",
        output_action="log_only",
        stream_guard_mode="disallow",
    )

    assert config.mode is GuardMode.BOTH
    assert config.output_action is GuardAction.LOG_ONLY

    with pytest.raises(ValueError):
        WardenConfig(timeout_seconds=0)

    with pytest.raises(ValueError):
        WardenConfig(max_rewrites=-1)

    with pytest.raises(ValueError):
        RAGConfig(provider=object(), top_k=0)

    with pytest.raises(ValueError):
        RAGConfig(provider=object(), minimum_score=1.5)
