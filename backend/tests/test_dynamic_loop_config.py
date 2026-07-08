"""Tests for DynamicLoopConfig."""

import pytest
from pydantic import ValidationError

from deerflow.config.dynamic_loop_config import DynamicLoopConfig


def test_dynamic_loop_config_defaults_are_conservative() -> None:
    cfg = DynamicLoopConfig()

    assert cfg.enabled is True
    assert cfg.stagnation_warn_steps == 3
    assert cfg.stagnation_hard_limit == 6
    assert cfg.max_tool_calls_without_answer == 40
    assert cfg.wrap_up_when_recursion_remaining == 8
    assert cfg.min_new_info_chars == 120
    assert cfg.force_wrap_up_on_non_interactive_stagnation is True


def test_dynamic_loop_rejects_hard_limit_below_warn_steps() -> None:
    with pytest.raises(ValidationError, match="stagnation_hard_limit must be >= stagnation_warn_steps"):
        DynamicLoopConfig(stagnation_warn_steps=4, stagnation_hard_limit=3)


@pytest.mark.parametrize(
    "field",
    [
        "stagnation_warn_steps",
        "stagnation_hard_limit",
        "max_tool_calls_without_answer",
        "wrap_up_when_recursion_remaining",
        "min_new_info_chars",
    ],
)
def test_dynamic_loop_rejects_non_positive_thresholds(field: str) -> None:
    with pytest.raises(ValidationError):
        DynamicLoopConfig(**{field: 0})
