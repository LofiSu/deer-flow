"""Configuration for dynamic loop controller middleware."""

from pydantic import BaseModel, Field, model_validator


class DynamicLoopConfig(BaseModel):
    """Run-level dynamic loop control for the lead agent."""

    enabled: bool = Field(
        default=True,
        description="Whether to enable run-level dynamic loop control for the lead agent.",
    )
    stagnation_warn_steps: int = Field(
        default=3,
        ge=1,
        description="Consecutive no-progress tool results before injecting strategy guidance.",
    )
    stagnation_hard_limit: int = Field(
        default=6,
        ge=1,
        description="Consecutive no-progress tool results before forcing a final answer.",
    )
    max_tool_calls_without_answer: int = Field(
        default=40,
        ge=1,
        description="Tool-call count in one run before injecting wrap-up guidance.",
    )
    wrap_up_when_recursion_remaining: int = Field(
        default=8,
        ge=1,
        description="Approximate remaining LangGraph super-step budget at which to inject wrap-up guidance.",
    )
    min_new_info_chars: int = Field(
        default=120,
        ge=1,
        description="Minimum non-error text length that counts as new information without another salient signal.",
    )
    force_wrap_up_on_non_interactive_stagnation: bool = Field(
        default=True,
        description="Whether stalled non-interactive runs should be guided to complete without asking for clarification.",
    )

    @model_validator(mode="after")
    def validate_thresholds(self) -> "DynamicLoopConfig":
        if self.stagnation_hard_limit < self.stagnation_warn_steps:
            raise ValueError("stagnation_hard_limit must be >= stagnation_warn_steps")
        return self
