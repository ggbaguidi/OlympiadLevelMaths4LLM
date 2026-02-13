# pylint: disable=broad-exception-caught,missing-function-docstring,line-too-long,missing-module-docstring,import-outside-toplevel,invalid-name,too-many-instance-attributes
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttemptStats:
    """Lightweight stats about a single attempt/run."""

    token_count: int = 0
    python_calls: int = 0
    python_errors: int = 0
    # Heuristic count of python tool calls that likely invoked Lean/Lake.
    lean_calls: int = 0
    # Count of python tool calls that timed out.
    timeout_count: int = 0
    # Whether the attempt was cut off by the absolute per-problem deadline.
    deadline_exceeded: bool = False

    # Optional confidence proxy from model logprobs.
    # When enabled, the solver computes a mean per-token entropy (lower is more confident).
    # Default is +inf meaning "unknown / not computed".
    mean_entropy: float = float("inf")

    # Optional: if set, require a verification marker in tool output to count as verified.
    # None => use the legacy heuristic (python_calls>0 and python_errors==0).
    verification_marker_found: bool | None = None

    # Phase 3: Enhanced verification fields
    # Whether the tool output passed enhanced verification
    tool_output_verified: bool = False
    # Confidence score from verification (0.0 to 1.0)
    verification_confidence: float = 0.0
    # Type of error detected during verification (if any)
    verification_error_type: str | None = None
    # List of warnings from verification
    verification_warnings: list[str] | None = None
    # Numerical value extracted from tool output (if any)
    extracted_numerical_value: float | int | None = None

    # Last error encountered during tool execution (if any).
    last_error: str | None = None

    @property
    def tool_verified(self) -> bool:
        """Heuristic: attempt used the tool and tool produced no errors OR explicitly verified."""
        # If the model explicitly said "VERIFY_OK", trust it even if there were errors.
        if self.verification_marker_found:
            return True

        # Phase 3: Enhanced verification takes precedence
        if self.tool_output_verified:
            return True

        # Legacy/Fallback: requires no errors.
        if self.python_calls <= 0 or self.python_errors > 0:
            return False

        if self.verification_marker_found is None:
            return True
        return bool(self.verification_marker_found)

    @property
    def had_timeout(self) -> bool:
        """Heuristic: attempt had at least one timeout."""

        return self.timeout_count > 0


@dataclass(frozen=True)
class AttemptResult:
    """Shared schema for an attempt.

    Different runners (mock chat, OpenAI/vLLM, Kaggle) can populate these fields.
    """

    attempt: int
    answer: int | str | None
    stats: AttemptStats = AttemptStats()
    # Optional: final assistant text (or last chunk) for auditing.
    output_text: str | None = None
    # Optional: label for the attempt (e.g., which prompt family / strategy pack).
    tag: str | None = None
