from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttemptStats:
    """Lightweight stats about a single attempt/run."""

    token_count: int = 0
    python_calls: int = 0
    python_errors: int = 0

    @property
    def tool_verified(self) -> bool:
        """Heuristic: attempt used the tool and tool produced no errors."""

        return self.python_calls > 0 and self.python_errors == 0


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
