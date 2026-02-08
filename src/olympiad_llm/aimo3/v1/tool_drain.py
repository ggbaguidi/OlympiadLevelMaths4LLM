from __future__ import annotations

"""Utilities for handling multiple tool calls emitted in one assistant completion.

Some models can emit multiple tool-call messages before the runtime returns any
results. If the orchestrator only executes the *last* tool call, subsequent code
may assume variables from earlier calls exist and crash.

This module is intentionally dependency-free and works on duck-typed messages.
"""

from dataclasses import dataclass
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class ToolCall:
    recipient: str
    text: str
    message: Any


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _get_first_text_content(message: Any) -> str:
    # Harmony messages typically have message.content: list[TextContent]
    content = _get_attr(message, "content", None)
    if not content:
        return ""
    try:
        first = content[0]
    except Exception:
        return ""
    txt = _get_attr(first, "text", "")
    return str(txt or "")


def iter_tool_calls(messages: Iterable[Any], *, recipient: str) -> Iterable[ToolCall]:
    for m in messages:
        if _get_attr(m, "recipient", None) != recipient:
            continue
        yield ToolCall(recipient=recipient, text=_get_first_text_content(m), message=m)


def drain_tool_calls(
    messages: list[Any],
    *,
    recipient: str,
    execute: Callable[[Any], Any],
    call_cap: int | None = None,
) -> list[Any]:
    """Execute all tool calls for `recipient` in message order.

    - `execute(message)` should perform the tool call and return an output object
      (often a tool message or list of tool messages).
    - If `call_cap` is not None, raise RuntimeError when cap would be exceeded.

    Returns a list of outputs in the same order as tool calls.
    """

    outputs: list[Any] = []
    calls = 0
    for call in iter_tool_calls(messages, recipient=recipient):
        if call_cap is not None and (calls + 1) > int(call_cap):
            raise RuntimeError("tool_call_cap_exceeded")
        outputs.append(execute(call.message))
        calls += 1
    return outputs
