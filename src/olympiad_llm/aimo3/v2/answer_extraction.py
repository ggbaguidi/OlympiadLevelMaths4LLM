# pylint: disable=broad-exception-caught,missing-function-docstring,line-too-long,missing-module-docstring,import-outside-toplevel,invalid-name,too-many-instance-attributes
from __future__ import annotations

import re
from dataclasses import dataclass

_LATEX_WRAPPER_RE = re.compile(
    r"^\s*\\(?P<cmd>text|mathrm|mathbf|textbf|bf|operatorname)\s*\{(?P<body>.*)\}\s*$",
    flags=re.DOTALL,
)

_LATEX_SPACING_RE = re.compile(r"\\[,;!]|\\quad|\\qquad")

_STRICT_INT_RE = re.compile(r"^\s*([+-]?[0-9][0-9,]*)\s*$")
_STRICT_INTISH_FLOAT_RE = re.compile(r"^\s*([+-]?[0-9][0-9,]*)\.0+\s*$")
_SIMPLE_FRAC_RE = re.compile(
    r"^\s*\\frac\s*\{\s*([+-]?[0-9][0-9,]*)\s*\}\s*\{\s*([+-]?[0-9][0-9,]*)\s*\}\s*$"
)
_OPERATOR_ANS_RE = re.compile(
    r"\\operatorname\s*\{\s*(?:ans|answer)\s*\}\s*\(\s*([+-]?[0-9][0-9,]*)\s*\)",
    flags=re.IGNORECASE,
)
_INT_TOKEN_RE = re.compile(r"([+-]?[0-9][0-9,]*)")

# Fallback patterns when the model forgets boxing.
#
# Common failure mode: the model outputs something like
#   "Final answer is $1,234$."
#   "Final Answer: **1234**"
#   "final answer is \(1234\)"
#   "Final answer is \boxed{1234}" (boxing will be handled earlier, but keep this robust).
#   "Thus, the answer is 1234."
_FINAL_INT_HINT_RE = re.compile(
    r"(?:final(?:\s+answer)?|thus\s+(?:the\s+)?answer|answer|ans)\s*"  # hint
    r"(?:should\s+be|must\s+be|is|=|:)?\s*"  # connector
    r"(?:(?:indeed|exactly|simply|just)\s+)*"  # optional emphasis words
    r"(?:\\boxed\s*\{\s*)?"  # optional \boxed{
    r"(?:\\text\s*\{\s*)?"  # optional \text{
    r"(?:\*\*|\$|\\\(|\\\[)?\s*"  # optional markdown/LaTeX opener
    r"([+-]?[0-9][0-9,]*)",  # capture integer
    flags=re.IGNORECASE,
)
_TEXT_ANSWER_TRAIL_INT_RE = re.compile(
    r"\\text\s*\{[^}]*?(?:final|answer|ans)[^}]*\}\s*([+-]?[0-9][0-9,]*)",
    flags=re.IGNORECASE,
)
_LATEX_WRAPPED_INT_RE = re.compile(
    r"\\(?:\(|\[)\s*(?:\\displaystyle\s*)?([+-]?[0-9][0-9,]*)\s*\\(?:\)|\])",
    flags=re.IGNORECASE,
)
_LATEX_WRAPPED_SEGMENT_RE = re.compile(
    r"\\(?:\(|\[)\s*(.*?)\s*\\(?:\)|\])",
    flags=re.IGNORECASE | re.DOTALL,
)
_DOLLAR_MATH_SEGMENT_RE = re.compile(
    r"\${1,2}\s*(.*?)\s*\${1,2}",
    flags=re.DOTALL,
)
_ANY_INT_RE = re.compile(r"\b([+-]?[0-9][0-9,]*)\b")


def _iter_braced_command_contents(text: str, command: str) -> list[str]:
    """Return contents of all occurrences of '\\{command}{...}'.

    Uses a small brace-matching parser (regex breaks on nested braces).
    """

    t = text or ""
    out: list[str] = []
    i = 0
    n = len(t)
    needle = f"\\{command}"
    while i < n:
        j = t.find(needle, i)
        if j < 0:
            break

        k = j + len(needle)
        # Skip optional whitespace.
        while k < n and t[k].isspace():
            k += 1
        if k >= n or t[k] != "{":
            i = j + 1
            continue

        # Parse balanced braces starting at this '{'.
        depth = 0
        start = None
        end = None
        for p in range(k, n):
            ch = t[p]
            if ch == "{":
                depth += 1
                if depth == 1:
                    start = p + 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = p
                    break
                if depth < 0:
                    break
        if start is not None and end is not None and end >= start:
            out.append(t[start:end])
            i = end + 1
        else:
            i = j + 1

    return out


def _iter_boxed_contents(text: str) -> list[str]:
    """Return contents of all common boxed answer forms."""

    out: list[str] = []
    for cmd in ("boxed", "fbox", "mbox"):
        out.extend(_iter_braced_command_contents(text, cmd))
    return out


def _clean_box_content(content: str) -> str:
    """Best-effort cleanup for common LaTeX wrappers/spacers around integers."""

    c = (content or "").strip()

    # Remove common spacing commands.
    c = _LATEX_SPACING_RE.sub(" ", c)
    c = c.replace("\\displaystyle", " ")
    c = c.replace("\\left", " ")
    c = c.replace("\\right", " ")
    c = c.replace("{,}", ",")

    # Iteratively unwrap common text wrappers: \text{123} -> 123.
    # (This also helps with \mathrm{...}, \mathbf{...}, etc.)
    for _ in range(5):
        m = _LATEX_WRAPPER_RE.match(c)
        if not m:
            break
        c = (m.group("body") or "").strip()

    # Strip outer braces if someone wrote \boxed{{123}}.
    for _ in range(3):
        cc = c.strip()
        if cc.startswith("{") and cc.endswith("}"):
            c = cc[1:-1].strip()
        else:
            break

    return c


def _extract_int_from_fragment_with_rule(fragment: str) -> tuple[int | None, str | None]:
    """Best-effort parse of an integer-like answer from a fragment with rule label."""

    c = _clean_box_content(fragment)
    if not c:
        return None, None

    # Pattern: \operatorname{Ans}(1234)
    m = _OPERATOR_ANS_RE.search(c)
    if m:
        raw = (m.group(1) or "").replace(",", "")
        with_value = None
        try:
            with_value = int(raw)
        except ValueError:
            with_value = None
        if with_value is not None:
            return with_value, "operator_ans"

    # Unwrap common outer delimiters repeatedly.
    for _ in range(4):
        cc = c.strip()
        if cc.startswith("(") and cc.endswith(")") and len(cc) >= 2:
            c = cc[1:-1].strip()
            continue
        if cc.startswith("{") and cc.endswith("}") and len(cc) >= 2:
            c = cc[1:-1].strip()
            continue
        break

    m = _STRICT_INT_RE.match(c)
    if m:
        raw = (m.group(1) or "").replace(",", "")
        try:
            return int(raw), "strict_int"
        except ValueError:
            pass

    # Accept integer-looking floats like 98449.0
    m = _STRICT_INTISH_FLOAT_RE.match(c)
    if m:
        raw = (m.group(1) or "").replace(",", "")
        try:
            return int(raw), "intish_float"
        except ValueError:
            pass

    # Accept simple exact integer fractions like \frac{196898}{2}
    m = _SIMPLE_FRAC_RE.match(c)
    if m:
        num_raw = (m.group(1) or "").replace(",", "")
        den_raw = (m.group(2) or "").replace(",", "")
        try:
            num = int(num_raw)
            den = int(den_raw)
        except ValueError:
            num = den = 0
        if den != 0 and num % den == 0:
            return num // den, "simple_fraction"

    # Fallback: choose the longest integer token in the fragment.
    toks = _INT_TOKEN_RE.findall(c)
    if toks:
        toks_sorted = sorted(
            enumerate(toks),
            key=lambda item: (len(item[1].replace(",", "").lstrip("+-")), item[0]),
        )
        cand = toks_sorted[-1][1].replace(",", "")
        try:
            return int(cand), "longest_int_token"
        except ValueError:
            return None, None

    return None, None


def _extract_int_from_fragment(fragment: str) -> int | None:
    val, _rule = _extract_int_from_fragment_with_rule(fragment)
    return val


@dataclass(frozen=True)
class AnswerExtractor:
    """Extract answers from model text.

    The AIMO scoring format expects an integer in a box: \boxed{n}.
    """

    aimo_lo: int = 0
    aimo_hi: int = 99999
    strict_fallback: bool = True  # Only use hint-based fallback, not any integer

    def extract_boxed_int_with_rule(self, text: str) -> tuple[int | None, str | None]:
        """Return (value, rule) for last valid boxed answer in range."""

        contents = _iter_boxed_contents(text or "")
        if not contents:
            return None, None

        for raw_content in reversed(contents):
            val, rule = _extract_int_from_fragment_with_rule(raw_content)
            if val is None:
                continue
            if self.aimo_lo <= val <= self.aimo_hi:
                return val, f"boxed:{rule or 'unknown'}"

        return None, None

    def extract_boxed_int(self, text: str) -> int | None:
        """Return the last valid boxed int in [aimo_lo, aimo_hi], else None."""
        val, _rule = self.extract_boxed_int_with_rule(text)
        return val

    def extract_int_fallback_with_rule(self, text: str) -> tuple[int | None, str | None]:
        """Return (value, rule) for likely final integer answer if unboxed."""

        t = text or ""
        hinted = list(_FINAL_INT_HINT_RE.findall(t))
        hinted_text_wrapper = list(_TEXT_ANSWER_TRAIL_INT_RE.findall(t))
        tail = t[-2000:]
        latex_wrapped = list(_LATEX_WRAPPED_INT_RE.findall(tail))
        latex_wrapped_segments = list(_LATEX_WRAPPED_SEGMENT_RE.findall(tail))
        dollar_math_segments = list(_DOLLAR_MATH_SEGMENT_RE.findall(tail))

        candidates: list[tuple[str, str]]
        if self.strict_fallback:
            candidates = (
                [(x, "hint_final_int") for x in hinted]
                + [(x, "hint_text_answer_trail") for x in hinted_text_wrapper]
                + [(x, "latex_wrapped_int") for x in latex_wrapped]
                + [(x, "latex_wrapped_segment") for x in latex_wrapped_segments]
                + [(x, "dollar_math_segment") for x in dollar_math_segments]
            )
        else:
            if (
                hinted
                or hinted_text_wrapper
                or latex_wrapped
                or latex_wrapped_segments
                or dollar_math_segments
            ):
                candidates = (
                    [(x, "hint_final_int") for x in hinted]
                    + [(x, "hint_text_answer_trail") for x in hinted_text_wrapper]
                    + [(x, "latex_wrapped_int") for x in latex_wrapped]
                    + [(x, "latex_wrapped_segment") for x in latex_wrapped_segments]
                    + [(x, "dollar_math_segment") for x in dollar_math_segments]
                )
            else:
                candidates = [(x, "any_int") for x in _ANY_INT_RE.findall(t)]

        if not candidates:
            return None, None

        for raw, source in reversed(candidates):
            val, rule = _extract_int_from_fragment_with_rule(raw)
            if val is None:
                continue
            if self.aimo_lo <= val <= self.aimo_hi:
                return val, f"fallback:{source}:{rule or 'unknown'}"
        return None, None

    def extract_int_fallback(self, text: str) -> int | None:
        """Return a likely final integer answer even if unboxed.

        Strategy (general, conservative):
        1) Prefer integers that appear near an "answer" hint.
        2) Otherwise fall back to the last integer in-range in the text.

        This helps recover from formatting failures without changing the model prompt.
        """

        val, _rule = self.extract_int_fallback_with_rule(text)
        return val
