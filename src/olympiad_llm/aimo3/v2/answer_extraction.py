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
    r"(?:final|thus\s+answer|answer|ans)\s*(?:is|=|:)?\s*"  # hint
    r"(?:\\boxed\s*\{\s*)?"  # optional \boxed{
    r"(?:\\text\s*\{\s*)?"  # optional \text{
    r"(?:\*\*|\$|\\\(|\\\[)?\s*"  # optional markdown/LaTeX opener
    r"([+-]?[0-9][0-9,]*)",  # capture integer
    flags=re.IGNORECASE,
)
_ANY_INT_RE = re.compile(r"\b([+-]?[0-9][0-9,]*)\b")


def _iter_boxed_contents(text: str) -> list[str]:
    """Return contents of all occurrences of \boxed{...}.

    Uses a small brace-matching parser (regex breaks on nested braces).
    """

    t = text or ""
    out: list[str] = []
    i = 0
    n = len(t)
    needle = "\\boxed"
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


def _clean_box_content(content: str) -> str:
    """Best-effort cleanup for common LaTeX wrappers/spacers around integers."""

    c = (content or "").strip()

    # Remove common spacing commands.
    c = _LATEX_SPACING_RE.sub(" ", c)
    c = c.replace("\\displaystyle", " ")
    c = c.replace("\\left", " ")
    c = c.replace("\\right", " ")

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


@dataclass(frozen=True)
class AnswerExtractor:
    """Extract answers from model text.

    The AIMO scoring format expects an integer in a box: \boxed{n}.
    """

    aimo_lo: int = 0
    aimo_hi: int = 99999
    strict_fallback: bool = True  # Only use hint-based fallback, not any integer

    def extract_boxed_int(self, text: str) -> int | None:
        """Return the last valid \boxed{int} in [aimo_lo, aimo_hi], else None."""

        contents = _iter_boxed_contents(text or "")
        if not contents:
            return None

        for raw_content in reversed(contents):
            cleaned = _clean_box_content(raw_content)

            # Strict path: the cleaned content is just an integer.
            m = _STRICT_INT_RE.match(cleaned)
            if m:
                raw = (m.group(1) or "").replace(",", "")
                try:
                    val = int(raw)
                except ValueError:
                    val = None
                if val is not None and self.aimo_lo <= val <= self.aimo_hi:
                    return val

            # Fallback path: pull the most plausible integer token from the content.
            toks = _INT_TOKEN_RE.findall(cleaned)
            if not toks:
                continue
            # Prefer the longest token (e.g., avoid picking the trailing "5" in "10^{5}").
            toks_sorted = sorted(
                enumerate(toks),
                key=lambda item: (len(item[1].replace(",", "").lstrip("+-")), item[0]),
            )
            cand = toks_sorted[-1][1]
            cand2 = cand.replace(",", "")
            try:
                val = int(cand2)
            except ValueError:
                continue
            if self.aimo_lo <= val <= self.aimo_hi:
                return val

        return None

    def extract_int_fallback(self, text: str) -> int | None:
        """Return a likely final integer answer even if unboxed.

        Strategy (general, conservative):
        1) Prefer integers that appear near an "answer" hint.
        2) Otherwise fall back to the last integer in-range in the text.

        This helps recover from formatting failures without changing the model prompt.
        """

        t = text or ""
        hinted = list(_FINAL_INT_HINT_RE.findall(t))
        # If strict_fallback is enabled, only use hint-based matches (avoid random integers)
        if self.strict_fallback:
            candidates = hinted
        else:
            candidates = hinted if hinted else list(_ANY_INT_RE.findall(t))
        if not candidates:
            return None

        # Choose the *last* in-range integer (closest to the end of the response).
        for raw in reversed(candidates):
            raw2 = raw.replace(",", "")
            try:
                val = int(raw2)
            except ValueError:
                continue
            if self.aimo_lo <= val <= self.aimo_hi:
                return val
        return None

    def extract_boxed_content(self, text: str) -> str | None:
        """Return the *content* of the last \boxed{...} (not necessarily int)."""

        contents = _iter_boxed_contents(text or "")
        if not contents:
            return None
        return contents[-1].strip()

    def normalize_final_answer_text(self, text: str) -> str:
        """If the answer contains \boxed{...}, return the boxed content; else return stripped text."""

        boxed = self.extract_boxed_content(text)
        return boxed if boxed is not None else (text or "").strip()
