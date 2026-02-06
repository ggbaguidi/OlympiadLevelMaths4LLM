from __future__ import annotations

"""Helpers for dealing with python tool timeouts.

These are dependency-free utilities that can be unit-tested without importing
openai_harmony or spinning up a Jupyter kernel.

We support two mechanisms:
1) A timeout directive in code:
   - First non-empty line like: "# timeout: 120" or "#timeout=120"
2) Parsing our sandbox timeout error string:
   - "[ERROR] Execution timed out after 60.0s. TIP: For expensive computations, add '# timeout: 120' as the FIRST line of your code."
"""

import re
from typing import Optional


_TIMEOUT_DIRECTIVE_RE = re.compile(
    r"^\s*#\s*timeout\s*[:=]\s*(?P<s>\d+(?:\.\d+)?)\s*$", re.IGNORECASE
)
_TIMEOUT_ERROR_RE = re.compile(
    # Accept both formats:
    #   "... after 60.0s" (current sandbox)
    #   "... after 60.0 seconds" (legacy / external kernels)
    r"Execution\s+timed\s+out\s+after\s+(?P<s>\d+(?:\.\d+)?)\s*(?:s|sec(?:ond)?s?)\b\.?",
    re.IGNORECASE,
)


def parse_timeout_directive(code: str, *, max_scan_lines: int = 5) -> Optional[float]:
    if not code:
        return None
    lines = code.splitlines()
    for line in lines[: max(0, int(max_scan_lines))]:
        if not line.strip():
            continue
        m = _TIMEOUT_DIRECTIVE_RE.match(line)
        if not m:
            return None
        try:
            return float(m.group("s"))
        except Exception:
            return None
    return None


def parse_timeout_error(text: str) -> Optional[float]:
    if not text:
        return None
    m = _TIMEOUT_ERROR_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group("s"))
    except Exception:
        return None
