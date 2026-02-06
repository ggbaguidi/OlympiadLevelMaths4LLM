from __future__ import annotations

"""Best-effort rewrites for python-tool snippets.

Goal: make the python tool more robust to version-dependent APIs by applying
safe, minimal, *stdlib-only* rewrites before executing code in the sandbox.

This module is deliberately conservative:
- prefers token-based rewriting over naive string replacement
- avoids touching strings/comments

Current rewrites:
 Replace `sp.valuation` with a small compatibility wrapper `_aimo3_valuation`
  that tries the integer p-adic valuation first, then falls back to
  `prime_valuation` for numberfield contexts.

 Replace `sp.Circle(A,B,C)` with `_aimo3_circle3(A,B,C)` where `_aimo3_circle3`
    attempts to call `sp.Circle.from_three_points` if available, otherwise falls
    back to `sp.Circle(A,B,C)`.
"""

import io
import tokenize
from typing import Iterable


_AIMO3_VALUATION_ALIAS = "_aimo3_valuation"
_AIMO3_INT_VALUATION_ALIAS = "_aimo3_int_valuation"
_AIMO3_PRIME_VALUATION_ALIAS = "_aimo3_prime_valuation"

_AIMO3_CIRCLE3_ALIAS = "_aimo3_circle3"


# Auto-import mappings for common bare names that models often use without imports.
# Maps bare NAME token -> import statement to inject.
_AUTO_IMPORTS: dict[str, str] = {
    # itertools
    "combinations": "from itertools import combinations",
    "permutations": "from itertools import permutations",
    "product": "from itertools import product",
    "combinations_with_replacement": "from itertools import combinations_with_replacement",
    "chain": "from itertools import chain",
    "groupby": "from itertools import groupby",
    "accumulate": "from itertools import accumulate",
    # collections
    "Counter": "from collections import Counter",
    "defaultdict": "from collections import defaultdict",
    "deque": "from collections import deque",
    "OrderedDict": "from collections import OrderedDict",
    # math
    "gcd": "from math import gcd",
    "lcm": "from math import lcm",
    "factorial": "from math import factorial",
    "comb": "from math import comb",
    "perm": "from math import perm",
    "isqrt": "from math import isqrt",
    "ceil": "from math import ceil",
    "floor": "from math import floor",
    "log": "from math import log",
    "log2": "from math import log2",
    "log10": "from math import log10",
    "sqrt": "from math import sqrt",
    # sympy number theory
    "crt": "from sympy.ntheory.modular import crt",
    "mod_inverse": "from sympy import mod_inverse",
    "divisors": "from sympy import divisors",
    "factorint": "from sympy import factorint",
    "isprime": "from sympy import isprime",
    "nextprime": "from sympy import nextprime",
    "primerange": "from sympy import primerange",
    "primefactors": "from sympy import primefactors",
    "totient": "from sympy import totient",
    "mobius": "from sympy import mobius",
    # functools
    "reduce": "from functools import reduce",
    "lru_cache": "from functools import lru_cache",
    "cache": "from functools import cache",
    # fractions
    "Fraction": "from fractions import Fraction",
    # heapq
    "heappush": "from heapq import heappush",
    "heappop": "from heapq import heappop",
    "heapify": "from heapq import heapify",
    # bisect
    "bisect_left": "from bisect import bisect_left",
    "bisect_right": "from bisect import bisect_right",
}

_AIMO3_VALUATION_BLOCK = (
    f"from sympy.polys.numberfields import prime_valuation as {_AIMO3_PRIME_VALUATION_ALIAS}\n"
    f"def {_AIMO3_INT_VALUATION_ALIAS}(a, p):\n"
    '    """Best-effort integer p-adic valuation.\n\n'
    "    Returns the exponent $v$ such that $p^v$ divides $a$.\n"
    "    For $a=0$, returns sympy.oo (infinite valuation).\n\n"
    "    This is implemented directly to avoid relying on SymPy's internal\n"
    "    helper locations, which have changed across versions.\n"
    '    """\n'
    "    import sympy as sp\n"
    "    a = sp.Integer(a)\n"
    "    p = sp.Integer(p)\n"
    "    if a == 0:\n"
    "        return sp.oo\n"
    "    if p in (0, 1, -1):\n"
    "        return 0\n"
    "    k = 0\n"
    "    while a % p == 0:\n"
    "        a //= p\n"
    "        k += 1\n"
    "    return k\n"
    f"def {_AIMO3_VALUATION_ALIAS}(a, p):\n"
    '    """Compatibility valuation helper.\n\n'
    "    - If p is an integer-like (int or sympy.Integer), use integer p-adic valuation.\n"
    "    - Otherwise, fall back to SymPy's numberfield prime_valuation.\n"
    '    """\n'
    "    try:\n"
    "        import sympy as sp\n"
    "        if isinstance(p, (int, sp.Integer)):\n"
    "            return _aimo3_int_valuation(a, int(p))\n"
    "    except Exception:\n"
    "        pass\n"
    f"    return {_AIMO3_PRIME_VALUATION_ALIAS}(a, p)\n"
)


_AIMO3_CIRCLE3_BLOCK = (
    f"def {_AIMO3_CIRCLE3_ALIAS}(p, q, r):\n"
    '    """Compatibility helper for circles through three points.\n\n'
    "    Uses `Circle.from_three_points` when available; otherwise falls back to\n"
    "    `Circle(p,q,r)` which is supported on many SymPy versions.\n"
    '    """\n'
    "    try:\n"
    "        import sympy as sp\n"
    "        Circle = sp.Circle\n"
    "        if hasattr(Circle, 'from_three_points'):\n"
    "            try:\n"
    "                return Circle.from_three_points(p, q, r)\n"
    "            except Exception:\n"
    "                pass\n"
    "        return Circle(p, q, r)\n"
    "    except Exception:\n"
    "        # If SymPy isn't available for some reason, fail loudly.\n"
    "        return sp.Circle(p, q, r)\n"
)


def _is_name(tok: tokenize.TokenInfo, s: str) -> bool:
    return tok.type == tokenize.NAME and tok.string == s


def _is_op(tok: tokenize.TokenInfo, s: str) -> bool:
    return tok.type == tokenize.OP and tok.string == s


def _insert_import_after_header_lines(code: str, import_line: str) -> str:
    """Insert `import_line` after leading blank/comment lines.

    This keeps common headers like:
    - encoding comment
    - `# timeout: N`
    - other comments

    If the import already exists (exact line match), return unchanged.
    """

    if import_line in code:
        return code

    lines = code.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s == "" or s.startswith("#"):
            i += 1
            continue
        break

    insert = import_line + ("\n" if (not import_line.endswith("\n")) else "")
    return "".join(lines[:i] + [insert] + lines[i:])


def _find_bare_names_needing_imports(src: str) -> set[str]:
    """Find bare NAME tokens that match _AUTO_IMPORTS keys and likely need importing.

    This uses tokenization to avoid matching names in strings/comments.
    We also check that the name is not:
    - preceded by '.' (attribute access)
    - preceded by 'def' or 'class' (function/class definition)
    - followed by '=' (assignment target)
    """

    needed: set[str] = set()

    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except tokenize.TokenError:
        return needed

    # Track names that appear to be defined.
    # - Names followed by '=' (assignment)
    # - Names preceded by 'def' or 'class' (function/class definition)
    defined_names: set[str] = set()

    i = 0
    while i < len(toks):
        t = toks[i]
        if t.type == tokenize.NAME and t.string in _AUTO_IMPORTS:
            # Check if preceded by '.' (attribute access) -> skip entirely.
            if i > 0 and toks[i - 1].type == tokenize.OP and toks[i - 1].string == ".":
                i += 1
                continue
            # Check if preceded by 'def' or 'class' (function/class definition) -> mark as defined.
            if (
                i > 0
                and toks[i - 1].type == tokenize.NAME
                and toks[i - 1].string in ("def", "class")
            ):
                defined_names.add(t.string)
                i += 1
                continue
            # Check if followed by '=' (assignment target) -> mark as defined.
            if (
                i + 1 < len(toks)
                and toks[i + 1].type == tokenize.OP
                and toks[i + 1].string == "="
            ):
                defined_names.add(t.string)
            else:
                needed.add(t.string)
        i += 1

    # Remove names that are defined in the code.
    return needed - defined_names


def _inject_missing_imports(code: str) -> str:
    """Inject missing imports for common bare names.

    Scans the code for bare NAME tokens that match _AUTO_IMPORTS keys,
    and injects the corresponding import statements if not already present.
    """

    needed = _find_bare_names_needing_imports(code)
    if not needed:
        return code

    result = code
    for name in sorted(needed):  # Sort for deterministic output.
        import_stmt = _AUTO_IMPORTS[name]
        # Only inject if not already present.
        if import_stmt not in result:
            result = _insert_import_after_header_lines(result, import_stmt)

    return result


def _rewrite_sp_valuation_tokens(src: str) -> str:
    """Replace token sequence `sp . valuation` with `_aimo3_valuation`."""

    # Tokenize / untokenize preserves formatting reasonably well.
    out_tokens: list[tokenize.TokenInfo] = []
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))

    i = 0
    while i < len(toks):
        t = toks[i]

        # Look for NAME 'sp' OP '.' NAME 'valuation'
        if (
            t.type == tokenize.NAME
            and t.string == "sp"
            and i + 2 < len(toks)
            and toks[i + 1].type == tokenize.OP
            and toks[i + 1].string == "."
            and toks[i + 2].type == tokenize.NAME
            and toks[i + 2].string == "valuation"
        ):
            # Replace the 3-token sequence with a single NAME token.
            out_tokens.append(
                tokenize.TokenInfo(
                    type=tokenize.NAME,
                    string=_AIMO3_VALUATION_ALIAS,
                    start=t.start,
                    end=toks[i + 2].end,
                    line=t.line,
                )
            )
            i += 3
            continue

        out_tokens.append(t)
        i += 1

    return tokenize.untokenize(out_tokens)


def _rewrite_prime_valuation_alias_to_wrapper(src: str) -> str:
    """Rewrite `_aimo3_prime_valuation` calls to `_aimo3_valuation`.

    Backward compatibility: older rewrites emitted `_aimo3_prime_valuation(...)`.
    That alias crashes on integer valuation use-cases; the wrapper dispatches.
    """

    if _AIMO3_PRIME_VALUATION_ALIAS not in src:
        return src

    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    out: list[tokenize.TokenInfo] = []
    for t in toks:
        if t.type == tokenize.NAME and t.string == _AIMO3_PRIME_VALUATION_ALIAS:
            out.append(
                tokenize.TokenInfo(
                    type=tokenize.NAME,
                    string=_AIMO3_VALUATION_ALIAS,
                    start=t.start,
                    end=t.end,
                    line=t.line,
                )
            )
        else:
            out.append(t)
    return tokenize.untokenize(out)


def _count_top_level_commas(
    tokens: list[tokenize.TokenInfo], start_i: int, end_i: int
) -> int:
    """Count commas at nesting depth 0 within (start_i, end_i) token indices."""

    depth = 0
    commas = 0
    for t in tokens[start_i:end_i]:
        if _is_op(t, "(") or _is_op(t, "[") or _is_op(t, "{"):
            depth += 1
        elif _is_op(t, ")") or _is_op(t, "]") or _is_op(t, "}"):
            depth = max(0, depth - 1)
        elif depth == 0 and _is_op(t, ","):
            commas += 1
    return commas


def _rewrite_sp_circle_three_points(src: str) -> str:
    """Rewrite `sp.Circle(A,B,C)` into `_aimo3_circle3(A,B,C)`.

    Motivation: SymPy versions differ in whether `Circle.from_three_points` exists.
    `_aimo3_circle3` tries it when available but falls back to `Circle(p,q,r)`.

    This rewrite is conservative:
    - only applies to the exact `sp.Circle(...)` call form
    - only when there are exactly 3 top-level args (2 commas)
    - does not touch strings/comments (token-based)
    - does not touch already-rewritten `_aimo3_circle3(...)`
    """

    if "sp.Circle" not in src:
        return src

    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    # Use (tok_type, tok_string) pairs for untokenize to avoid relying on positional
    # metadata when injecting new tokens.
    out: list[tuple[int, str]] = []
    changed = False
    i = 0
    while i < len(toks):
        t = toks[i]

        # Match: sp . Circle
        if (
            _is_name(t, "sp")
            and i + 2 < len(toks)
            and _is_op(toks[i + 1], ".")
            and _is_name(toks[i + 2], "Circle")
        ):
            # Only rewrite when followed by a call: (...)
            j = i + 3
            if j < len(toks) and _is_op(toks[j], "("):
                # Find matching ')'
                depth = 0
                k = j
                while k < len(toks):
                    if _is_op(toks[k], "("):
                        depth += 1
                    elif _is_op(toks[k], ")"):
                        depth -= 1
                        if depth == 0:
                            break
                    k += 1

                if k < len(toks) and _is_op(toks[k], ")"):
                    # tokens between j+1 and k are the arg list (with nesting)
                    commas = _count_top_level_commas(toks, j + 1, k)
                    if commas == 2:
                        # Emit: _aimo3_circle3
                        out.append((tokenize.NAME, _AIMO3_CIRCLE3_ALIAS))
                        changed = True
                        i += 3
                        continue

        out.append((t.type, t.string))
        i += 1

    if not changed:
        return src
    return tokenize.untokenize(out)


def rewrite_python_tool_code(code: str) -> str:
    """Apply all configured rewrites to python-tool code."""

    src = str(code or "")
    rewritten = src

    # First: inject missing imports for common bare names.
    # This runs before other rewrites to ensure imports are available.
    rewritten = _inject_missing_imports(rewritten)

    if "sp.Circle" in rewritten:
        rewritten = _rewrite_sp_circle_three_points(rewritten)

    # If helper is referenced, ensure helper block is present.
    if (
        _AIMO3_CIRCLE3_ALIAS in rewritten
        and f"def {_AIMO3_CIRCLE3_ALIAS}" not in rewritten
    ):
        rewritten = _insert_import_after_header_lines(rewritten, _AIMO3_CIRCLE3_BLOCK)

    if "sp.valuation" in rewritten:
        rewritten = _rewrite_sp_valuation_tokens(rewritten)

    # Back-compat: older rewrite emitted `_aimo3_prime_valuation(...)`.
    # Only apply if the wrapper isn't already defined in the snippet; otherwise we risk
    # rewriting user-provided helper code and breaking idempotence.
    if (
        _AIMO3_PRIME_VALUATION_ALIAS in rewritten
        and f"def {_AIMO3_VALUATION_ALIAS}" not in rewritten
    ):
        rewritten = _rewrite_prime_valuation_alias_to_wrapper(rewritten)

    # If wrapper is referenced, ensure helper block is present.
    # IMPORTANT: avoid modifying user-provided wrapper definitions; if the function is
    # already defined (even with a different body), leave it untouched for idempotence.
    if (
        _AIMO3_VALUATION_ALIAS in rewritten
        and f"def {_AIMO3_VALUATION_ALIAS}" not in rewritten
    ):
        rewritten = _insert_import_after_header_lines(rewritten, _AIMO3_VALUATION_BLOCK)

    return rewritten
