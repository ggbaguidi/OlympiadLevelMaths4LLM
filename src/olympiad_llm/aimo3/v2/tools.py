# pylint: disable=broad-exception-caught,missing-function-docstring,line-too-long,missing-module-docstring,import-outside-toplevel,invalid-name,too-many-instance-attributes
from __future__ import annotations

import re

import threading

from .sandbox import AIMO3Sandbox
from .require import _require_harmony
from .verification import ToolOutputVerifier

_TIMEOUT_DIRECTIVE_RE = re.compile(
    r"^\s*#\s*timeout\s*[:=]\s*(?P<s>\d+(?:\.\d+)?)\s*$",
    re.IGNORECASE,
)

Z3_TOOL_PROMPT = """Use this tool to execute Z3 SMT solver code for constraint solving and theorem proving.
The environment has 'from z3 import *' already imported.

When to use Z3:
- Integer/real constraints and Diophantine equations
- Combinatorial problems (scheduling, assignment)
- Proving mathematical propositions
- Finding solutions to systems of constraints

Z3 API examples:
- Integers: x = Int('x'), y = Int('y')
- Reals: x = Real('x'), y = Real('y')  
- Booleans: p = Bool('p')
- Solve: solve(x + y == 10, x > 0)
- Optimize: opt = Optimize(); opt.add(x + y == 10); opt.maximize(x)
- Check sat: s = Solver(); s.add(x**2 == 2); s.check()
- Get model: s.model()[x]

Always use print() to output results. The final answer must be a non-negative integer between 0 and 99999."""


class AIMO3Tool:
    """Bridges Harmony tool-call messages to a sandboxed Jupyter kernel, with backend selection."""

    def __init__(
        self,
        local_jupyter_timeout: float,
        tool_prompt: str,
        sandbox: AIMO3Sandbox | None = None,
        tool_timeout_cap_s: float | None = None,
        z3_enabled: bool = False,
        enable_verification: bool = True,
    ):
        self._h = _require_harmony()
        self._local_jupyter_timeout = float(local_jupyter_timeout)
        self._tool_prompt = tool_prompt
        self._tool_timeout_cap_s = (
            None if tool_timeout_cap_s is None else float(tool_timeout_cap_s)
        )
        self._jupyter_session = sandbox
        self._owns_session = sandbox is None
        self._execution_lock = threading.Lock()
        self._init_lock = threading.Lock()
        self._z3_enabled = z3_enabled
        self._enable_verification = enable_verification
        self._verifier = ToolOutputVerifier() if enable_verification else None

    def _ensure_session(self) -> None:
        if self._jupyter_session is None:
            with self._init_lock:
                if self._jupyter_session is None:
                    self._jupyter_session = AIMO3Sandbox(
                        timeout=self._local_jupyter_timeout
                    )

    def _decide_backend(self, code: str) -> str:
        """Decide which backend to use for the given code."""
        if not self._z3_enabled:
            return "python"

        lines = code.split("\n")
        first_line = lines[0].strip().lower() if lines else ""

        # Check for explicit Z3 marker in first line
        if first_line.startswith("# z3") or first_line.startswith("#z3"):
            return "z3"

        lower_code = code.lower()
        # Z3 keywords
        z3_keywords = [
            "z3.",
            "from z3 import",
            "solver(",
            "optimize(",
            "solve(",
        ]
        if any(keyword in lower_code for keyword in z3_keywords):
            return "z3"

        # SymPy keywords
        sympy_keywords = ["sympy.", "sp."]
        if any(keyword in lower_code for keyword in sympy_keywords):
            return "sympy"

        return "python"

    @staticmethod
    def _ensure_last_print(code: str) -> str:
        # Best-effort UX: if the user ends their tool snippet with a simple expression,
        # auto-wrap it in print(...). This helps avoid the common warning:
        #   [WARN] No output. Use print() to see results.
        #
        # Safety: do NOT rewrite multi-line blocks (function/class defs, loops, returns,
        # indented code, etc.) because that can change semantics or introduce SyntaxError.
        src = str(code or "")
        stripped = src.strip("\n")
        if not stripped.strip():
            return src

        lines = stripped.split("\n")
        if not lines:
            return src

        raw_last_line = lines[-1]
        last = raw_last_line.strip()
        if not last or last.startswith("#"):
            return src

        # If the last line is indented, it's almost certainly inside a block.
        indent = raw_last_line[: len(raw_last_line) - len(raw_last_line.lstrip())]
        if indent:
            return src

        # Heuristic: avoid rewriting statements/headers.
        statement_prefixes = (
            "return",
            "def ",
            "class ",
            "for ",
            "while ",
            "if ",
            "elif ",
            "else",
            "try",
            "except",
            "finally",
            "with ",
            "import ",
            "from ",
            "raise",
            "assert",
            "pass",
            "break",
            "continue",
            "yield",
            "del ",
            "global ",
            "nonlocal ",
            "@",
        )
        lower_last = last.lower()
        if lower_last.endswith(":") or lower_last.startswith(statement_prefixes):
            return src

        # Skip Jupyter magic commands (! for shell, % for magic)
        # e.g., "!pip install foo" or "%timeit foo()"
        if last.startswith("!") or last.startswith("%"):
            return src

        # If it already prints, do nothing.
        if last.startswith("print(") or "print" in last:
            return src

        # CRITICAL: Do not wrap assignment statements - this causes SyntaxError:
        # print(x = foo()) is invalid (= looks like keyword argument)
        # Check for assignment: contains '=' but not '==', '!=', '<=', '>=', '+=', etc.
        if "=" in last:
            # Match standalone = (assignment) but not compound operators
            if re.search(r"(?<![=!<>+\-*/%&|^])=(?!=)", last):
                return src

        # Strip trailing comments before wrapping - otherwise print(x # comment) is invalid
        # because the # hides the closing parenthesis
        expr = last
        comment = ""
        if "#" in last:
            # Find the comment part (not inside a string)
            # Simple heuristic: split on # and check if it's likely a comment
            hash_idx = last.find("#")
            # Check if # is inside quotes (very rough check)
            before_hash = last[:hash_idx]
            if before_hash.count('"') % 2 == 0 and before_hash.count("'") % 2 == 0:
                expr = last[:hash_idx].rstrip()
                comment = "  " + last[hash_idx:]

        if not expr:
            return src

        # Only apply to single-line snippets or when the last line is a top-level expression.
        lines[-1] = f"print({expr}){comment}"
        return "\n".join(lines)

    @staticmethod
    def _parse_timeout_directive(
        code: str | None, *, max_scan_lines: int = 5
    ) -> float | None:
        """Parse first non-empty '# timeout: N' directive (seconds)."""
        src = str(code or "")
        if not src:
            return None
        lines = src.splitlines()
        for line in lines[: max(0, int(max_scan_lines))]:
            if not line.strip():
                continue
            match = _TIMEOUT_DIRECTIVE_RE.match(line)
            if not match:
                return None
            try:
                return float(match.group("s"))
            except Exception:  # noqa: BLE001
                return None
        return None

    @property
    def instruction(self) -> str:
        return self._tool_prompt

    @property
    def tool_config(self):
        ToolNamespaceConfig = self._h["ToolNamespaceConfig"]

        if self._z3_enabled:
            # Combine both Python and Z3 info into a single tool config
            combined_description = (
                f"{self._tool_prompt}\n\n"
                f"Z3 SMT SOLVER: You can also use Z3 for constraint solving. "
                f"Best for Diophantine equations, combinatorial problems, and proving propositions.\n\n"
                f"To use Z3, prefix your code with '# z3' on the first line. "
                f"Examples: x = Int('x'); solve(x**2 == 2)"
            )
            return ToolNamespaceConfig(
                name="python", description=combined_description, tools=[]
            )

        return ToolNamespaceConfig(
            name="python", description=self._tool_prompt, tools=[]
        )

    def _make_response(
        self, output: str, channel: str | None = None, backend: str = "python"
    ):
        TextContent = self._h["TextContent"]
        Author = self._h["Author"]
        Message = self._h["Message"]
        Role = self._h["Role"]
        content = TextContent(text=output)
        author = Author(role=Role.TOOL, name=backend)
        msg = Message(author=author, content=[content]).with_recipient("assistant")
        if channel:
            msg = msg.with_channel(channel)
        return msg

    def process_sync_plus(
        self,
        message,
        timeout_override_s: float | None = None,
        expected_answer: int | None = None,
    ):
        self._ensure_session()
        raw_script = message.content[0].text

        backend = self._decide_backend(raw_script)

        # Apply best-effort rewrites for known API mismatches before execution.
        final_script = self._ensure_last_print(raw_script)

        timeout_s: float | None = None
        if timeout_override_s is not None:
            timeout_s = float(timeout_override_s)
        else:
            timeout_s = self._parse_timeout_directive(raw_script)

        if timeout_s is not None and self._tool_timeout_cap_s is not None:
            timeout_s = min(float(timeout_s), float(self._tool_timeout_cap_s))

        with self._execution_lock:
            output = self._jupyter_session.execute(final_script, timeout=timeout_s)

        # Phase 3: Enhanced verification
        verification_info = ""
        if self._enable_verification and self._verifier:
            vresult = self._verifier.verify_output(output, expected_answer)
            if not vresult.is_valid:
                verification_info = f"\n[VERIFICATION WARNING] {vresult.error_type}"
                if vresult.error_details:
                    verification_info += f": {vresult.error_details}"
            elif vresult.warnings:
                verification_info = "\n[VERIFICATION NOTICE] " + "; ".join(
                    vresult.warnings
                )

            # Append verification info to output
            if verification_info:
                output = output + verification_info

        return [
            self._make_response(
                output, channel=getattr(message, "channel", None), backend=backend
            )
        ]

    def close(self) -> None:
        if self._jupyter_session is not None and self._owns_session:
            self._jupyter_session.close()
        self._jupyter_session = None

    def __del__(self) -> None:
        self.close()
