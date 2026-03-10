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

Z3_TOOL_PROMPT = """Use this tool to run Z3 code for exact constraint solving.
The environment already has 'from z3 import *'.

Use Z3 for:
- Diophantine equations and exact integer constraints
- Combinatorial search with hard constraints
- Logical consistency or impossibility checks

Keep the model small, print only decisive results, and always use print()."""


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

        if first_line.startswith("# z3") or first_line.startswith("#z3"):
            return "z3"

        lower_code = code.lower()
        z3_keywords = [
            "z3.",
            "from z3 import",
            "solver(",
            "optimize(",
            "solve(",
        ]
        if any(keyword in lower_code for keyword in z3_keywords):
            return "z3"

        return "python"

    @staticmethod
    def _ensure_last_print(code: str) -> str:
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

        indent = raw_last_line[: len(raw_last_line) - len(raw_last_line.lstrip())]
        if indent:
            return src

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

        if last.startswith("!") or last.startswith("%"):
            return src

        if last.startswith("print(") or "print" in last:
            return src

        if "=" in last:
            if re.search(r"(?<![=!<>+\-*/%&|^])=(?!=)", last):
                return src

        expr = last
        comment = ""
        if "#" in last:
            hash_idx = last.find("#")
            before_hash = last[:hash_idx]
            if before_hash.count('"') % 2 == 0 and before_hash.count("'") % 2 == 0:
                expr = last[:hash_idx].rstrip()
                comment = "  " + last[hash_idx:]

        if not expr:
            return src

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
            combined_description = (
                f"{self._tool_prompt}\n\n"
                f"Z3 is also available for exact constraints and combinatorial search. "
                f"To use it, put '# z3' on the first line."
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

    def _augment_output_with_verification(
        self, output: str, expected_answer: int | None
    ) -> str:
        if not self._enable_verification or self._verifier is None:
            return output

        try:
            verdict = self._verifier.verify_output(
                output, expected_answer=expected_answer
            )
        except Exception:  # noqa: BLE001
            return output

        notices: list[str] = [
            (
                "[VERIFICATION NOTICE] TOOL_OUTPUT_VALID"
                if verdict.is_valid
                else "[VERIFICATION NOTICE] TOOL_OUTPUT_INVALID"
            )
        ]
        if verdict.error_type:
            notices.append(f"[VERIFICATION NOTICE] ERROR_TYPE={verdict.error_type}")
        if verdict.warnings:
            notices.append(
                "[VERIFICATION NOTICE] WARNINGS=" + "; ".join(verdict.warnings[:3])
            )

        if expected_answer is not None and verdict.numerical_value is not None:
            try:
                if abs(float(verdict.numerical_value) - float(expected_answer)) < 1e-9:
                    notices.append("VERIFY_OK")
                else:
                    notices.append("VERIFY_FAIL")
            except Exception:  # noqa: BLE001
                pass

        suffix = "\n".join(notices)
        if not suffix:
            return output
        if output and not output.endswith("\n"):
            return output + "\n" + suffix
        return output + suffix

    def process_sync_plus(
        self,
        message,
        timeout_override_s: float | None = None,
        expected_answer: int | None = None,
    ):
        self._ensure_session()
        raw_script = message.content[0].text

        backend = self._decide_backend(raw_script)

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
        output = self._augment_output_with_verification(output, expected_answer)

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
