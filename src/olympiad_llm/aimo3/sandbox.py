from __future__ import annotations

import contextlib
import os
import queue
import re
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass

from .errors import OptionalDependencyError


def _require_jupyter_client():
    try:
        from jupyter_client import KernelManager  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise OptionalDependencyError(
            "AIMO3Sandbox requires 'jupyter_client'. Install extras: pip install .[aimo3]"
        ) from e
    return KernelManager


@dataclass
class SandboxExecResult:
    stdout: str
    stderr: str
    timed_out: bool = False


class AIMO3Sandbox:
    """Persistent Python execution sandbox using a Jupyter kernel.

    This mirrors the notebook approach but is packaged for reuse.
    """

    def __init__(self, timeout: float):
        KernelManager = _require_jupyter_client()

        self._default_timeout = float(timeout)
        self._owns_kernel = False
        self._client = None
        self._km = None
        self._connection_file: str | None = None

        # IMPORTANT: don't hardcode ports.
        # In managed notebook runtimes (Kaggle/Colab), fixed port ranges are often already in use.
        # Let jupyter_client pick ephemeral free ports.
        #
        # Still, rare port-collision races can occur (or stale kernels may hold ports after Ctrl-C).
        # Be robust: retry a few times on startup failures.
        max_start_attempts = 6
        last_err: Exception | None = None
        for attempt in range(1, max_start_attempts + 1):
            km = None
            client = None
            connection_file = None
            try:
                # IMPORTANT: KernelManager defaults to a connection_file name derived from the
                # *current* process PID (e.g. kernel-<pid>.json). If we start multiple kernels
                # concurrently from one Python process (as we do when filling the sandbox pool),
                # those KernelManagers can race/overwrite the same connection file, causing the
                # child kernels to pick up identical ports and fail with "Address already in use".
                #
                # Fix: always use a unique connection file per sandbox instance.
                connection_file = os.path.join(
                    tempfile.gettempdir(),
                    f"aimo3-kernel-{os.getpid()}-{uuid.uuid4().hex}.json",
                )
                km = KernelManager(connection_file=connection_file)
                km.start_kernel(extra_arguments=["--Application.log_level=CRITICAL"])
                client = km.blocking_client()
                client.start_channels()
                client.wait_for_ready(timeout=self._default_timeout)

                self._km = km
                self._client = client
                self._owns_kernel = True
                self._connection_file = connection_file
                last_err = None
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                # Best-effort cleanup before retrying.
                with contextlib.suppress(Exception):
                    if client is not None:
                        client.stop_channels()
                with contextlib.suppress(Exception):
                    if km is not None:
                        km.shutdown_kernel(now=True)
                with contextlib.suppress(Exception):
                    if km is not None:
                        km.cleanup_resources()
                with contextlib.suppress(Exception):
                    if connection_file is not None and os.path.exists(connection_file):
                        os.remove(connection_file)

                # Small backoff; collisions tend to resolve quickly.
                time.sleep(min(0.25, 0.05 * attempt))

        if self._client is None or self._km is None:
            raise RuntimeError(
                f"Failed to start Jupyter kernel after {max_start_attempts} attempts: {last_err}"
            )

        # Preload common math stack.
        self.execute(
            "import sys\n"
            "# Increase limit for large integer string conversion (Python 3.11+)\n"
            "try:\n"
            "    sys.set_int_max_str_digits(0)  # 0 = unlimited\n"
            "except AttributeError:\n"
            "    pass  # Python < 3.11\n"
            "import os\n"
            '_lean_bin = os.environ.get("AIMO3_LEAN_BIN_DIR")\n'
            'if _lean_bin and _lean_bin not in os.environ.get("PATH", ""):\n'
            '    os.environ["PATH"] = _lean_bin + os.pathsep + os.environ.get("PATH", "")\n'
            "def aimo3_verify(ok=True):\n"
            "    if ok:\n"
            '        print("VERIFY_OK")\n'
            "    else:\n"
            '        print("VERIFY_FAIL")\n'
            "import math\n"
            "import numpy\n"
            "import sympy\n"
            "import random\n"
            "import itertools\n"
            "import collections\n"
            "import fractions\n"
            "import sympy as sp\n"
            "import numpy as np\n"
            "import mpmath as mp\n"
            "from fractions import Fraction\n"
            "import mpmath\n"
            "mpmath.mp.dps = 64\n"
            "try:\n"
            "    import ortools  # noqa: F401\n"
            "    from ortools.sat.python import cp_model  # noqa: F401\n"
            "except Exception:\n"
            "    pass\n"
        )

    @staticmethod
    def _format_error(traceback: list[str]) -> str:
        # Jupyter/IPython traceback formats vary across runtimes.
        # In some environments frames look like:
        #   File "<ipython-input-...>", line ...
        # In others they look like:
        #   File "/tmp/ipykernel_1234/....py", line ...
        #
        # Keep relevant frames (ipython-input OR ipykernel) so we don't accidentally
        # drop the entire traceback and return the misleading "[WARN] No output".
        clean_lines: list[str] = []
        for frame in traceback:
            clean_frame = re.sub(r"\x1b\[[0-9;]*m", "", frame)
            if 'File "' in clean_frame:
                if ("ipython-input" not in clean_frame) and (
                    "ipykernel" not in clean_frame
                ):
                    continue
            clean_lines.append(clean_frame)

        if clean_lines:
            return "".join(clean_lines)

        # Fallback: return original traceback (ANSI-stripped) if filtering removed everything.
        return "".join(re.sub(r"\x1b\[[0-9;]*m", "", f) for f in traceback)

    @staticmethod
    def _add_error_hints(error_text: str) -> str:
        """Add helpful hints for common errors to guide the model."""
        hints = []

        # Dict slicing error: fvals[:10] on a dict
        if "KeyError: slice(" in error_text:
            hints.append(
                "TIP: You're trying to slice a dict like a list. "
                "Use list(d.items())[:10] or {k: d[k] for k in list(d.keys())[:10]}"
            )

        # Integer string conversion limit (Python 3.11+)
        if "Exceeds the limit" in error_text and "int_max_str_digits" in error_text:
            hints.append(
                "TIP: Large integer printing limit. Add: import sys; sys.set_int_max_str_digits(0)"
            )

        # SyntaxError: incomplete input - often caused by comment inside parentheses
        if (
            "SyntaxError: incomplete input" in error_text
            or "SyntaxError: '(' was never closed" in error_text
        ):
            hints.append(
                "TIP: Check for unclosed parentheses. A common mistake is putting a comment "
                "inside a function call like print(x # comment) - the # hides the closing )."
            )

        # NameError for common functions/modules
        if "NameError" in error_text:
            if "fractions" in error_text:
                hints.append(
                    "TIP: Add 'import fractions' or 'from fractions import Fraction'"
                )
            elif "gcd" in error_text:
                hints.append("TIP: Add 'from math import gcd'")
            elif "combinations" in error_text or "permutations" in error_text:
                hints.append(
                    "TIP: Add 'from itertools import combinations, permutations'"
                )
            elif "factorial" in error_text:
                hints.append("TIP: Add 'from math import factorial'")
            elif "Counter" in error_text:
                hints.append("TIP: Add 'from collections import Counter'")

        # mpmath findroot tolerance error - tolerance is too strict at high precision
        if (
            "ValueError" in error_text
            and "Could not find root within given tolerance" in error_text
        ):
            hints.append(
                "TIP: mpmath findroot tolerance is too strict. Either:\n"
                "  1. Lower precision: mp.dps = 18\n"
                "  2. Set explicit tolerance: mp.findroot(f, x0, tol=1e-12)\n"
                "  3. Use verify=False: mp.findroot(f, x0, verify=False)"
            )

        if hints:
            return error_text.rstrip() + "\n\n" + "\n".join(hints)
        return error_text

    def execute(self, code: str, timeout: float | None = None) -> str:
        if self._client is None or self._km is None:
            return "[ERROR] Sandbox is closed."

        client = self._client
        effective_timeout = (
            float(timeout) if timeout is not None else self._default_timeout
        )

        msg_id = client.execute(
            code,
            store_history=True,
            allow_stdin=False,
            stop_on_error=False,
        )

        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > effective_timeout:
                with contextlib.suppress(Exception):
                    self._km.interrupt_kernel()
                # Helpful error message that guides the model to use timeout directive
                hint = ""
                if effective_timeout < 60:
                    hint = " TIP: For expensive computations, add '# timeout: 120' as the FIRST line of your code."
                return (
                    f"[ERROR] Execution timed out after {effective_timeout:.0f}s.{hint}"
                )

            try:
                msg = client.get_iopub_msg(timeout=1.0)
            except queue.Empty:
                continue

            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            msg_type = msg.get("msg_type")
            content = msg.get("content", {})

            if msg_type == "stream":
                text = content.get("text", "")
                if content.get("name") == "stdout":
                    stdout_parts.append(text)
                else:
                    stderr_parts.append(text)

            elif msg_type == "error":
                tb = content.get("traceback", [])
                formatted_error = self._format_error(tb)
                # Add helpful hints for common errors
                formatted_error = self._add_error_hints(formatted_error)
                stderr_parts.append(formatted_error)

            elif msg_type in {"execute_result", "display_data"}:
                data = content.get("data", {})
                text = data.get("text/plain")
                if text:
                    stdout_parts.append(text if text.endswith("\n") else text + "\n")

            elif msg_type == "status" and content.get("execution_state") == "idle":
                break

        stdout = "".join(stdout_parts)
        stderr = "".join(stderr_parts)
        if stderr:
            return f"{stdout.rstrip()}\n{stderr}" if stdout else stderr
        return (
            stdout
            if stdout.strip()
            else "[WARN] No output. Use print() to see results."
        )

    def reset(self) -> None:
        self.execute("%reset -f")
        self.execute("import gc; gc.collect()")
        self.execute(
            "import sys\n"
            "# Increase limit for large integer string conversion (Python 3.11+)\n"
            "try:\n"
            "    sys.set_int_max_str_digits(0)  # 0 = unlimited\n"
            "except AttributeError:\n"
            "    pass  # Python < 3.11\n"
            "import os\n"
            '_lean_bin = os.environ.get("AIMO3_LEAN_BIN_DIR")\n'
            'if _lean_bin and _lean_bin not in os.environ.get("PATH", ""):\n'
            '    os.environ["PATH"] = _lean_bin + os.pathsep + os.environ.get("PATH", "")\n'
            "def aimo3_verify(ok=True):\n"
            "    if ok:\n"
            '        print("VERIFY_OK")\n'
            "    else:\n"
            '        print("VERIFY_FAIL")\n'
            "import math\n"
            "import numpy\n"
            "import sympy\n"
            "import random\n"
            "import itertools\n"
            "import collections\n"
            "import fractions\n"
            "import sympy as sp\n"
            "import numpy as np\n"
            "import mpmath as mp\n"
            "from fractions import Fraction\n"
            "import mpmath\n"
            "mpmath.mp.dps = 18\n"
            "try:\n"
            "    import ortools  # noqa: F401\n"
            "    from ortools.sat.python import cp_model  # noqa: F401\n"
            "except Exception:\n"
            "    pass\n"
        )

    def close(self) -> None:
        with contextlib.suppress(Exception):
            if self._client:
                self._client.stop_channels()
        if self._owns_kernel and self._km is not None:
            with contextlib.suppress(Exception):
                self._km.shutdown_kernel(now=True)
            with contextlib.suppress(Exception):
                self._km.cleanup_resources()
        with contextlib.suppress(Exception):
            if self._connection_file is not None and os.path.exists(
                self._connection_file
            ):
                os.remove(self._connection_file)
        self._client = None
        self._km = None
        self._connection_file = None

    def __del__(self) -> None:  # noqa: D401
        self.close()
