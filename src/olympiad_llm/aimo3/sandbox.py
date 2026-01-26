from __future__ import annotations

import contextlib
import queue
import re
import threading
import time
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

        # IMPORTANT: don't hardcode ports.
        # In managed notebook runtimes (Kaggle/Colab), fixed port ranges are often already in use.
        # Let jupyter_client pick ephemeral free ports to avoid ZMQError: Address already in use.
        self._km = KernelManager()

        # Start kernel quietly.
        self._km.start_kernel(extra_arguments=["--Application.log_level=CRITICAL"])
        self._client = self._km.blocking_client()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=self._default_timeout)
        self._owns_kernel = True

        # Preload common math stack.
        self.execute(
            'import os\n'
            '_lean_bin = os.environ.get("AIMO3_LEAN_BIN_DIR")\n'
            'if _lean_bin and _lean_bin not in os.environ.get("PATH", ""):\n'
            '    os.environ["PATH"] = _lean_bin + os.pathsep + os.environ.get("PATH", "")\n'
            'import math\n'
            'import numpy\n'
            'import sympy\n'
            'import random\n'
            'import itertools\n'
            'import collections\n'
            "import sympy as sp\n"
            "import numpy as np\n"
            "import mpmath as mp\n"
            "from fractions import Fraction\n"
            'import mpmath\n'
            'mpmath.mp.dps = 128\n'
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
                if ("ipython-input" not in clean_frame) and ("ipykernel" not in clean_frame):
                    continue
            clean_lines.append(clean_frame)

        if clean_lines:
            return "".join(clean_lines)

        # Fallback: return original traceback (ANSI-stripped) if filtering removed everything.
        return "".join(re.sub(r"\x1b\[[0-9;]*m", "", f) for f in traceback)

    def execute(self, code: str, timeout: float | None = None) -> str:
        if self._client is None or self._km is None:
            return "[ERROR] Sandbox is closed."

        client = self._client
        effective_timeout = float(timeout) if timeout is not None else self._default_timeout

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
                return f"[ERROR] Execution timed out after {effective_timeout} seconds"

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
                stderr_parts.append(self._format_error(tb))

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
        return stdout if stdout.strip() else "[WARN] No output. Use print() to see results."

    def reset(self) -> None:
        self.execute("%reset -f")
        self.execute("import gc; gc.collect()")
        self.execute(
            'import os\n'
            '_lean_bin = os.environ.get("AIMO3_LEAN_BIN_DIR")\n'
            'if _lean_bin and _lean_bin not in os.environ.get("PATH", ""):\n'
            '    os.environ["PATH"] = _lean_bin + os.pathsep + os.environ.get("PATH", "")\n'
            'import math\n'
            'import numpy\n'
            'import sympy\n'
            'import random\n'
            'import itertools\n'
            'import collections\n'
            "import sympy as sp\n"
            "import numpy as np\n"
            "import mpmath as mp\n"
            "from fractions import Fraction\n"
            'import mpmath\n'
            'mpmath.mp.dps = 128\n'
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
        self._client = None
        self._km = None

    def __del__(self) -> None:  # noqa: D401
        self.close()
