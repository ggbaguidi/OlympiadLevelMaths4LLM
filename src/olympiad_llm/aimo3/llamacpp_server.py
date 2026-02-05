from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import time
from dataclasses import dataclass

from .config import AIMO3Config


@dataclass
class LlamaCppServer:
    """Start/stop a llama.cpp OpenAI-compatible server as a subprocess."""

    cfg: AIMO3Config
    port: int = 8000
    log_path: str = "llamacpp_server.log"

    process: subprocess.Popen | None = None
    _log_file: object | None = None

    def start(self) -> None:
        if not self.cfg.model_path:
            raise ValueError(
                "AIMO3Config.model_path is empty. Set env AIMO3_MODEL_PATH or pass it explicitly."
            )

        cmd = [
            sys.executable,
            "-m",
            "llama_cpp.server",
            "--model",
            self.cfg.model_path,
            "--model_alias",
            self.cfg.served_model_name,
            "--host",
            "0.0.0.0",
            "--port",
            str(self.port),
            "--n_ctx",
            str(self.cfg.context_tokens),
        ]
        
        # Add GPU layers if configured (usually -1 for all)
        n_gpu_layers = getattr(self.cfg, "llama_cpp_n_gpu_layers", -1)
        if n_gpu_layers != 0:
            cmd.extend(["--n_gpu_layers", str(n_gpu_layers)])

        # Add batch size if configured
        if self.cfg.batch_size > 0:
             cmd.extend(["--n_batch", str(self.cfg.batch_size)])

        self._log_file = open(self.log_path, "w", encoding="utf-8")
        self.process = subprocess.Popen(
            cmd,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=os.environ.copy(),
        )

    def wait_ready(self, client, timeout_s: float | None = None) -> None:
        """Wait until the server responds to `client.models.list()`."""
        if self.process is None:
            raise RuntimeError("Server not started")

        timeout_s = float(timeout_s) if timeout_s is not None else float(self.cfg.server_timeout)
        start = time.time()
        while time.time() - start < timeout_s:
            rc = self.process.poll()
            if rc is not None:
                with contextlib.suppress(Exception):
                    if self._log_file:
                        self._log_file.flush()
                try:
                    with open(self.log_path, "r", encoding="utf-8") as f:
                        logs = f.read()
                except Exception:  # noqa: BLE001
                    logs = "(failed to read logs)"

                raise RuntimeError(f"llama.cpp server died with code {rc}. Logs:\n{logs}")

            try:
                # llama-cpp-python server startup can be slow to accept connections
                client.models.list()
                return
            except Exception:  # noqa: BLE001
                time.sleep(1)

        # Timeout handling
        tail = "(no logs)"
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
            tail_lines = lines[-80:] if lines else []
            tail = "\n".join(tail_lines) if tail_lines else "(empty log)"
        except Exception:  # noqa: BLE001
            tail = "(failed to read logs)"

        with contextlib.suppress(Exception):
            self.stop()

        raise RuntimeError(
            "llama.cpp server failed to start (timeout).\n"
            f"Recent logs (tail):\n{tail}"
        )

    def stop(self) -> None:
        if self.process is not None:
            with contextlib.suppress(Exception):
                self.process.terminate()
            with contextlib.suppress(Exception):
                self.process.wait(timeout=10)
        if self._log_file is not None:
            with contextlib.suppress(Exception):
                self._log_file.close()
        self.process = None
        self._log_file = None

    def __del__(self) -> None:
        self.stop()
