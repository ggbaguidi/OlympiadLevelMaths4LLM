from __future__ import annotations

import contextlib
import ctypes.util
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .config import AIMO3Config


@dataclass
class VLLMServer:
    """Start/stop a vLLM OpenAI-compatible server as a subprocess."""

    cfg: AIMO3Config
    port: int = 8000
    log_path: str = "vllm_server.log"

    process: subprocess.Popen | None = None
    _log_file: object | None = None

    @staticmethod
    def _cuda_visible_devices_allows_gpu() -> bool:
        cvd = os.getenv("CUDA_VISIBLE_DEVICES")
        if cvd is None:
            return True
        v = cvd.strip().lower()
        # Common ways to disable CUDA visibility.
        return v not in {"", "-1", "none", "no", "false"}

    @staticmethod
    def _cuda_driver_present() -> bool:
        # Fast checks that don't require importing torch.
        if not VLLMServer._cuda_visible_devices_allows_gpu():
            return False

        lib = ctypes.util.find_library("cuda")
        has_lib = lib is not None or os.path.exists("/usr/lib/x86_64-linux-gnu/libcuda.so.1")
        has_dev = any(
            os.path.exists(p)
            for p in (
                "/dev/nvidiactl",
                "/dev/nvidia0",
                "/proc/driver/nvidia/version",
            )
        )

        # If either side is missing, CUDA won't work for vLLM.
        return bool(has_lib and has_dev)

    def _preload_model_weights(self) -> None:
        """Best-effort OS page-cache warmup for large checkpoints.

        This mirrors the high-LB Kaggle notebook trick: reading shard files once before
        starting vLLM reduces random stalls and first-token latency on cold starts.

        Note: If you used `cleanup.environ_setup_parallel(warm_model=True)`, the warmup
        already happened in parallel with pip install - this will be a fast no-op (cached).
        """

        if not bool(getattr(self.cfg, "preload_model_weights", False)):
            return

        model_path = str(getattr(self.cfg, "model_path", "") or "")
        if not model_path or not os.path.isdir(model_path):
            return

        # Enumerate shard files.
        files: list[str] = []
        for root, _dirs, names in os.walk(model_path):
            for name in names:
                p = os.path.join(root, name)
                if os.path.isfile(p):
                    files.append(p)
        if not files:
            return

        workers = int(getattr(self.cfg, "preload_model_workers", 8) or 8)
        workers = max(1, workers)
        # Avoid silly oversubscription.
        with contextlib.suppress(Exception):
            workers = min(workers, max(1, (os.cpu_count() or 1)))

        def _read_file(path: str) -> None:
            # Read in 1GiB-ish chunks; content is discarded.
            try:
                with open(path, "rb") as f:
                    while f.read(1024 * 1024 * 1024):
                        pass
            except Exception:
                # Best-effort warmup; ignore unreadable files.
                return

        start = time.time()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_read_file, files))
        _ = time.time() - start

    def start(self) -> None:
        if not self.cfg.model_path:
            raise ValueError(
                "AIMO3Config.model_path is empty. Set env AIMO3_MODEL_PATH (Kaggle) or pass it explicitly."
            )

        if bool(getattr(self.cfg, "require_cuda", True)) and not self._cuda_driver_present():
            raise RuntimeError(
                "CUDA/NVIDIA driver not detected (e.g., libcuda.so.1 missing or GPU runtime disabled). "
                "vLLM cannot start in this environment.\n\n"
                "Kaggle: enable a GPU accelerator (Notebook Settings → Accelerator → GPU) and restart the session.\n"
                "Local: install NVIDIA drivers + CUDA runtime, and ensure /dev/nvidia* devices are present.\n\n"
                "If you *intentionally* want to try starting anyway, set AIMO3_REQUIRE_CUDA=0 (may still fail)."
            )

        # Optional: warm OS page cache for model files.
        with contextlib.suppress(Exception):
            self._preload_model_weights()

        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--seed",
            str(self.cfg.seed),
            "--model",
            self.cfg.model_path,
            "--served-model-name",
            self.cfg.served_model_name,
            "--tensor-parallel-size",
            "1",
            "--max-num-seqs",
            str(self.cfg.batch_size),
            "--gpu-memory-utilization",
            str(self.cfg.gpu_memory_utilization),
            "--host",
            "0.0.0.0",
            "--port",
            str(self.port),
            "--dtype",
            self.cfg.dtype,
            "--kv-cache-dtype",
            self.cfg.kv_cache_dtype,
            "--max-model-len",
            str(self.cfg.context_tokens),
            "--stream-interval",
            str(self.cfg.stream_interval),
            "--async-scheduling",
            "--enable-prefix-caching",
            "--disable-log-stats",
            "--disable-log-requests",
        ]

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

                hint = self._hint_from_logs(logs)
                if hint:
                    raise RuntimeError(f"vLLM server died with code {rc}.\n\n{hint}\n\nLogs:\n{logs}")

                raise RuntimeError(f"vLLM server died with code {rc}. Logs:\n{logs}")

            try:
                client.models.list()
                return
            except Exception:  # noqa: BLE001
                time.sleep(1)

        # Timeout: include recent log tail to make diagnosis faster in notebooks.
        tail = "(no logs)"
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
            tail_lines = lines[-80:] if lines else []
            tail = "\n".join(tail_lines) if tail_lines else "(empty log)"
        except Exception:  # noqa: BLE001
            tail = "(failed to read logs)"

        # Avoid leaking a half-started vLLM process that can keep VRAM allocated.
        with contextlib.suppress(Exception):
            self.stop()

        raise RuntimeError(
            "vLLM server failed to start (timeout).\n"
            "Tip: model loading can exceed the default timeout on Kaggle cold starts; consider setting AIMO3_SERVER_TIMEOUT=900 (or higher for 120B-class checkpoints).\n\n"
            f"Recent logs (tail):\n{tail}"
        )

    @staticmethod
    def _hint_from_logs(logs: str) -> str | None:
        """Return an actionable hint for common startup failures (best-effort)."""

        if not logs:
            return None

        # Common: GPU memory already consumed (often by another vLLM that was started earlier).
        if "Free memory on device" in logs and "desired GPU memory utilization" in logs:
            # Example:
            # ValueError: Free memory on device (10.47/79.44 GiB) on startup is less than desired GPU memory utilization (0.96, 76.26 GiB).
            m = re.search(r"Free memory on device \((\d+(?:\.\d+)?)/(\d+(?:\.\d+)?) GiB\)", logs)
            free_gib = float(m.group(1)) if m else None
            total_gib = float(m.group(2)) if m else None
            suggested = None
            if free_gib is not None and total_gib and total_gib > 0:
                # Leave a small headroom to avoid fragmentation surprises.
                suggested = max(0.1, min(0.95, (free_gib / total_gib) - 0.05))

            suggestion_line = (
                f"Set AIMO3_GPU_MEMORY_UTILIZATION={suggested:.2f} (based on detected free/total VRAM)"
                if suggested is not None
                else "Set AIMO3_GPU_MEMORY_UTILIZATION to something like 0.80–0.90"
            )

            return (
                "Startup failed due to insufficient *free* GPU memory for the requested KV-cache allocation.\n"
                "This usually happens when:\n"
                "- another vLLM server from a previous cell run is still holding VRAM, or\n"
                "- you requested a very large max context / max_num_seqs.\n\n"
                "Fixes (pick 1–3):\n"
                f"- {suggestion_line}\n"
                "- Reduce AIMO3_CONTEXT_TOKENS (e.g. 32768 or 16384 instead of 65536)\n"
                "- Reduce AIMO3_BATCH_SIZE (max-num-seqs) (e.g. 64 or 32 instead of 256)\n"
                "- Restart the notebook kernel/session to clear any leaked vLLM processes\n"
                "- Keep AIMO3_REUSE_EXISTING_SERVER=1 and avoid re-creating the solver multiple times"
            )

        # Common: CUDA/driver missing (CPU runtime).
        if "libcuda.so.1" in logs or "Failed to infer device type" in logs:
            return (
                "Startup failed because CUDA/NVIDIA driver is not available in this runtime.\n"
                "Kaggle: enable GPU accelerator and restart the session."
            )

        return None

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
