"""vLLM OpenAI-compatible server subprocess management for AIMO-3 solver."""

# pylint: disable=broad-exception-caught,missing-function-docstring,line-too-long

from __future__ import annotations

import contextlib
import ctypes.util
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .config import AIMO3Config


_WARMED_MODEL_PATHS: set[str] = set()
_WARMED_MODEL_PATHS_LOCK = threading.Lock()


@dataclass
class VLLMServer:
    """Start/stop a vLLM OpenAI-compatible server as a subprocess."""

    cfg: AIMO3Config
    port: int = 8000
    log_path: str = "vllm_server.log"

    process: subprocess.Popen | None = None
    _log_file: object | None = None

    _MAX_ERROR_LOG_CHARS: int = 3500

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
        has_lib = lib is not None or os.path.exists(
            "/usr/lib/x86_64-linux-gnu/libcuda.so.1"
        )
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

        model_key = os.path.realpath(model_path)
        with _WARMED_MODEL_PATHS_LOCK:
            if model_key in _WARMED_MODEL_PATHS:
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
        with _WARMED_MODEL_PATHS_LOCK:
            _WARMED_MODEL_PATHS.add(model_key)

    def start(self) -> None:
        if not self.cfg.model_path:
            raise ValueError(
                "AIMO3Config.model_path is empty. Set env AIMO3_MODEL_PATH (Kaggle) or pass it explicitly."
            )

        if (
            bool(getattr(self.cfg, "require_cuda", True))
            and not self._cuda_driver_present()
        ):
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
            "--no-enable-log-requests",
        ]

        if bool(getattr(self.cfg, "vllm_trust_remote_code", False)):
            cmd.append("--trust-remote-code")

        if bool(getattr(self.cfg, "vllm_enable_chunked_prefill", False)):
            cmd.append("--enable-chunked-prefill")

        if bool(getattr(self.cfg, "vllm_enable_auto_tool_choice", False)):
            cmd.append("--enable-auto-tool-choice")

        tool_call_parser = str(getattr(self.cfg, "vllm_tool_call_parser", "") or "")
        if tool_call_parser:
            cmd.extend(["--tool-call-parser", tool_call_parser])

        reasoning_parser_plugin = str(
            getattr(self.cfg, "vllm_reasoning_parser_plugin", "") or ""
        )
        if reasoning_parser_plugin:
            cmd.extend(["--reasoning-parser-plugin", reasoning_parser_plugin])

        reasoning_parser = str(getattr(self.cfg, "vllm_reasoning_parser", "") or "")
        if reasoning_parser:
            cmd.extend(["--reasoning-parser", reasoning_parser])

        attention_backend = str(getattr(self.cfg, "vllm_attention_backend", "") or "")
        if attention_backend:
            cmd.extend(["--attention-backend", attention_backend])

        max_cudagraph_capture_size = int(
            getattr(self.cfg, "vllm_max_cudagraph_capture_size", 0) or 0
        )
        if max_cudagraph_capture_size > 0:
            cmd.extend(
                [
                    "--max-cudagraph-capture-size",
                    str(max_cudagraph_capture_size),
                ]
            )

        self._log_file = open(self.log_path, "w", encoding="utf-8")
        child_env = os.environ.copy()
        child_env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        self.process = subprocess.Popen(
            cmd,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=child_env,
        )

    def wait_ready(self, client, timeout_s: float | None = None) -> None:
        """Wait until the server responds to `client.models.list()`."""
        if self.process is None:
            raise RuntimeError("Server not started")

        timeout_s = (
            float(timeout_s)
            if timeout_s is not None
            else float(self.cfg.server_timeout)
        )
        start = time.time()
        sleep_s = 0.25
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
                excerpt = self._log_excerpt(logs, self.log_path)
                if hint:
                    raise RuntimeError(
                        f"vLLM server died with code {rc}.\n\n{hint}\n\nRecent logs:\n{excerpt}"
                    )

                raise RuntimeError(
                    f"vLLM server died with code {rc}. Recent logs:\n{excerpt}"
                )

            try:
                client.models.list()
                return
            except Exception:  # noqa: BLE001
                time.sleep(sleep_s)
                sleep_s = min(1.0, sleep_s * 1.5)

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
    def _log_excerpt(
        logs: str,
        log_path: str = "vllm_server.log",
        max_chars: int = _MAX_ERROR_LOG_CHARS,
    ) -> str:
        """Return a compact, high-signal log excerpt for notebook / gRPC errors."""

        if not logs:
            return "(no logs)"

        cleaned = logs.strip()
        if len(cleaned) <= max_chars:
            return cleaned

        lines = [line.rstrip() for line in cleaned.splitlines() if line.strip()]
        error_markers = (
            "traceback",
            "error:",
            "runtimeerror",
            "valueerror",
            "exception",
            "failed",
            "oom",
            "outofmemory",
            "cuda",
        )
        key_lines: list[str] = []
        for line in lines:
            lower = line.lower()
            if any(marker in lower for marker in error_markers):
                if line not in key_lines:
                    key_lines.append(line)
            if len(key_lines) >= 12:
                break

        header = (
            f"(log truncated; full log in {os.path.abspath(log_path)})\n"
            "--- key lines ---\n"
        )
        body_parts: list[str] = []
        if key_lines:
            body_parts.append("\n".join(key_lines))

        head = "\n".join(lines[:10])
        tail = "\n".join(lines[-20:])
        if head:
            body_parts.append("--- log head ---\n" + head)
        if tail:
            body_parts.append("--- log tail ---\n" + tail)

        excerpt = header + "\n\n".join(body_parts)
        if len(excerpt) <= max_chars:
            return excerpt

        tail_len = max(0, max_chars - len(header) - 32)
        return header + excerpt[-tail_len:]

    @staticmethod
    def _hint_from_logs(logs: str) -> str | None:
        """Return an actionable hint for common startup failures (best-effort)."""

        if not logs:
            return None

        def _oom_tuning_hint() -> str:
            free_gib = None
            total_gib = None
            requested_gib = None
            requested_mib = None
            gpu_util = None
            max_model_len = None
            max_num_seqs = None

            mem_match = re.search(
                r"GPU\s+\d+\s+has a total capacity of\s+(\d+(?:\.\d+)?)\s+GiB\s+of which\s+(\d+(?:\.\d+)?)\s+(GiB|MiB)\s+is free",
                logs,
                re.IGNORECASE,
            )
            if mem_match:
                total_gib = float(mem_match.group(1))
                free_value = float(mem_match.group(2))
                free_unit = mem_match.group(3).lower()
                free_gib = free_value if free_unit == "gib" else free_value / 1024.0

            alloc_match = re.search(
                r"Tried to allocate\s+(\d+(?:\.\d+)?)\s+(GiB|MiB)",
                logs,
                re.IGNORECASE,
            )
            if alloc_match:
                alloc_value = float(alloc_match.group(1))
                alloc_unit = alloc_match.group(2).lower()
                if alloc_unit == "gib":
                    requested_gib = alloc_value
                else:
                    requested_mib = alloc_value
                    requested_gib = alloc_value / 1024.0

            util_match = re.search(
                r"['\"]gpu_memory_utilization['\"]\s*:\s*(\d+(?:\.\d+)?)",
                logs,
                re.IGNORECASE,
            )
            if util_match:
                gpu_util = float(util_match.group(1))

            max_model_len_match = re.search(
                r"['\"]max_model_len['\"]\s*:\s*(\d+)",
                logs,
                re.IGNORECASE,
            ) or re.search(r"Using max model len\s+(\d+)", logs, re.IGNORECASE)
            if max_model_len_match:
                max_model_len = int(max_model_len_match.group(1))

            max_num_seqs_match = re.search(
                r"['\"]max_num_seqs['\"]\s*:\s*(\d+)",
                logs,
                re.IGNORECASE,
            )
            if max_num_seqs_match:
                max_num_seqs = int(max_num_seqs_match.group(1))

            current_cfg_lines: list[str] = []
            if gpu_util is not None:
                current_cfg_lines.append(
                    f"- Observed gpu_memory_utilization={gpu_util:.2f}"
                )
            if max_model_len is not None:
                current_cfg_lines.append(
                    f"- Observed max_model_len={max_model_len}"
                )
            if max_num_seqs is not None:
                current_cfg_lines.append(f"- Observed max_num_seqs={max_num_seqs}")

            suggestion_lines = []
            if gpu_util is not None:
                lowered_util = max(0.5, min(0.92, gpu_util - 0.06))
                suggestion_lines.append(
                    f"- Lower AIMO3_GPU_MEMORY_UTILIZATION first (for example {lowered_util:.2f} instead of {gpu_util:.2f})"
                )
            else:
                suggestion_lines.append(
                    "- Lower AIMO3_GPU_MEMORY_UTILIZATION first (try 0.88-0.92)"
                )

            if max_model_len is not None and max_model_len > 65536:
                suggestion_lines.append(
                    "- Reduce AIMO3_CONTEXT_TOKENS (128000 is very aggressive for a 120B checkpoint; try 65536 or 32768)"
                )
            else:
                suggestion_lines.append(
                    "- Reduce AIMO3_CONTEXT_TOKENS if you do not truly need the current context window"
                )

            if max_num_seqs is not None and max_num_seqs >= 128:
                suggestion_lines.append(
                    "- Reduce AIMO3_BATCH_SIZE / max-num-seqs (try 64 or 32)"
                )
            else:
                suggestion_lines.append(
                    "- Reduce AIMO3_BATCH_SIZE / max-num-seqs if you are parallelizing many requests"
                )

            suggestion_lines.extend(
                [
                    "- If this still fails, lower AIMO3_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE or set it to 0",
                    "- Restart the notebook/session to clear any leftover vLLM processes holding VRAM",
                    "- Optionally set PYTORCH_ALLOC_CONF=expandable_segments:True to reduce fragmentation",
                ]
            )

            memory_line = ""
            if free_gib is not None and total_gib is not None:
                memory_line = (
                    f"The GPU had only about {free_gib:.2f} GiB free out of {total_gib:.2f} GiB at the point of failure.\n"
                )
            request_line = ""
            if requested_gib is not None:
                if requested_mib is not None and requested_mib < 1024.0:
                    request_line = (
                        f"The failing allocation was about {requested_mib:.0f} MiB.\n"
                    )
                else:
                    request_line = (
                        f"The failing allocation was about {requested_gib:.2f} GiB.\n"
                    )

            current_cfg_block = ""
            if current_cfg_lines:
                current_cfg_block = "Current startup settings seen in the log:\n" + "\n".join(
                    current_cfg_lines
                ) + "\n\n"

            return (
                "Startup failed because the model ran out of GPU memory during weight post-processing / layout conversion, not because the path is invalid.\n"
                f"{memory_line}"
                f"{request_line}"
                f"{current_cfg_block}"
                "This usually means the checkpoint barely fits, then tips over once vLLM allocates extra scratch/KV/cudagraph memory.\n\n"
                "Fixes (try in this order):\n"
                + "\n".join(suggestion_lines)
            )

        def _cache_blocks_hint() -> str:
            gpu_util = None
            max_model_len = None
            max_num_seqs = None

            util_match = re.search(
                r"['\"]gpu_memory_utilization['\"]\s*:\s*(\d+(?:\.\d+)?)",
                logs,
                re.IGNORECASE,
            )
            if util_match:
                gpu_util = float(util_match.group(1))

            max_model_len_match = re.search(
                r"['\"]max_model_len['\"]\s*:\s*(\d+)",
                logs,
                re.IGNORECASE,
            ) or re.search(r"Using max model len\s+(\d+)", logs, re.IGNORECASE)
            if max_model_len_match:
                max_model_len = int(max_model_len_match.group(1))

            max_num_seqs_match = re.search(
                r"['\"]max_num_seqs['\"]\s*:\s*(\d+)",
                logs,
                re.IGNORECASE,
            )
            if max_num_seqs_match:
                max_num_seqs = int(max_num_seqs_match.group(1))

            current_cfg_lines: list[str] = []
            if gpu_util is not None:
                current_cfg_lines.append(
                    f"- Observed gpu_memory_utilization={gpu_util:.2f}"
                )
            if max_model_len is not None:
                current_cfg_lines.append(
                    f"- Observed max_model_len={max_model_len}"
                )
            if max_num_seqs is not None:
                current_cfg_lines.append(f"- Observed max_num_seqs={max_num_seqs}")

            suggestion_lines = []
            if max_num_seqs is not None and max_num_seqs > 32:
                suggestion_lines.append(
                    "- Reduce AIMO3_BATCH_SIZE / max-num-seqs first (for large 120B checkpoints, try 32 or even 16)"
                )
            else:
                suggestion_lines.append(
                    "- Reduce AIMO3_BATCH_SIZE / max-num-seqs first if you are parallelizing many requests"
                )

            if max_model_len is not None and max_model_len > 32768:
                suggestion_lines.append(
                    "- Reduce AIMO3_CONTEXT_TOKENS (try 32768 or 16384)"
                )
            else:
                suggestion_lines.append(
                    "- Reduce AIMO3_CONTEXT_TOKENS if you do not truly need the current context window"
                )

            suggestion_lines.append(
                "- Only if the GPU still has plenty of free VRAM after a clean restart, consider raising AIMO3_GPU_MEMORY_UTILIZATION slightly"
            )
            suggestion_lines.append(
                "- Keep AIMO3_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE=0 to avoid extra startup reserve"
            )
            suggestion_lines.append(
                "- Restart the notebook/session to clear any leftover vLLM processes holding VRAM"
            )

            current_cfg_block = ""
            if current_cfg_lines:
                current_cfg_block = (
                    "Current startup settings seen in the log:\n"
                    + "\n".join(current_cfg_lines)
                    + "\n\n"
                )

            return (
                "Startup failed because vLLM could load the model weights but could not reserve enough KV-cache blocks for the requested serving configuration.\n"
                f"{current_cfg_block}"
                "This usually means max_num_seqs and/or max_model_len are still too large for the checkpoint size on this GPU.\n\n"
                "Fixes (try in this order):\n"
                + "\n".join(suggestion_lines)
            )

        # Common: GPU memory already consumed (often by another vLLM that was started earlier).
        if "Free memory on device" in logs and "desired GPU memory utilization" in logs:
            # Example:
            # ValueError: Free memory on device (10.47/79.44 GiB) on startup is less than desired GPU memory utilization (0.96, 76.26 GiB).
            m = re.search(
                r"Free memory on device \((\d+(?:\.\d+)?)/(\d+(?:\.\d+)?) GiB\)", logs
            )
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

        # Common: model loads, but KV-cache block reservation still fails.
        if "cache blocks" in logs.lower() or "No available memory for the cache blocks" in logs:
            return _cache_blocks_hint()

        # Common: load-time OOM while swizzling / processing quantized weights.
        if (
            "Failed to load model - not enough GPU memory" in logs
            or "torch.OutOfMemoryError" in logs
            or "CUDA out of memory" in logs
        ):
            return _oom_tuning_hint()

        # Common: CUDA/driver missing (CPU runtime).
        if "libcuda.so.1" in logs or "Failed to infer device type" in logs:
            return (
                "Startup failed because CUDA/NVIDIA driver is not available in this runtime.\n"
                "Kaggle: enable GPU accelerator and restart the session."
            )

        # Common: older vLLM build cannot load newer ModelOpt/NVFP4 checkpoints.
        if "ModelOpt currently only supports" in logs and "hf_quant_config.json" in logs:
            version_match = re.search(r"vLLM API server version\s+([0-9][^\s]*)", logs)
            detected_version = version_match.group(1) if version_match else None
            version_line = (
                f"Detected vLLM version: {detected_version}.\n"
                if detected_version
                else ""
            )
            return (
                "Startup failed because this checkpoint uses a newer ModelOpt quantization configuration than the installed vLLM build understands.\n"
                f"{version_line}"
                "For NVIDIA Nemotron 3 Super, NVIDIA's deployment guide pins vLLM 0.17.1; Kaggle wheels using vLLM 0.11.2 are too old for this NVFP4 checkpoint.\n\n"
                "Fixes:\n"
                "- Upgrade the runtime to a newer vLLM build compatible with Nemotron 3 Super / ModelOpt NVFP4 (recommended)\n"
                "- If you're using offline Kaggle wheels, replace the bundled vLLM 0.11.2 wheel with a newer compatible build before starting the server\n"
                "- If upgrading vLLM is impossible in this environment, use a backend the model card documents for this checkpoint instead (for example SGLang or TensorRT-LLM)\n\n"
                "This is a backend compatibility issue, not a bad max context / memory flag."
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
