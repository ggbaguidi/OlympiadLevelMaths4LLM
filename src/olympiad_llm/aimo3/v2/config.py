"""Configuration for the AIMO-3 solver orchestration loop."""

# pylint: disable=broad-exception-caught
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AIMO3Config:
    """Configuration for the AIMO-3 solver loop.

    Defaults are taken from the notebook (`aimo-3.py`) but converted to a
    dataclass with explicit types.
    """

    # Prompts
    system_prompt: str = (
        "You are a world-class International Mathematical Olympiad (IMO) competitor. "
        "The final answer must be a non-negative integer between 0 and 99999. "
        "You must place the final integer answer inside \\boxed{}."
    )
    tool_prompt: str = (
        "Use this tool to execute Python code. The environment is a stateful Jupyter notebook. "
        "You must use print() to output results."
    )
    preference_prompt: str = (
        "You have access to `math`, `numpy` and `sympy` to solve the problem."
    )

    # Notebook display / logging
    # If True, show a table of attempts (candidate answers + stats + snippet) after solving.
    display_candidates: bool = True
    # How many chars of assistant text to store per attempt (tail buffer).
    capture_attempt_text_chars: int = 8000
    # How many chars to show in the display table.
    display_attempt_text_chars: int = 600
    # Max number of attempts rows to display (after all attempts are done).
    display_max_rows: int = 12

    # Observability / tracing
    # If enabled, append a JSON line per solved problem to trace_path.
    trace_enabled: bool = False
    trace_path: str = "aimo3_trace.jsonl"
    # If True, include the full problem text in the trace. Off by default to avoid leakage.
    trace_include_problem_text: bool = False

    # If True, delete (reset) the trace file at solver startup.
    # Useful in notebooks where you want a fresh trace on each kernel restart.
    trace_reset_on_start: bool = True

    # Optional: record a lightweight snapshot of the sandbox environment at solve start.
    # This is useful when debugging version-dependent behavior (e.g., sympy API differences).
    trace_env_enabled: bool = False
    # Comma-separated list of import names to query for __version__ inside the sandbox.
    # Example: "sympy,numpy,mpmath,jupyter_client,ortools"
    trace_env_packages: str = "sympy,numpy,mpmath"

    # Model/server
    served_model_name: str = "gpt-oss"
    model_path: str = ""  # set via env AIMO3_MODEL_PATH in Kaggle
    kv_cache_dtype: str = "fp8_e4m3"
    dtype: str = "auto"

    # Combined logging path for vLLM or other server logs
    server_log_path: str = "server.log"

    # Optional: warm OS page cache by reading model shards before starting vLLM.
    # This reduces cold-start stalls and first-token latency in notebook runtimes.
    preload_model_weights: bool = False
    preload_model_workers: int = 8

    # Time budgets (seconds)
    high_problem_timeout: float = 900.0
    base_problem_timeout: float = 300.0
    notebook_limit: float = 17520.0
    server_timeout: float = 180.0
    session_timeout: float = 960.0
    jupyter_timeout: float = 30.0
    sandbox_timeout: float = 5.0

    # Budget allocator assumes a fixed number of remaining problems.
    # Set to 1 when debugging a single hard problem locally.
    problems_total: int = 50

    # Server reuse / probing
    # If True, attempt to connect to an already-running OpenAI-compatible server
    # on the configured port and reuse it instead of starting another vLLM process.
    reuse_existing_server: bool = True
    server_probe_timeout: float = 2.0
    server_probe_attempts: int = 3

    # Decoding + orchestration
    stream_interval: int = 200
    context_tokens: int = 65536
    search_tokens: int = 1024
    buffer_tokens: int = 512
    # If enabled, request top-k logprobs from vLLM and compute a mean-token entropy
    # to slightly bias ranking toward more confident attempts.
    entropy_weighting_enabled: bool = False
    top_logprobs: int = 5
    batch_size: int = 256
    early_stop: int = 3
    early_stop_min_verified: int = 0

    attempts: int = 8
    workers: int = 16

    # Sandbox state policy
    # By default we reset the sandbox between attempts to avoid cross-attempt contamination.
    # Set to False only for debugging / interactive workflows where you want to reuse
    # functions/variables across attempts.
    sandbox_reset_between_attempts: bool = True

    # Sandbox pool configuration
    kernel_init_workers: int = 4
    sandbox_pool_size: int = 8
    sandbox_create_on_exhaustion: bool = True

    # Strict extraction mode
    strict_fallback_extraction: bool = True

    # Tracing attempts
    trace_attempts_enabled: bool = False

    turns: int = 128
    seed: int = 3

    gpu_memory_utilization: float = 0.96
    temperature: float = 0.95
    min_p: float = 0.05
    top_p: float = 1.0  # Nucleus sampling (1.0 = disabled)
    top_k: int = -1  # Top-k sampling (-1 = disabled)

    # Hardware requirements
    # vLLM (as used in Kaggle) typically requires an NVIDIA GPU with a working driver.
    # If True and no CUDA driver/GPU is detected, we fail fast with a helpful error.
    require_cuda: bool = True

    @staticmethod
    def from_env() -> "AIMO3Config":
        """Create a config by reading from environment variables, falling back to defaults."""

        def _env_present(name: str) -> bool:
            raw = os.getenv(name)
            return raw is not None and bool(raw.strip())

        def _env_float(name: str, default: float) -> float:
            raw = os.getenv(name)
            if raw is None or not raw.strip():
                return float(default)
            try:
                return float(raw)
            except Exception:  # noqa: BLE001
                return float(default)

        def _env_int(name: str, default: int) -> int:
            raw = os.getenv(name)
            if raw is None or not raw.strip():
                return int(default)
            try:
                return int(float(raw))
            except Exception:  # noqa: BLE001
                return int(default)

        strict_fallback_extraction = os.getenv(
            "AIMO3_STRICT_FALLBACK_EXTRACTION", "1"
        ).strip().lower() not in {"0", "false", "no"}

        seed = _env_int("AIMO3_SEED", AIMO3Config.seed)
        preference_prompt = os.getenv(
            "AIMO3_PREFERENCE_PROMPT", AIMO3Config.preference_prompt
        )
        tool_prompt = os.getenv("AIMO3_TOOL_PROMPT", AIMO3Config.tool_prompt)
        system_prompt = os.getenv("AIMO3_SYSTEM_PROMPT", AIMO3Config.system_prompt)

        # Profile presets (apply only when the corresponding env var is NOT explicitly set).
        # This makes it easy to reduce orchestration steps without rewriting many env vars.
        profile = (os.getenv("AIMO3_PROFILE", "") or "").strip().lower()
        if profile not in {"", "default", "full", "lean"}:
            profile = ""

        model_path = os.path.expanduser(os.getenv("AIMO3_MODEL_PATH", ""))
        served_model_name = os.getenv("AIMO3_SERVED_MODEL_NAME", "gpt-oss")

        reuse = os.getenv("AIMO3_REUSE_EXISTING_SERVER", "1").strip().lower() not in {
            "0",
            "false",
            "no",
        }

        trace_enabled = os.getenv("AIMO3_TRACE", "0").strip().lower() not in {
            "0",
            "false",
            "no",
        }
        trace_path = os.path.expanduser(
            (os.getenv("AIMO3_TRACE_PATH", AIMO3Config.trace_path) or "").strip()
            or AIMO3Config.trace_path
        )
        trace_include_problem_text = os.getenv(
            "AIMO3_TRACE_INCLUDE_PROBLEM_TEXT", "0"
        ).strip().lower() not in {"0", "false", "no"}
        trace_reset_on_start = os.getenv(
            "AIMO3_TRACE_RESET_ON_START", "1"
        ).strip().lower() not in {"0", "false", "no"}
        trace_attempts_enabled = os.getenv(
            "AIMO3_TRACE_ATTEMPTS", "0"
        ).strip().lower() not in {"0", "false", "no"}
        trace_env_enabled = os.getenv("AIMO3_TRACE_ENV", "0").strip().lower() not in {
            "0",
            "false",
            "no",
        }
        trace_env_packages = (
            os.getenv("AIMO3_TRACE_ENV_PACKAGES", AIMO3Config.trace_env_packages) or ""
        ).strip() or AIMO3Config.trace_env_packages
        disp = os.getenv("AIMO3_DISPLAY_CANDIDATES", "1").strip().lower() not in {
            "0",
            "false",
            "no",
        }
        require_cuda = os.getenv("AIMO3_REQUIRE_CUDA", "1").strip().lower() not in {
            "0",
            "false",
            "no",
        }

        # Startup perf knobs
        preload_model_weights = os.getenv(
            "AIMO3_PRELOAD_MODEL_WEIGHTS", "0"
        ).strip().lower() not in {"0", "false", "no"}
        preload_model_workers = _env_int(
            "AIMO3_PRELOAD_MODEL_WORKERS", AIMO3Config.preload_model_workers
        )

        # Core solver knobs
        attempts = _env_int("AIMO3_ATTEMPTS", AIMO3Config.attempts)
        workers = _env_int("AIMO3_WORKERS", AIMO3Config.workers)
        early_stop = _env_int("AIMO3_EARLY_STOP", AIMO3Config.early_stop)
        early_stop_min_verified = _env_int(
            "AIMO3_EARLY_STOP_MIN_VERIFIED", AIMO3Config.early_stop_min_verified
        )

        turns = _env_int("AIMO3_TURNS", AIMO3Config.turns)

        # Time budgets
        base_problem_timeout = _env_float(
            "AIMO3_BASE_PROBLEM_TIMEOUT", AIMO3Config.base_problem_timeout
        )
        high_problem_timeout = _env_float(
            "AIMO3_HIGH_PROBLEM_TIMEOUT", AIMO3Config.high_problem_timeout
        )
        notebook_limit = _env_float("AIMO3_NOTEBOOK_LIMIT", AIMO3Config.notebook_limit)

        # Tooling timeouts
        jupyter_timeout = _env_float(
            "AIMO3_JUPYTER_TIMEOUT", AIMO3Config.jupyter_timeout
        )
        sandbox_timeout = _env_float(
            "AIMO3_SANDBOX_TIMEOUT", AIMO3Config.sandbox_timeout
        )

        problems_total = _env_int("AIMO3_PROBLEMS_TOTAL", AIMO3Config.problems_total)

        # Startup knobs (useful in Kaggle when model load can take >3 minutes).
        # If the user didn't set AIMO3_SERVER_TIMEOUT explicitly and
        # we're using a Kaggle input model,
        # default to a longer timeout (large checkpoints can easily take 6-12 minutes to load).
        raw_server_timeout = os.getenv("AIMO3_SERVER_TIMEOUT")
        if raw_server_timeout is None or not raw_server_timeout.strip():
            if model_path.startswith("/kaggle/input/"):
                server_timeout = 900.0
            else:
                server_timeout = float(AIMO3Config.server_timeout)
        else:
            server_timeout = _env_float(
                "AIMO3_SERVER_TIMEOUT", AIMO3Config.server_timeout
            )
        context_tokens = _env_int("AIMO3_CONTEXT_TOKENS", AIMO3Config.context_tokens)
        batch_size = _env_int("AIMO3_BATCH_SIZE", AIMO3Config.batch_size)
        gpu_mem = _env_float(
            "AIMO3_GPU_MEMORY_UTILIZATION", AIMO3Config.gpu_memory_utilization
        )

        # Decoding knobs
        temperature = _env_float("AIMO3_TEMPERATURE", AIMO3Config.temperature)
        min_p = _env_float("AIMO3_MIN_P", AIMO3Config.min_p)
        top_p = _env_float("AIMO3_TOP_P", AIMO3Config.top_p)
        top_k = _env_int("AIMO3_TOP_K", AIMO3Config.top_k)

        return AIMO3Config(
            seed=seed,
            system_prompt=system_prompt,
            tool_prompt=tool_prompt,
            preference_prompt=preference_prompt,
            model_path=model_path,
            served_model_name=served_model_name,
            preload_model_weights=preload_model_weights,
            preload_model_workers=preload_model_workers,
            reuse_existing_server=reuse,
            display_candidates=disp,
            trace_enabled=trace_enabled,
            trace_path=trace_path,
            trace_include_problem_text=trace_include_problem_text,
            trace_reset_on_start=trace_reset_on_start,
            trace_env_enabled=trace_env_enabled,
            trace_env_packages=trace_env_packages,
            trace_attempts_enabled=trace_attempts_enabled,
            require_cuda=require_cuda,
            server_timeout=server_timeout,
            context_tokens=context_tokens,
            batch_size=batch_size,
            gpu_memory_utilization=gpu_mem,
            sandbox_reset_between_attempts=AIMO3Config.sandbox_reset_between_attempts,
            temperature=temperature,
            min_p=min_p,
            top_p=top_p,
            top_k=top_k,
            attempts=attempts,
            workers=workers,
            early_stop=early_stop,
            early_stop_min_verified=early_stop_min_verified,
            base_problem_timeout=base_problem_timeout,
            high_problem_timeout=high_problem_timeout,
            notebook_limit=notebook_limit,
            jupyter_timeout=jupyter_timeout,
            sandbox_timeout=sandbox_timeout,
            problems_total=problems_total,
            turns=turns,
            strict_fallback_extraction=strict_fallback_extraction,
        )
