from __future__ import annotations

import os
from dataclasses import dataclass

from .prompts import ENHANCED_TOOL_INSTRUCTION, TIR_PROMPT_STANDARD


@dataclass(frozen=True)
class AIMO3Config:
    """Configuration for the AIMO-3 solver loop.

    Defaults are taken from the notebook (`aimo-3.py`) but converted to a
    dataclass with explicit types.
    """

    # Prompts
    system_prompt: str = TIR_PROMPT_STANDARD
    tool_prompt: str = ENHANCED_TOOL_INSTRUCTION
    preference_prompt: str = (
        "Use `math`, `numpy`, `sympy`, `mpmath`, `scipy`, `itertools` and `collections` to solve the problem."
    )

    # Heuristics / strategy augmentation
    wickelgren_strategies_enabled: bool = True

    # Attempt-level protocol (lemmas + verification gate)
    protocol_enabled: bool = True

    # Notebook display / logging
    # If True, show a table of attempts (candidate answers + stats + snippet) after solving.
    display_candidates: bool = True
    # How many chars of assistant text to store per attempt (tail buffer).
    capture_attempt_text_chars: int = 8000
    # How many chars to show in the display table.
    display_attempt_text_chars: int = 600
    # Max number of attempts rows to display (after all attempts are done).
    display_max_rows: int = 12

    # Model/server
    served_model_name: str = "gpt-oss"
    model_path: str = ""  # set via env AIMO3_MODEL_PATH in Kaggle
    kv_cache_dtype: str = "fp8_e4m3"
    dtype: str = "auto"

    # Time budgets (seconds)
    high_problem_timeout: float = 900.0
    base_problem_timeout: float = 300.0
    notebook_limit: float = 17520.0
    server_timeout: float = 180.0
    session_timeout: float = 960.0
    jupyter_timeout: float = 10.0
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
    batch_size: int = 256
    early_stop: int = 3
    attempts: int = 8
    workers: int = 16
    # Concurrency used only during *kernel creation*. High values can cause port races.
    kernel_init_workers: int = 4
    turns: int = 128
    seed: int = 3

    gpu_memory_utilization: float = 0.96
    temperature: float = 0.95
    min_p: float = 0.05

    # Hardware requirements
    # vLLM (as used in Kaggle) typically requires an NVIDIA GPU with a working driver.
    # If True and no CUDA driver/GPU is detected, we fail fast with a helpful error.
    require_cuda: bool = True

    # Second-stage verification
    second_stage_verify_enabled: bool = True
    second_stage_verify_top_k: int = 2

    # Reserve part of the per-problem time budget for verification.
    # This prevents the common failure mode: spending the entire budget on generation,
    # then skipping verification due to insufficient remaining time.
    verification_reserve_fraction: float = 0.15
    verification_reserve_cap: float = 120.0
    verification_reserve_min: float = 10.0

    second_stage_verify_trigger_votes_gap: int = 1
    second_stage_verify_trigger_if_no_verified: bool = True

    second_stage_verify_min_remaining: float = 12.0
    second_stage_verify_budget_cap: float = 45.0
    second_stage_verify_budget_fraction: float = 0.60
    second_stage_verify_min_effective_time: float = 3.0

    second_stage_verify_repeats_threshold: float = 25.0
    second_stage_verify_repeats_low: int = 1
    second_stage_verify_repeats_high: int = 2
    second_stage_verify_workers_cap: int = 2
    second_stage_verify_attempt_base: int = 10_000

    @staticmethod
    def from_env() -> "AIMO3Config":
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

        model_path = os.getenv("AIMO3_MODEL_PATH", "")
        served_model_name = os.getenv("AIMO3_SERVED_MODEL_NAME", "gpt-oss")
        reuse = os.getenv("AIMO3_REUSE_EXISTING_SERVER", "1").strip().lower() not in {"0", "false", "no"}
        wick = os.getenv("AIMO3_WICKELGREN", "1").strip().lower() not in {"0", "false", "no"}
        proto = os.getenv("AIMO3_PROTOCOL", "1").strip().lower() not in {"0", "false", "no"}
        disp = os.getenv("AIMO3_DISPLAY_CANDIDATES", "1").strip().lower() not in {"0", "false", "no"}
        require_cuda = os.getenv("AIMO3_REQUIRE_CUDA", "1").strip().lower() not in {"0", "false", "no"}

        # Core solver knobs
        attempts = _env_int("AIMO3_ATTEMPTS", AIMO3Config.attempts)
        workers = _env_int("AIMO3_WORKERS", AIMO3Config.workers)
        early_stop = _env_int("AIMO3_EARLY_STOP", AIMO3Config.early_stop)

        # Time budgets
        base_problem_timeout = _env_float("AIMO3_BASE_PROBLEM_TIMEOUT", AIMO3Config.base_problem_timeout)
        high_problem_timeout = _env_float("AIMO3_HIGH_PROBLEM_TIMEOUT", AIMO3Config.high_problem_timeout)
        notebook_limit = _env_float("AIMO3_NOTEBOOK_LIMIT", AIMO3Config.notebook_limit)

        # Tooling timeouts
        jupyter_timeout = _env_float("AIMO3_JUPYTER_TIMEOUT", AIMO3Config.jupyter_timeout)
        sandbox_timeout = _env_float("AIMO3_SANDBOX_TIMEOUT", AIMO3Config.sandbox_timeout)

        # Verification knobs
        second_stage_top_k = _env_int("AIMO3_SECOND_STAGE_TOP_K", AIMO3Config.second_stage_verify_top_k)
        second_stage_cap = _env_float("AIMO3_SECOND_STAGE_BUDGET_CAP", AIMO3Config.second_stage_verify_budget_cap)
        second_stage_fraction = _env_float(
            "AIMO3_SECOND_STAGE_BUDGET_FRACTION", AIMO3Config.second_stage_verify_budget_fraction
        )

        verify_reserve_fraction = _env_float(
            "AIMO3_VERIFY_RESERVE_FRACTION", AIMO3Config.verification_reserve_fraction
        )
        verify_reserve_cap = _env_float("AIMO3_VERIFY_RESERVE_CAP", AIMO3Config.verification_reserve_cap)
        verify_reserve_min = _env_float("AIMO3_VERIFY_RESERVE_MIN", AIMO3Config.verification_reserve_min)

        problems_total = _env_int("AIMO3_PROBLEMS_TOTAL", AIMO3Config.problems_total)

        # Startup knobs (useful in Kaggle when model load can take >3 minutes).
        # If the user didn't set AIMO3_SERVER_TIMEOUT explicitly and we're using a Kaggle input model,
        # default to a longer timeout (large checkpoints can easily take 6-12 minutes to load).
        raw_server_timeout = os.getenv("AIMO3_SERVER_TIMEOUT")
        if raw_server_timeout is None or not raw_server_timeout.strip():
            if model_path.startswith("/kaggle/input/"):
                server_timeout = 900.0
            else:
                server_timeout = float(AIMO3Config.server_timeout)
        else:
            server_timeout = _env_float("AIMO3_SERVER_TIMEOUT", AIMO3Config.server_timeout)
        context_tokens = _env_int("AIMO3_CONTEXT_TOKENS", AIMO3Config.context_tokens)
        batch_size = _env_int("AIMO3_BATCH_SIZE", AIMO3Config.batch_size)
        gpu_mem = _env_float("AIMO3_GPU_MEMORY_UTILIZATION", AIMO3Config.gpu_memory_utilization)
        kernel_init_workers = _env_int("AIMO3_KERNEL_INIT_WORKERS", AIMO3Config.kernel_init_workers)

        return AIMO3Config(
            model_path=model_path,
            served_model_name=served_model_name,
            reuse_existing_server=reuse,
            wickelgren_strategies_enabled=wick,
            protocol_enabled=proto,
            display_candidates=disp,
            require_cuda=require_cuda,
            server_timeout=server_timeout,
            context_tokens=context_tokens,
            batch_size=batch_size,
            gpu_memory_utilization=gpu_mem,
            kernel_init_workers=kernel_init_workers,
            attempts=attempts,
            workers=workers,
            early_stop=early_stop,
            base_problem_timeout=base_problem_timeout,
            high_problem_timeout=high_problem_timeout,
            notebook_limit=notebook_limit,
            jupyter_timeout=jupyter_timeout,
            sandbox_timeout=sandbox_timeout,
            second_stage_verify_top_k=second_stage_top_k,
            second_stage_verify_budget_cap=second_stage_cap,
            second_stage_verify_budget_fraction=second_stage_fraction,
            verification_reserve_fraction=verify_reserve_fraction,
            verification_reserve_cap=verify_reserve_cap,
            verification_reserve_min=verify_reserve_min,
            problems_total=problems_total,
        )
