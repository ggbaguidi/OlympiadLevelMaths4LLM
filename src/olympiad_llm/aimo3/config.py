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
        "You have access to `math`, `numpy`, `sympy`, `mpmath`, `scipy`, `itertools`, and `collections` for:\n\n"
        "# Symbolic Computation (sympy):\n"
        "- Algebraic manipulation and simplification\n"
        "- Solving equations and systems of equations\n"
        "- Symbolic differentiation and integration\n"
        "- Number theory functions (primes, divisors, modular arithmetic)\n"
        "- Polynomial operations and factorization\n"
        "- Working with mathematical expressions symbolically\n\n"
        "# Numerical Computation (numpy):\n"
        "- Array operations and linear algebra\n"
        "- Efficient numerical calculations for large datasets\n"
        "- Matrix operations and eigenvalue problems\n"
        "- Statistical computations\n\n"

        "# High-precision / numerical analysis (mpmath):\n"
        "- High-precision floating-point arithmetic\n"
        "- Numerical integration/summation and special functions\n"
        "- Use mp.mp.dps to increase precision when needed\n\n"

        "# Scientific computing (scipy) (import explicitly if needed):\n"
        "- Optimization, root finding, numerical integration\n"
        "- Linear algebra routines, statistics, special functions\n\n"

        "# Discrete / combinatorics helpers (itertools, collections):\n"
        "- Efficient iteration over combinations/permutations/products\n"
        "- Counters, deques, default dicts for counting and graph/DP problems\n\n"
        "# Mathematical Functions (math):\n"
        "- Standard mathematical functions (trig, log, exp)\n"
        "- Constants like pi and e\n"
        "- Basic operations for single values\n\n"
        "Best Practices:\n"
        "- Use sympy for exact symbolic answers when possible\n"
        "- Use numpy for numerical verification and large-scale computation\n"
        "- Use mpmath for high-precision numeric checks when floating error matters\n"
        "- Combine symbolic and numerical approaches: derive symbolically, verify numerically\n"
        "- Keep tool code small and print intermediate checkpoints\n"
        "- Document your computational strategy clearly\n"
        "- Validate computational results against known cases or theoretical bounds"
    )

    # Heuristics / strategy augmentation
    wickelgren_strategies_enabled: bool = True

    # Strategy packs (general: diversify attempt styles)
    # Modes:
    # - "off": use only the generic pack (same behavior as before)
    # - "round_robin": cycle through enabled packs across attempts
    # - "auto": enable topic packs only when simple keyword cues are present
    strategy_pack_mode: str = "round_robin"
    # Comma-separated list of enabled packs. Known packs include: generic, fe_combi
    strategy_packs: str = "generic,fe_combi"

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

    # Observability / tracing
    # If enabled, append a JSON line per solved problem to trace_path.
    trace_enabled: bool = False
    trace_path: str = "aimo3_trace.jsonl"
    # If True, include the full problem text in the trace. Off by default to avoid leakage.
    trace_include_problem_text: bool = False

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
    # Early-stop quality guardrails
    # Require at least this many "clean tool" attempts (python_calls>0 and python_errors==0)
    # for the leading candidate before we stop early.
    # Set to 0 to revert to vote-only early stopping.
    early_stop_min_verified: int = 1
    attempts: int = 8
    workers: int = 16
    # Concurrency used only during *kernel creation*. High values can cause port races.
    kernel_init_workers: int = 4
    # How many sandbox kernels to keep warm in the pool.
    # Creating `workers` kernels up-front can be expensive/unreliable in notebook runtimes.
    sandbox_pool_size: int = 4
    # If the pool is exhausted, allow creating an ephemeral sandbox for that attempt.
    sandbox_create_on_exhaustion: bool = True

    # Tool-error recovery (general robustness)
    # Abort an attempt early if it is repeatedly failing tool calls, to avoid wasting tokens/time.
    abort_attempt_after_python_errors: int = 4
    abort_attempt_after_consecutive_python_errors: int = 3

    # If a sandbox produced many tool errors, consider it "poisoned" and recycle it.
    # Recycling means closing it and creating a fresh sandbox to keep the pool healthy.
    recycle_sandbox_after_python_errors: int = 4

    # Recovery attempts (general robustness)
    # If an attempt aborts due to tool errors (or is very error-heavy), optionally schedule
    # an extra "recovery" attempt to salvage a valid answer.
    recovery_attempts_enabled: bool = True
    recovery_attempts_cap: int = 2
    recovery_trigger_python_errors: int = 3
    recovery_min_remaining_s: float = 15.0

    # Recovery attempt style:
    # - "auto": choose based on observed tool instability
    # - "no_tool": strongly discourage tool use (and enforce cap=0)
    # - "micro_tool": allow a small number of tool calls (cap > 0)
    recovery_mode: str = "auto"
    recovery_micro_tool_call_cap: int = 2

    # Extraction/format recovery (general): if the model produced lots of tokens but we couldn't
    # extract an integer answer, schedule a short attempt focused purely on producing the final
    # boxed integer.
    format_recovery_enabled: bool = True
    format_recovery_cap: int = 1
    format_recovery_trigger_tokens: int = 2000
    format_recovery_min_remaining_s: float = 20.0

    # Tie-break verification (general): if ranking/second-stage verification is inconclusive,
    # run one short discriminating attempt comparing top candidates.
    tiebreak_enabled: bool = True
    tiebreak_min_remaining_s: float = 25.0
    tiebreak_budget_cap_s: float = 35.0
    turns: int = 128
    seed: int = 3

    gpu_memory_utilization: float = 0.96
    temperature: float = 0.95
    min_p: float = 0.05

    # Optional: per-role temperature schedule (general)
    # If a value is None, the solver will fall back to `temperature`.
    temperature_exploration: float | None = 0.95
    temperature_main: float | None = 0.70
    temperature_code: float | None = 0.65
    temperature_verification: float | None = 0.20
    temperature_formatting: float | None = 0.10

    # How many early attempts should run in "exploration" temperature mode.
    exploration_attempts: int = 2

    # Hardware requirements
    # vLLM (as used in Kaggle) typically requires an NVIDIA GPU with a working driver.
    # If True and no CUDA driver/GPU is detected, we fail fast with a helpful error.
    require_cuda: bool = True

    # Second-stage verification
    second_stage_verify_enabled: bool = True
    second_stage_verify_top_k: int = 2
    # Second-stage verification strength
    # Require the model to include a marker in its assistant text when it has actually run checks.
    # This is an internal protocol (not the final Kaggle answer), so adding a marker is safe.
    second_stage_verify_marker: str = "VERIFIED_OK"
    second_stage_verify_require_marker: bool = True

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
        strategy_pack_mode = (os.getenv("AIMO3_STRATEGY_PACK_MODE", AIMO3Config.strategy_pack_mode) or "").strip()
        if not strategy_pack_mode:
            strategy_pack_mode = AIMO3Config.strategy_pack_mode
        strategy_packs = (os.getenv("AIMO3_STRATEGY_PACKS", AIMO3Config.strategy_packs) or "").strip()
        if not strategy_packs:
            strategy_packs = AIMO3Config.strategy_packs
        proto = os.getenv("AIMO3_PROTOCOL", "1").strip().lower() not in {"0", "false", "no"}
        disp = os.getenv("AIMO3_DISPLAY_CANDIDATES", "1").strip().lower() not in {"0", "false", "no"}
        require_cuda = os.getenv("AIMO3_REQUIRE_CUDA", "1").strip().lower() not in {"0", "false", "no"}

        trace_enabled = os.getenv("AIMO3_TRACE", "0").strip().lower() not in {"0", "false", "no"}
        trace_path = os.getenv("AIMO3_TRACE_PATH", AIMO3Config.trace_path)
        trace_include_problem_text = (
            os.getenv("AIMO3_TRACE_INCLUDE_PROBLEM", "0").strip().lower() not in {"0", "false", "no"}
        )

        # Core solver knobs
        attempts = _env_int("AIMO3_ATTEMPTS", AIMO3Config.attempts)
        workers = _env_int("AIMO3_WORKERS", AIMO3Config.workers)
        early_stop = _env_int("AIMO3_EARLY_STOP", AIMO3Config.early_stop)
        early_stop_min_verified = _env_int("AIMO3_EARLY_STOP_MIN_VERIFIED", AIMO3Config.early_stop_min_verified)

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
        sandbox_pool_size = _env_int("AIMO3_SANDBOX_POOL_SIZE", AIMO3Config.sandbox_pool_size)
        sandbox_create_on_exhaustion = (
            os.getenv("AIMO3_SANDBOX_CREATE_ON_EXHAUSTION", "1").strip().lower() not in {"0", "false", "no"}
        )

        abort_attempt_after_python_errors = _env_int(
            "AIMO3_ABORT_ATTEMPT_AFTER_PYTHON_ERRORS", AIMO3Config.abort_attempt_after_python_errors
        )
        abort_attempt_after_consecutive_python_errors = _env_int(
            "AIMO3_ABORT_ATTEMPT_AFTER_CONSECUTIVE_PYTHON_ERRORS",
            AIMO3Config.abort_attempt_after_consecutive_python_errors,
        )
        recycle_sandbox_after_python_errors = _env_int(
            "AIMO3_RECYCLE_SANDBOX_AFTER_PYTHON_ERRORS", AIMO3Config.recycle_sandbox_after_python_errors
        )

        recovery_attempts_enabled = (
            os.getenv("AIMO3_RECOVERY_ATTEMPTS_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
        )
        recovery_attempts_cap = _env_int("AIMO3_RECOVERY_ATTEMPTS_CAP", AIMO3Config.recovery_attempts_cap)
        recovery_trigger_python_errors = _env_int(
            "AIMO3_RECOVERY_TRIGGER_PYTHON_ERRORS", AIMO3Config.recovery_trigger_python_errors
        )
        recovery_min_remaining_s = _env_float(
            "AIMO3_RECOVERY_MIN_REMAINING_S", AIMO3Config.recovery_min_remaining_s
        )
        recovery_mode = (os.getenv("AIMO3_RECOVERY_MODE", AIMO3Config.recovery_mode) or "").strip().lower()
        if recovery_mode not in {"auto", "no_tool", "micro_tool"}:
            recovery_mode = AIMO3Config.recovery_mode
        recovery_micro_tool_call_cap = _env_int(
            "AIMO3_RECOVERY_MICRO_TOOL_CALL_CAP", AIMO3Config.recovery_micro_tool_call_cap
        )

        # Decoding knobs
        temperature = _env_float("AIMO3_TEMPERATURE", AIMO3Config.temperature)
        min_p = _env_float("AIMO3_MIN_P", AIMO3Config.min_p)

        # Per-role temperatures (optional)
        temperature_exploration = _env_float(
            "AIMO3_TEMPERATURE_EXPLORATION", float(AIMO3Config.temperature_exploration or temperature)
        )
        temperature_main = _env_float("AIMO3_TEMPERATURE_MAIN", float(AIMO3Config.temperature_main or temperature))
        temperature_code = _env_float("AIMO3_TEMPERATURE_CODE", float(AIMO3Config.temperature_code or temperature))
        temperature_verification = _env_float(
            "AIMO3_TEMPERATURE_VERIFICATION", float(AIMO3Config.temperature_verification or temperature)
        )
        temperature_formatting = _env_float(
            "AIMO3_TEMPERATURE_FORMATTING", float(AIMO3Config.temperature_formatting or temperature)
        )
        exploration_attempts = _env_int("AIMO3_EXPLORATION_ATTEMPTS", AIMO3Config.exploration_attempts)

        format_recovery_enabled = (
            os.getenv("AIMO3_FORMAT_RECOVERY_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
        )
        format_recovery_cap = _env_int("AIMO3_FORMAT_RECOVERY_CAP", AIMO3Config.format_recovery_cap)
        format_recovery_trigger_tokens = _env_int(
            "AIMO3_FORMAT_RECOVERY_TRIGGER_TOKENS", AIMO3Config.format_recovery_trigger_tokens
        )
        format_recovery_min_remaining_s = _env_float(
            "AIMO3_FORMAT_RECOVERY_MIN_REMAINING_S", AIMO3Config.format_recovery_min_remaining_s
        )

        tiebreak_enabled = os.getenv("AIMO3_TIEBREAK_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
        tiebreak_min_remaining_s = _env_float(
            "AIMO3_TIEBREAK_MIN_REMAINING_S", AIMO3Config.tiebreak_min_remaining_s
        )
        tiebreak_budget_cap_s = _env_float("AIMO3_TIEBREAK_BUDGET_CAP_S", AIMO3Config.tiebreak_budget_cap_s)

        second_stage_verify_marker = os.getenv(
            "AIMO3_SECOND_STAGE_MARKER", AIMO3Config.second_stage_verify_marker
        ).strip() or AIMO3Config.second_stage_verify_marker
        second_stage_verify_require_marker = (
            os.getenv("AIMO3_SECOND_STAGE_REQUIRE_MARKER", "1").strip().lower() not in {"0", "false", "no"}
        )

        return AIMO3Config(
            model_path=model_path,
            served_model_name=served_model_name,
            reuse_existing_server=reuse,
            wickelgren_strategies_enabled=wick,
            strategy_pack_mode=strategy_pack_mode,
            strategy_packs=strategy_packs,
            protocol_enabled=proto,
            display_candidates=disp,
            trace_enabled=trace_enabled,
            trace_path=trace_path,
            trace_include_problem_text=trace_include_problem_text,
            require_cuda=require_cuda,
            server_timeout=server_timeout,
            context_tokens=context_tokens,
            batch_size=batch_size,
            gpu_memory_utilization=gpu_mem,
            kernel_init_workers=kernel_init_workers,
            sandbox_pool_size=sandbox_pool_size,
            sandbox_create_on_exhaustion=sandbox_create_on_exhaustion,
            abort_attempt_after_python_errors=abort_attempt_after_python_errors,
            abort_attempt_after_consecutive_python_errors=abort_attempt_after_consecutive_python_errors,
            recycle_sandbox_after_python_errors=recycle_sandbox_after_python_errors,
            recovery_attempts_enabled=recovery_attempts_enabled,
            recovery_attempts_cap=recovery_attempts_cap,
            recovery_trigger_python_errors=recovery_trigger_python_errors,
            recovery_min_remaining_s=recovery_min_remaining_s,
            recovery_mode=recovery_mode,
            recovery_micro_tool_call_cap=recovery_micro_tool_call_cap,
            format_recovery_enabled=format_recovery_enabled,
            format_recovery_cap=format_recovery_cap,
            format_recovery_trigger_tokens=format_recovery_trigger_tokens,
            format_recovery_min_remaining_s=format_recovery_min_remaining_s,
            tiebreak_enabled=tiebreak_enabled,
            tiebreak_min_remaining_s=tiebreak_min_remaining_s,
            tiebreak_budget_cap_s=tiebreak_budget_cap_s,
            temperature=temperature,
            min_p=min_p,
            temperature_exploration=temperature_exploration,
            temperature_main=temperature_main,
            temperature_code=temperature_code,
            temperature_verification=temperature_verification,
            temperature_formatting=temperature_formatting,
            exploration_attempts=exploration_attempts,
            attempts=attempts,
            workers=workers,
            early_stop=early_stop,
            early_stop_min_verified=early_stop_min_verified,
            base_problem_timeout=base_problem_timeout,
            high_problem_timeout=high_problem_timeout,
            notebook_limit=notebook_limit,
            jupyter_timeout=jupyter_timeout,
            sandbox_timeout=sandbox_timeout,
            second_stage_verify_top_k=second_stage_top_k,
            second_stage_verify_budget_cap=second_stage_cap,
            second_stage_verify_budget_fraction=second_stage_fraction,
            second_stage_verify_marker=second_stage_verify_marker,
            second_stage_verify_require_marker=second_stage_verify_require_marker,
            verification_reserve_fraction=verify_reserve_fraction,
            verification_reserve_cap=verify_reserve_cap,
            verification_reserve_min=verify_reserve_min,
            problems_total=problems_total,
        )
