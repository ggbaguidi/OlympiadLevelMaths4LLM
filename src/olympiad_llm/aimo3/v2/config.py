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

    # Adaptive budget extension settings for the TimeBudgetTracker.
    adaptive_budget_flex_pool_fraction: float = 0.15
    adaptive_budget_max_extension: float = 2.0
    adaptive_budget_hardness_trigger: float = 0.5
    adaptive_budget_min_distinct: int = 3

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

    # Easy-exit: aggressive early stop for problems solved quickly with verified support.
    easy_exit_enabled: bool = True
    easy_exit_time_threshold_s: float = 60.0
    easy_exit_min_votes: int = 3
    easy_exit_min_verified: int = 2

    # Minimum number of generated tokens before the streaming \boxed{} extraction
    # kicks in.  This prevents the solver from latching onto intermediate boxed
    # expressions that appear early in the model's chain-of-thought (e.g. restating
    # the problem or referencing a value from continuation context).  Answers are
    # still caught *after* the turn loop via the "final" channel and the
    # last-resort scan, so easy problems are not affected.
    min_tokens_before_stream_extraction: int = 1500

    attempts: int = 8
    workers: int = 16

    # Sandbox state policy
    # By default we reset the sandbox between attempts to avoid cross-attempt contamination.
    # Set to False only for debugging / interactive workflows where you want to reuse
    # functions/variables across attempts.
    sandbox_reset_between_attempts: bool = True

    # Sandbox pool configuration
    kernel_init_workers: int = 2
    sandbox_pool_size: int = 8

    # Strict extraction mode
    strict_fallback_extraction: bool = True

    # Verification marker policy.
    # When False (default), ``tool_verified`` uses the legacy heuristic:
    #   python_calls > 0 **and** python_errors == 0  →  verified.
    # When True, the model must explicitly print ``VERIFY_OK`` in its
    # tool output for the attempt to count as verified.
    require_verification_marker: bool = False

    turns: int = 128
    seed: int = 3

    gpu_memory_utilization: float = 0.96
    temperature: float = 0.95
    min_p: float = 0.05
    top_p: float = 1.0  # Nucleus sampling (1.0 = disabled)
    top_k: int = -1  # Top-k sampling (-1 = disabled)

    # ---------- Answer-conditional verification phase ----------
    # After the generation phase, stress-test the top candidate answers by
    # running short, focused verification attempts (substitution, counterexample,
    # alternative-method checks).  This catches "wrong but popular" answers on
    # hard problems without materially increasing wall-clock time.
    verify_phase_enabled: bool = True
    # Max wall-clock seconds for the entire verification phase.
    verify_timeout_s: float = 60.0
    # Max completion tokens per verification attempt (short & focused).
    verify_max_tokens: int = 16384
    # Number of parallel verification attempts per candidate answer.
    verify_attempts_per_candidate: int = 3
    # How many of the top-ranked distinct candidates to verify.
    verify_top_k_candidates: int = 3
    # Only trigger verification when the top answer has fewer than this many votes.
    # If consensus is already strong, skip verification to save time.
    verify_trigger_max_votes: int = 4
    # Minimum remaining time (seconds) to even attempt verification.
    # If the budget is tighter than this, skip.
    verify_min_remaining_s: float = 90.0
    # Temperature for verification attempts (lower = more deterministic checks).
    verify_temperature: float = 0.6

    # Hardware requirements
    # vLLM (as used in Kaggle) typically requires an NVIDIA GPU with a working driver.
    # If True and no CUDA driver/GPU is detected, we fail fast with a helpful error.
    require_cuda: bool = True

    @staticmethod
    def from_env() -> "AIMO3Config":
        """Create a config by reading from environment variables, falling back to defaults."""

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

        require_verification_marker = os.getenv(
            "AIMO3_PYTHON_TOOL_VERIFY_REQUIRE_MARKER", "0"
        ).strip().lower() not in {"0", "false", "no"}

        seed = _env_int("AIMO3_SEED", AIMO3Config.seed)
        preference_prompt = os.getenv(
            "AIMO3_PREFERENCE_PROMPT", AIMO3Config.preference_prompt
        )
        tool_prompt = os.getenv("AIMO3_TOOL_PROMPT", AIMO3Config.tool_prompt)
        system_prompt = os.getenv("AIMO3_SYSTEM_PROMPT", AIMO3Config.system_prompt)

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
        min_tokens_before_stream_extraction = _env_int(
            "AIMO3_MIN_TOKENS_BEFORE_STREAM_EXTRACTION",
            AIMO3Config.min_tokens_before_stream_extraction,
        )

        # Easy-exit tuning
        easy_exit_enabled = os.getenv(
            "AIMO3_EASY_EXIT_ENABLED", "1"
        ).strip().lower() not in {"0", "false", "no"}
        easy_exit_time_threshold_s = _env_float(
            "AIMO3_EASY_EXIT_TIME_THRESHOLD", AIMO3Config.easy_exit_time_threshold_s
        )
        easy_exit_min_votes = _env_int(
            "AIMO3_EASY_EXIT_MIN_VOTES", AIMO3Config.easy_exit_min_votes
        )
        easy_exit_min_verified_ee = _env_int(
            "AIMO3_EASY_EXIT_MIN_VERIFIED", AIMO3Config.easy_exit_min_verified
        )

        # Verification phase tuning
        verify_phase_enabled = os.getenv(
            "AIMO3_VERIFY_PHASE_ENABLED", "1"
        ).strip().lower() not in {"0", "false", "no"}
        verify_timeout_s = _env_float(
            "AIMO3_VERIFY_TIMEOUT", AIMO3Config.verify_timeout_s
        )
        verify_max_tokens = _env_int(
            "AIMO3_VERIFY_MAX_TOKENS", AIMO3Config.verify_max_tokens
        )
        verify_attempts_per_candidate = _env_int(
            "AIMO3_VERIFY_ATTEMPTS_PER_CANDIDATE",
            AIMO3Config.verify_attempts_per_candidate,
        )
        verify_top_k_candidates = _env_int(
            "AIMO3_VERIFY_TOP_K_CANDIDATES", AIMO3Config.verify_top_k_candidates
        )
        verify_trigger_max_votes = _env_int(
            "AIMO3_VERIFY_TRIGGER_MAX_VOTES", AIMO3Config.verify_trigger_max_votes
        )
        verify_min_remaining_s = _env_float(
            "AIMO3_VERIFY_MIN_REMAINING", AIMO3Config.verify_min_remaining_s
        )
        verify_temperature = _env_float(
            "AIMO3_VERIFY_TEMPERATURE", AIMO3Config.verify_temperature
        )

        # Adaptive budget tuning
        adaptive_budget_flex_pool_fraction = _env_float(
            "AIMO3_ADAPTIVE_BUDGET_FLEX_POOL_FRACTION",
            AIMO3Config.adaptive_budget_flex_pool_fraction,
        )
        adaptive_budget_max_extension = _env_float(
            "AIMO3_ADAPTIVE_BUDGET_MAX_EXTENSION",
            AIMO3Config.adaptive_budget_max_extension,
        )
        adaptive_budget_hardness_trigger = _env_float(
            "AIMO3_ADAPTIVE_BUDGET_HARDNESS_TRIGGER",
            AIMO3Config.adaptive_budget_hardness_trigger,
        )
        adaptive_budget_min_distinct = _env_int(
            "AIMO3_ADAPTIVE_BUDGET_MIN_DISTINCT",
            AIMO3Config.adaptive_budget_min_distinct,
        )

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
            easy_exit_enabled=easy_exit_enabled,
            easy_exit_time_threshold_s=easy_exit_time_threshold_s,
            easy_exit_min_votes=easy_exit_min_votes,
            easy_exit_min_verified=easy_exit_min_verified_ee,
            base_problem_timeout=base_problem_timeout,
            high_problem_timeout=high_problem_timeout,
            notebook_limit=notebook_limit,
            jupyter_timeout=jupyter_timeout,
            sandbox_timeout=sandbox_timeout,
            problems_total=problems_total,
            adaptive_budget_flex_pool_fraction=adaptive_budget_flex_pool_fraction,
            adaptive_budget_max_extension=adaptive_budget_max_extension,
            adaptive_budget_hardness_trigger=adaptive_budget_hardness_trigger,
            adaptive_budget_min_distinct=adaptive_budget_min_distinct,
            turns=turns,
            min_tokens_before_stream_extraction=min_tokens_before_stream_extraction,
            strict_fallback_extraction=strict_fallback_extraction,
            require_verification_marker=require_verification_marker,
            verify_phase_enabled=verify_phase_enabled,
            verify_timeout_s=verify_timeout_s,
            verify_max_tokens=verify_max_tokens,
            verify_attempts_per_candidate=verify_attempts_per_candidate,
            verify_top_k_candidates=verify_top_k_candidates,
            verify_trigger_max_votes=verify_trigger_max_votes,
            verify_min_remaining_s=verify_min_remaining_s,
            verify_temperature=verify_temperature,
        )
