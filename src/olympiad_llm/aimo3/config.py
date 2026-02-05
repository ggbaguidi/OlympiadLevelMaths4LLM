from __future__ import annotations

import os
from dataclasses import dataclass

from .prompts import ENHANCED_TOOL_INSTRUCTION, TIR_PROMPT_STANDARD, PREFERENCE_PROMPT


@dataclass(frozen=True)
class AIMO3Config:
    """Configuration for the AIMO-3 solver loop.

    Defaults are taken from the notebook (`aimo-3.py`) but converted to a
    dataclass with explicit types.
    """

    # Prompts
    system_prompt: str = TIR_PROMPT_STANDARD
    tool_prompt: str = ENHANCED_TOOL_INSTRUCTION
    preference_prompt: str = PREFERENCE_PROMPT

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
    # If True, shuffle cards per-problem (deterministic but different order per problem).
    # Ensures full coverage within each problem while varying exploration order across problems.
    shuffle_cards: bool = True

    # Attempt-level protocol (lemmas + verification gate)
    protocol_enabled: bool = True

    # Prompt selection (first-stage rotation)
    # Comma-separated list of prompt kinds to disable.
    # Known kinds: standard, code_first, analytic, verification.
    # Example: AIMO3_DISABLE_PROMPTS="verification,analytic"
    disabled_prompts: str = ""

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

    # If enabled, also record per-attempt transcripts to the same JSONL trace.
    # Notes:
    # - By default we do NOT record hidden analysis/CoT.
    # - We record user-visible channels (final/commentary) + python tool I/O.
    trace_attempts_enabled: bool = False
    # Hard cap on total characters stored per attempt transcript payload.
    trace_attempts_max_chars: int = 20000

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

    # Inference server backend: "vllm" (default) or "llama_cpp"
    inference_backend: str = "vllm"
    # Number of GPU layers to offload for llama.cpp (-1 = all, 0 = CPU only)
    llama_cpp_n_gpu_layers: int = -1

    # Optional: warm OS page cache by reading model shards before starting vLLM.
    # This reduces cold-start stalls and first-token latency in notebook runtimes.
    preload_model_weights: bool = False
    preload_model_workers: int = 8

    # Optional: Lean4 toolchain bootstrap (offline Kaggle).
    # If enabled, the solver will attempt to locate/extract a Lean toolchain archive
    # (e.g. lean-<ver>-linux.tar.gz) and make `lean`/`lake` available on PATH.
    #
    # This is NOT required for normal operation; it only matters if you want to
    # call `lean`/`lake` from inside the python tool.
    lean_toolchain_enabled: bool = False
    # Kaggle dataset mount directory containing the tar.gz (e.g. /kaggle/input/<dataset>)
    lean_toolchain_dataset_dir: str = ""
    # Full path to the tar.gz (overrides dataset_dir if set)
    lean_toolchain_archive_path: str = ""
    # Optional: specific filename inside dataset_dir (if multiple archives exist)
    lean_toolchain_archive_name: str = ""
    # Writable directory to extract into (defaults to /kaggle/working/lean4 when present)
    lean_toolchain_work_dir: str = ""
    # Print setup diagnostics (useful while debugging Kaggle mounts)
    lean_toolchain_verbose: bool = False

    # Time budgets (seconds)
    high_problem_timeout: float = 900.0
    base_problem_timeout: float = 300.0
    notebook_limit: float = 17520.0
    server_timeout: float = 180.0
    session_timeout: float = 960.0
    jupyter_timeout: float = 30.0
    sandbox_timeout: float = 5.0

    # Python tool execution timeout handling
    # The sandbox has a default timeout (`jupyter_timeout`). If a tool call times out,
    # the solver can optionally retry once with a longer timeout (bounded by a cap).
    python_tool_timeout_cap_s: float = 180.0
    python_tool_timeout_retry_enabled: bool = True
    python_tool_timeout_retry_multiplier: float = 2.0
    python_tool_timeout_retry_min_remaining_s: float = 5.0

    # Tool verification marker policy (first-stage attempts)
    # If required, a tool run is only considered verified if the python output contains the marker.
    python_tool_verify_marker: str = "VERIFY_OK"
    python_tool_verify_require_marker: bool = True

    # Timeout handling policy
    # If a python tool call times out, it's often a sign of a "wedged" kernel or an overly heavy computation.
    # These toggles let us fail fast and keep the pool healthy.
    abort_attempt_on_python_timeout: bool = True
    recycle_sandbox_on_python_timeout: bool = True

    # Budget allocator assumes a fixed number of remaining problems.
    # Set to 1 when debugging a single hard problem locally.
    problems_total: int = 50

    # Adaptive budget extension: reserve a flex pool for hard problems.
    # When a problem shows hardness signals (no consensus after 50% of base budget),
    # draw extra time from the flex pool.
    adaptive_budget_enabled: bool = True
    # Fraction of total notebook time reserved as flex pool
    adaptive_budget_flex_pool_fraction: float = 0.15
    # Max extension multiplier (e.g., 2.0 = can double base budget for hard problems)
    adaptive_budget_max_extension: float = 2.0
    # Trigger extension check after this fraction of base budget spent
    adaptive_budget_hardness_trigger: float = 0.5
    # Min distinct answers to consider "no consensus"
    adaptive_budget_min_distinct: int = 3
    # Consensus detection: minimum total answers before checking
    adaptive_budget_consensus_min_answers: int = 3
    # Consensus detection: minimum votes for one answer to count as consensus
    adaptive_budget_consensus_min_votes: int = 2

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
    # Early-stop quality guardrails
    # Require at least this many "clean tool" attempts (python_calls>0 and python_errors==0)
    # for the leading candidate before we stop early.
    # Set to 0 to revert to vote-only early stopping.
    early_stop_min_verified: int = 1
    
    # Easy problem early exit: if we get consensus quickly, stop and bank time
    # This helps accumulate flex pool time for harder problems
    easy_exit_enabled: bool = True
    # Time threshold: if we reach consensus within this many seconds, exit early
    easy_exit_time_threshold_s: float = 60.0
    # Min votes for easy exit (can be lower than regular early_stop since we require verified)
    easy_exit_min_votes: int = 3
    # Require at least this many verified attempts for easy exit
    easy_exit_min_verified: int = 2
    
    attempts: int = 8
    workers: int = 16

    # Phase scheduling (optional)
    # If > 0, then until we either (a) find at least one extracted integer answer, or
    # (b) this many seconds have elapsed for the current problem, we prioritize
    # code-first / tool-heavy prompting and postpone proof-y prompts.
    #
    # This is a cheap way to reduce wasted “long proof with no boxed answer” tokens early.
    code_first_phase_s: float = 0.0
    # Concurrency used only during *kernel creation*. High values can cause port races.
    kernel_init_workers: int = 4
    # How many sandbox kernels to keep warm in the pool.
    # Creating `workers` kernels up-front can be expensive/unreliable in notebook runtimes.
    sandbox_pool_size: int = 4
    # If the pool is exhausted, allow creating an ephemeral sandbox for that attempt.
    sandbox_create_on_exhaustion: bool = True

    # Sandbox state policy
    # By default we reset the sandbox between attempts to avoid cross-attempt contamination.
    # Set to False only for debugging / interactive workflows where you want to reuse
    # functions/variables across attempts.
    sandbox_reset_between_attempts: bool = True

    # Tool-error recovery (general robustness)
    # Abort an attempt early if it is repeatedly failing tool calls, to avoid wasting tokens/time.
    abort_attempt_after_python_errors: int = 4
    abort_attempt_after_consecutive_python_errors: int = 3

    # If a sandbox produced many tool errors, consider it "poisoned" and recycle it.
    # Recycling means closing it and creating a fresh sandbox to keep the pool healthy.
    recycle_sandbox_after_python_errors: int = 4

    # Constraint Discovery (novel): force the model to analyze problem structure before solving.
    # This helps with hard problems where jumping straight to computation leads to wrong approaches.
    # The model must first identify: answer type, constraints, impossible values, candidate techniques.
    constraint_discovery_enabled: bool = True
    # Use the dedicated constraint discovery prompt for some attempts
    constraint_discovery_prompt_fraction: float = 0.25  # 25% of attempts use discovery prompt
    # Inject a discovery prefix into the user prompt for all attempts
    constraint_discovery_prefix_enabled: bool = True

    # Working Memory Scratchpad (novel): force explicit state tracking between reasoning steps.
    # Prevents circular reasoning, tracks attempted approaches, maintains focus on goals.
    scratchpad_enabled: bool = True
    # Use the dedicated scratchpad prompt for some attempts
    scratchpad_prompt_fraction: float = 0.20  # 20% of attempts use scratchpad prompt
    # Inject lightweight scratchpad reminder into user prompts
    scratchpad_reminder_enabled: bool = True

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

    # Answer extraction: if True, only extract unboxed integers when near explicit "answer" language.
    # If False, falls back to any integer in text (can pick up random numbers from reasoning).
    strict_fallback_extraction: bool = True

    # Extraction/format recovery (general): if the model produced lots of tokens but we couldn't
    # extract an integer answer, schedule a short attempt focused purely on producing the final
    # boxed integer.
    format_recovery_enabled: bool = True
    format_recovery_cap: int = 1
    format_recovery_trigger_tokens: int = 2000
    format_recovery_min_remaining_s: float = 20.0

    # Contradiction-driven retry (novel): when attempts produce wildly different answers,
    # it signals a fundamental misunderstanding. Trigger a special retry that explicitly
    # tells the model about the disagreement and asks it to re-read the problem carefully.
    contradiction_retry_enabled: bool = True
    # Trigger when top answer has <= this many votes AND there are >= N distinct answers
    contradiction_retry_max_top_votes: int = 2
    contradiction_retry_min_distinct_answers: int = 3
    contradiction_retry_min_remaining_s: float = 45.0
    contradiction_retry_budget_cap_s: float = 90.0
    # Conclusion prompting (per-attempt): when the model is generating many tokens without
    # concluding, inject a nudge message asking it to finalize. This helps with the common
    # failure mode: model does 20k+ tokens of computation but never outputs \boxed{}.
    conclude_nudge_enabled: bool = True
    # Token threshold to trigger the nudge. Set to 0 to disable.
    conclude_nudge_tokens: int = 16000
    # Only send one nudge per attempt.
    conclude_nudge_once: bool = True
    # Hard-problem robustness: when we ask the model to conclude (nudge/finalization),
    # prepend the *last N* python tool outputs. This helps prevent the model from
    # "losing the thread" after long tool-heavy exploration.
    recent_python_outputs_in_conclusion_enabled: bool = True
    recent_python_outputs_in_conclusion_n: int = 5
    # Per-output character cap (0 => unlimited). Keep this bounded to avoid blowing
    # up context on verbose tracebacks / large prints.
    recent_python_outputs_in_conclusion_max_chars: int = 2000
    # Finalization (per-attempt): if an attempt did tool work but never emitted a clean final
    # boxed integer, do one short “final answer only” completion to force synthesis.
    # This helps with the common failure mode: last tool call ran, then the attempt ends
    # without a final answer line.
    finalize_answer_enabled: bool = True
    finalize_answer_max_tokens: int = 128
    finalize_answer_min_remaining_s: float = 3.0

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
    top_p: float = 1.0  # Nucleus sampling (1.0 = disabled)
    top_k: int = -1     # Top-k sampling (-1 = disabled)

            # Optional: per-role temperature schedule (general)
    # If a value is None, the solver will fall back to `temperature`.
    temperature_exploration: float | None = 0.95
    temperature_main: float | None = 0.70
    temperature_code: float | None = 0.65
    temperature_verification: float | None = 0.15  # Decreased from 0.20
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
    verification_reserve_fraction: float = 0.20  # Increased from 0.15
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

    # Math knowledge retriever (RAG): inject relevant mathematical concepts from a
    # pre-built knowledge base into the prompt. Helps with theorems, definitions, etc.
    # The knowledge base must be pre-computed using scripts/extract_math_concepts.py
    retriever_enabled: bool = False  # Disabled by default (requires knowledge base)
    retriever_knowledge_base_path: str = ""  # Path to knowledge base directory
    retriever_model_path: str = ""  # Local path to sentence-transformer model (for offline/Kaggle)
    retriever_cpu_only: bool = True  # Force CPU-only inference (recommended for Kaggle)
    retriever_top_k: int = 5  # Number of concepts to retrieve
    retriever_min_score: float = 0.35  # Minimum similarity threshold
    retriever_include_examples: bool = True  # Include example problems
    retriever_include_definitions: bool = True  # Include definitions
    retriever_warmup_on_init: bool = True  # Pre-load embedding model at solver init

    # Adversarial debate verification (novel): when second-stage verify is inconclusive,
    # run an adversarial debate where one agent critiques and another defends.
    # This catches subtle errors that simple verification misses.
    adversarial_debate_enabled: bool = True
    # Trigger when: second-stage verify didn't decide AND we have time AND top answers are close
    adversarial_debate_min_remaining_s: float = 30.0
    adversarial_debate_budget_cap_s: float = 60.0
    # Number of critique-defend rounds
    adversarial_debate_rounds: int = 1
    # If critique finds flaw, run arbiter to pick between original and revised answer
    adversarial_debate_use_arbiter: bool = True

    @staticmethod
    def from_env() -> "AIMO3Config":
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
        
        seed = _env_int("AIMO3_SEED", AIMO3Config.seed)
        preference_prompt = os.getenv("AIMO3_PREFERENCE_PROMPT", AIMO3Config.preference_prompt)
        tool_prompt = os.getenv("AIMO3_TOOL_PROMPT", AIMO3Config.tool_prompt)
        system_prompt = os.getenv("AIMO3_SYSTEM_PROMPT", AIMO3Config.system_prompt)

        disabled_prompts = (os.getenv("AIMO3_DISABLE_PROMPTS", "") or "").strip()

        # Profile presets (apply only when the corresponding env var is NOT explicitly set).
        # This makes it easy to reduce orchestration steps without rewriting many env vars.
        profile = (os.getenv("AIMO3_PROFILE", "") or "").strip().lower()
        if profile not in {"", "default", "full", "lean"}:
            profile = ""

        model_path = os.path.expanduser(os.getenv("AIMO3_MODEL_PATH", ""))
        served_model_name = os.getenv("AIMO3_SERVED_MODEL_NAME", "gpt-oss")
        inference_backend = (os.getenv("AIMO3_INFERENCE_BACKEND", AIMO3Config.inference_backend) or "").strip()
        llama_cpp_n_gpu_layers = _env_int("AIMO3_LLAMA_CPP_N_GPU_LAYERS", AIMO3Config.llama_cpp_n_gpu_layers)

        reuse = os.getenv("AIMO3_REUSE_EXISTING_SERVER", "1").strip().lower() not in {"0", "false", "no"}
        wick = os.getenv("AIMO3_WICKELGREN", "1").strip().lower() not in {"0", "false", "no"}
        strategy_pack_mode = (os.getenv("AIMO3_STRATEGY_PACK_MODE", AIMO3Config.strategy_pack_mode) or "").strip()
        if not strategy_pack_mode:
            strategy_pack_mode = AIMO3Config.strategy_pack_mode
        strategy_packs = (os.getenv("AIMO3_STRATEGY_PACKS", AIMO3Config.strategy_packs) or "").strip()
        if not strategy_packs:
            strategy_packs = AIMO3Config.strategy_packs
        shuffle_cards = os.getenv("AIMO3_SHUFFLE_CARDS", "1").strip().lower() not in {"0", "false", "no"}

        trace_enabled = os.getenv("AIMO3_TRACE", "0").strip().lower() not in {"0", "false", "no"}
        trace_path = os.path.expanduser((os.getenv("AIMO3_TRACE_PATH", AIMO3Config.trace_path) or "").strip() or AIMO3Config.trace_path)
        trace_include_problem_text = (
            os.getenv("AIMO3_TRACE_INCLUDE_PROBLEM_TEXT", "0").strip().lower() not in {"0", "false", "no"}
        )
        trace_reset_on_start = (
            os.getenv("AIMO3_TRACE_RESET_ON_START", "1").strip().lower() not in {"0", "false", "no"}
        )
        trace_attempts_enabled = os.getenv("AIMO3_TRACE_ATTEMPTS", "0").strip().lower() not in {"0", "false", "no"}
        trace_attempts_max_chars = _env_int("AIMO3_TRACE_ATTEMPTS_MAX_CHARS", AIMO3Config.trace_attempts_max_chars)
        trace_env_enabled = os.getenv("AIMO3_TRACE_ENV", "0").strip().lower() not in {"0", "false", "no"}
        trace_env_packages = (os.getenv("AIMO3_TRACE_ENV_PACKAGES", AIMO3Config.trace_env_packages) or "").strip() or AIMO3Config.trace_env_packages
        proto = os.getenv("AIMO3_PROTOCOL", "1").strip().lower() not in {"0", "false", "no"}
        disp = os.getenv("AIMO3_DISPLAY_CANDIDATES", "1").strip().lower() not in {"0", "false", "no"}
        require_cuda = os.getenv("AIMO3_REQUIRE_CUDA", "1").strip().lower() not in {"0", "false", "no"}

        # Startup perf knobs
        preload_model_weights = (
            os.getenv("AIMO3_PRELOAD_MODEL_WEIGHTS", "0").strip().lower() not in {"0", "false", "no"}
        )
        preload_model_workers = _env_int("AIMO3_PRELOAD_MODEL_WORKERS", AIMO3Config.preload_model_workers)

        # Optional: Lean toolchain bootstrap (offline Kaggle).
        lean_toolchain_enabled = (
            os.getenv("AIMO3_LEAN_TOOLCHAIN_ENABLED", "0").strip().lower() not in {"0", "false", "no"}
        )
        lean_toolchain_dataset_dir = (os.getenv("AIMO3_LEAN_DATASET_DIR", "") or "").strip()
        lean_toolchain_archive_path = (os.getenv("AIMO3_LEAN_ARCHIVE_PATH", "") or "").strip()
        lean_toolchain_archive_name = (os.getenv("AIMO3_LEAN_ARCHIVE_NAME", "") or "").strip()
        lean_toolchain_work_dir = (os.getenv("AIMO3_LEAN_WORK_DIR", "") or "").strip()
        lean_toolchain_verbose = os.getenv("AIMO3_LEAN_VERBOSE", "0").strip().lower() not in {"0", "false", "no"}

        # Core solver knobs
        attempts = _env_int("AIMO3_ATTEMPTS", AIMO3Config.attempts)
        workers = _env_int("AIMO3_WORKERS", AIMO3Config.workers)
        early_stop = _env_int("AIMO3_EARLY_STOP", AIMO3Config.early_stop)
        early_stop_min_verified = _env_int("AIMO3_EARLY_STOP_MIN_VERIFIED", AIMO3Config.early_stop_min_verified)

        # Easy problem early exit
        easy_exit_enabled = (
            os.getenv("AIMO3_EASY_EXIT_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
        )
        easy_exit_time_threshold_s = _env_float(
            "AIMO3_EASY_EXIT_TIME_THRESHOLD_S", AIMO3Config.easy_exit_time_threshold_s
        )
        easy_exit_min_votes = _env_int("AIMO3_EASY_EXIT_MIN_VOTES", AIMO3Config.easy_exit_min_votes)
        easy_exit_min_verified = _env_int("AIMO3_EASY_EXIT_MIN_VERIFIED", AIMO3Config.easy_exit_min_verified)

        turns = _env_int("AIMO3_TURNS", AIMO3Config.turns)

        # Phase scheduling
        code_first_phase_s = _env_float("AIMO3_CODE_FIRST_PHASE_S", AIMO3Config.code_first_phase_s)

        # Time budgets
        base_problem_timeout = _env_float("AIMO3_BASE_PROBLEM_TIMEOUT", AIMO3Config.base_problem_timeout)
        high_problem_timeout = _env_float("AIMO3_HIGH_PROBLEM_TIMEOUT", AIMO3Config.high_problem_timeout)
        notebook_limit = _env_float("AIMO3_NOTEBOOK_LIMIT", AIMO3Config.notebook_limit)

        # Tooling timeouts
        jupyter_timeout = _env_float("AIMO3_JUPYTER_TIMEOUT", AIMO3Config.jupyter_timeout)
        sandbox_timeout = _env_float("AIMO3_SANDBOX_TIMEOUT", AIMO3Config.sandbox_timeout)

        python_tool_timeout_cap_s = _env_float(
            "AIMO3_PYTHON_TOOL_TIMEOUT_CAP_S", AIMO3Config.python_tool_timeout_cap_s
        )
        python_tool_timeout_retry_enabled = (
            os.getenv("AIMO3_PYTHON_TOOL_TIMEOUT_RETRY_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
        )
        python_tool_timeout_retry_multiplier = _env_float(
            "AIMO3_PYTHON_TOOL_TIMEOUT_RETRY_MULT", AIMO3Config.python_tool_timeout_retry_multiplier
        )
        python_tool_timeout_retry_min_remaining_s = _env_float(
            "AIMO3_PYTHON_TOOL_TIMEOUT_RETRY_MIN_REMAINING_S",
            AIMO3Config.python_tool_timeout_retry_min_remaining_s,
        )

        python_tool_verify_marker = (
            os.getenv("AIMO3_PYTHON_TOOL_VERIFY_MARKER", AIMO3Config.python_tool_verify_marker) or ""
        ).strip() or AIMO3Config.python_tool_verify_marker
        python_tool_verify_require_marker = (
            os.getenv("AIMO3_PYTHON_TOOL_VERIFY_REQUIRE_MARKER", "1").strip().lower() not in {"0", "false", "no"}
        )

        abort_attempt_on_python_timeout = (
            os.getenv("AIMO3_ABORT_ATTEMPT_ON_PYTHON_TIMEOUT", "1").strip().lower() not in {"0", "false", "no"}
        )
        recycle_sandbox_on_python_timeout = (
            os.getenv("AIMO3_RECYCLE_SANDBOX_ON_PYTHON_TIMEOUT", "1").strip().lower() not in {"0", "false", "no"}
        )

        # Verification knobs
        second_stage_verify_enabled = (
            os.getenv("AIMO3_SECOND_STAGE_VERIFY_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
        )
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

        # Adaptive budget extension
        adaptive_budget_enabled = (
            os.getenv("AIMO3_ADAPTIVE_BUDGET_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
        )
        adaptive_budget_flex_pool_fraction = _env_float(
            "AIMO3_ADAPTIVE_BUDGET_FLEX_POOL_FRACTION", AIMO3Config.adaptive_budget_flex_pool_fraction
        )
        adaptive_budget_max_extension = _env_float(
            "AIMO3_ADAPTIVE_BUDGET_MAX_EXTENSION", AIMO3Config.adaptive_budget_max_extension
        )
        adaptive_budget_hardness_trigger = _env_float(
            "AIMO3_ADAPTIVE_BUDGET_HARDNESS_TRIGGER", AIMO3Config.adaptive_budget_hardness_trigger
        )
        adaptive_budget_min_distinct = _env_int(
            "AIMO3_ADAPTIVE_BUDGET_MIN_DISTINCT", AIMO3Config.adaptive_budget_min_distinct
        )
        adaptive_budget_consensus_min_answers = _env_int(
            "AIMO3_ADAPTIVE_BUDGET_CONSENSUS_MIN_ANSWERS", AIMO3Config.adaptive_budget_consensus_min_answers
        )
        adaptive_budget_consensus_min_votes = _env_int(
            "AIMO3_ADAPTIVE_BUDGET_CONSENSUS_MIN_VOTES", AIMO3Config.adaptive_budget_consensus_min_votes
        )

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
        sandbox_reset_between_attempts = (
            os.getenv("AIMO3_SANDBOX_RESET_BETWEEN_ATTEMPTS", "1").strip().lower() not in {"0", "false", "no"}
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

        # Constraint discovery (novel approach for hard problems)
        constraint_discovery_enabled = (
            os.getenv("AIMO3_CONSTRAINT_DISCOVERY_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
        )
        constraint_discovery_prompt_fraction = _env_float(
            "AIMO3_CONSTRAINT_DISCOVERY_PROMPT_FRACTION", AIMO3Config.constraint_discovery_prompt_fraction
        )
        constraint_discovery_prefix_enabled = (
            os.getenv("AIMO3_CONSTRAINT_DISCOVERY_PREFIX_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
        )

        # Working memory scratchpad (novel approach for multi-step reasoning)
        scratchpad_enabled = (
            os.getenv("AIMO3_SCRATCHPAD_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
        )
        scratchpad_prompt_fraction = _env_float(
            "AIMO3_SCRATCHPAD_PROMPT_FRACTION", AIMO3Config.scratchpad_prompt_fraction
        )
        scratchpad_reminder_enabled = (
            os.getenv("AIMO3_SCRATCHPAD_REMINDER_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
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
        top_p = _env_float("AIMO3_TOP_P", AIMO3Config.top_p)
        top_k = _env_int("AIMO3_TOP_K", AIMO3Config.top_k)

        entropy_weighting_enabled = (
            os.getenv("AIMO3_ENTROPY_WEIGHTING_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
        )
        top_logprobs = _env_int("AIMO3_TOP_LOGPROBS", AIMO3Config.top_logprobs)

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

        # Contradiction-driven retry
        contradiction_retry_enabled = (
            os.getenv("AIMO3_CONTRADICTION_RETRY_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
        )
        contradiction_retry_max_top_votes = _env_int(
            "AIMO3_CONTRADICTION_RETRY_MAX_TOP_VOTES", AIMO3Config.contradiction_retry_max_top_votes
        )
        contradiction_retry_min_distinct_answers = _env_int(
            "AIMO3_CONTRADICTION_RETRY_MIN_DISTINCT_ANSWERS", AIMO3Config.contradiction_retry_min_distinct_answers
        )
        contradiction_retry_min_remaining_s = _env_float(
            "AIMO3_CONTRADICTION_RETRY_MIN_REMAINING_S", AIMO3Config.contradiction_retry_min_remaining_s
        )
        contradiction_retry_budget_cap_s = _env_float(
            "AIMO3_CONTRADICTION_RETRY_BUDGET_CAP_S", AIMO3Config.contradiction_retry_budget_cap_s
        )

        finalize_answer_enabled = (
            os.getenv("AIMO3_FINALIZE_ANSWER_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
        )
        finalize_answer_max_tokens = _env_int(
            "AIMO3_FINALIZE_ANSWER_MAX_TOKENS", AIMO3Config.finalize_answer_max_tokens
        )
        finalize_answer_min_remaining_s = _env_float(
            "AIMO3_FINALIZE_ANSWER_MIN_REMAINING_S", AIMO3Config.finalize_answer_min_remaining_s
        )

        # Conclusion nudge (intra-attempt prompting to force synthesis)
        conclude_nudge_enabled = (
            os.getenv("AIMO3_CONCLUDE_NUDGE_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
        )
        conclude_nudge_tokens = _env_int("AIMO3_CONCLUDE_NUDGE_TOKENS", AIMO3Config.conclude_nudge_tokens)
        conclude_nudge_once = (
            os.getenv("AIMO3_CONCLUDE_NUDGE_ONCE", "1").strip().lower() not in {"0", "false", "no"}
        )
        recent_python_outputs_in_conclusion_enabled = (
            os.getenv("AIMO3_RECENT_PYTHON_OUTPUTS_IN_CONCLUSION_ENABLED", "1").strip().lower()
            not in {"0", "false", "no"}
        )
        recent_python_outputs_in_conclusion_n = _env_int(
            "AIMO3_RECENT_PYTHON_OUTPUTS_IN_CONCLUSION_N", AIMO3Config.recent_python_outputs_in_conclusion_n
        )
        recent_python_outputs_in_conclusion_max_chars = _env_int(
            "AIMO3_RECENT_PYTHON_OUTPUTS_IN_CONCLUSION_MAX_CHARS",
            AIMO3Config.recent_python_outputs_in_conclusion_max_chars,
        )

        tiebreak_enabled = os.getenv("AIMO3_TIEBREAK_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
        tiebreak_min_remaining_s = _env_float(
            "AIMO3_TIEBREAK_MIN_REMAINING_S", AIMO3Config.tiebreak_min_remaining_s
        )
        tiebreak_budget_cap_s = _env_float("AIMO3_TIEBREAK_BUDGET_CAP_S", AIMO3Config.tiebreak_budget_cap_s)

        # Math knowledge retriever (RAG) configuration
        retriever_enabled = (
            os.getenv("AIMO3_RETRIEVER_ENABLED", "0").strip().lower() not in {"0", "false", "no"}
        )
        retriever_knowledge_base_path = os.path.expanduser((os.getenv("AIMO3_RETRIEVER_KB_PATH", "") or "").strip())
        retriever_model_path = os.path.expanduser((os.getenv("AIMO3_RETRIEVER_MODEL_PATH", "") or "").strip())
        retriever_top_k = _env_int("AIMO3_RETRIEVER_TOP_K", AIMO3Config.retriever_top_k)
        retriever_min_score = _env_float("AIMO3_RETRIEVER_MIN_SCORE", AIMO3Config.retriever_min_score)
        retriever_include_examples = (
            os.getenv("AIMO3_RETRIEVER_INCLUDE_EXAMPLES", "1").strip().lower() not in {"0", "false", "no"}
        )
        retriever_include_definitions = (
            os.getenv("AIMO3_RETRIEVER_INCLUDE_DEFINITIONS", "1").strip().lower() not in {"0", "false", "no"}
        )
        retriever_cpu_only = (
            os.getenv("AIMO3_RETRIEVER_CPU_ONLY", "1").strip().lower() not in {"0", "false", "no"}
        )
        retriever_warmup_on_init = (
            os.getenv("AIMO3_RETRIEVER_WARMUP_ON_INIT", "1").strip().lower() not in {"0", "false", "no"}
        )

        # Adversarial debate configuration
        adversarial_debate_enabled = (
            os.getenv("AIMO3_ADVERSARIAL_DEBATE_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
        )
        adversarial_debate_min_remaining_s = _env_float(
            "AIMO3_ADVERSARIAL_DEBATE_MIN_REMAINING_S", AIMO3Config.adversarial_debate_min_remaining_s
        )
        adversarial_debate_budget_cap_s = _env_float(
            "AIMO3_ADVERSARIAL_DEBATE_BUDGET_CAP_S", AIMO3Config.adversarial_debate_budget_cap_s
        )
        adversarial_debate_rounds = _env_int(
            "AIMO3_ADVERSARIAL_DEBATE_ROUNDS", AIMO3Config.adversarial_debate_rounds
        )
        adversarial_debate_use_arbiter = (
            os.getenv("AIMO3_ADVERSARIAL_DEBATE_USE_ARBITER", "1").strip().lower() not in {"0", "false", "no"}
        )

        second_stage_verify_marker = os.getenv(
            "AIMO3_SECOND_STAGE_MARKER", AIMO3Config.second_stage_verify_marker
        ).strip() or AIMO3Config.second_stage_verify_marker
        second_stage_verify_require_marker = (
            os.getenv("AIMO3_SECOND_STAGE_REQUIRE_MARKER", "1").strip().lower() not in {"0", "false", "no"}
        )

        # Apply lean defaults if requested.
        # Only override knobs if the user didn't explicitly set them.
        if profile == "lean":
            if not _env_present("AIMO3_ATTEMPTS"):
                attempts = 4
            if not _env_present("AIMO3_WORKERS"):
                workers = max(4, min(8, attempts))
            if not _env_present("AIMO3_TURNS"):
                turns = 64

            if not _env_present("AIMO3_WICKELGREN"):
                wick = False
            if not _env_present("AIMO3_STRATEGY_PACKS"):
                strategy_packs = "generic"
            if not _env_present("AIMO3_STRATEGY_PACK_MODE"):
                strategy_pack_mode = "off"

            if not _env_present("AIMO3_RECOVERY_ATTEMPTS_CAP"):
                recovery_attempts_cap = 1
            if not _env_present("AIMO3_TIEBREAK_ENABLED"):
                tiebreak_enabled = False

        return AIMO3Config(
            seed=seed,
            system_prompt=system_prompt,
            tool_prompt=tool_prompt,
            preference_prompt=preference_prompt,
            disabled_prompts=disabled_prompts,
            model_path=model_path,
            served_model_name=served_model_name,
            preload_model_weights=preload_model_weights,
            preload_model_workers=preload_model_workers,
            reuse_existing_server=reuse,
            inference_backend=inference_backend,
            llama_cpp_n_gpu_layers=llama_cpp_n_gpu_layers,
            wickelgren_strategies_enabled=wick,
            strategy_pack_mode=strategy_pack_mode,
            strategy_packs=strategy_packs,
            shuffle_cards=shuffle_cards,
            protocol_enabled=proto,
            display_candidates=disp,
            trace_enabled=trace_enabled,
            trace_path=trace_path,
            trace_include_problem_text=trace_include_problem_text,
            trace_reset_on_start=trace_reset_on_start,
            trace_attempts_enabled=trace_attempts_enabled,
            trace_attempts_max_chars=trace_attempts_max_chars,
            trace_env_enabled=trace_env_enabled,
            trace_env_packages=trace_env_packages,
            require_cuda=require_cuda,
            lean_toolchain_enabled=lean_toolchain_enabled,
            lean_toolchain_dataset_dir=lean_toolchain_dataset_dir,
            lean_toolchain_archive_path=lean_toolchain_archive_path,
            lean_toolchain_archive_name=lean_toolchain_archive_name,
            lean_toolchain_work_dir=lean_toolchain_work_dir,
            lean_toolchain_verbose=lean_toolchain_verbose,
            server_timeout=server_timeout,
            context_tokens=context_tokens,
            batch_size=batch_size,
            gpu_memory_utilization=gpu_mem,
            kernel_init_workers=kernel_init_workers,
            sandbox_pool_size=sandbox_pool_size,
            sandbox_create_on_exhaustion=sandbox_create_on_exhaustion,
            sandbox_reset_between_attempts=sandbox_reset_between_attempts,
            abort_attempt_after_python_errors=abort_attempt_after_python_errors,
            abort_attempt_after_consecutive_python_errors=abort_attempt_after_consecutive_python_errors,
            recycle_sandbox_after_python_errors=recycle_sandbox_after_python_errors,
            constraint_discovery_enabled=constraint_discovery_enabled,
            constraint_discovery_prompt_fraction=constraint_discovery_prompt_fraction,
            constraint_discovery_prefix_enabled=constraint_discovery_prefix_enabled,
            scratchpad_enabled=scratchpad_enabled,
            scratchpad_prompt_fraction=scratchpad_prompt_fraction,
            scratchpad_reminder_enabled=scratchpad_reminder_enabled,
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
            contradiction_retry_enabled=contradiction_retry_enabled,
            contradiction_retry_max_top_votes=contradiction_retry_max_top_votes,
            contradiction_retry_min_distinct_answers=contradiction_retry_min_distinct_answers,
            contradiction_retry_min_remaining_s=contradiction_retry_min_remaining_s,
            contradiction_retry_budget_cap_s=contradiction_retry_budget_cap_s,
            finalize_answer_enabled=finalize_answer_enabled,
            finalize_answer_max_tokens=finalize_answer_max_tokens,
            finalize_answer_min_remaining_s=finalize_answer_min_remaining_s,
            conclude_nudge_enabled=conclude_nudge_enabled,
            conclude_nudge_tokens=conclude_nudge_tokens,
            conclude_nudge_once=conclude_nudge_once,
            recent_python_outputs_in_conclusion_enabled=recent_python_outputs_in_conclusion_enabled,
            recent_python_outputs_in_conclusion_n=recent_python_outputs_in_conclusion_n,
            recent_python_outputs_in_conclusion_max_chars=recent_python_outputs_in_conclusion_max_chars,
            tiebreak_enabled=tiebreak_enabled,
            tiebreak_min_remaining_s=tiebreak_min_remaining_s,
            tiebreak_budget_cap_s=tiebreak_budget_cap_s,
            retriever_enabled=retriever_enabled,
            retriever_knowledge_base_path=retriever_knowledge_base_path,
            retriever_model_path=retriever_model_path,
            retriever_cpu_only=retriever_cpu_only,
            retriever_top_k=retriever_top_k,
            retriever_min_score=retriever_min_score,
            retriever_include_examples=retriever_include_examples,
            retriever_include_definitions=retriever_include_definitions,
            retriever_warmup_on_init=retriever_warmup_on_init,
            adversarial_debate_enabled=adversarial_debate_enabled,
            adversarial_debate_min_remaining_s=adversarial_debate_min_remaining_s,
            adversarial_debate_budget_cap_s=adversarial_debate_budget_cap_s,
            adversarial_debate_rounds=adversarial_debate_rounds,
            adversarial_debate_use_arbiter=adversarial_debate_use_arbiter,
            temperature=temperature,
            min_p=min_p,
            top_p=top_p,
            top_k=top_k,
            entropy_weighting_enabled=entropy_weighting_enabled,
            top_logprobs=top_logprobs,
            temperature_exploration=temperature_exploration,
            temperature_main=temperature_main,
            temperature_code=temperature_code,
            temperature_verification=temperature_verification,
            temperature_formatting=temperature_formatting,
            exploration_attempts=exploration_attempts,
            attempts=attempts,
            workers=workers,
            code_first_phase_s=code_first_phase_s,
            early_stop=early_stop,
            early_stop_min_verified=early_stop_min_verified,
            easy_exit_enabled=easy_exit_enabled,
            easy_exit_time_threshold_s=easy_exit_time_threshold_s,
            easy_exit_min_votes=easy_exit_min_votes,
            easy_exit_min_verified=easy_exit_min_verified,
            turns=turns,
            base_problem_timeout=base_problem_timeout,
            high_problem_timeout=high_problem_timeout,
            notebook_limit=notebook_limit,
            jupyter_timeout=jupyter_timeout,
            sandbox_timeout=sandbox_timeout,
            python_tool_timeout_cap_s=python_tool_timeout_cap_s,
            python_tool_timeout_retry_enabled=python_tool_timeout_retry_enabled,
            python_tool_timeout_retry_multiplier=python_tool_timeout_retry_multiplier,
            python_tool_timeout_retry_min_remaining_s=python_tool_timeout_retry_min_remaining_s,
            python_tool_verify_marker=python_tool_verify_marker,
            python_tool_verify_require_marker=python_tool_verify_require_marker,
            abort_attempt_on_python_timeout=abort_attempt_on_python_timeout,
            recycle_sandbox_on_python_timeout=recycle_sandbox_on_python_timeout,
            second_stage_verify_enabled=second_stage_verify_enabled,
            second_stage_verify_top_k=second_stage_top_k,
            second_stage_verify_budget_cap=second_stage_cap,
            second_stage_verify_budget_fraction=second_stage_fraction,
            second_stage_verify_marker=second_stage_verify_marker,
            second_stage_verify_require_marker=second_stage_verify_require_marker,
            verification_reserve_fraction=verify_reserve_fraction,
            verification_reserve_cap=verify_reserve_cap,
            verification_reserve_min=verify_reserve_min,
            problems_total=problems_total,
            adaptive_budget_enabled=adaptive_budget_enabled,
            adaptive_budget_flex_pool_fraction=adaptive_budget_flex_pool_fraction,
            adaptive_budget_max_extension=adaptive_budget_max_extension,
            adaptive_budget_hardness_trigger=adaptive_budget_hardness_trigger,
            adaptive_budget_min_distinct=adaptive_budget_min_distinct,
            adaptive_budget_consensus_min_answers=adaptive_budget_consensus_min_answers,
            adaptive_budget_consensus_min_votes=adaptive_budget_consensus_min_votes,
        )
