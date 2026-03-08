from __future__ import annotations

import threading
import time
from types import MethodType

from olympiad_llm.aimo3.v2.attempts import AttemptResult, AttemptStats
from olympiad_llm.aimo3.v2.budget import TimeBudgetTracker
from olympiad_llm.aimo3.v2.config import AIMO3Config
from olympiad_llm.aimo3.v2.meta_learning import AdaptiveHyperparameters, ProblemFeatures
from olympiad_llm.aimo3.v2.solver import AIMO3Solver
from olympiad_llm.aimo3.v2.tools import AIMO3Tool
from olympiad_llm.aimo3.v2.trace import TraceRecorder
from olympiad_llm.aimo3.v2.verification import ToolOutputVerifier


class _DummyBudget:
    def __init__(self) -> None:
        self.time_remaining_s = 9999.0
        self.problems_remaining = 50

    def compute_budget(self) -> float:
        return 5.0

    def status_summary(self) -> str:
        return "ok"

    def request_no_answer_extension(self, *, time_spent_s: float, current_budget_s: float) -> float:
        _ = time_spent_s
        _ = current_budget_s
        return 2.0

    def record_solve(self, time_used_s: float, allocated_budget_s: float | None = None) -> None:
        _ = time_used_s
        _ = allocated_budget_s
        self.problems_remaining -= 1


def test_second_wave_uses_integer_attempt_index() -> None:
    solver = object.__new__(AIMO3Solver)
    solver.cfg = AIMO3Config(
        attempts=1,
        workers=1,
        verify_phase_enabled=False,
        display_candidates=False,
        trace_enabled=False,
        meta_learning_enabled=False,
    )
    solver._budget_tracker = _DummyBudget()
    solver._trace = TraceRecorder(enabled=False, path="tmp_ignore.jsonl")
    solver.problems_remaining = 50

    call_indices: list[int | str] = []

    def _adapt(self, problem: str) -> tuple[None, dict]:
        _ = problem
        return None, {}

    def _build(self, attempt_index: int, problem_text: str | None = None, used_strategies=None, preferred_strategy=None):
        _ = problem_text
        _ = used_strategies
        _ = preferred_strategy
        return "dev-prompt", f"tag-{attempt_index}", "strat"

    def _process(self, problem: str, developer_prompt: str, attempt_index: int, attempt_tag: str | None, stop_event, deadline: float, **kwargs) -> AttemptResult:
        _ = problem
        _ = developer_prompt
        _ = attempt_tag
        _ = stop_event
        _ = deadline
        _ = kwargs
        call_indices.append(attempt_index)
        ans = None if len(call_indices) == 1 else 123
        return AttemptResult(attempt=int(attempt_index) + 1, answer=ans, stats=AttemptStats())

    solver._adapt_problem_hyperparameters = MethodType(_adapt, solver)
    solver._build_attempt_prompt = MethodType(_build, solver)
    solver._process_attempt = MethodType(_process, solver)
    solver._display_candidates = MethodType(lambda self, attempts: None, solver)
    solver._should_early_stop = MethodType(lambda self, detailed, *_args, **_kwargs: False, solver)
    solver._should_run_verification = MethodType(lambda self, ranked, time_remaining_s: False, solver)
    solver._update_meta_learning_from_problem_outcome = MethodType(
        lambda self, **kwargs: None, solver
    )

    final_answer = solver.solve_problem("dummy problem")
    assert final_answer == 123
    assert len(call_indices) == 2
    assert all(isinstance(idx, int) for idx in call_indices)


def test_verify_candidates_returns_uniform_tuple_schema() -> None:
    solver = object.__new__(AIMO3Solver)
    solver.cfg = AIMO3Config(
        workers=1,
        verify_top_k_candidates=2,
        verify_attempts_per_candidate=1,
    )
    solver._run_verify_attempt = MethodType(
        lambda self, problem, ans, strategy_template, attempt_seed, deadline, problem_id=None: {
            "candidate": ans,
            "verdict": "UNKNOWN",
            "alt_answer": None,
            "error": None,
        },
        solver,
    )
    ranked = [
        (11, {"votes": 2, "verified": 1, "entropy_score": 0.0}),
        (12, {"votes": 1, "verified": 0, "entropy_score": 0.0}),
    ]
    out = solver._verify_candidates("p", ranked, deadline=time.time() + 10.0)
    assert out
    assert all(isinstance(row, tuple) and len(row) == 2 for row in out)
    assert all("verify_correct" in row[1] and "verify_incorrect" in row[1] for row in out)


def test_record_attempt_trace_emits_attempt_end_event() -> None:
    solver = object.__new__(AIMO3Solver)
    solver.cfg = AIMO3Config(
        trace_enabled=True,
        trace_attempts_enabled=True,
        trace_attempts_max_chars=1000,
    )
    events: list[dict] = []
    solver._trace = type(
        "TraceSink",
        (),
        {
            "record": staticmethod(lambda ev: events.append(ev)),
            "include_problem_text": False,
        },
    )()
    solver._truncate = AIMO3Solver._truncate

    attempt = AttemptResult(
        attempt=1,
        answer=42,
        stats=AttemptStats(token_count=10, python_calls=1, python_errors=0),
        output_text="done",
        tag="wickelgren:test",
        python_calls_text=("print(1)",),
        python_outputs_text=("1",),
    )
    solver._record_attempt_trace("pid-1", attempt)
    assert len(events) == 1
    ev = events[0]
    assert ev["event"] == "attempt_end"
    assert ev["problem_id"] == "pid-1"
    assert ev["attempt"] == 1
    assert ev["answer"] == 42
    assert ev["python_calls_text"] == ["print(1)"]


def test_budget_carryover_disabled_via_env(monkeypatch) -> None:
    monkeypatch.setenv("AIMO3_BUDGET_STRATEGY", "cumulative")
    monkeypatch.setenv("AIMO3_CARRYOVER_ENABLED", "0")
    monkeypatch.setenv("AIMO3_CUMULATIVE_DISTRIBUTE", "0")
    tracker = TimeBudgetTracker(total_budget_s=1000.0, total_problems=10, base_timeout_s=100.0)
    tracker.record_solve(time_used_s=50.0, allocated_budget_s=100.0)
    assert tracker.carryover_enabled is False
    assert tracker.carryover_pool_s == 0.0


def test_adaptive_hparams_fallback_to_general_for_uncertain_features() -> None:
    hparams = AdaptiveHyperparameters(
        default_config=AIMO3Config(attempts=16, temperature=1.0, early_stop=5)
    )
    cfg = hparams.get_config(ProblemFeatures())
    assert cfg["preferred_strategy"] is None
    assert cfg["attempts"] == 16
    assert cfg["temperature"] == 1.0
    assert cfg["early_stop"] == 5


def test_adaptive_hparams_preserve_user_knobs_for_typed_defaults() -> None:
    hparams = AdaptiveHyperparameters(
        default_config=AIMO3Config(attempts=16, temperature=1.0, early_stop=5)
    )
    cfg = hparams.get_config(
        ProblemFeatures(has_modular_arithmetic=1.0, has_number_theory=0.5)
    )
    assert cfg["attempts"] == 16
    assert cfg["temperature"] == 1.0
    assert cfg["early_stop"] == 5
    assert cfg["preferred_strategy"] == "modular_arithmetic"


def test_build_attempt_prompt_uses_answer_only_first_wave() -> None:
    solver = object.__new__(AIMO3Solver)
    solver.cfg = AIMO3Config(
        system_prompt="full-prompt",
        answer_only_prompt="answer-only-prompt",
        answer_only_attempts=4,
        wickelgren_strategies_enabled=False,
        meta_learning_enabled=False,
    )

    first_prompt, first_tag, first_strategy = solver._build_attempt_prompt(0)
    later_prompt, later_tag, later_strategy = solver._build_attempt_prompt(4)

    assert first_prompt == "answer-only-prompt"
    assert first_tag == "answer-only"
    assert first_strategy is None
    assert later_prompt == "full-prompt"
    assert later_tag is None
    assert later_strategy is None


def test_solve_problem_stops_before_full_wave_when_answer_only_consensus_hits() -> None:
    solver = object.__new__(AIMO3Solver)
    solver.cfg = AIMO3Config(
        attempts=4,
        workers=2,
        answer_only_attempts=2,
        early_stop=1,
        early_stop_min_verified=0,
        verify_phase_enabled=False,
        display_candidates=False,
        trace_enabled=False,
        meta_learning_enabled=False,
    )
    solver._budget_tracker = _DummyBudget()
    solver._trace = TraceRecorder(enabled=False, path="tmp_ignore.jsonl")
    solver.problems_remaining = 50

    call_indices: list[int] = []

    def _adapt(self, problem: str) -> tuple[None, dict]:
        _ = problem
        return None, {}

    def _build(self, attempt_index: int, problem_text: str | None = None, used_strategies=None, preferred_strategy=None):
        _ = problem_text
        _ = used_strategies
        _ = preferred_strategy
        if attempt_index < self.cfg.answer_only_attempts:
            return "answer-only", "answer-only", None
        return "full", f"tag-{attempt_index}", "strat"

    def _process(self, problem: str, developer_prompt: str, attempt_index: int, attempt_tag: str | None, stop_event, deadline: float, **kwargs) -> AttemptResult:
        _ = problem
        _ = developer_prompt
        _ = attempt_tag
        _ = stop_event
        _ = deadline
        _ = kwargs
        call_indices.append(attempt_index)
        return AttemptResult(
            attempt=attempt_index + 1,
            answer=42,
            stats=AttemptStats(),
        )

    solver._adapt_problem_hyperparameters = MethodType(_adapt, solver)
    solver._build_attempt_prompt = MethodType(_build, solver)
    solver._process_attempt = MethodType(_process, solver)
    solver._display_candidates = MethodType(lambda self, attempts: None, solver)
    solver._should_early_stop = MethodType(
        lambda self, detailed, *_args, **_kwargs: any(r.answer is not None for r in detailed),
        solver,
    )
    solver._should_run_verification = MethodType(lambda self, ranked, time_remaining_s: False, solver)
    solver._update_meta_learning_from_problem_outcome = MethodType(
        lambda self, **kwargs: None, solver
    )

    final_answer = solver.solve_problem("dummy problem")

    assert final_answer == 42
    assert call_indices
    assert all(idx < solver.cfg.answer_only_attempts for idx in call_indices)


def test_startup_runtime_overlaps_kernel_init_with_server_wait() -> None:
    solver = object.__new__(AIMO3Solver)
    solver.cfg = AIMO3Config(
        reuse_existing_server=False,
        server_probe_attempts=1,
        server_probe_timeout=0.01,
        session_timeout=1.0,
        meta_learning_enabled=False,
    )
    solver.port = 8000
    solver.server = None
    solver.client = None

    init_started = threading.Event()
    wait_ready_called = threading.Event()

    def _init(self) -> None:
        init_started.set()
        assert wait_ready_called.wait(1.0)

    solver._initialize_kernels = MethodType(_init, solver)
    solver._probe_server_ready = MethodType(lambda self, client, attempts, sleep_s=0.5: False, solver)

    events: list[str] = []

    class _FakeServer:
        def __init__(self, cfg, port) -> None:
            _ = cfg
            _ = port

        def start(self) -> None:
            events.append("start")

        def wait_ready(self, client) -> None:
            _ = client
            assert init_started.wait(1.0)
            events.append("wait_ready")
            wait_ready_called.set()

    class _FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    import olympiad_llm.aimo3.v2.solver as solver_module

    original_server_cls = solver_module.VLLMServer
    solver_module.VLLMServer = _FakeServer
    try:
        solver._startup_runtime("http://0.0.0.0:8000/v1", _FakeOpenAI)
    finally:
        solver_module.VLLMServer = original_server_cls

    assert events == ["start", "wait_ready"]
    assert isinstance(solver.client, _FakeOpenAI)


def test_config_from_env_reads_answer_only_attempts(monkeypatch) -> None:
    monkeypatch.setenv("AIMO3_ANSWER_ONLY_ATTEMPTS", "3")
    monkeypatch.setenv("AIMO3_ANSWER_ONLY_PROMPT", "box-only")
    cfg = AIMO3Config.from_env()
    assert cfg.answer_only_attempts == 3
    assert cfg.answer_only_prompt == "box-only"


def test_tool_output_verification_notice_integration() -> None:
    tool = object.__new__(AIMO3Tool)
    tool._enable_verification = True
    tool._verifier = ToolOutputVerifier()
    tool._jupyter_session = None
    tool._owns_session = False
    out = tool._augment_output_with_verification("answer: 42", expected_answer=42)
    assert "[VERIFICATION NOTICE] TOOL_OUTPUT_VALID" in out
    assert "VERIFY_OK" in out
