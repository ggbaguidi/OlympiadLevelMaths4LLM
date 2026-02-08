# pylint: disable=broad-exception-caught,missing-function-docstring,line-too-long,missing-module-docstring,import-outside-toplevel,invalid-name,too-many-instance-attributes
"""AIMO-3 multi-attempt solver (ported and modularized).

This module intentionally keeps imports *lazy* so that the base project can be
installed without the heavy AIMO-3 stack.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import queue
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from dataclasses import dataclass

from .budget import TimeBudgetTracker
from .answer_extraction import AnswerExtractor
from .attempts import AttemptResult, AttemptStats

from .config import AIMO3Config
from .sandbox import AIMO3Sandbox
from .trace import TraceRecorder, stable_problem_id
from .vllm_server import VLLMServer
from .require import _require_harmony, _require_openai
from .template import AIMO3Template
from .tools import AIMO3Tool


def rank_candidates(
    results: list,
    filter_to_verified_if_any: bool = True,
) -> list:
    """Rank candidate answers by votes and verification status.

    Returns a list of (answer, info_dict) tuples sorted by quality.
    """
    if not results:
        return []

    groups: dict = defaultdict(
        lambda: {"votes": 0, "verified": 0, "entropy_score": 0.0}
    )

    for r in results:
        ans = r.answer if isinstance(r, AttemptResult) else r.get("Answer")
        if ans is None:
            continue
        g = groups[ans]
        g["votes"] += 1

        if isinstance(r, AttemptResult):
            if r.stats.tool_verified:
                g["verified"] += 1
            ent = r.stats.mean_entropy
        else:
            ent = r.get("Entropy", float("inf"))

        if ent != float("inf") and ent > 0:
            g["entropy_score"] += 1.0 / max(ent, 1e-9)

    if not groups:
        return []

    has_any_verified = any(g["verified"] > 0 for g in groups.values())
    if filter_to_verified_if_any and has_any_verified:
        groups = {k: v for k, v in groups.items() if v["verified"] > 0}

    ranked = sorted(
        groups.items(),
        key=lambda kv: (kv[1]["verified"], kv[1]["votes"], kv[1]["entropy_score"]),
        reverse=True,
    )
    return [(ans, data) for ans, data in ranked]


@dataclass
class AIMO3Solver:
    """AIMO-3 multi-attempt solver with streaming and tool use."""
    cfg: AIMO3Config
    port: int = 8000

    @staticmethod
    def _truncate(text: str | None, max_chars: int) -> str:
        """Truncate text to max_chars, keeping the tail (most recent output)."""
        s = str(text or "")
        if len(s) <= max_chars:
            return s
        return "..." + s[-(max_chars - 3) :]

    def _attempt_to_row(self, r: AttemptResult) -> dict:
        snippet = self._truncate(
            r.output_text, int(self.cfg.display_attempt_text_chars)
        )
        ent = None
        with contextlib.suppress(Exception):
            v = float(getattr(r.stats, "mean_entropy", float("inf")))
            if v != float("inf") and v > 0.0:
                ent = v
        return {
            "Attempt": r.attempt,
            "Answer": r.answer,
            "ToolVerified": bool(r.stats.tool_verified),
            "PyCalls": int(r.stats.python_calls),
            "Timeouts": int(getattr(r.stats, "timeout_count", 0) or 0),
            "PyErrors": int(r.stats.python_errors),
            "LeanCalls": int(getattr(r.stats, "lean_calls", 0) or 0),
            "Tokens": int(r.stats.token_count),
            "Entropy": ent,
            "Snippet": snippet,
        }

    def _display_candidates(self, attempts: list[AttemptResult]) -> None:
        """Display attempt candidates in notebooks (best-effort).

        - Uses pandas + IPython.display if available.
        - Falls back to plain printing.
        """

        if not bool(self.cfg.display_candidates):
            return

        rows = [self._attempt_to_row(r) for r in attempts]
        if not rows:
            return

        # Keep output manageable.
        max_rows = max(1, int(self.cfg.display_max_rows))
        rows = rows[:max_rows]

        # Prefer notebook display.
        try:
            import pandas as pd  # type: ignore

            df = pd.DataFrame(rows)
            try:
                from IPython.display import display  # type: ignore

                display(df)
            except Exception:  # noqa: BLE001
                print(df.to_string(index=False))
        except Exception:  # noqa: BLE001
            for row in rows:
                print(
                    f"Attempt {row['Attempt']}: ans={row['Answer']} "
                    f"verified={row['ToolVerified']} calls={row['PyCalls']} errors={row['PyErrors']} tokens={row['Tokens']}\n"
                    f"  {row['Snippet']}\n"
                )

    def _sandbox_env_snapshot(self) -> dict | None:
        """Best-effort: query one sandbox for python + package versions.

        Uses a borrowed sandbox from the pool to reflect the actual kernel environment.
        Returns a small dict (JSON-serializable) or None on failure.
        """

        if not hasattr(self, "sandbox_pool"):
            return None
        # Only meaningful if the optional dependency stack is present.
        sb: AIMO3Sandbox | None = None
        try:
            sb = self.sandbox_pool.get(
                timeout=float(getattr(self.cfg, "sandbox_timeout", 0.0) or 0.0) or 0.5
            )
        except Exception:
            sb = None

        if sb is None:
            return None

        try:
            pkg_raw = str(getattr(self.cfg, "trace_env_packages", "") or "")
            packages = [p.strip() for p in pkg_raw.split(",") if p.strip()]

            # Emit a single JSON line as the *last* line, so we can parse robustly.
            code = (
                "import json, sys\n"
                "try:\n"
                "    import platform\n"
                "    _platform = platform.platform()\n"
                "except Exception:\n"
                "    _platform = None\n"
                "def _ver(name):\n"
                "    try:\n"
                "        m = __import__(name)\n"
                "    except Exception:\n"
                "        return None\n"
                "    return getattr(m, '__version__', None)\n"
                f"_names = {packages!r}\n"
                "_pkgs = {n: _ver(n) for n in _names}\n"
                "_info = {\n"
                "  'python': {'version': sys.version, 'executable': sys.executable},\n"
                "  'platform': _platform,\n"
                "  'packages': _pkgs,\n"
                "}\n"
                "print(json.dumps(_info, ensure_ascii=False))\n"
            )
            out = sb.execute(
                code,
                timeout=min(
                    2.0, float(getattr(self.cfg, "jupyter_timeout", 10.0) or 10.0)
                ),
            )

            # Find the last JSON-looking line.
            last_json = None
            for line in str(out or "").splitlines()[::-1]:
                s = line.strip()
                if s.startswith("{") and s.endswith("}"):
                    last_json = s
                    break
            if not last_json:
                return {"error": "no_json", "raw": self._truncate(str(out or ""), 1000)}
            try:
                parsed = json.loads(last_json)
            except Exception:  # noqa: BLE001
                return {"error": "bad_json", "raw": self._truncate(last_json, 1000)}

            # Keep payload bounded.
            if isinstance(parsed, dict):
                # Avoid huge sys.version strings if something went weird.
                with contextlib.suppress(Exception):
                    py = parsed.get("python")
                    if isinstance(py, dict) and "version" in py:
                        py["version"] = str(py.get("version", ""))[:4000]
                return parsed
            return {"error": "unexpected_type"}
        finally:
            with contextlib.suppress(Exception):
                self.sandbox_pool.put(sb)

    def _should_early_stop(
        self, detailed: list[AttemptResult], time_spent_s: float = float("inf")
    ) -> bool:
        """Quality-aware early stop.

        Default behavior requires at least one clean tool run for the leading candidate
        before early stopping. This reduces the "wrong but popular" failure mode.

        Easy exit mode: if we get consensus quickly (<60s) with verified support,
        stop aggressively to bank time for harder problems.
        """

        ranked_all = rank_candidates(detailed, filter_to_verified_if_any=False)
        if not ranked_all:
            return False

        top_ans, top_d = ranked_all[0]
        _ = top_ans
        votes = int(top_d.get("votes", 0))
        verified = int(top_d.get("verified", 0))

        # Easy exit: aggressive early stop for problems solved quickly with good verification
        if (
            bool(getattr(self.cfg, "easy_exit_enabled", True))
            and time_spent_s
            < float(getattr(self.cfg, "easy_exit_time_threshold_s", 60.0))
            and votes >= int(getattr(self.cfg, "easy_exit_min_votes", 3))
            and verified >= int(getattr(self.cfg, "easy_exit_min_verified", 2))
        ):
            return True

        # Standard early stop logic
        need = max(0, int(self.cfg.early_stop_min_verified))
        target_votes = max(1, int(self.cfg.early_stop))

        if votes < target_votes:
            return False
        if need <= 0:
            return True
        return verified >= need

    @staticmethod
    def _probe_server_ready(client, attempts: int, sleep_s: float = 0.5) -> bool:
        """Return True if an OpenAI-compatible server responds to models.list()."""

        for _ in range(max(1, int(attempts))):
            try:
                client.models.list()
                return True
            except Exception:  # noqa: BLE001
                time.sleep(float(sleep_s))
        return False

    def __post_init__(self) -> None:
        # Helpful env defaults from notebook.
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        # Avoid accidental network attempts when model_path is a local Kaggle input.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        # Optional seed setting if transformers is present.
        with contextlib.suppress(Exception):
            from transformers import set_seed  # type: ignore

            set_seed(self.cfg.seed)

        OpenAI = _require_openai()
        h = _require_harmony()

        # Keep Harmony symbols available for code paths that need to construct
        # Message/TextContent objects (e.g., llama.cpp plain-text fallbacks).
        self._h = h

        # If the user provided a filesystem path, validate it early.
        # This avoids accidentally "succeeding" by reusing an unrelated running server.
        mp_raw = str(getattr(self.cfg, "model_path", "") or "")
        if mp_raw:
            mp_expanded = os.path.expanduser(mp_raw)
            looks_like_path = mp_expanded.startswith(("/", "./", "../"))
            if looks_like_path and not os.path.exists(mp_expanded):
                raise ValueError(f"model_path does not exist: {mp_expanded}")

        self.template = AIMO3Template()
        self.encoding = h["load_harmony_encoding"](
            h["HarmonyEncodingName"].HARMONY_GPT_OSS
        )
        self.Role = h["Role"]
        self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()

        ServerClass = VLLMServer

        self.server = ServerClass(cfg=self.cfg, port=self.port)

        self.base_url = f"http://0.0.0.0:{self.port}/v1"

        # If a server is already running on this port, reuse it.
        if bool(self.cfg.reuse_existing_server):
            probe_client = OpenAI(
                base_url=self.base_url,
                api_key="sk-local",
                timeout=self.cfg.server_probe_timeout,
            )
            if self._probe_server_ready(
                probe_client, attempts=self.cfg.server_probe_attempts
            ):
                # Reuse: don't start a new process.
                self.server = None
                self.client = OpenAI(
                    base_url=self.base_url,
                    api_key="sk-local",
                    timeout=self.cfg.session_timeout,
                )
            else:
                self.server = ServerClass(cfg=self.cfg, port=self.port)
                self.server.start()
                self.client = OpenAI(
                    base_url=self.base_url,
                    api_key="sk-local",
                    timeout=self.cfg.session_timeout,
                )
                self.server.wait_ready(self.client)
        else:
            self.server.start()
            self.client = OpenAI(
                base_url=self.base_url,
                api_key="sk-local",
                timeout=self.cfg.session_timeout,
            )
            self.server.wait_ready(self.client)

        self._initialize_kernels()
        self.notebook_start_time = time.time()
        self.problems_remaining = int(self.cfg.problems_total)

        # Dynamic time budgeting: track actual solve times to adjust per-problem budgets
        self._budget_tracker = TimeBudgetTracker(
            total_budget_s=float(self.cfg.notebook_limit),
            total_problems=int(self.cfg.problems_total),
            base_timeout_s=float(self.cfg.base_problem_timeout),
            high_timeout_s=float(self.cfg.high_problem_timeout),
            # Adaptive extension settings
            flex_pool_fraction=float(
                getattr(self.cfg, "adaptive_budget_flex_pool_fraction", 0.15)
            ),
            max_extension_multiplier=float(
                getattr(self.cfg, "adaptive_budget_max_extension", 2.0)
            ),
            hardness_trigger_fraction=float(
                getattr(self.cfg, "adaptive_budget_hardness_trigger", 0.5)
            ),
            hardness_min_distinct_answers=int(
                getattr(self.cfg, "adaptive_budget_min_distinct", 3)
            ),
        )

        # Notebook-friendly tracing behavior: optionally reset the trace file at startup.
        if bool(getattr(self.cfg, "trace_enabled", False)) and bool(
            getattr(self.cfg, "trace_reset_on_start", False)
        ):
            with contextlib.suppress(Exception):
                p = str(
                    getattr(self.cfg, "trace_path", "aimo3_trace.jsonl")
                    or "aimo3_trace.jsonl"
                )
                if p and os.path.exists(p):
                    os.remove(p)

        self._trace = TraceRecorder(
            enabled=bool(getattr(self.cfg, "trace_enabled", False)),
            path=str(getattr(self.cfg, "trace_path", "aimo3_trace.jsonl")),
            include_problem_text=bool(
                getattr(self.cfg, "trace_include_problem_text", False)
            ),
        )

    def close(self) -> None:
        if hasattr(self, "sandbox_pool"):
            while not self.sandbox_pool.empty():
                with contextlib.suppress(Exception):
                    sb = self.sandbox_pool.get_nowait()
                    sb.close()
        if hasattr(self, "server") and self.server is not None:
            with contextlib.suppress(Exception):
                self.server.stop()

    def __del__(self) -> None:
        self.close()

    def _initialize_kernels(self) -> None:
        self.sandbox_pool: queue.Queue[AIMO3Sandbox] = queue.Queue()

        pool_size = max(1, min(int(self.cfg.sandbox_pool_size), int(self.cfg.workers)))

        # Keep track of how many kernels we created for the pool.
        self._sandbox_pool_target_size = pool_size

        def _create():
            return AIMO3Sandbox(timeout=self.cfg.jupyter_timeout)

        init_workers = max(1, min(int(self.cfg.kernel_init_workers), pool_size))

        # Creating many kernels in parallel can trigger port selection races in notebook runtimes.
        # Limit concurrency during initialization, while still creating the full pool size.
        created = 0
        with ThreadPoolExecutor(max_workers=init_workers) as ex:
            futures = [ex.submit(_create) for _ in range(pool_size)]
            for f in as_completed(futures):
                try:
                    sb = f.result()
                except Exception:  # noqa: BLE001
                    # Transient kernel startup issues happen; we'll fill the pool below.
                    continue
                self.sandbox_pool.put(sb)
                created += 1

        # Best-effort: fill any missing slots sequentially (reduces port collision races).
        # Keep this bounded so we don't hang forever on a broken environment.
        missing = max(0, int(pool_size) - int(created))
        fill_attempts = 0
        while missing > 0 and fill_attempts < max(2 * int(pool_size), 4):
            fill_attempts += 1
            try:
                self.sandbox_pool.put(_create())
                missing -= 1
            except Exception:  # noqa: BLE001
                # Give the OS a moment to release ports.
                time.sleep(0.05)

    @property
    def _extractor(self) -> AnswerExtractor:
        # Centralize extraction behavior (range, formatting).
        strict = bool(getattr(self.cfg, "strict_fallback_extraction", True))
        return AnswerExtractor(aimo_lo=0, aimo_hi=99999, strict_fallback=strict)

    @staticmethod
    def _compute_mean_entropy(logprobs_buffer: list) -> float:
        """Compute mean per-token entropy from streaming logprobs."""
        if not logprobs_buffer:
            return float("inf")

        total_entropy = 0.0
        token_count = 0

        for top_logprobs_dict in logprobs_buffer:
            if not isinstance(top_logprobs_dict, dict):
                continue
            if not top_logprobs_dict:
                continue

            token_entropy = 0.0
            for _tok, log_prob in top_logprobs_dict.items():
                prob = math.exp(log_prob)
                if prob > 0:
                    token_entropy -= prob * math.log2(prob)

            total_entropy += token_entropy
            token_count += 1

        if token_count == 0:
            return float("inf")

        return total_entropy / token_count

    def _process_attempt(
        self,
        problem: str,
        developer_prompt: str,
        attempt_index: int,
        attempt_tag: str | None,
        stop_event: threading.Event,
        deadline: float,
        problem_id: str | None = None,
        retriever_used: bool = False,
    ) -> AttemptResult:
        """Run a single solver attempt with streaming completions and tool execution."""

        if stop_event.is_set() or time.time() > deadline:
            return AttemptResult(
                attempt=attempt_index + 1,
                answer=None,
                stats=AttemptStats(),
                tag=attempt_tag,
            )

        sandbox = None
        local_tool = None
        python_calls = 0
        python_errors = 0
        lean_calls = 0
        timeout_count = 0
        total_tokens = 0
        final_answer = None
        logprobs_buffer: list = []
        text_tail: deque = deque(maxlen=int(self.cfg.capture_attempt_text_chars))
        verification_marker_found = False

        attempt_seed = int(math.pow(self.cfg.seed + attempt_index, 2))
        Conversation = self._h["Conversation"]
        Role = self.Role

        try:
            sandbox = self.sandbox_pool.get(timeout=self.cfg.sandbox_timeout)

            local_tool = AIMO3Tool(
                local_jupyter_timeout=self.cfg.jupyter_timeout,
                tool_prompt=self.cfg.tool_prompt,
                sandbox=sandbox,
            )

            messages = self.template.apply_chat_template(
                developer_prompt, problem, local_tool.tool_config
            )
            conversation = Conversation.from_messages(messages)

            for _turn in range(self.cfg.turns):
                if stop_event.is_set() or time.time() > deadline:
                    break

                prompt_ids = self.encoding.render_conversation_for_completion(
                    conversation, Role.ASSISTANT
                )
                max_tokens = self.cfg.context_tokens - len(prompt_ids)

                if max_tokens < self.cfg.buffer_tokens:
                    break

                extra = {
                    "min_p": self.cfg.min_p,
                    "stop_token_ids": self.stop_token_ids,
                    "return_token_ids": True,
                }
                if self.cfg.top_k > 0:
                    extra["top_k"] = self.cfg.top_k

                stream = self.client.completions.create(
                    model=self.cfg.served_model_name,
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                    logprobs=(
                        self.cfg.top_logprobs
                        if self.cfg.entropy_weighting_enabled
                        else None
                    ),
                    max_tokens=max_tokens,
                    prompt=prompt_ids,
                    seed=attempt_seed,
                    stream=True,
                    extra_body=extra,
                )

                try:
                    token_buffer: list = []
                    text_chunks: list[str] = []

                    for chunk in stream:
                        if stop_event.is_set() or time.time() > deadline:
                            break

                        choice = chunk.choices[0]
                        new_tokens = choice.token_ids
                        new_text = choice.text

                        if new_tokens:
                            token_buffer.extend(new_tokens)
                            total_tokens += len(new_tokens)
                            text_chunks.append(new_text)
                            text_tail.append(new_text)

                            if self.cfg.entropy_weighting_enabled:
                                chunk_lp = choice.logprobs
                                if (
                                    chunk_lp is not None
                                    and chunk_lp.top_logprobs
                                ):
                                    logprobs_buffer.extend(
                                        chunk_lp.top_logprobs
                                    )

                        if "}" in (new_text or ""):
                            search_text = "".join(
                                text_chunks[-self.cfg.search_tokens :]
                            )
                            answer = self._extractor.extract_boxed_int(
                                search_text
                            )
                            if answer is not None:
                                final_answer = answer
                                break

                finally:
                    stream.close()

                if final_answer is not None:
                    break

                if not token_buffer:
                    break

                new_messages = (
                    self.encoding.parse_messages_from_completion_tokens(
                        token_buffer, Role.ASSISTANT
                    )
                )
                if not new_messages:
                    break

                conversation.messages = conversation.messages + list(new_messages)
                last_message = new_messages[-1]

                if last_message.channel == "final":
                    answer_text = last_message.content[0].text
                    final_answer = self._extractor.extract_boxed_int(
                        answer_text
                    )
                    if final_answer is None:
                        final_answer = self._extractor.extract_int_fallback(
                            answer_text
                        )
                    break

                if last_message.recipient == "python":
                    python_calls += 1
                    tool_responses = local_tool.process_sync_plus(last_message)
                    response_text = tool_responses[0].content[0].text

                    if (
                        response_text.startswith("[ERROR]")
                        or "Traceback" in response_text
                        or "Error:" in response_text
                    ):
                        python_errors += 1
                        if "timed out" in response_text.lower():
                            timeout_count += 1

                    if "VERIFY_OK" in response_text:
                        verification_marker_found = True

                    # Heuristic: detect Lean tool use inside Python code.
                    with contextlib.suppress(Exception):
                        code_text = (
                            last_message.content[0].text or ""
                        ).lower()
                        if "lean" in code_text or "lake" in code_text:
                            lean_calls += 1

                    conversation.messages = conversation.messages + list(tool_responses)

        except Exception:  # noqa: BLE001
            python_errors += 1

        finally:
            if sandbox is not None:
                if bool(self.cfg.sandbox_reset_between_attempts):
                    with contextlib.suppress(Exception):
                        sandbox.reset()
                with contextlib.suppress(Exception):
                    self.sandbox_pool.put(sandbox)

        mean_entropy = self._compute_mean_entropy(logprobs_buffer)
        output_text = "".join(text_tail)

        return AttemptResult(
            attempt=attempt_index + 1,
            answer=final_answer,
            stats=AttemptStats(
                token_count=total_tokens,
                python_calls=python_calls,
                python_errors=python_errors,
                lean_calls=lean_calls,
                timeout_count=timeout_count,
                mean_entropy=mean_entropy,
                verification_marker_found=(
                    verification_marker_found if python_calls > 0 else None
                ),
            ),
            output_text=output_text,
            tag=attempt_tag,
        )

    def solve_problem(self, problem: str) -> int:
        """Solve a single problem with multiple parallel attempts and answer ranking."""

        print(f"\nProblem: {problem}\n")

        user_input = f"{problem} {self.cfg.preference_prompt}"

        # Compute budget using adaptive tracker.
        budget = self._budget_tracker.compute_budget()
        deadline = time.time() + budget
        problem_start = time.time()
        problem_id = stable_problem_id(problem)

        print(f"Budget: {budget:.2f}s | {self._budget_tracker.status_summary()}\n")

        # Record trace start.
        self._trace.record(
            {
                "event": "solve_start",
                "problem_id": problem_id,
                "budget_s": budget,
                "problem": problem if self._trace.include_problem_text else None,
            }
        )

        # Optional: snapshot sandbox environment.
        env_snapshot = None
        if bool(getattr(self.cfg, "trace_env_enabled", False)):
            with contextlib.suppress(Exception):
                env_snapshot = self._sandbox_env_snapshot()

        # Build task list: (developer_prompt, attempt_index, tag).
        tasks = [
            (self.cfg.system_prompt, i, None) for i in range(self.cfg.attempts)
        ]

        detailed_results: list[AttemptResult] = []
        stop_event = threading.Event()
        executor = ThreadPoolExecutor(max_workers=self.cfg.workers)

        try:
            futures = []
            for dev_prompt, attempt_idx, tag in tasks:
                f = executor.submit(
                    self._process_attempt,
                    user_input,
                    dev_prompt,
                    attempt_idx,
                    tag,
                    stop_event,
                    deadline,
                    problem_id=problem_id,
                )
                futures.append(f)

            for future in as_completed(futures):
                try:
                    result = future.result()
                    detailed_results.append(result)

                    time_spent = time.time() - problem_start
                    if self._should_early_stop(detailed_results, time_spent):
                        stop_event.set()
                        for f in futures:
                            f.cancel()
                        break

                except Exception as exc:  # noqa: BLE001
                    print(f"Future failed: {exc}")
                    continue

        finally:
            stop_event.set()
            executor.shutdown(wait=True, cancel_futures=True)

        time_used = time.time() - problem_start
        self._budget_tracker.record_solve(time_used)
        self.problems_remaining = self._budget_tracker.problems_remaining

        # Display candidates.
        self._display_candidates(detailed_results)

        # Select answer via ranking.
        ranked = rank_candidates(detailed_results, filter_to_verified_if_any=True)

        if ranked:
            final_answer = ranked[0][0]
            data = ranked[0][1]
            print(
                f"\nFinal Answer: {final_answer} "
                f"(votes={data['votes']}, verified={data['verified']})\n"
            )
        else:
            final_answer = 0
            print("\nFinal Answer: 0 (no valid candidates)\n")

        # Record trace end.
        self._trace.record(
            {
                "event": "solve_end",
                "problem_id": problem_id,
                "answer": final_answer,
                "time_s": time_used,
                "attempts_total": len(detailed_results),
                "attempts_with_answer": sum(
                    1 for r in detailed_results if r.answer is not None
                ),
                "ranking": [
                    {
                        "answer": a,
                        "votes": d["votes"],
                        "verified": d["verified"],
                    }
                    for a, d in ranked[:5]
                ]
                if ranked
                else [],
                "env": env_snapshot,
            }
        )

        return int(final_answer) if final_answer is not None else 0
