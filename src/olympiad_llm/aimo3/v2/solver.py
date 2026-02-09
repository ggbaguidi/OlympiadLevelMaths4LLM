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
from .template import AIMO3Template, VERIFY_STRATEGIES
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
            v = float(r.stats.mean_entropy)
            if v != float("inf") and v > 0.0:
                ent = v
        return {
            "Attempt": r.attempt,
            "Answer": r.answer,
            "ToolVerified": bool(r.stats.tool_verified),
            "PyCalls": int(r.stats.python_calls),
            "Timeouts": int(r.stats.timeout_count),
            "PyErrors": int(r.stats.python_errors),
            "LeanCalls": int(r.stats.lean_calls),
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

        # Sort: attempts with an answer first, then by attempt number.
        rows.sort(key=lambda r: (r["Answer"] is None, r["Attempt"]))

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
                timeout=max(self.cfg.sandbox_timeout, 0.5)
            )
        except Exception:
            sb = None

        if sb is None:
            return None

        try:
            pkg_raw = self.cfg.trace_env_packages or ""
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
                timeout=min(2.0, self.cfg.jupyter_timeout),
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

        _, top_d = ranked_all[0]
        votes = int(top_d.get("votes", 0))
        verified = int(top_d.get("verified", 0))

        # Easy exit: aggressive early stop for problems solved quickly with good verification
        if (
            self.cfg.easy_exit_enabled
            and time_spent_s < self.cfg.easy_exit_time_threshold_s
            and votes >= self.cfg.easy_exit_min_votes
            and verified >= self.cfg.easy_exit_min_verified
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

    # ------------------------------------------------------------------
    # Answer-conditional verification phase
    # ------------------------------------------------------------------

    def _should_run_verification(
        self,
        ranked: list,
        time_remaining_s: float,
    ) -> bool:
        """Decide whether the verification phase should trigger."""
        if not self.cfg.verify_phase_enabled:
            return False
        if not ranked:
            return False
        if time_remaining_s < self.cfg.verify_min_remaining_s:
            return False
        # Skip if consensus is already strong.
        top_votes = ranked[0][1].get("votes", 0)
        if top_votes > self.cfg.verify_trigger_max_votes:
            return False
        return True

    def _run_verify_attempt(
        self,
        problem: str,
        candidate_answer: int,
        strategy_template: str,
        attempt_seed: int,
        deadline: float,
    ) -> dict:
        """Run a single short verification attempt.

        Returns a dict with keys:
          - candidate: the answer being verified
          - verdict: "CORRECT" | "INCORRECT" | "UNKNOWN"
          - alt_answer: int | None  (if the verifier found a different answer)
        """
        result = {
            "candidate": candidate_answer,
            "verdict": "UNKNOWN",
            "alt_answer": None,
        }

        if time.time() > deadline:
            return result

        sandbox = None
        try:
            sandbox = self.sandbox_pool.get(timeout=self.cfg.sandbox_timeout)

            local_tool = AIMO3Tool(
                local_jupyter_timeout=self.cfg.jupyter_timeout,
                tool_prompt=self.cfg.tool_prompt,
                sandbox=sandbox,
            )

            # Build the verification prompt.
            verify_user_text = strategy_template.format(
                answer=candidate_answer,
                problem=problem,
            )
            verify_dev_prompt = (
                "You are a world-class mathematical verifier. "
                "Your ONLY job is to check whether the proposed answer is correct. "
                "Use Python to compute — do NOT just reason about it. "
                "Be concise. The final answer must be a non-negative integer between 0 and 99999."
            )

            messages = self.template.apply_chat_template(
                verify_dev_prompt, verify_user_text, local_tool.tool_config
            )

            Conversation = self._h["Conversation"]
            conversation = Conversation.from_messages(messages)

            text_parts: list[str] = []

            for _turn in range(32):  # Cap turns for verification.
                if time.time() > deadline:
                    break

                prompt_ids = self.encoding.render_conversation_for_completion(
                    conversation, self.Role
                )
                max_tokens = min(
                    self.cfg.verify_max_tokens,
                    self.cfg.context_tokens - len(prompt_ids),
                )
                if max_tokens < self.cfg.buffer_tokens:
                    break

                extra = {
                    "min_p": self.cfg.min_p,
                    "stop_token_ids": self.stop_token_ids,
                    "return_token_ids": True,
                }

                stream = self.client.completions.create(
                    model=self.cfg.served_model_name,
                    temperature=self.cfg.verify_temperature,
                    top_p=self.cfg.top_p,
                    max_tokens=max_tokens,
                    prompt=prompt_ids,
                    seed=attempt_seed,
                    stream=True,
                    extra_body=extra,
                )

                try:
                    token_buffer: list = []
                    for chunk in stream:
                        if time.time() > deadline:
                            break
                        choice = chunk.choices[0]
                        new_tokens = choice.token_ids
                        new_text = choice.text
                        if new_tokens:
                            token_buffer.extend(new_tokens)
                            text_parts.append(new_text)
                finally:
                    stream.close()

                if not token_buffer:
                    break

                new_messages = (
                    self.encoding.parse_messages_from_completion_tokens(
                        token_buffer, self.Role
                    )
                )
                if not new_messages:
                    break

                conversation.messages = conversation.messages + list(new_messages)
                last_message = new_messages[-1]

                # Handle tool calls during verification.
                if last_message.recipient == "python":
                    tool_responses = local_tool.process_sync_plus(last_message)
                    conversation.messages = conversation.messages + list(tool_responses)
                    continue

                # If it's a "final" channel or no more tool calls, we're done.
                break

            # Parse the verification verdict from accumulated text.
            full_text = "\n".join(text_parts)
            upper = full_text.upper()

            # 1. Exact match: "VERDICT: CORRECT" / "VERDICT: INCORRECT"
            if "VERDICT: CORRECT" in upper or "VERDICT:CORRECT" in upper:
                result["verdict"] = "CORRECT"
            elif "VERDICT: INCORRECT" in upper or "VERDICT:INCORRECT" in upper:
                result["verdict"] = "INCORRECT"
            else:
                # 2. Fuzzy match: look for strong signal phrases.
                #    Check for INCORRECT first (more specific) to avoid false
                #    positives from "the answer is correct" inside longer text.
                incorrect_signals = [
                    "THE ANSWER IS INCORRECT",
                    "THE PROPOSED ANSWER IS INCORRECT",
                    "THE PROPOSED ANSWER IS WRONG",
                    "THIS IS INCORRECT",
                    "ANSWER IS WRONG",
                    "NOT CORRECT",
                ]
                correct_signals = [
                    "THE ANSWER IS CORRECT",
                    "THE PROPOSED ANSWER IS CORRECT",
                    "CONFIRMED CORRECT",
                    "THIS IS CORRECT",
                    "I CONFIRM THE ANSWER",
                    "ANSWER IS VERIFIED",
                ]
                if any(sig in upper for sig in incorrect_signals):
                    result["verdict"] = "INCORRECT"
                elif any(sig in upper for sig in correct_signals):
                    result["verdict"] = "CORRECT"
                else:
                    # 3. Fallback: if the verifier produced a \boxed{} answer,
                    #    compare it with the candidate.
                    verifier_answer = self._extractor.extract_boxed_int(full_text)
                    if verifier_answer is not None:
                        if verifier_answer == candidate_answer:
                            result["verdict"] = "CORRECT"
                        else:
                            result["verdict"] = "INCORRECT"
                            result["alt_answer"] = verifier_answer

            # Extract alt answer for any INCORRECT verdict (if not already set).
            if result["verdict"] == "INCORRECT" and result["alt_answer"] is None:
                alt = self._extractor.extract_boxed_int(full_text)
                if alt is not None and alt != candidate_answer:
                    result["alt_answer"] = alt

            # Log full output for UNKNOWN verdicts to debug prompt compliance.
            if result["verdict"] == "UNKNOWN":
                print(f"  [Verify UNKNOWN] Candidate {candidate_answer} — full output:\n{full_text}...")

        except Exception:  # noqa: BLE001
            pass  # Verification attempt failed — verdict stays UNKNOWN.

        finally:
            if sandbox is not None:
                with contextlib.suppress(Exception):
                    sandbox.reset()
                with contextlib.suppress(Exception):
                    self.sandbox_pool.put(sandbox)

        return result

    def _verify_candidates(
        self,
        problem: str,
        ranked: list,
        deadline: float,
    ) -> list:
        """Run the answer-conditional verification phase.

        Takes the top-K ranked candidates, runs short parallel verification
        attempts for each, and returns a re-ranked list of (answer, info_dict)
        tuples.  The info_dict is augmented with 'verify_correct' / 'verify_incorrect'
        counts.

        If verification proves one candidate and disproves others, the proven
        candidate is promoted regardless of original vote count.
        """
        top_k = min(self.cfg.verify_top_k_candidates, len(ranked))
        candidates_to_check = ranked[:top_k]
        per_candidate = self.cfg.verify_attempts_per_candidate
        n_strategies = len(VERIFY_STRATEGIES)

        # Build all verification tasks.
        verify_tasks = []
        for cand_idx, (answer, _data) in enumerate(candidates_to_check):
            for v_idx in range(per_candidate):
                strategy = VERIFY_STRATEGIES[(cand_idx * per_candidate + v_idx) % n_strategies]
                seed = (self.cfg.seed + 1000 + cand_idx * 100 + v_idx) ** 2
                verify_tasks.append((answer, strategy, seed))

        # Run all verification attempts in parallel.
        verify_results: list[dict] = []
        executor = ThreadPoolExecutor(
            max_workers=min(len(verify_tasks), self.cfg.workers)
        )
        try:
            futures = []
            for answer, strategy, seed in verify_tasks:
                f = executor.submit(
                    self._run_verify_attempt,
                    problem,
                    answer,
                    strategy,
                    seed,
                    deadline,
                )
                futures.append(f)

            for future in as_completed(futures):
                try:
                    verify_results.append(future.result())
                except Exception:  # noqa: BLE001
                    continue
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

        # Aggregate verdicts per candidate.
        verify_scores: dict[int, dict] = defaultdict(
            lambda: {"correct": 0, "incorrect": 0, "unknown": 0, "alt_answers": []}
        )
        for vr in verify_results:
            cand = vr["candidate"]
            v = vr["verdict"]
            if v == "CORRECT":
                verify_scores[cand]["correct"] += 1
            elif v == "INCORRECT":
                verify_scores[cand]["incorrect"] += 1
                if vr["alt_answer"] is not None:
                    verify_scores[cand]["alt_answers"].append(vr["alt_answer"])
            else:
                verify_scores[cand]["unknown"] += 1

        # Log verification results.
        for cand, scores in verify_scores.items():
            print(
                f"  [Verify] Candidate {cand}: "
                f"correct={scores['correct']}, "
                f"incorrect={scores['incorrect']}, "
                f"unknown={scores['unknown']}"
            )

        # Re-rank: augment the original ranking data with verification info
        # and re-sort.  Verification results are weighted heavily.
        augmented = []
        for answer, data in ranked:
            vs = verify_scores.get(answer)
            if vs:
                data = dict(data)  # Copy to avoid mutating the original.
                data["verify_correct"] = vs["correct"]
                data["verify_incorrect"] = vs["incorrect"]
            else:
                data = dict(data)
                data["verify_correct"] = 0
                data["verify_incorrect"] = 0
            augmented.append((answer, data))

        # Sort: net verification score first, then verified votes, then raw votes.
        augmented.sort(
            key=lambda kv: (
                kv[1].get("verify_correct", 0) - kv[1].get("verify_incorrect", 0),
                kv[1]["verified"],
                kv[1]["votes"],
            ),
            reverse=True,
        )

        # Check if any alternative answer emerged consistently from verifiers.
        # If multiple verifiers independently proposed the same alt answer and
        # it wasn't already in our candidate set, inject it.
        all_alt_answers: list[int] = []
        for vs in verify_scores.values():
            all_alt_answers.extend(vs["alt_answers"])
        if all_alt_answers:
            from collections import Counter as _Counter

            alt_counts = _Counter(all_alt_answers)
            best_alt, best_alt_count = alt_counts.most_common(1)[0]
            existing_answers = {a for a, _ in augmented}
            # Need at least 2 independent verifiers to propose the same alt.
            if best_alt not in existing_answers and best_alt_count >= 2:
                print(
                    f"  [Verify] Injecting alt answer {best_alt} "
                    f"(proposed by {best_alt_count} verifiers)"
                )
                augmented.insert(
                    0,
                    (
                        best_alt,
                        {
                            "votes": best_alt_count,
                            "verified": best_alt_count,
                            "verify_correct": best_alt_count,
                            "verify_incorrect": 0,
                            "entropy_score": 0.0,
                        },
                    ),
                )

        return augmented

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

        self.base_url = f"http://0.0.0.0:{self.port}/v1"
        self.server = None  # set below only if we need to start one

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
                self.client = OpenAI(
                    base_url=self.base_url,
                    api_key="sk-local",
                    timeout=self.cfg.session_timeout,
                )
            else:
                self.server = VLLMServer(cfg=self.cfg, port=self.port)
                self.server.start()
                self.client = OpenAI(
                    base_url=self.base_url,
                    api_key="sk-local",
                    timeout=self.cfg.session_timeout,
                )
                self.server.wait_ready(self.client)
        else:
            self.server = VLLMServer(cfg=self.cfg, port=self.port)
            self.server.start()
            self.client = OpenAI(
                base_url=self.base_url,
                api_key="sk-local",
                timeout=self.cfg.session_timeout,
            )
            self.server.wait_ready(self.client)

        self._initialize_kernels()
        self.problems_remaining = int(self.cfg.problems_total)

        # Dynamic time budgeting: track actual solve times to adjust per-problem budgets
        self._budget_tracker = TimeBudgetTracker(
            total_budget_s=float(self.cfg.notebook_limit),
            total_problems=int(self.cfg.problems_total),
            base_timeout_s=float(self.cfg.base_problem_timeout),
            high_timeout_s=float(self.cfg.high_problem_timeout),
            flex_pool_fraction=self.cfg.adaptive_budget_flex_pool_fraction,
            max_extension_multiplier=self.cfg.adaptive_budget_max_extension,
            hardness_trigger_fraction=self.cfg.adaptive_budget_hardness_trigger,
            hardness_min_distinct_answers=self.cfg.adaptive_budget_min_distinct,
        )

        # Notebook-friendly tracing behavior: optionally reset the trace file at startup.
        # Cache the answer extractor (avoid re-creating on every call).
        self._cached_extractor = AnswerExtractor(
            aimo_lo=0,
            aimo_hi=99999,
            strict_fallback=bool(self.cfg.strict_fallback_extraction),
        )

        if self.cfg.trace_enabled and self.cfg.trace_reset_on_start:
            with contextlib.suppress(Exception):
                p = self.cfg.trace_path
                if p and os.path.exists(p):
                    os.remove(p)

        self._trace = TraceRecorder(
            enabled=self.cfg.trace_enabled,
            path=self.cfg.trace_path,
            include_problem_text=self.cfg.trace_include_problem_text,
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
                # Give the OS a moment to release ports (exponential backoff).
                time.sleep(0.1 * fill_attempts)

    @property
    def _extractor(self) -> AnswerExtractor:
        return self._cached_extractor

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
        continuation_context: str | None = None,
    ) -> AttemptResult:
        """Run a single solver attempt with streaming completions and tool execution.

        If *continuation_context* is provided (non-empty string), it is injected
        as an additional user message after the problem statement, giving the model
        a summary of a previous incomplete attempt so it can continue rather than
        restart from scratch.
        """

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

        attempt_seed = (self.cfg.seed + attempt_index) ** 2
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

            # Inject continuation context from a previous incomplete wave.
            if continuation_context:
                Message = self._h["Message"]
                continuation_msg = Message.from_role_and_content(
                    Role.USER,
                    (
                        "A previous attempt on this problem ran out of time before finding an answer. "
                        "Here is the end of its reasoning — use it to continue, not restart:\n\n"
                        + continuation_context
                    ),
                )
                messages.append(continuation_msg)

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

                        if (
                            "}" in (new_text or "")
                            and total_tokens
                            >= self.cfg.min_tokens_before_stream_extraction
                        ):
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
            pass  # Infrastructure failure (e.g. vLLM stream error), not a Python tool error

        finally:
            if sandbox is not None:
                if bool(self.cfg.sandbox_reset_between_attempts):
                    with contextlib.suppress(Exception):
                        sandbox.reset()
                with contextlib.suppress(Exception):
                    self.sandbox_pool.put(sandbox)

        # Last-resort extraction: the model used the tool and generated many
        # tokens but never emitted a clean "final" channel message.  Scan the
        # accumulated tail text for a \boxed{} or fallback integer.
        if final_answer is None and text_tail:
            full_tail = "".join(text_tail)
            final_answer = self._extractor.extract_boxed_int(full_tail)
            if final_answer is None:
                final_answer = self._extractor.extract_int_fallback(full_tail)

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
                    verification_marker_found
                    if python_calls > 0
                    and self.cfg.require_verification_marker
                    else None
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
        if self.cfg.trace_env_enabled:
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

        # --- Adaptive retry: if every attempt returned None, request an
        #     extension from the flex pool and run a second wave. -----------
        has_any_answer = any(r.answer is not None for r in detailed_results)
        if not has_any_answer:
            time_spent = time.time() - problem_start
            extension = self._budget_tracker.request_no_answer_extension(
                time_spent_s=time_spent,
                current_budget_s=budget,
            )
            if extension > 0:
                new_deadline = time.time() + extension

                # Pick the most-progressed wave-1 attempt to use as continuation.
                best_w1 = max(
                    detailed_results,
                    key=lambda r: (
                        r.stats.python_calls,
                        r.stats.token_count,
                    ),
                )
                cont_ctx = best_w1.output_text or None

                print(
                    f"[Adaptive] No answer found — granted {extension:.0f}s "
                    f"extension (flex pool). Running second wave "
                    f"{'with' if cont_ctx else 'without'} continuation context...\n"
                )

                stop_event_2 = threading.Event()
                executor_2 = ThreadPoolExecutor(max_workers=self.cfg.workers)
                # Use fresh attempt indices so seeds differ from the first wave.
                wave2_start = self.cfg.attempts
                try:
                    futures_2 = []
                    for i in range(self.cfg.attempts):
                        f2 = executor_2.submit(
                            self._process_attempt,
                            user_input,
                            self.cfg.system_prompt,
                            wave2_start + i,
                            None,
                            stop_event_2,
                            new_deadline,
                            continuation_context=cont_ctx,
                        )
                        futures_2.append(f2)

                    for future in as_completed(futures_2):
                        try:
                            result = future.result()
                            detailed_results.append(result)

                            time_spent_2 = time.time() - problem_start
                            if self._should_early_stop(
                                detailed_results, time_spent_2
                            ):
                                stop_event_2.set()
                                for f2 in futures_2:
                                    f2.cancel()
                                break
                        except Exception:  # noqa: BLE001
                            continue
                finally:
                    stop_event_2.set()
                    executor_2.shutdown(wait=True, cancel_futures=True)

        time_used = time.time() - problem_start
        self._budget_tracker.record_solve(time_used)
        self.problems_remaining = self._budget_tracker.problems_remaining

        # Display candidates.
        self._display_candidates(detailed_results)

        # Select answer via ranking.
        ranked = rank_candidates(detailed_results, filter_to_verified_if_any=True)

        # --- Answer-conditional verification phase ---
        # Use the UNFILTERED ranking so that all strong candidates (including
        # those without tool-verification) get a fair shot at being checked.
        ranked_for_verify = rank_candidates(
            detailed_results, filter_to_verified_if_any=False
        )
        time_remaining_for_verify = (
            self._budget_tracker.time_remaining_s - time_used
        )
        if self._should_run_verification(ranked_for_verify, time_remaining_for_verify):
            verify_deadline = time.time() + min(
                self.cfg.verify_timeout_s, time_remaining_for_verify * 0.8
            )
            print("[Verify] Weak consensus — running verification phase...")
            ranked = self._verify_candidates(
                problem, ranked_for_verify, verify_deadline
            )
            # Update time_used to include verification.
            time_used = time.time() - problem_start

        if ranked:
            final_answer = ranked[0][0]
            data = ranked[0][1]
            verify_info = ""
            if "verify_correct" in data:
                verify_info = (
                    f", verify_ok={data['verify_correct']}"
                    f", verify_fail={data['verify_incorrect']}"
                )
            print(
                f"\nFinal Answer: {final_answer} "
                f"(votes={data['votes']}, verified={data['verified']}"
                f"{verify_info})\n"
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
                        "verify_correct": d.get("verify_correct"),
                        "verify_incorrect": d.get("verify_incorrect"),
                    }
                    for a, d in ranked[:5]
                ]
                if ranked
                else [],
                "env": env_snapshot,
            }
        )

        return int(final_answer) if final_answer is not None else 0
