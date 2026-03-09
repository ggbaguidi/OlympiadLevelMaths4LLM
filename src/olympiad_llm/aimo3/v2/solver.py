# pylint: disable=broad-exception-caught,missing-function-docstring,line-too-long,missing-module-docstring,import-outside-toplevel,invalid-name,too-many-instance-attributes,missing-class-docstring
"""AIMO-3 multi-attempt solver (optimized)."""
from __future__ import annotations
import contextlib
import json
import math
import os
import queue
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
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
from .wickelgren import (
    GENERIC_STRATEGY_CARDS,
    augment_developer_prompt_with_meta,
    init_math_retriever_from_cfg,
)

_INF = float("inf")
_VERDICT_RE = re.compile(r"VERDICT\s*:\s*(CORRECT|INCORRECT)", re.IGNORECASE)
_INC_SIGNALS = frozenset(
    {
        "THE ANSWER IS INCORRECT",
        "THE PROPOSED ANSWER IS INCORRECT",
        "THE PROPOSED ANSWER IS WRONG",
        "THIS IS INCORRECT",
        "ANSWER IS WRONG",
        "NOT CORRECT",
    }
)
_COR_SIGNALS = frozenset(
    {
        "THE ANSWER IS CORRECT",
        "THE PROPOSED ANSWER IS CORRECT",
        "CONFIRMED CORRECT",
        "THIS IS CORRECT",
        "I CONFIRM THE ANSWER",
        "ANSWER IS VERIFIED",
    }
)


def _magnitude_bucket(x: int) -> int:
    return 0 if x == 0 else int(math.log10(abs(x) + 1))


def _detect_magnitude_outlier(groups: dict) -> tuple[bool, int | None, set[int]]:
    if len(groups) < 3:
        return False, None, set()

    bucket_votes: Counter = Counter()
    bucket_answers: dict = defaultdict(list)

    for ans, data in groups.items():
        bucket = _magnitude_bucket(ans)
        bucket_votes[bucket] += data["votes"]
        bucket_answers[bucket].append(ans)

    if len(bucket_votes) < 2:
        return False, None, set()

    dominant_bucket, dominant_votes = bucket_votes.most_common(1)[0]
    outlier_answers = {
        a
        for b, answers in bucket_answers.items()
        if b >= dominant_bucket + 2
        for a in answers
    }
    total_votes = sum(bucket_votes.values())

    if outlier_answers and dominant_votes >= total_votes * 0.5:
        return True, dominant_bucket, outlier_answers
    return False, dominant_bucket, set()


def rank_candidates(
    results: list,
    filter_to_verified_if_any: bool = True,
    magnitude_aware: bool = True,
    ranking_strategy: str = "verified_then_votes",
) -> list:
    if not results:
        return []

    groups = defaultdict(
        lambda: {
            "votes": 0,
            "verified": 0,
            "entropy_score": 0.0,
            "timeout_attempts": 0,
            "deadline_exceeded_attempts": 0,
        }
    )

    for r in results:
        ans = r.answer if isinstance(r, AttemptResult) else r.get("Answer")
        if ans is None or not isinstance(ans, int):
            continue

        g = groups[ans]
        g["votes"] += 1

        if isinstance(r, AttemptResult):
            if r.stats.tool_verified:
                g["verified"] += 1
            if r.stats.had_timeout:
                g["timeout_attempts"] += 1
            if r.stats.deadline_exceeded:
                g["deadline_exceeded_attempts"] += 1
            ent = r.stats.mean_entropy
        else:
            if r.get("ToolVerified"):
                g["verified"] += 1
            if r.get("Timeouts", 0) > 0:
                g["timeout_attempts"] += 1
            ent = r.get("Entropy", _INF)

        if ent != _INF and ent > 0:
            g["entropy_score"] += 1.0 / max(ent, 1e-9)

    if not groups:
        return []

    strategy = (ranking_strategy or "verified_then_votes").strip().lower()
    if strategy == "votes_then_entropy":
        return sorted(
            ((ans, data) for ans, data in groups.items()),
            key=lambda x: (
                x[1]["votes"],
                x[1]["entropy_score"],
                -x[1]["timeout_attempts"],
                -x[1]["deadline_exceeded_attempts"],
            ),
            reverse=True,
        )

    is_suspicious, _, outlier_answers = _detect_magnitude_outlier(groups)

    should_filter = filter_to_verified_if_any and not (
        magnitude_aware and is_suspicious
    )
    if should_filter and any(g["verified"] > 0 for g in groups.values()):
        groups = {k: v for k, v in groups.items() if v["verified"] > 0}

    if strategy not in {"verified_then_votes", "votes_then_verified"}:
        strategy = "verified_then_votes"

    def _sort_key(item):
        ans, data = item
        mag_v = 1 if magnitude_aware and ans in outlier_answers else 0
        mag_vote = 3 if magnitude_aware and ans in outlier_answers else 0

        if strategy == "votes_then_verified":
            return (
                data["votes"] + mag_vote,
                int(data["verified"] > 0) + mag_v,
                data["verified"] + mag_v,
                data["entropy_score"],
                -data["timeout_attempts"],
                -data["deadline_exceeded_attempts"],
            )
        return (
            int(data["verified"] > 0) + mag_v,
            data["verified"] + mag_v,
            data["votes"] + mag_vote,
            data["entropy_score"],
            -data["timeout_attempts"],
            -data["deadline_exceeded_attempts"],
        )

    return sorted(
        ((ans, data) for ans, data in groups.items()), key=_sort_key, reverse=True
    )


@dataclass
class AIMO3Solver:
    cfg: AIMO3Config
    port: int = 8000

    def _startup_runtime(self, base_url: str, OpenAI) -> None:
        with ThreadPoolExecutor(max_workers=1) as ex:
            kernels_future = ex.submit(self._initialize_kernels)
            try:
                probe_client = OpenAI(
                    base_url=base_url,
                    api_key="sk-local",
                    timeout=self.cfg.server_probe_timeout,
                )

                if self.cfg.reuse_existing_server and self._probe_server_ready(
                    probe_client,
                    self.cfg.server_probe_attempts,
                ):
                    self.client = OpenAI(
                        base_url=base_url,
                        api_key="sk-local",
                        timeout=self.cfg.session_timeout,
                    )
                    self.server = None
                else:
                    self.server = VLLMServer(cfg=self.cfg, port=self.port)
                    self.server.start()
                    self.client = OpenAI(
                        base_url=base_url,
                        api_key="sk-local",
                        timeout=self.cfg.session_timeout,
                    )
                    self.server.wait_ready(self.client)

                kernels_future.result()
            except Exception:
                with contextlib.suppress(Exception):
                    self.close()
                raise

    @classmethod
    def _extract_verdict_label(cls, text: str | None) -> str | None:
        if not text:
            return None
        matches = _VERDICT_RE.findall(text)
        return (
            matches[-1].upper()
            if matches and matches[-1].upper() in {"CORRECT", "INCORRECT"}
            else None
        )

    @staticmethod
    def _truncate(text: str | None, max_chars: int) -> str:
        s = text or ""
        return s if len(s) <= max_chars else "..." + s[-(max_chars - 3) :]

    def _attempt_to_row(self, r: AttemptResult) -> dict:
        ent = None
        with contextlib.suppress(Exception):
            v = float(r.stats.mean_entropy)
            if v != _INF and v > 0.0:
                ent = v
        return {
            "Attempt": r.attempt,
            "Answer": r.answer,
            "ToolVerified": bool(r.stats.tool_verified),
            "PyCalls": r.stats.python_calls,
            "Timeouts": r.stats.timeout_count,
            "PyErrors": r.stats.python_errors,
            "Tokens": r.stats.token_count,
            "Entropy": ent,
            "Snippet": self._truncate(
                r.output_text, self.cfg.display_attempt_text_chars
            ),
            "LastError": r.stats.last_error,
        }

    def _display_candidates(self, attempts: list[AttemptResult]) -> None:
        if not self.cfg.display_candidates or not attempts:
            return

        rows = sorted(
            (self._attempt_to_row(r) for r in attempts),
            key=lambda r: (r["Answer"] is None, r["Attempt"]),
        )[: max(1, self.cfg.display_max_rows)]

        try:
            import pandas as pd

            df = pd.DataFrame(rows)
            try:
                from IPython.display import display

                display(df)
            except Exception:
                print(df.to_string(index=False))
        except Exception:
            for row in rows:
                print(
                    f"Attempt {row['Attempt']}: ans={row['Answer']} verified={row['ToolVerified']} "
                    f"calls={row['PyCalls']} errors={row['PyErrors']} tokens={row['Tokens']}\n  {row['Snippet']}\n"
                )

    def _sandbox_env_snapshot(self) -> dict | None:
        if not hasattr(self, "sandbox_pool"):
            return None

        sb = None
        try:
            sb = self.sandbox_pool.get(timeout=max(self.cfg.sandbox_timeout, 0.5))
        except Exception:
            return None

        if sb is None:
            return None

        try:
            packages = [
                p.strip()
                for p in (self.cfg.trace_env_packages or "").split(",")
                if p.strip()
            ]
            code = f"""import json, sys
def _ver(n):
    try: return __import__(n).__version__
    except: return None
print(json.dumps({{'python': {{'version': sys.version[:400], 'executable': sys.executable}},
'packages': {{n: _ver(n) for n in {packages!r}}}}}, ensure_ascii=False))"""

            out = sb.execute(code, timeout=min(2.0, self.cfg.jupyter_timeout))
            for line in reversed((out or "").splitlines()):
                s = line.strip()
                if s.startswith("{") and s.endswith("}"):
                    return json.loads(s)
            return {"error": "no_json"}
        finally:
            with contextlib.suppress(Exception):
                self.sandbox_pool.put(sb)

    def _should_early_stop(
        self,
        detailed: list[AttemptResult],
        time_spent_s: float = _INF,
        early_stop_target: int | None = None,
    ) -> bool:
        ranked = rank_candidates(
            detailed,
            filter_to_verified_if_any=False,
            magnitude_aware=self.cfg.magnitude_aware_ranking_enabled,
            ranking_strategy=self.cfg.ranking_strategy,
        )
        if not ranked:
            return False

        _, top_d = ranked[0]
        votes, verified = top_d["votes"], top_d["verified"]
        top_answer = ranked[0][0]

        if self.cfg.early_stop_require_computed_support and not any(
            isinstance(r.answer, int)
            and r.answer == top_answer
            and r.stats.python_calls > 0
            and r.stats.python_errors == 0
            and not r.stats.deadline_exceeded
            for r in detailed
        ):
            return False

        if (
            self.cfg.easy_exit_enabled
            and time_spent_s < self.cfg.easy_exit_time_threshold_s
            and votes >= self.cfg.easy_exit_min_votes
            and verified >= self.cfg.easy_exit_min_verified
        ):
            return True

        need = max(0, self.cfg.early_stop_min_verified)
        target = max(
            1, self.cfg.early_stop if early_stop_target is None else early_stop_target
        )
        return votes >= target and (need <= 0 or verified >= need)

    def _should_run_verification(self, ranked: list, time_remaining_s: float) -> bool:
        if (
            getattr(self, "_verify_runtime_disabled", False)
            or not self.cfg.verify_phase_enabled
            or not ranked
        ):
            return False

        adaptive_min = min(
            self.cfg.verify_min_remaining_s, self.cfg.base_problem_timeout * 0.25
        )
        if time_remaining_s < adaptive_min:
            return False
        return ranked[0][1].get("votes", 0) <= self.cfg.verify_trigger_max_votes

    def _run_verify_attempt(
        self,
        problem: str,
        candidate_answer: int,
        strategy_template: str,
        attempt_seed: int,
        deadline: float,
        problem_id: str | None = None,
    ) -> dict:
        result = {
            "candidate": candidate_answer,
            "verdict": "UNKNOWN",
            "alt_answer": None,
            "error": None,
        }
        _ = problem_id
        if time.time() > deadline:
            return result

        sandbox = None
        try:
            sandbox = self.sandbox_pool.get(timeout=self.cfg.sandbox_timeout)
            local_tool = AIMO3Tool(
                local_jupyter_timeout=self.cfg.jupyter_timeout,
                tool_prompt=self.cfg.tool_prompt,
                sandbox=sandbox,
                z3_enabled=self.cfg.z3_tool_enabled,
            )

            messages = self.template.apply_chat_template(
                "Check only whether the proposed integer is correct. "
                "Use Python to compute, not verbal intuition. Be brief.",
                strategy_template.format(answer=candidate_answer, problem=problem),
                local_tool.tool_config,
            )

            conversation = self._h["Conversation"].from_messages(messages)
            text_parts, transcript = [], []

            for _ in range(32):
                if time.time() > deadline:
                    break

                prompt_ids = self.encoding.render_conversation_for_completion(
                    conversation, self.Role.ASSISTANT
                )
                max_tokens = min(
                    self.cfg.verify_max_tokens,
                    self.cfg.context_tokens - len(prompt_ids),
                )
                if max_tokens < self.cfg.buffer_tokens:
                    break

                stream = self.client.completions.create(
                    model=self.cfg.served_model_name,
                    temperature=self.cfg.verify_temperature,
                    top_p=self.cfg.top_p,
                    max_tokens=max_tokens,
                    prompt=prompt_ids,
                    seed=attempt_seed,
                    stream=True,
                    extra_body={
                        "min_p": self.cfg.min_p,
                        "stop_token_ids": self.stop_token_ids,
                        "return_token_ids": True,
                    },
                )

                try:
                    token_buffer = []
                    for chunk in stream:
                        if time.time() > deadline:
                            break
                        choice = chunk.choices[0]
                        if choice.token_ids:
                            token_buffer.extend(choice.token_ids)
                            text_parts.append(choice.text)
                finally:
                    stream.close()

                if not token_buffer:
                    break

                new_msgs = self.encoding.parse_messages_from_completion_tokens(
                    token_buffer, self.Role.ASSISTANT
                )
                if not new_msgs:
                    break

                conversation.messages = conversation.messages + list(new_msgs)
                last_msg = new_msgs[-1]

                if last_msg.recipient == "python":
                    transcript.append(str(last_msg.content[0].text or ""))
                    tool_resp = local_tool.process_sync_plus(last_msg)
                    transcript.append(str(tool_resp[0].content[0].text or ""))
                    conversation.messages = conversation.messages + list(tool_resp)
                else:
                    break

            full_text = "\n".join(
                filter(str.strip, ("\n".join(transcript), "\n".join(text_parts)))
            ).upper()

            parsed = self._extract_verdict_label(full_text)
            if parsed:
                result["verdict"] = parsed
            elif any(s in full_text for s in _INC_SIGNALS):
                result["verdict"] = "INCORRECT"
            elif any(s in full_text for s in _COR_SIGNALS):
                result["verdict"] = "CORRECT"
            else:
                result["verdict"] = "UNKNOWN"
                verifier_ans = self._extractor.extract_boxed_int(full_text)
                if verifier_ans is not None:
                    result["verdict"] = (
                        "CORRECT" if verifier_ans == candidate_answer else "INCORRECT"
                    )
                    if verifier_ans != candidate_answer:
                        result["alt_answer"] = verifier_ans

            if result["verdict"] == "INCORRECT" and result["alt_answer"] is None:
                alt = self._extractor.extract_boxed_int(full_text)
                if alt is not None and alt != candidate_answer:
                    result["alt_answer"] = alt

            if result["verdict"] == "UNKNOWN":
                print(
                    f"  [Verify UNKNOWN] Candidate {candidate_answer} — full output:\n{full_text}..."
                )

        except Exception as exc:  # noqa: BLE001
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if sandbox:
                with contextlib.suppress(Exception):
                    sandbox.reset()
                    self.sandbox_pool.put(sandbox)
        return result

    def _verify_candidates(
        self, problem: str, ranked: list, deadline: float, problem_id: str | None = None
    ) -> list:
        top_k = min(self.cfg.verify_top_k_candidates, len(ranked))
        candidates, per_cand = ranked[:top_k], self.cfg.verify_attempts_per_candidate
        n_strat = len(VERIFY_STRATEGIES)

        tasks = [
            (
                ans,
                VERIFY_STRATEGIES[(i * per_cand + j) % n_strat],
                (self.cfg.seed + 1000 + i * 100 + j) ** 2,
            )
            for i, (ans, _) in enumerate(candidates)
            for j in range(per_cand)
        ]

        results = []
        with ThreadPoolExecutor(max_workers=min(len(tasks), self.cfg.workers)) as ex:
            futures = [
                ex.submit(
                    self._run_verify_attempt,
                    problem,
                    ans,
                    strat,
                    seed,
                    deadline,
                    problem_id,
                )
                for ans, strat, seed in tasks
            ]
            for f in as_completed(futures):
                with contextlib.suppress(Exception):
                    results.append(f.result())

        scores = defaultdict(
            lambda: {"correct": 0, "incorrect": 0, "unknown": 0, "alt_answers": []}
        )
        for r in results:
            v = r["verdict"]
            if v == "CORRECT":
                scores[r["candidate"]]["correct"] += 1
            elif v == "INCORRECT":
                scores[r["candidate"]]["incorrect"] += 1
                if r["alt_answer"] is not None:
                    scores[r["candidate"]]["alt_answers"].append(r["alt_answer"])
            else:
                scores[r["candidate"]]["unknown"] += 1

        for cand, sc in scores.items():
            print(
                f"  [Verify] Candidate {cand}: correct={sc['correct']}, incorrect={sc['incorrect']}, unknown={sc['unknown']}"
            )

        augmented = [
            (
                ans,
                {
                    **data,
                    "verify_correct": scores[ans]["correct"],
                    "verify_incorrect": scores[ans]["incorrect"],
                },
            )
            for ans, data in ranked
        ]

        augmented.sort(
            key=lambda kv: (
                kv[1].get("verify_correct", 0) - kv[1].get("verify_incorrect", 0),
                kv[1]["verified"],
                kv[1]["votes"],
            ),
            reverse=True,
        )

        all_alts = [a for vs in scores.values() for a in vs["alt_answers"]]
        if all_alts:
            alt_cnt = Counter(all_alts)
            best_alt, cnt = alt_cnt.most_common(1)[0]
            if best_alt not in {a for a, _ in augmented} and cnt >= 2:
                print(
                    f"  [Verify] Injecting alt answer {best_alt} (proposed by {cnt} verifiers)"
                )
                augmented.insert(
                    0,
                    (
                        best_alt,
                        {
                            "votes": cnt,
                            "verified": cnt,
                            "verify_correct": cnt,
                            "verify_incorrect": 0,
                            "entropy_score": 0.0,
                        },
                    ),
                )
        return augmented

    @staticmethod
    def _probe_server_ready(client, attempts: int, sleep_s: float = 0.5) -> bool:
        for _ in range(max(1, attempts)):
            try:
                client.models.list()
                return True
            except Exception:
                time.sleep(sleep_s)
        return False

    def __post_init__(self) -> None:
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        with contextlib.suppress(Exception):
            from transformers import set_seed

            set_seed(self.cfg.seed)

        self._h = _require_harmony()

        mp = self.cfg.model_path
        if mp and not os.path.exists(os.path.expanduser(mp)):
            raise ValueError(f"model_path does not exist: {mp}")

        self.template = AIMO3Template()
        self._wickelgren_retriever = init_math_retriever_from_cfg(self.cfg)
        enc = self._h["load_harmony_encoding"](
            self._h["HarmonyEncodingName"].HARMONY_GPT_OSS
        )
        self.encoding = enc
        self.Role = self._h["Role"]
        self.stop_token_ids = enc.stop_tokens_for_assistant_actions()

        base_url = f"http://0.0.0.0:{self.port}/v1"
        OpenAI = _require_openai()
        self._startup_runtime(base_url, OpenAI)
        self.problems_remaining = self.cfg.problems_total

        self._budget_tracker = TimeBudgetTracker(
            total_budget_s=self.cfg.notebook_limit,
            total_problems=self.cfg.problems_total,
            base_timeout_s=self.cfg.base_problem_timeout,
            high_timeout_s=self.cfg.high_problem_timeout,
            flex_pool_fraction=self.cfg.adaptive_budget_flex_pool_fraction,
            max_extension_multiplier=self.cfg.adaptive_budget_max_extension,
            hardness_trigger_fraction=self.cfg.adaptive_budget_hardness_trigger,
            hardness_min_distinct_answers=self.cfg.adaptive_budget_min_distinct,
        )

        self._cached_extractor = AnswerExtractor(
            aimo_lo=0,
            aimo_hi=99999,
            strict_fallback=self.cfg.strict_fallback_extraction,
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
        self._verify_runtime_disabled = False
        self._meta_embedder = None
        self._adaptive_hparams = None

        if self.cfg.meta_learning_enabled:
            with contextlib.suppress(Exception):
                from .meta_learning import (
                    AdaptiveHyperparameters,
                    get_global_bandit,
                    get_global_embedder,
                )

                strat_names = [c.name for c in GENERIC_STRATEGY_CARDS]
                exp_file = (
                    Path(self.cfg.meta_learning_experience_file).expanduser()
                    if self.cfg.meta_learning_experience_file.strip()
                    else None
                )
                get_global_bandit(
                    strategy_names=strat_names,
                    exploration_factor=self.cfg.meta_learning_exploration,
                    similarity_threshold=self.cfg.meta_learning_similarity_threshold,
                    experience_file=exp_file,
                )
                self._meta_embedder = get_global_embedder()
                self._adaptive_hparams = AdaptiveHyperparameters(self.cfg)

    def close(self) -> None:
        if hasattr(self, "sandbox_pool"):
            while not self.sandbox_pool.empty():
                with contextlib.suppress(Exception):
                    self.sandbox_pool.get_nowait().close()
        if getattr(self, "server", None):
            with contextlib.suppress(Exception):
                self.server.stop()

    def __del__(self) -> None:
        self.close()

    def _initialize_kernels(self) -> None:
        self.sandbox_pool: queue.Queue = queue.Queue()
        pool_sz = max(1, min(self.cfg.sandbox_pool_size, self.cfg.workers))

        def _create():
            return AIMO3Sandbox(timeout=self.cfg.jupyter_timeout)

        with ThreadPoolExecutor(
            max_workers=max(1, min(self.cfg.kernel_init_workers, pool_sz))
        ) as ex:
            futures = [ex.submit(_create) for _ in range(pool_sz)]
            for f in as_completed(futures):
                with contextlib.suppress(Exception):
                    self.sandbox_pool.put(f.result())

        for _ in range(max(0, pool_sz - self.sandbox_pool.qsize())):
            with contextlib.suppress(Exception):
                self.sandbox_pool.put(_create())

    @property
    def _extractor(self) -> AnswerExtractor:
        return self._cached_extractor

    def _build_attempt_prompt(
        self,
        attempt_index: int,
        problem_text: str | None = None,
        used_strategies: list | None = None,
        preferred_strategy: str | None = None,
    ) -> tuple:
        if attempt_index < max(0, getattr(self.cfg, "answer_only_attempts", 0)):
            return self.cfg.answer_only_prompt, "answer-only", None

        dev_prompt = self.cfg.system_prompt
        strat_name, tag = None, None

        if self.cfg.wickelgren_strategies_enabled:
            dev_prompt, meta = augment_developer_prompt_with_meta(
                dev_prompt,
                attempt_index=attempt_index,
                problem_text=problem_text,
                retriever=self._wickelgren_retriever,
                retriever_top_k=self.cfg.retriever_top_k,
                retriever_min_score=self.cfg.retriever_min_score,
                retriever_include_examples=self.cfg.retriever_include_examples,
                retriever_include_definitions=self.cfg.retriever_include_definitions,
                used_strategies=used_strategies,
                meta_learning_enabled=self.cfg.meta_learning_enabled,
                meta_learning_experience_file=self.cfg.meta_learning_experience_file,
                meta_learning_exploration=self.cfg.meta_learning_exploration,
                meta_learning_similarity_threshold=self.cfg.meta_learning_similarity_threshold,
                preferred_strategy=preferred_strategy,
            )
            strat_name = (meta.get("card") or "").strip() or None
            tag = f"wickelgren:{strat_name or 'unknown'}"
            if meta.get("retriever_used"):
                tag += f"|rag={meta.get('retriever_results', 0)}|rag_backend={meta.get('retriever_backend', 'unknown')}"
        return dev_prompt, tag, strat_name

    @staticmethod
    def _strategy_from_attempt_tag(tag: str | None) -> str | None:
        raw = (tag or "").strip()
        return (
            raw.split(":", 1)[-1].split("|")[0].strip()
            if raw.startswith("wickelgren:")
            else None
        )

    def _adapt_problem_hyperparameters(self, problem_text: str) -> tuple:
        if (
            not self.cfg.meta_learning_enabled
            or not self._meta_embedder
            or not self._adaptive_hparams
        ):
            return None, {}
        try:
            feats = self._meta_embedder.embed(problem_text)
            cfg = self._adaptive_hparams.get_config(feats)
            return (feats, dict(cfg)) if isinstance(cfg, dict) else (None, {})
        except Exception:
            return None, {}

    def _update_meta_learning_from_problem_outcome(
        self,
        *,
        problem_features: Any | None,
        adaptive_cfg_used: dict,
        detailed_results: list[AttemptResult],
        ranked: list,
        time_used_s: float,
    ) -> None:
        if not self.cfg.meta_learning_enabled or problem_features is None or not ranked:
            return

        top_ans, top_data = ranked[0]
        verify_correct = int(top_data.get("verify_correct", 0) or 0)
        verify_incorrect = int(top_data.get("verify_incorrect", 0) or 0)
        verified_votes = int(top_data.get("verified", 0) or 0)
        top_votes = int(top_data.get("votes", 0) or 0)
        verification_decisive = verify_correct > verify_incorrect and verify_correct > 0
        strong_internal_support = verified_votes >= 2 and top_votes >= 3
        allow_learning = verification_decisive or strong_internal_support
        if not allow_learning:
            return
        if not any(
            isinstance(r.answer, int) and r.answer == top_ans for r in detailed_results
        ):
            return

        with contextlib.suppress(Exception):
            from .meta_learning import get_global_bandit

            strat_names = [c.name for c in GENERIC_STRATEGY_CARDS]
            exp_file = (
                Path(self.cfg.meta_learning_experience_file).expanduser()
                if self.cfg.meta_learning_experience_file.strip()
                else None
            )
            bandit = get_global_bandit(
                strategy_names=strat_names,
                exploration_factor=self.cfg.meta_learning_exploration,
                similarity_threshold=self.cfg.meta_learning_similarity_threshold,
                experience_file=exp_file,
            )

            winning = [
                r
                for r in detailed_results
                if isinstance(r.answer, int) and r.answer == top_ans
            ]
            first_succ = min((r.attempt for r in winning), default=1) if winning else 1

            for r in detailed_results:
                strat = self._strategy_from_attempt_tag(r.tag)
                if not strat:
                    continue
                is_succ = isinstance(r.answer, int) and r.answer == top_ans
                bandit.update(
                    problem_features=problem_features,
                    strategy_name=strat,
                    success=is_succ,
                    attempts_to_success=first_succ if is_succ else max(1, r.attempt),
                    time_spent=time_used_s,
                )

        if adaptive_cfg_used and self._adaptive_hparams:
            with contextlib.suppress(Exception):
                self._adaptive_hparams.update_from_outcome(
                    problem_features=problem_features,
                    config_used=adaptive_cfg_used,
                    success=allow_learning,
                    time_spent=time_used_s,
                )

    @staticmethod
    def _compute_mean_entropy(logprobs_buffer: list) -> float:
        if not logprobs_buffer:
            return _INF
        total, count = 0.0, 0
        for lp in logprobs_buffer:
            if not lp:
                continue
            te = 0.0
            for _, log_p in lp.items():
                p = math.exp(log_p)
                if p > 0:
                    te -= p * math.log2(p)
            total += te
            count += 1
        return _INF if count == 0 else total / count

    def _record_attempt_trace(self, problem_id: str, result: AttemptResult) -> None:
        if not self.cfg.trace_attempts_enabled:
            return

        remaining = max(0, int(self.cfg.trace_attempts_max_chars))

        def _cap(items: tuple[str, ...] | list[str]) -> list[str]:
            nonlocal remaining
            out: list[str] = []
            for raw in items:
                if remaining <= 0:
                    break
                s = str(raw or "")
                if len(s) > remaining:
                    keep = max(0, remaining - 3)
                    out.append(("..." + s[-keep:]) if keep > 0 else "")
                    remaining = 0
                    break
                out.append(s)
                remaining -= len(s)
            return out

        calls = _cap(list(result.python_calls_text or ()))
        outs = _cap(list(result.python_outputs_text or ()))
        text_payload = ""
        if remaining > 0:
            text_payload = self._truncate(result.output_text, remaining)

        self._trace.record(
            {
                "event": "attempt_end",
                "problem_id": problem_id,
                "attempt": int(result.attempt),
                "tag": result.tag,
                "answer": result.answer,
                "token_count": int(result.stats.token_count),
                "python_calls": int(result.stats.python_calls),
                "python_errors": int(result.stats.python_errors),
                "timeout_count": int(result.stats.timeout_count),
                "deadline_exceeded": bool(result.stats.deadline_exceeded),
                "tool_verified": bool(result.stats.tool_verified),
                "last_error": result.stats.last_error,
                "python_calls_text": calls,
                "python_outputs_text": outs,
                "output_text": text_payload,
            }
        )

    def _run_attempt_batch(
        self,
        *,
        user_input: str,
        task_specs: list[tuple[str, int, str | None]],
        results: list[AttemptResult],
        deadline: float,
        problem_start: float,
        early_stop_target: int,
        problem_id: str | None = None,
        continuation_context: str | None = None,
        temperature: float | None = None,
    ) -> bool:
        if not task_specs:
            return False

        stop_evt = threading.Event()
        with ThreadPoolExecutor(
            max_workers=max(1, min(self.cfg.workers, len(task_specs)))
        ) as ex:
            futures = [
                ex.submit(
                    self._process_attempt,
                    user_input,
                    dev_p,
                    idx,
                    tag,
                    stop_evt,
                    deadline,
                    problem_id=problem_id,
                    continuation_context=continuation_context,
                    temperature=temperature,
                )
                for dev_p, idx, tag in task_specs
            ]

            for f in as_completed(futures):
                try:
                    r = f.result()
                    results.append(r)
                    self._record_attempt_trace(problem_id or "", r)
                    if self._should_early_stop(
                        results,
                        time.time() - problem_start,
                        early_stop_target,
                    ):
                        stop_evt.set()
                        for ff in futures:
                            ff.cancel()
                        return True
                except Exception as e:
                    print(f"Future failed: {e}")

        return False

    def _process_attempt(
        self,
        problem: str,
        developer_prompt: str,
        attempt_index: int,
        attempt_tag: str | None,
        stop_event: threading.Event,
        deadline: float,
        problem_id: str | None = None,
        continuation_context: str | None = None,
        temperature: float | None = None,
    ) -> AttemptResult:
        if stop_event.is_set() or time.time() > deadline:
            return AttemptResult(
                attempt=attempt_index + 1,
                answer=None,
                stats=AttemptStats(deadline_exceeded=time.time() > deadline),
                tag=attempt_tag,
            )

        sandbox, python_calls, python_errors = None, 0, 0
        last_error, timeout_count, total_tokens = None, 0, 0
        final_answer, logprobs_buf = None, []
        text_tail, transcript_calls, transcript_outs = [], [], []
        verification_found, deadline_exceeded = False, False
        tool_verified = False

        attempt_seed = (self.cfg.seed + attempt_index) ** 2
        temp = self.cfg.temperature if temperature is None else temperature
        _ = problem_id

        try:
            sandbox = self.sandbox_pool.get(timeout=self.cfg.sandbox_timeout)

            tool_p = (
                self.cfg.tool_prompt
                + (
                    "\n\nZ3 SMT SOLVER AVAILABLE: You can use 'from z3 import *' for constraint solving."
                    " Best for Diophantine equations, combinatorial problems. Example: x = Int('x'); solve(x**2 == 2)"
                )
                if self.cfg.z3_tool_enabled
                else self.cfg.tool_prompt
            )

            local_tool = AIMO3Tool(
                local_jupyter_timeout=self.cfg.jupyter_timeout,
                tool_prompt=tool_p,
                sandbox=sandbox,
                z3_enabled=self.cfg.z3_tool_enabled,
            )

            messages = self.template.apply_chat_template(
                developer_prompt, problem, local_tool.tool_config
            )

            if continuation_context:
                msg = self._h["Message"].from_role_and_content(
                    self.Role.USER,
                    "Continue from this partial work; do not restart from scratch. "
                    "Keep only useful progress and discard bad leads quickly:\n\n"
                    + continuation_context,
                )
                messages.append(msg)

            conversation = self._h["Conversation"].from_messages(messages)

            for _ in range(self.cfg.turns):
                if stop_event.is_set() or time.time() > deadline:
                    deadline_exceeded = True
                    break

                prompt_ids = self.encoding.render_conversation_for_completion(
                    conversation, self.Role.ASSISTANT
                )
                max_tok = self.cfg.context_tokens - len(prompt_ids)
                if max_tok < self.cfg.buffer_tokens:
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
                    temperature=temp,
                    top_p=self.cfg.top_p,
                    logprobs=(
                        self.cfg.top_logprobs
                        if self.cfg.entropy_weighting_enabled
                        else None
                    ),
                    max_tokens=max_tok,
                    prompt=prompt_ids,
                    seed=attempt_seed,
                    stream=True,
                    extra_body=extra,
                )

                try:
                    token_buf, text_buf = [], []
                    for chunk in stream:
                        if stop_event.is_set() or time.time() > deadline:
                            deadline_exceeded = True
                            break
                        choice = chunk.choices[0]
                        if choice.token_ids:
                            token_buf.extend(choice.token_ids)
                            total_tokens += len(choice.token_ids)
                            text_buf.append(choice.text)
                            text_tail.append(choice.text)
                            if self.cfg.entropy_weighting_enabled and choice.logprobs:
                                logprobs_buf.extend(choice.logprobs.top_logprobs or [])

                        if (
                            "}" in (choice.text or "")
                            and total_tokens
                            >= self.cfg.min_tokens_before_stream_extraction
                        ):
                            ans = self._extractor.extract_boxed_int(
                                "".join(text_buf[-self.cfg.search_tokens :])
                            )
                            if ans is not None:
                                final_answer = ans
                                break
                finally:
                    stream.close()

                if final_answer is not None or not token_buf:
                    break

                new_msgs = self.encoding.parse_messages_from_completion_tokens(
                    token_buf, self.Role.ASSISTANT
                )
                if not new_msgs:
                    break

                conversation.messages = conversation.messages + list(new_msgs)
                last_msg = new_msgs[-1]

                if last_msg.channel == "final":
                    final_answer = self._extractor.extract_boxed_int(
                        last_msg.content[0].text
                    ) or self._extractor.extract_int_fallback(last_msg.content[0].text)
                    break

                if last_msg.recipient in ("python", "z3"):
                    python_calls += 1
                    transcript_calls.append(str(last_msg.content[0].text or ""))
                    tool_resp = local_tool.process_sync_plus(
                        last_msg, expected_answer=final_answer
                    )
                    resp_text = str(tool_resp[0].content[0].text or "")
                    transcript_outs.append(resp_text)

                    if (
                        resp_text.startswith("[ERROR]")
                        or "Traceback" in resp_text
                        or "Error:" in resp_text
                    ):
                        python_errors += 1
                        if "timed out" in resp_text.lower():
                            timeout_count += 1
                        last_error = resp_text[:500]

                    if "VERIFY_OK" in resp_text:
                        verification_found = True
                    if "[VERIFICATION NOTICE] TOOL_OUTPUT_VALID" in resp_text:
                        tool_verified = True
                    if (
                        "[VERIFICATION NOTICE] TOOL_OUTPUT_INVALID" in resp_text
                        and not resp_text.startswith("[ERROR]")
                    ):
                        if self.cfg.require_verification_marker:
                            python_errors += 1
                        if last_error is None and self.cfg.require_verification_marker:
                            last_error = "Tool output verification marked invalid."

                    conversation.messages = conversation.messages + list(tool_resp)

        except Exception as exc:  # noqa: BLE001
            if last_error is None:
                last_error = f"[INTERNAL_ERROR] {type(exc).__name__}: {exc}"[:500]
        finally:
            if sandbox:
                if self.cfg.sandbox_reset_between_attempts:
                    with contextlib.suppress(Exception):
                        sandbox.reset()
                with contextlib.suppress(Exception):
                    self.sandbox_pool.put(sandbox)

        if final_answer is None and text_tail:
            full = "".join(text_tail)
            final_answer = self._extractor.extract_boxed_int(
                full
            ) or self._extractor.extract_int_fallback(full)

        mean_ent = self._compute_mean_entropy(logprobs_buf)

        return AttemptResult(
            attempt=attempt_index + 1,
            answer=final_answer,
            stats=AttemptStats(
                token_count=total_tokens,
                python_calls=python_calls,
                python_errors=python_errors,
                timeout_count=timeout_count,
                mean_entropy=mean_ent,
                verification_marker_found=(
                    True
                    if (verification_found or tool_verified)
                    else (False if self.cfg.require_verification_marker else None)
                ),
                deadline_exceeded=deadline_exceeded,
                last_error=last_error,
            ),
            output_text="".join(text_tail),
            tag=attempt_tag,
            python_calls_text=tuple(transcript_calls),
            python_outputs_text=tuple(transcript_outs),
        )

    def solve_problem(self, problem: str) -> int:
        print(f"\nProblem: {problem}\n")

        user_input = f"{problem} {self.cfg.preference_prompt}"
        problem_feats, adaptive_cfg = self._adapt_problem_hyperparameters(problem)

        attempts_for_prob = max(1, adaptive_cfg.get("attempts", self.cfg.attempts))
        temp_for_prob = adaptive_cfg.get("temperature", self.cfg.temperature)
        early_stop_prob = max(1, adaptive_cfg.get("early_stop", self.cfg.early_stop))
        pref_strat = adaptive_cfg.get("preferred_strategy")

        budget = self._budget_tracker.compute_budget()
        deadline = time.time() + budget
        problem_start = time.time()
        problem_id = stable_problem_id(problem)

        print(f"Budget: {budget:.2f}s | {self._budget_tracker.status_summary()}\n")

        self._trace.record(
            {
                "event": "solve_start",
                "problem_id": problem_id,
                "budget_s": budget,
                "problem": problem if self._trace.include_problem_text else None,
            }
        )

        env_snap = self._sandbox_env_snapshot() if self.cfg.trace_env_enabled else None

        if adaptive_cfg:
            print(
                f"[Meta] attempts={attempts_for_prob}, temperature={temp_for_prob:.2f}, "
                f"early_stop={early_stop_prob}, preferred={pref_strat or 'none'}"
            )

        used_strats = [] if self.cfg.meta_learning_track_strategies else None
        results: list[AttemptResult] = []
        answer_only_count = min(
            max(0, self.cfg.answer_only_attempts),
            attempts_for_prob,
        )

        def _build_task_specs(start_idx: int, end_idx: int) -> list[tuple[str, int, str | None]]:
            specs: list[tuple[str, int, str | None]] = []
            for i in range(start_idx, end_idx):
                dev_p, tag, strat_n = self._build_attempt_prompt(
                    i,
                    problem,
                    used_strats,
                    pref_strat if i == 0 else None,
                )
                specs.append((dev_p, i, tag))
                if used_strats and strat_n and strat_n not in used_strats:
                    used_strats.append(strat_n)
            return specs

        stopped_early = False
        if answer_only_count > 0:
            stopped_early = self._run_attempt_batch(
                user_input=user_input,
                task_specs=_build_task_specs(0, answer_only_count),
                results=results,
                deadline=deadline,
                problem_start=problem_start,
                early_stop_target=early_stop_prob,
                problem_id=problem_id,
                temperature=temp_for_prob,
            )

        if not stopped_early and answer_only_count < attempts_for_prob:
            stopped_early = self._run_attempt_batch(
                user_input=user_input,
                task_specs=_build_task_specs(answer_only_count, attempts_for_prob),
                results=results,
                deadline=deadline,
                problem_start=problem_start,
                early_stop_target=early_stop_prob,
                problem_id=problem_id,
                temperature=temp_for_prob,
            )

        if not any(r.answer is not None for r in results):
            time_spent = time.time() - problem_start
            ext = self._budget_tracker.request_no_answer_extension(
                time_spent_s=time_spent, current_budget_s=budget
            )
            if ext > 0:
                new_dead = time.time() + ext
                best_w1 = max(
                    results,
                    key=lambda r: (r.stats.python_calls, r.stats.token_count),
                    default=None,
                )
                cont = best_w1.output_text if best_w1 else None

                print(
                    f"[Adaptive] No answer found — granted {ext:.0f}s extension. Running second wave "
                    f"{'with' if cont else 'without'} continuation context...\n"
                )

                second_wave_specs = _build_task_specs(
                    attempts_for_prob,
                    attempts_for_prob * 2,
                )
                self._run_attempt_batch(
                    user_input=user_input,
                    task_specs=second_wave_specs,
                    results=results,
                    deadline=new_dead,
                    problem_start=problem_start,
                    early_stop_target=early_stop_prob,
                    problem_id=problem_id,
                    continuation_context=cont,
                    temperature=temp_for_prob,
                )

        time_used = time.time() - problem_start
        self._display_candidates(results)

        ranked = rank_candidates(
            results,
            filter_to_verified_if_any=self.cfg.filter_to_verified_if_any,
            magnitude_aware=self.cfg.magnitude_aware_ranking_enabled,
            ranking_strategy=self.cfg.ranking_strategy,
        )

        ranked_for_v = rank_candidates(
            results,
            filter_to_verified_if_any=False,
            magnitude_aware=self.cfg.magnitude_aware_ranking_enabled,
            ranking_strategy=self.cfg.ranking_strategy,
        )

        if self._should_run_verification(
            ranked_for_v, self._budget_tracker.time_remaining_s - time_used
        ):
            v_dead = time.time() + min(
                self.cfg.verify_timeout_s,
                (self._budget_tracker.time_remaining_s - time_used) * 0.8,
            )
            print("[Verify] Weak consensus — running verification phase...")
            ranked = self._verify_candidates(problem, ranked_for_v, v_dead, problem_id)

            if (
                not any(
                    d.get("verify_correct", 0) + d.get("verify_incorrect", 0) > 0
                    for _, d in ranked
                )
                and self.cfg.verify_disable_globally_if_all_unknown
            ):
                self._verify_runtime_disabled = True
                print(
                    "[Verify] No decisive verification signal. Disabling for remaining problems."
                )
            time_used = time.time() - problem_start

        if ranked:
            final_ans = ranked[0][0]
            data = ranked[0][1]
            vinfo = (
                f", verify_ok={data['verify_correct']}, verify_fail={data['verify_incorrect']}"
                if "verify_correct" in data
                else ""
            )
            print(
                f"\nFinal Answer: {final_ans} (votes={data['votes']}, verified={data['verified']}{vinfo})\n"
            )
        else:
            final_ans = 0
            print("\nFinal Answer: 0 (no valid candidates)\n")

        self._update_meta_learning_from_problem_outcome(
            problem_features=problem_feats,
            adaptive_cfg_used=adaptive_cfg,
            detailed_results=results,
            ranked=ranked,
            time_used_s=time_used,
        )

        try:
            self._budget_tracker.record_solve(time_used, allocated_budget_s=budget)
        except TypeError:
            self._budget_tracker.record_solve(time_used)
        self.problems_remaining = self._budget_tracker.problems_remaining

        self._trace.record(
            {
                "event": "solve_end",
                "problem_id": problem_id,
                "answer": final_ans,
                "time_s": time_used,
                "attempts_total": len(results),
                "attempts_with_answer": sum(1 for r in results if r.answer is not None),
                "ranking": (
                    [
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
                    else []
                ),
                "env": env_snap,
            }
        )

        return final_ans if final_ans is not None else 0
