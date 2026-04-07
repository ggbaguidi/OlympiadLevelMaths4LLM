# pylint: disable=broad-exception-caught,missing-function-docstring,line-too-long,missing-module-docstring,import-outside-toplevel,invalid-name,too-many-instance-attributes,missing-class-docstring
"""AIMO-3 multi-attempt solver (optimized)."""
from __future__ import annotations
import contextlib
import difflib
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
from .llamacpp_server import LlamaCppServer
from .config import AIMO3Config
from .sandbox import AIMO3Sandbox
from .trace import TraceRecorder, stable_problem_id
from .vllm_server import VLLMServer
from .require import _require_harmony, _require_openai
from .template import AIMO3Template
from .tools import AIMO3Tool
from .agent_memory import init_agent_memory_from_cfg
from .reasoning_framework import augment_prompt_with_reasoning_framework
from .wickelgren import (
    GENERIC_STRATEGY_CARDS,
    augment_developer_prompt_with_meta,
    init_math_retriever_from_cfg,
)

_INF = float("inf")


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


@dataclass(frozen=True)
class AttemptPortfolioProfile:
    name: str
    title: str
    instructions: tuple[str, ...]


@dataclass
class AIMO3Solver:
    cfg: AIMO3Config
    port: int = 8000
    sandbox_pool: queue.Queue | None = None

    def _backend_name(self) -> str:
        backend = str(getattr(self.cfg, "inference_backend", "vllm") or "vllm")
        return backend.strip().lower() or "vllm"

    def _server_class(self):
        return LlamaCppServer if self._backend_name() == "llama_cpp" else VLLMServer

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
                    self.server = self._server_class()(cfg=self.cfg, port=self.port)
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

    @staticmethod
    def _truncate(text: str | None, max_chars: int) -> str:
        s = text or ""
        return s if len(s) <= max_chars else "..." + s[-(max_chars - 3) :]

    @staticmethod
    def _normalize_repetition_text(text: str, *, tail_chars: int = 1200) -> str:
        s = str(text or "").strip().lower()
        if not s:
            return ""
        s = re.sub(r"\s+", " ", s)
        if len(s) > tail_chars:
            s = s[-tail_chars:]
        return s

    @staticmethod
    def _clip_single_line(text: str | None, max_chars: int = 180) -> str:
        s = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(s) <= max_chars:
            return s
        return s[: max(0, max_chars - 3)].rstrip() + "..."

    @staticmethod
    def _infer_problem_domain(problem_text: str | None) -> str:
        text = (problem_text or "").lower()
        if any(
            token in text
            for token in (
                "triangle",
                "circle",
                "angle",
                "tangent",
                "circumcircle",
                "incircle",
                "perpendicular",
                "geometry",
            )
        ):
            return "geometry"
        if any(
            token in text
            for token in (
                "count",
                "ways",
                "permutation",
                "arrangement",
                "subset",
                "coloring",
                "tournament",
            )
        ):
            return "combinatorics"
        if any(
            token in text
            for token in (
                "mod",
                "modulo",
                "remainder",
                "divisible",
                "prime",
                "gcd",
                "lcm",
                "valuation",
            )
        ):
            return "number_theory"
        if any(
            token in text
            for token in ("sequence", "recurrence", "term", "a_n", "f_n", "recursive")
        ):
            return "recurrence"
        return "general"

    def _portfolio_profile_for_attempt(
        self, attempt_index: int, problem_text: str | None = None
    ) -> AttemptPortfolioProfile:
        domain = self._infer_problem_domain(problem_text)
        direct_exact = AttemptPortfolioProfile(
            name="direct_exact",
            title="Primary Exact Route",
            instructions=(
                "Find the shortest exact derivation first: invariant, recurrence, factorization, or exact reduction.",
                "Do not open with broad exploration; commit quickly to one exact route and compute its decisive quantity.",
            ),
        )
        constructive_program = AttemptPortfolioProfile(
            name="constructive_program",
            title="Program-First Exact Search",
            instructions=(
                "Translate the problem into the smallest exact program that can expose the real structure.",
                "Use tiny cases only to discover a recurrence/state compression, then jump to the exact full computation.",
            ),
        )
        alternative_reframe = AttemptPortfolioProfile(
            name="alternative_reframe",
            title="Alternative Reframing",
            instructions=(
                "Assume the first obvious formulation is a trap; seek a different model such as complement counting, reverse process, or different variables.",
                "Do not repeat the default route unless it becomes exact and decisive very quickly.",
            ),
        )

        if domain == "geometry":
            specialist = AttemptPortfolioProfile(
                name="geometry_specialist",
                title="Geometry Reduction",
                instructions=(
                    "Normalize the figure aggressively: coordinates, vectors, directed lengths, or exact trig only if they simplify the geometry.",
                    "Target one eliminable quantity and derive a compact exact relation before any large symbolic expansion.",
                ),
            )
        elif domain == "combinatorics":
            specialist = AttemptPortfolioProfile(
                name="combinatorics_specialist",
                title="Combinatorics Structure",
                instructions=(
                    "Model the objects with a recurrence, bijection, inclusion-exclusion, or generating-function style state description.",
                    "Avoid raw enumeration unless it immediately collapses to a tiny state space.",
                ),
            )
        elif domain in {"number_theory", "recurrence"}:
            specialist = AttemptPortfolioProfile(
                name="number_theory_specialist",
                title="Arithmetic Structure",
                instructions=(
                    "Lean on valuations, congruences, multiplicative structure, or exact recurrence transformations.",
                    "Prefer exact integer algorithms such as CRT, matrix powering, or divisor structure over symbolic wandering.",
                ),
            )
        else:
            specialist = AttemptPortfolioProfile(
                name="algebra_specialist",
                title="Algebraic Reduction",
                instructions=(
                    "Choose a substitution or normalization that reduces the problem to one exact polynomial, recurrence, or invariant.",
                    "Keep the symbolic object small and compute only what decides the final integer.",
                ),
            )

        profiles = (
            direct_exact,
            constructive_program,
            alternative_reframe,
            specialist,
        )
        return profiles[int(attempt_index) % len(profiles)]

    @staticmethod
    def _render_portfolio_profile(profile: AttemptPortfolioProfile) -> str:
        lines = [
            "[META_PORTFOLIO_PROFILE]",
            "Use this as attempt-specific steering only.",
            f"Profile: {profile.name}",
            f"Focus: {profile.title}",
        ]
        for idx, item in enumerate(profile.instructions, start=1):
            lines.append(f"{idx}. {item}")
        lines.append("[/META_PORTFOLIO_PROFILE]")
        return "\n".join(lines)

    @staticmethod
    def _attach_portfolio_tag(tag: str | None, profile_name: str) -> str:
        base = (tag or "").strip()
        if not base:
            return f"portfolio:{profile_name}"
        return base + f"|profile={profile_name}"

    @staticmethod
    def _portfolio_name_from_tag(tag: str | None) -> str | None:
        raw = (tag or "").strip()
        if raw.startswith("portfolio:"):
            return raw.split(":", 1)[-1].split("|", 1)[0].strip() or None
        match = re.search(r"(?:^|\|)profile=([^|]+)", raw)
        if match:
            name = (match.group(1) or "").strip()
            return name or None
        return None

    def _temperature_for_attempt(self, base_temperature: float, attempt_index: int) -> float:
        answer_only_count = max(0, getattr(self.cfg, "answer_only_attempts", 0))
        if attempt_index < answer_only_count:
            return float(base_temperature)
        relative_attempt_index = max(0, int(attempt_index) - int(answer_only_count))

        raw = str(getattr(self.cfg, "portfolio_temperature_schedule", "") or "").strip()
        if raw:
            temps: list[float] = []
            for part in raw.split(","):
                part = part.strip()
                if not part:
                    continue
                with contextlib.suppress(Exception):
                    temps.append(max(0.0, float(part)))
            if temps:
                idx = min(relative_attempt_index, len(temps) - 1)
                return float(temps[idx])

        return float(base_temperature)

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return float(difflib.SequenceMatcher(a=a, b=b).ratio())

    def _repetition_watchdog_action(
        self,
        *,
        assistant_repetition_streak: int,
        tool_repeat_streak: int,
        coached: bool,
    ) -> str | None:
        if not bool(getattr(self.cfg, "repetition_watchdog_enabled", True)):
            return None

        hard = max(1, int(getattr(self.cfg, "repetition_hard_streak", 3) or 3))
        soft_raw = max(1, int(getattr(self.cfg, "repetition_soft_streak", 2) or 2))
        soft = min(soft_raw, hard)
        tool_hard = max(
            1,
            int(
                getattr(self.cfg, "repetition_tool_repeat_hard_streak", 2)
                or 2
            ),
        )

        if assistant_repetition_streak >= hard or tool_repeat_streak >= tool_hard:
            return "abort"
        if (not coached) and assistant_repetition_streak >= soft:
            return "coach"
        return None

    def _can_try_stream_boxed_extraction(self, new_text: str, total_tokens: int) -> bool:
        if "}" not in str(new_text or ""):
            return False
        if bool(getattr(self.cfg, "early_boxed_exit_enabled", False)):
            return True
        return int(total_tokens) >= int(
            getattr(self.cfg, "min_tokens_before_stream_extraction", 1500) or 1500
        )

    def _extract_tool_final_answer(self, text: str) -> tuple[int | None, str | None]:
        if not bool(getattr(self.cfg, "tool_final_answer_marker_enabled", True)):
            return None, None
        raw = str(text or "")
        match = re.search(r"FINAL_ANSWER\s*[:=]\s*(-?\d+)", raw, flags=re.IGNORECASE)
        if not match:
            return None, None
        try:
            val = int(match.group(1))
        except Exception:  # noqa: BLE001
            return None, None
        if not (0 <= val <= 99999):
            return None, None
        return val, "tool_final_marker"

    @staticmethod
    def _runtime_failure_hint_for_error(error_text: str) -> tuple[str, str] | None:
        s = str(error_text or "").strip()
        if not s:
            return None
        lo = s.lower()

        if "timed out" in lo or "timeout" in lo:
            return (
                "Timeout hotspot",
                "Split the computation into smaller exact checks and prune earlier before heavy loops.",
            )
        if "syntaxerror" in lo or "indentationerror" in lo or "unmatched" in lo:
            return (
                "Generated code must parse",
                "Check parentheses/indentation and ensure generated code is syntactically valid before execution.",
            )
        if "attributeerror" in lo and ("sympy" in lo or "has no attribute" in lo):
            return (
                "SymPy API mismatch",
                "Do not assume helper names exist at top level; verify imports/APIs before use.",
            )
        if (
            "polynomialerror" in lo
            or "keyerror: 1/x" in lo
            or ("1/x" in lo and ("poly" in lo or "polynomial" in lo))
        ):
            return (
                "Polynomial cleanup first",
                "Normalize rational expressions and separate numerator/denominator before Poly/factor routines.",
            )
        if "no sign change" in lo or ("findroot" in lo and "valueerror" in lo):
            return (
                "Bracket root search first",
                "Verify a sign-change bracket on an interval before calling root finders.",
            )
        if "indexerror" in lo:
            return (
                "Index bounds check",
                "Guard array/list access with explicit bounds checks before indexing.",
            )
        if "typeerror" in lo and (
            "none" in lo
            or "cannot convert symbols to int" in lo
            or "not supported between instances" in lo
        ):
            return (
                "Type normalization",
                "Filter None/symbolic values and normalize types before numeric comparisons/conversions.",
            )
        return None

    @staticmethod
    def _build_runtime_failure_memory_block(
        results: list[AttemptResult],
        max_items: int = 3,
    ) -> str:
        if not results or max_items <= 0:
            return ""

        rows: list[tuple[str, str, str]] = []
        seen_titles: set[str] = set()

        for r in reversed(results):
            err = str(getattr(r.stats, "last_error", "") or "").strip()
            if not err and int(getattr(r.stats, "timeout_count", 0) or 0) > 0:
                err = "[ERROR] Execution timed out"
            if not err:
                continue

            hint = AIMO3Solver._runtime_failure_hint_for_error(err)
            if hint is None:
                continue

            title, guidance = hint
            if title in seen_titles:
                continue
            seen_titles.add(title)

            evidence = re.sub(r"\s+", " ", err).strip()
            if len(evidence) > 120:
                evidence = evidence[:117].rstrip() + "..."
            rows.append((title, guidance, evidence))
            if len(rows) >= max_items:
                break

        if not rows:
            return ""

        lines = [
            "[META_RUNTIME_FAILURE_MEMORY]",
            "From recent attempts in this same problem. Use as guardrails only.",
        ]
        for i, (title, guidance, evidence) in enumerate(rows, start=1):
            lines.append(f"{i}. {title}: {guidance} (seen: {evidence})")
        lines.append("[/META_RUNTIME_FAILURE_MEMORY]")
        return "\n".join(lines)

    @staticmethod
    def _append_runtime_failure_memory(dev_prompt: str, runtime_block: str) -> str:
        rb = str(runtime_block or "").strip()
        if not rb:
            return dev_prompt
        base = str(dev_prompt or "").rstrip()
        return rb if not base else base + "\n\n" + rb

    def _extract_useful_tool_output(self, result: AttemptResult) -> str | None:
        for raw in reversed(tuple(result.python_outputs_text or ())):
            s = str(raw or "").strip()
            if not s:
                continue
            if (
                s.startswith("[ERROR]")
                or "Traceback" in s
                or "Error:" in s
                or "Exception" in s
            ):
                continue
            lines = [line.strip() for line in s.splitlines() if line.strip()]
            if not lines:
                continue
            sample = " | ".join(lines[-2:])
            return self._clip_single_line(sample, 220)
        return None

    def _build_portfolio_continuation_context(self, results: list[AttemptResult]) -> str:
        if not bool(getattr(self.cfg, "portfolio_enabled", False)):
            return ""
        if not results:
            return ""

        ranked = rank_candidates(
            results,
            filter_to_verified_if_any=False,
            magnitude_aware=self.cfg.magnitude_aware_ranking_enabled,
            ranking_strategy=self.cfg.ranking_strategy,
        )

        lines = [
            "[META_PORTFOLIO_STATE]",
            "Hints from earlier attempts. Treat them as hypotheses, not ground truth.",
        ]

        if ranked:
            lines.append("Candidate answers so far:")
            for idx, (ans, data) in enumerate(ranked[:3], start=1):
                support_bits = [f"votes={data['votes']}"]
                if int(data.get("verified", 0) or 0) > 0:
                    support_bits.append(f"clean_python={data['verified']}")
                if int(data.get("timeout_attempts", 0) or 0) > 0:
                    support_bits.append(f"timeouts={data['timeout_attempts']}")
                lines.append(f"{idx}. {ans} ({', '.join(support_bits)})")

        useful_rows: list[str] = []
        seen_outputs: set[str] = set()
        scored_results = sorted(
            results,
            key=lambda r: (
                int(bool(r.stats.tool_verified)),
                int(bool(r.answer is not None)),
                int(r.stats.python_calls),
                -int(r.attempt),
            ),
            reverse=True,
        )
        for r in scored_results:
            sample = self._extract_useful_tool_output(r)
            if not sample or sample in seen_outputs:
                continue
            seen_outputs.add(sample)
            profile = self._portfolio_name_from_tag(r.tag) or "default"
            label = f"attempt {r.attempt} [{profile}]"
            if isinstance(r.answer, int):
                label += f" -> {r.answer}"
            useful_rows.append(f"{len(useful_rows) + 1}. {label}: {sample}")
            if len(useful_rows) >= 3:
                break

        if useful_rows:
            lines.append("Useful exact artifacts:")
            lines.extend(useful_rows)

        failure_block = self._build_runtime_failure_memory_block(results, max_items=2)
        if failure_block:
            failure_lines = [
                line
                for line in failure_block.splitlines()
                if line
                and not line.startswith("[META_RUNTIME_FAILURE_MEMORY]")
                and not line.startswith("[/META_RUNTIME_FAILURE_MEMORY]")
                and not line.startswith("From recent attempts")
            ]
            if failure_lines:
                lines.append("Failure modes already seen:")
                lines.extend(failure_lines)

        lines.append("Next attempt requirements:")
        lines.append("1. Do not repeat prior code or prose verbatim.")
        lines.append(
            "2. Either derive the leading candidate by a materially different exact route or compute a quantity that decisively separates the candidates."
        )
        lines.append(
            "3. Prefer a short exact program that prints the deciding quantity or FINAL_ANSWER=<n>."
        )
        lines.append("[/META_PORTFOLIO_STATE]")

        block = "\n".join(lines)
        max_chars = max(400, int(getattr(self.cfg, "portfolio_summary_max_chars", 2200) or 2200))
        if len(block) <= max_chars:
            return block

        clipped = block[: max(0, max_chars - 32)].rstrip()
        if "\n" in clipped:
            clipped = clipped.rsplit("\n", 1)[0]
        return clipped + "\n[/META_PORTFOLIO_STATE]"

    @staticmethod
    def _attempt_has_failure_signal(result: AttemptResult) -> bool:
        st = result.stats
        return bool(
            int(getattr(st, "python_errors", 0) or 0) > 0
            or int(getattr(st, "timeout_count", 0) or 0) > 0
            or bool(getattr(st, "deadline_exceeded", False))
            or bool(str(getattr(st, "last_error", "") or "").strip())
        )

    @staticmethod
    def _attempt_has_timeout_signal(result: AttemptResult) -> bool:
        st = result.stats
        if int(getattr(st, "timeout_count", 0) or 0) > 0:
            return True
        err = str(getattr(st, "last_error", "") or "").lower()
        return "timed out" in err or "timeout" in err

    def _should_run_sequential_repair(
        self,
        results: list[AttemptResult],
        *,
        stopped_early: bool,
    ) -> bool:
        if stopped_early or not results:
            return False
        if not bool(getattr(self.cfg, "sequential_repair_enabled", True)):
            return False

        min_attempts = max(1, int(getattr(self.cfg, "sequential_repair_min_attempts", 2) or 2))
        if len(results) < min_attempts:
            return False

        timeout_only = bool(
            getattr(self.cfg, "sequential_repair_only_on_timeout", False)
        )
        signal_fn = (
            self._attempt_has_timeout_signal
            if timeout_only
            else self._attempt_has_failure_signal
        )

        fail_count = sum(1 for r in results if signal_fn(r))
        error_rate = fail_count / max(1, len(results))
        threshold = float(getattr(self.cfg, "sequential_repair_min_error_rate", 0.5) or 0.5)
        return error_rate >= threshold

    def _run_sequential_repair_attempts(
        self,
        *,
        user_input: str,
        problem_text: str,
        used_strategies: list[str] | None,
        preferred_strategy: str | None,
        results: list[AttemptResult],
        deadline: float,
        problem_start: float,
        early_stop_target: int,
        problem_id: str,
        temperature: float,
        next_attempt_idx: int,
    ) -> tuple[bool, int]:
        max_repairs = max(0, int(getattr(self.cfg, "sequential_repair_max_attempts", 2) or 2))
        if max_repairs <= 0:
            return False, next_attempt_idx

        print(f"[Adaptive] Running sequential repair pass ({max_repairs} max attempts)...")
        for _ in range(max_repairs):
            if time.time() > deadline:
                break

            dev_p, tag, strat_n = self._build_attempt_prompt(
                next_attempt_idx,
                problem_text,
                used_strategies,
                preferred_strategy if next_attempt_idx == 0 else None,
            )

            runtime_block = self._build_runtime_failure_memory_block(results)
            if runtime_block and tag != "answer-only":
                dev_p = self._append_runtime_failure_memory(dev_p, runtime_block)
            continuation_context = self._build_portfolio_continuation_context(results)

            r = self._process_attempt(
                user_input,
                dev_p,
                next_attempt_idx,
                tag,
                threading.Event(),
                deadline,
                problem_id=problem_id,
                continuation_context=continuation_context or None,
                temperature=temperature,
            )
            results.append(r)
            self._record_attempt_trace(problem_id, r)
            next_attempt_idx += 1

            if used_strategies and strat_n and strat_n not in used_strategies:
                used_strategies.append(strat_n)

            if self._should_early_stop(
                results,
                time.time() - problem_start,
                early_stop_target,
            ):
                return True, next_attempt_idx

        return False, next_attempt_idx

    def _prepare_completion_prompt(
        self, prompt_ids: list[int], extra: dict[str, Any]
    ) -> tuple[str | list[int], bool]:
        backend = self._backend_name()
        if backend == "llama_cpp":
            if int(extra.get("top_k", -1) or -1) < 0:
                extra["top_k"] = 40
            return self.encoding.decode(prompt_ids), False
        return prompt_ids, True

    def _completion_tokens_from_text(self, text: str) -> list[int]:
        with contextlib.suppress(Exception):
            return self.encoding.encode(text, allowed_special="all")
        with contextlib.suppress(Exception):
            return self.encoding.encode(text, disallowed_special=())
        return []

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
        self.sandbox_pool = queue.Queue()

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
        self._agent_memory_retriever = init_agent_memory_from_cfg(self.cfg)
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
        pool = getattr(self, "sandbox_pool", None)
        if pool is not None:
            while not pool.empty():
                with contextlib.suppress(Exception):
                    pool.get_nowait().close()
        if getattr(self, "server", None):
            with contextlib.suppress(Exception):
                self.server.stop()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()

    def _initialize_kernels(self) -> None:
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

        if self.cfg.reasoning_framework_enabled:
            dev_prompt = augment_prompt_with_reasoning_framework(
                dev_prompt,
                problem_text=problem_text,
            )

        use_meta_prompt = bool(
            self.cfg.wickelgren_strategies_enabled
            or getattr(self, "_wickelgren_retriever", None) is not None
            or getattr(self, "_agent_memory_retriever", None) is not None
        )

        if use_meta_prompt:
            dev_prompt, meta = augment_developer_prompt_with_meta(
                dev_prompt,
                attempt_index=attempt_index,
                problem_text=problem_text,
                agent_memory_retriever=getattr(self, "_agent_memory_retriever", None),
                agent_memory_skill_top_k=self.cfg.agent_memory_skill_top_k,
                agent_memory_failure_top_k=self.cfg.agent_memory_failure_top_k,
                agent_memory_min_score=self.cfg.agent_memory_min_score,
                retriever=getattr(self, "_wickelgren_retriever", None),
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
                include_strategy_card=self.cfg.wickelgren_strategies_enabled,
            )
            strat_name = (meta.get("card") or "").strip() or None
            if strat_name:
                tag = f"wickelgren:{strat_name}"
                if meta.get("retriever_used"):
                    tag += f"|rag={meta.get('retriever_results', 0)}|rag_backend={meta.get('retriever_backend', 'unknown')}"

        if bool(getattr(self.cfg, "portfolio_enabled", False)):
            portfolio_attempt_index = max(
                0,
                int(attempt_index)
                - max(0, int(getattr(self.cfg, "answer_only_attempts", 0) or 0)),
            )
            profile = self._portfolio_profile_for_attempt(
                portfolio_attempt_index, problem_text=problem_text
            )
            profile_block = self._render_portfolio_profile(profile)
            dev_prompt = (
                profile_block
                if not dev_prompt
                else dev_prompt.rstrip() + "\n\n" + profile_block
            )
            tag = self._attach_portfolio_tag(tag, profile.name)
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
        verified_votes = int(top_data.get("verified", 0) or 0)
        top_votes = int(top_data.get("votes", 0) or 0)
        allow_learning = verified_votes >= 2 and top_votes >= 3
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

        full_reasoning_payload = None
        if bool(getattr(self.cfg, "trace_full_reasoning_enabled", False)):
            fr = str(getattr(result, "full_reasoning_text", "") or "")
            max_chars = int(getattr(self.cfg, "trace_full_reasoning_max_chars", 0) or 0)
            if max_chars > 0 and len(fr) > max_chars:
                keep = max(0, max_chars - 3)
                fr = ("..." + fr[-keep:]) if keep > 0 else ""
            full_reasoning_payload = fr

        self._trace.record(
            {
                "event": "attempt_end",
                "problem_id": problem_id,
                "attempt": int(result.attempt),
                "tag": result.tag,
                "answer": result.answer,
                "extraction_rule": result.extraction_rule,
                "early_exit_reason": result.early_exit_reason,
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
                "full_reasoning_text": full_reasoning_payload,
            }
        )

    def _run_attempt_batch(
        self,
        *,
        user_input: str,
        task_specs: list[tuple[str, int, str | None, float]],
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
                    temperature=(task_temp if temperature is None else temperature),
                )
                for dev_p, idx, tag, task_temp in task_specs
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
        extraction_rule = None
        early_exit_reason = None
        text_tail, transcript_calls, transcript_outs = [], [], []
        deadline_exceeded = False
        recent_assistant_norm: list[str] = []
        recent_tool_call_norm: str | None = None
        assistant_repetition_streak = 0
        tool_repeat_streak = 0
        repetition_coached = False
        capture_full_reasoning = bool(
            getattr(self.cfg, "trace_full_reasoning_enabled", False)
        )
        full_reasoning_parts: list[str] = []

        if capture_full_reasoning:
            full_reasoning_parts.append(
                "[ATTEMPT_META]\n"
                f"attempt={attempt_index + 1}\n"
                f"tag={attempt_tag or ''}"
            )
            full_reasoning_parts.append(
                "[PROMPT_DEVELOPER]\n" + str(developer_prompt or "")
            )
            full_reasoning_parts.append("[PROMPT_USER]\n" + str(problem or ""))
            if continuation_context:
                full_reasoning_parts.append(
                    "[CONTINUATION_CONTEXT]\n" + str(continuation_context)
                )

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

                prompt_arg, use_stream = self._prepare_completion_prompt(
                    prompt_ids, extra
                )

                token_buf: list[int] = []
                text_buf: list[str] = []
                stream = None
                try:
                    if use_stream:
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
                            prompt=prompt_arg,
                            seed=attempt_seed,
                            stream=True,
                            extra_body=extra,
                        )

                        for chunk in stream:
                            if stop_event.is_set() or time.time() > deadline:
                                deadline_exceeded = True
                                break
                            choice = chunk.choices[0]
                            new_text = str(getattr(choice, "text", "") or "")
                            new_tokens = getattr(choice, "token_ids", None)
                            if new_tokens is None and new_text:
                                new_tokens = self._completion_tokens_from_text(new_text)
                            if new_tokens:
                                token_buf.extend(new_tokens)
                                total_tokens += len(new_tokens)
                            if new_text:
                                text_buf.append(new_text)
                                text_tail.append(new_text)
                                if self.cfg.entropy_weighting_enabled and choice.logprobs:
                                    logprobs_buf.extend(choice.logprobs.top_logprobs or [])

                            if self._can_try_stream_boxed_extraction(
                                new_text, total_tokens
                            ):
                                ans, rule = self._extractor.extract_boxed_int_with_rule(
                                    "".join(text_buf[-self.cfg.search_tokens :])
                                )
                                if ans is not None:
                                    final_answer = ans
                                    extraction_rule = rule
                                    early_exit_reason = "stream_boxed"
                                    break
                    else:
                        resp = self.client.completions.create(
                            model=self.cfg.served_model_name,
                            temperature=temp,
                            top_p=self.cfg.top_p,
                            max_tokens=max_tok,
                            prompt=prompt_arg,
                            seed=attempt_seed,
                            stream=False,
                            extra_body=extra,
                        )
                        new_text = str(getattr(resp.choices[0], "text", "") or "")
                        if new_text:
                            text_buf.append(new_text)
                            text_tail.append(new_text)
                            new_tokens = self._completion_tokens_from_text(new_text)
                            if new_tokens:
                                token_buf.extend(new_tokens)
                                total_tokens += len(new_tokens)

                            if self._can_try_stream_boxed_extraction(
                                new_text, total_tokens
                            ):
                                ans, rule = self._extractor.extract_boxed_int_with_rule(
                                    new_text[-self.cfg.search_tokens :]
                                )
                                if ans is not None:
                                    final_answer = ans
                                    extraction_rule = rule
                                    early_exit_reason = "stream_boxed"
                finally:
                    if stream is not None:
                        stream.close()

                if capture_full_reasoning and text_buf:
                    assistant_raw = "".join(text_buf).strip()
                    if assistant_raw:
                        full_reasoning_parts.append(
                            "[ASSISTANT_RAW]\n" + assistant_raw
                        )

                    norm = self._normalize_repetition_text(assistant_raw)
                    min_chars = max(
                        0,
                        int(getattr(self.cfg, "repetition_min_chars", 120) or 120),
                    )
                    if len(norm) >= min_chars and recent_assistant_norm:
                        sim = self._text_similarity(norm, recent_assistant_norm[-1])
                        threshold = max(
                            0.0,
                            min(
                                1.0,
                                float(
                                    getattr(
                                        self.cfg,
                                        "repetition_similarity_threshold",
                                        0.94,
                                    )
                                    or 0.94
                                ),
                            ),
                        )
                        if sim >= threshold:
                            assistant_repetition_streak += 1
                        else:
                            assistant_repetition_streak = 0
                    else:
                        assistant_repetition_streak = 0

                    recent_assistant_norm.append(norm)
                    if len(recent_assistant_norm) > 3:
                        recent_assistant_norm.pop(0)

                    action = self._repetition_watchdog_action(
                        assistant_repetition_streak=assistant_repetition_streak,
                        tool_repeat_streak=tool_repeat_streak,
                        coached=repetition_coached,
                    )
                    if action == "abort":
                        last_error = (
                            "[REPETITION_GUARD] Aborted attempt due to repeated reasoning pattern."
                        )
                        early_exit_reason = "repetition_guard"
                        break
                    if action == "coach":
                        repetition_coached = True
                        coach_msg = self._h["Message"].from_role_and_content(
                            self.Role.USER,
                            "You are repeating prior reasoning. Do not restate. "
                            "Either (1) run a materially different Python/Z3 check, or "
                            "(2) give final answer as \\boxed{n} now.",
                        )
                        conversation.messages = conversation.messages + [coach_msg]
                        continue

                if final_answer is not None or not token_buf:
                    break

                new_msgs = None
                if self._backend_name() == "llama_cpp":
                    with contextlib.suppress(Exception):
                        new_msgs = self.encoding.parse_messages_from_completion_tokens(
                            token_buf, self.Role.ASSISTANT, strict=False
                        )
                    if not new_msgs:
                        TextContent = self._h["TextContent"]
                        Author = self._h["Author"]
                        Message = self._h["Message"]
                        assistant_text = "".join(text_buf).strip()
                        content = (
                            [TextContent(text=assistant_text)] if assistant_text else []
                        )
                        author = Author(role=self.Role.ASSISTANT, name="assistant")
                        new_msgs = [Message(author=author, content=content)]
                else:
                    new_msgs = self.encoding.parse_messages_from_completion_tokens(
                        token_buf, self.Role.ASSISTANT
                    )
                if not new_msgs:
                    break

                conversation.messages = conversation.messages + list(new_msgs)
                last_msg = new_msgs[-1]

                if getattr(last_msg, "channel", None) == "final":
                    if capture_full_reasoning:
                        with contextlib.suppress(Exception):
                            full_reasoning_parts.append(
                                "[ASSISTANT_FINAL]\n" + str(last_msg.content[0].text or "")
                            )
                    boxed_ans, boxed_rule = self._extractor.extract_boxed_int_with_rule(
                        last_msg.content[0].text
                    )
                    if boxed_ans is not None:
                        final_answer = boxed_ans
                        extraction_rule = boxed_rule
                        early_exit_reason = "final_channel_boxed"
                    else:
                        fallback_ans, fallback_rule = self._extractor.extract_int_fallback_with_rule(
                            last_msg.content[0].text
                        )
                        final_answer = fallback_ans
                        extraction_rule = fallback_rule
                        if final_answer is not None:
                            early_exit_reason = "final_channel_fallback"
                    break

                if getattr(last_msg, "recipient", None) in ("python", "z3"):
                    python_calls += 1
                    tool_call_text = str(last_msg.content[0].text or "")
                    norm_tool_call = self._normalize_repetition_text(
                        tool_call_text, tail_chars=2000
                    )
                    if norm_tool_call and recent_tool_call_norm == norm_tool_call:
                        tool_repeat_streak += 1
                    else:
                        tool_repeat_streak = 0
                    recent_tool_call_norm = norm_tool_call

                    action = self._repetition_watchdog_action(
                        assistant_repetition_streak=assistant_repetition_streak,
                        tool_repeat_streak=tool_repeat_streak,
                        coached=repetition_coached,
                    )
                    if action == "abort":
                        last_error = (
                            "[REPETITION_GUARD] Aborted attempt due to repeated tool-call pattern."
                        )
                        early_exit_reason = "repetition_guard"
                        break

                    transcript_calls.append(tool_call_text)
                    if capture_full_reasoning and tool_call_text:
                        full_reasoning_parts.append("[TOOL_CALL]\n" + tool_call_text)
                    tool_resp = local_tool.process_sync_plus(last_msg)
                    resp_text = str(tool_resp[0].content[0].text or "")
                    transcript_outs.append(resp_text)
                    if capture_full_reasoning and resp_text:
                        full_reasoning_parts.append("[TOOL_OUTPUT]\n" + resp_text)

                    if (
                        resp_text.startswith("[ERROR]")
                        or "Traceback" in resp_text
                        or "Error:" in resp_text
                    ):
                        python_errors += 1
                        if "timed out" in resp_text.lower():
                            timeout_count += 1
                        last_error = resp_text[:500]

                    conversation.messages = conversation.messages + list(tool_resp)

                    marker_ans, marker_rule = self._extract_tool_final_answer(resp_text)
                    if marker_ans is not None:
                        final_answer = marker_ans
                        extraction_rule = marker_rule
                        early_exit_reason = "tool_final_marker"
                        break

        except Exception as exc:  # noqa: BLE001
            if last_error is None:
                last_error = f"[INTERNAL_ERROR] {type(exc).__name__}: {exc}"[:500]
            if capture_full_reasoning:
                full_reasoning_parts.append(
                    f"[INTERNAL_ERROR]\n{type(exc).__name__}: {exc}"
                )
        finally:
            if sandbox:
                if self.cfg.sandbox_reset_between_attempts:
                    with contextlib.suppress(Exception):
                        sandbox.reset()
                with contextlib.suppress(Exception):
                    self.sandbox_pool.put(sandbox)

        if final_answer is None and text_tail:
            full = "".join(text_tail)
            boxed_ans, boxed_rule = self._extractor.extract_boxed_int_with_rule(full)
            if boxed_ans is not None:
                final_answer = boxed_ans
                extraction_rule = boxed_rule
                early_exit_reason = "tail_boxed"
            else:
                fallback_ans, fallback_rule = self._extractor.extract_int_fallback_with_rule(
                    full
                )
                final_answer = fallback_ans
                extraction_rule = fallback_rule
                if final_answer is not None:
                    early_exit_reason = "tail_fallback"

        mean_ent = self._compute_mean_entropy(logprobs_buf)
        full_reasoning_text = None
        if capture_full_reasoning and full_reasoning_parts:
            full_reasoning_text = "\n\n".join(full_reasoning_parts)

        return AttemptResult(
            attempt=attempt_index + 1,
            answer=final_answer,
            extraction_rule=extraction_rule,
            early_exit_reason=early_exit_reason,
            stats=AttemptStats(
                token_count=total_tokens,
                python_calls=python_calls,
                python_errors=python_errors,
                timeout_count=timeout_count,
                mean_entropy=mean_ent,
                deadline_exceeded=deadline_exceeded,
                last_error=last_error,
            ),
            output_text="".join(text_tail),
            full_reasoning_text=full_reasoning_text,
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

        def _build_task_specs(
            start_idx: int, end_idx: int
        ) -> list[tuple[str, int, str | None, float]]:
            specs: list[tuple[str, int, str | None, float]] = []
            runtime_block = self._build_runtime_failure_memory_block(results)
            for i in range(start_idx, end_idx):
                dev_p, tag, strat_n = self._build_attempt_prompt(
                    i,
                    problem,
                    used_strats,
                    pref_strat if i == 0 else None,
                )
                if runtime_block and tag != "answer-only":
                    dev_p = self._append_runtime_failure_memory(dev_p, runtime_block)
                specs.append((dev_p, i, tag, self._temperature_for_attempt(temp_for_prob, i)))
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
            )

        next_attempt_idx = answer_only_count
        if not stopped_early and answer_only_count < attempts_for_prob:
            remaining_full_attempts = attempts_for_prob - answer_only_count
            scout_attempts = remaining_full_attempts
            if bool(getattr(self.cfg, "portfolio_enabled", False)):
                configured_scouts = int(
                    getattr(self.cfg, "portfolio_scout_attempts", 0) or 0
                )
                scout_attempts = (
                    min(remaining_full_attempts, configured_scouts)
                    if configured_scouts > 0
                    else min(remaining_full_attempts, 2)
                )
                scout_attempts = max(1, scout_attempts)

            scout_end = answer_only_count + scout_attempts
            stopped_early = self._run_attempt_batch(
                user_input=user_input,
                task_specs=_build_task_specs(answer_only_count, scout_end),
                results=results,
                deadline=deadline,
                problem_start=problem_start,
                early_stop_target=early_stop_prob,
                problem_id=problem_id,
            )
            next_attempt_idx = scout_end

            if not stopped_early and scout_end < attempts_for_prob:
                continuation_context = self._build_portfolio_continuation_context(results)
                stopped_early = self._run_attempt_batch(
                    user_input=user_input,
                    task_specs=_build_task_specs(scout_end, attempts_for_prob),
                    results=results,
                    deadline=deadline,
                    problem_start=problem_start,
                    early_stop_target=early_stop_prob,
                    problem_id=problem_id,
                    continuation_context=continuation_context or None,
                )
                next_attempt_idx = attempts_for_prob

        if self._should_run_sequential_repair(
            results,
            stopped_early=stopped_early,
        ):
            stopped_early, next_attempt_idx = self._run_sequential_repair_attempts(
                user_input=user_input,
                problem_text=problem,
                used_strategies=used_strats,
                preferred_strategy=pref_strat,
                results=results,
                deadline=deadline,
                problem_start=problem_start,
                early_stop_target=early_stop_prob,
                problem_id=problem_id,
                temperature=temp_for_prob,
                next_attempt_idx=next_attempt_idx,
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
                cont = self._build_portfolio_continuation_context(results)
                if not cont and best_w1 is not None:
                    cont = best_w1.output_text

                print(
                    f"[Adaptive] No answer found — granted {ext:.0f}s extension. Running second wave "
                    f"{'with' if cont else 'without'} continuation context...\n"
                )

                second_wave_specs = _build_task_specs(
                    next_attempt_idx,
                    next_attempt_idx + attempts_for_prob,
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
                )

        time_used = time.time() - problem_start
        self._display_candidates(results)

        ranked = rank_candidates(
            results,
            filter_to_verified_if_any=self.cfg.filter_to_verified_if_any,
            magnitude_aware=self.cfg.magnitude_aware_ranking_enabled,
            ranking_strategy=self.cfg.ranking_strategy,
        )

        if ranked:
            final_ans = ranked[0][0]
            data = ranked[0][1]
            print(
                f"\nFinal Answer: {final_ans} (votes={data['votes']}, verified={data['verified']})\n"
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
