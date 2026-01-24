from __future__ import annotations

"""AIMO-3 multi-attempt solver (ported and modularized).

This module intentionally keeps imports *lazy* so that the base project can be
installed without the heavy AIMO-3 stack.
"""

import contextlib
import math
import os
import queue
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

from .config import AIMO3Config
from .errors import OptionalDependencyError
from .prompts import TIR_PROMPT_ANALYTIC, TIR_PROMPT_CODE_FIRST, TIR_PROMPT_STANDARD, TIR_PROMPT_VERIFICATION, TIR_PROMPTS
from .sandbox import AIMO3Sandbox
from .vllm_server import VLLMServer
from .wickelgren import augment_system_prompt
from .protocol import with_protocol
from .ranking import rank_candidates
from .budget import compute_attempt_and_verify_deadlines

from .answer_extraction import AnswerExtractor
from .attempts import AttemptResult, AttemptStats


def _require_openai():
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise OptionalDependencyError("AIMO3Solver requires 'openai'. Install extras: pip install .[aimo3]") from e
    return OpenAI


def _require_harmony():
    try:
        from openai_harmony import (  # type: ignore
            HarmonyEncodingName,
            load_harmony_encoding,
            SystemContent,
            ReasoningEffort,
            ToolNamespaceConfig,
            Author,
            Message,
            Role,
            TextContent,
            Conversation,
        )
    except Exception as e:  # noqa: BLE001
        raise OptionalDependencyError(
            "AIMO3Solver requires 'openai_harmony'. Install it from your offline wheels or pip (package may be named openai-harmony)."
        ) from e

    return {
        "HarmonyEncodingName": HarmonyEncodingName,
        "load_harmony_encoding": load_harmony_encoding,
        "SystemContent": SystemContent,
        "ReasoningEffort": ReasoningEffort,
        "ToolNamespaceConfig": ToolNamespaceConfig,
        "Author": Author,
        "Message": Message,
        "Role": Role,
        "TextContent": TextContent,
        "Conversation": Conversation,
    }


class AIMO3Template:
    def __init__(self):
        self._h = _require_harmony()

    def get_system_content(self, system_prompt: str, tool_config):
        SystemContent = self._h["SystemContent"]
        ReasoningEffort = self._h["ReasoningEffort"]
        return (
            SystemContent.new()
            .with_model_identity(system_prompt)
            .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
            .with_tools(tool_config)
        )

    def apply_chat_template(self, system_prompt: str, user_prompt: str, tool_config):
        Message = self._h["Message"]
        Role = self._h["Role"]
        system_content = self.get_system_content(system_prompt, tool_config)
        system_message = Message.from_role_and_content(Role.SYSTEM, system_content)
        user_message = Message.from_role_and_content(Role.USER, user_prompt)
        return [system_message, user_message]


class AIMO3Tool:
    """Bridges Harmony tool-call messages to a sandboxed Jupyter kernel."""

    def __init__(self, local_jupyter_timeout: float, tool_prompt: str, sandbox: AIMO3Sandbox | None = None):
        self._h = _require_harmony()
        self._local_jupyter_timeout = float(local_jupyter_timeout)
        self._tool_prompt = tool_prompt
        self._jupyter_session = sandbox
        self._owns_session = sandbox is None
        self._execution_lock = threading.Lock()
        self._init_lock = threading.Lock()

    def _ensure_session(self) -> None:
        if self._jupyter_session is None:
            with self._init_lock:
                if self._jupyter_session is None:
                    self._jupyter_session = AIMO3Sandbox(timeout=self._local_jupyter_timeout)

    @staticmethod
    def _ensure_last_print(code: str) -> str:
        lines = code.strip().split("\n")
        if not lines:
            return code
        last = lines[-1].strip()
        if not last or last.startswith("#"):
            return code
        if "print" in last or last.startswith("import"):
            return code
        lines[-1] = f"print({last})"
        return "\n".join(lines)

    @property
    def instruction(self) -> str:
        return self._tool_prompt

    @property
    def tool_config(self):
        ToolNamespaceConfig = self._h["ToolNamespaceConfig"]
        return ToolNamespaceConfig(name="python", description=self.instruction, tools=[])

    def _make_response(self, output: str, channel: str | None = None):
        TextContent = self._h["TextContent"]
        Author = self._h["Author"]
        Message = self._h["Message"]
        Role = self._h["Role"]
        content = TextContent(text=output)
        author = Author(role=Role.TOOL, name="python")
        msg = Message(author=author, content=[content]).with_recipient("assistant")
        if channel:
            msg = msg.with_channel(channel)
        return msg

    def process_sync_plus(self, message):
        self._ensure_session()
        raw_script = message.content[0].text
        final_script = self._ensure_last_print(raw_script)
        with self._execution_lock:
            output = self._jupyter_session.execute(final_script)
        return [self._make_response(output, channel=getattr(message, "channel", None))]

    def close(self) -> None:
        if self._jupyter_session is not None and self._owns_session:
            self._jupyter_session.close()
        self._jupyter_session = None

    def __del__(self) -> None:
        self.close()


@dataclass
class AIMO3Solver:
    cfg: AIMO3Config
    port: int = 8000

    @staticmethod
    def _truncate(text: str | None, max_chars: int) -> str:
        if not text:
            return ""
        max_chars = int(max_chars)
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        return "…" + text[-max_chars:]

    def _attempt_to_row(self, r: AttemptResult) -> dict:
        snippet = self._truncate(r.output_text, int(self.cfg.display_attempt_text_chars))
        return {
            "Attempt": r.attempt,
            "Answer": r.answer,
            "ToolVerified": bool(r.stats.tool_verified),
            "PyCalls": int(r.stats.python_calls),
            "PyErrors": int(r.stats.python_errors),
            "Tokens": int(r.stats.token_count),
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

    @staticmethod
    def _has_verification_marker(text: str | None, marker: str) -> bool:
        if not marker:
            return False
        return marker in (text or "")

    def _should_early_stop(self, detailed: list[AttemptResult]) -> bool:
        """Quality-aware early stop.

        Default behavior requires at least one clean tool run for the leading candidate
        before early stopping. This reduces the "wrong but popular" failure mode.
        """

        need = max(0, int(self.cfg.early_stop_min_verified))
        target_votes = max(1, int(self.cfg.early_stop))
        ranked_all = rank_candidates(detailed, filter_to_verified_if_any=False)
        if not ranked_all:
            return False

        top_ans, top_d = ranked_all[0]
        _ = top_ans
        votes = int(top_d.get("votes", 0))
        verified = int(top_d.get("verified", 0))
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

        self.template = AIMO3Template()
        self.encoding = h["load_harmony_encoding"](h["HarmonyEncodingName"].HARMONY_GPT_OSS)
        self.Role = h["Role"]
        self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()

        self.server = VLLMServer(cfg=self.cfg, port=self.port)

        self.base_url = f"http://0.0.0.0:{self.port}/v1"

        # If a server is already running on this port, reuse it.
        if bool(self.cfg.reuse_existing_server):
            probe_client = OpenAI(base_url=self.base_url, api_key="sk-local", timeout=self.cfg.server_probe_timeout)
            if self._probe_server_ready(probe_client, attempts=self.cfg.server_probe_attempts):
                # Reuse: don't start a new process.
                self.server = None
                self.client = OpenAI(base_url=self.base_url, api_key="sk-local", timeout=self.cfg.session_timeout)
            else:
                self.server = VLLMServer(cfg=self.cfg, port=self.port)
                self.server.start()
                self.client = OpenAI(base_url=self.base_url, api_key="sk-local", timeout=self.cfg.session_timeout)
                self.server.wait_ready(self.client)
        else:
            self.server.start()
            self.client = OpenAI(base_url=self.base_url, api_key="sk-local", timeout=self.cfg.session_timeout)
            self.server.wait_ready(self.client)

        self._initialize_kernels()
        self.notebook_start_time = time.time()
        self.problems_remaining = int(self.cfg.problems_total)

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
        with ThreadPoolExecutor(max_workers=init_workers) as ex:
            futures = [ex.submit(_create) for _ in range(pool_size)]
            for f in as_completed(futures):
                self.sandbox_pool.put(f.result())

    @property
    def _extractor(self) -> AnswerExtractor:
        # Centralize extraction behavior (range, formatting).
        return AnswerExtractor(aimo_lo=0, aimo_hi=99999)

    def _process_attempt(
        self,
        problem: str,
        system_prompt: str,
        attempt_index: int,
        stop_event: threading.Event,
        deadline: float,
    ) -> AttemptResult:
        if stop_event.is_set() or time.time() > deadline:
            return AttemptResult(
                attempt=attempt_index + 1,
                answer=None,
                stats=AttemptStats(token_count=0, python_calls=0, python_errors=0),
                output_text=None,
            )

        local_tool: AIMO3Tool | None = None
        sandbox: AIMO3Sandbox | None = None
        borrowed_from_pool = False
        python_calls = 0
        python_errors = 0
        total_tokens = 0
        final_answer: int | None = None
        # Keep a tail buffer of the assistant text so we can display candidate solutions.
        text_tail: str = ""
        cap = int(self.cfg.capture_attempt_text_chars)

        attempt_seed = int(math.pow(self.cfg.seed + attempt_index, 2))

        try:
            try:
                sandbox = self.sandbox_pool.get(timeout=self.cfg.sandbox_timeout)
                borrowed_from_pool = True
            except queue.Empty:
                if bool(getattr(self.cfg, "sandbox_create_on_exhaustion", True)):
                    sandbox = AIMO3Sandbox(timeout=self.cfg.jupyter_timeout)
                    borrowed_from_pool = False
                else:
                    raise

            local_tool = AIMO3Tool(local_jupyter_timeout=self.cfg.jupyter_timeout, tool_prompt=self.cfg.tool_prompt, sandbox=sandbox)

            messages = self.template.apply_chat_template(system_prompt, problem, local_tool.tool_config)
            Conversation = _require_harmony()["Conversation"]
            conversation = Conversation.from_messages(messages)

            for _ in range(self.cfg.turns):
                if stop_event.is_set() or time.time() > deadline:
                    break

                prompt_ids = self.encoding.render_conversation_for_completion(conversation, self.Role.ASSISTANT)
                max_tokens = self.cfg.context_tokens - len(prompt_ids)
                if max_tokens < self.cfg.buffer_tokens:
                    break

                stream = self.client.completions.create(
                    model=self.cfg.served_model_name,
                    temperature=self.cfg.temperature,
                    max_tokens=max_tokens,
                    prompt=prompt_ids,
                    seed=attempt_seed,
                    stream=True,
                    extra_body={
                        "min_p": self.cfg.min_p,
                        "stop_token_ids": self.stop_token_ids,
                        "return_token_ids": True,
                    },
                    timeout=max(0.0, deadline - time.time()),
                )

                token_buffer: list[int] = []
                text_chunks: list[str] = []
                try:
                    for chunk in stream:
                        if stop_event.is_set() or time.time() > deadline:
                            break
                        new_tokens = chunk.choices[0].token_ids
                        new_text = chunk.choices[0].text
                        if new_tokens:
                            token_buffer.extend(new_tokens)
                            total_tokens += len(new_tokens)
                            text_chunks.append(new_text)
                            if new_text:
                                text_tail = (text_tail + new_text)
                                if cap > 0 and len(text_tail) > cap:
                                    text_tail = text_tail[-cap:]
                        if "}" in new_text:
                            search_text = "".join(text_chunks[-self.cfg.search_tokens :])
                            ans = self._extractor.extract_boxed_int(search_text)
                            if ans is not None:
                                final_answer = ans
                                break
                finally:
                    with contextlib.suppress(Exception):
                        stream.close()

                if final_answer is not None:
                    break
                if not token_buffer:
                    break

                new_messages = self.encoding.parse_messages_from_completion_tokens(token_buffer, self.Role.ASSISTANT)
                conversation.messages.extend(new_messages)
                last = new_messages[-1]

                if last.channel == "final":
                    answer_text = last.content[0].text
                    if answer_text:
                        text_tail = (text_tail + answer_text)
                        if cap > 0 and len(text_tail) > cap:
                            text_tail = text_tail[-cap:]
                    final_answer = self._extractor.extract_boxed_int(answer_text)
                    break

                if last.recipient == "python":
                    python_calls += 1
                    tool_responses = local_tool.process_sync_plus(last)
                    resp_text = tool_responses[0].content[0].text
                    if resp_text.startswith("[ERROR]") or "Traceback" in resp_text or "Error:" in resp_text:
                        python_errors += 1
                    conversation.messages.extend(tool_responses)

        except Exception:
            python_errors += 1
        finally:
            if local_tool is not None:
                local_tool.close()
            if sandbox is not None:
                if borrowed_from_pool:
                    with contextlib.suppress(Exception):
                        sandbox.reset()
                    self.sandbox_pool.put(sandbox)
                else:
                    with contextlib.suppress(Exception):
                        sandbox.close()

        return AttemptResult(
            attempt=attempt_index + 1,
            answer=final_answer,
            stats=AttemptStats(token_count=total_tokens, python_calls=python_calls, python_errors=python_errors),
            output_text=text_tail,
        )

    @staticmethod
    def _rank_answers(detailed_results: list[dict]) -> list[tuple[int, dict]]:
        # Compatibility wrapper around the dedicated module.
        return rank_candidates(detailed_results, filter_to_verified_if_any=True)

    def _second_stage_verify(self, user_input: str, candidates: list[int], verify_deadline: float) -> int | None:
        if not candidates:
            return None

        remaining = verify_deadline - time.time()
        if remaining <= float(self.cfg.second_stage_verify_min_effective_time):
            return None

        thr = float(self.cfg.second_stage_verify_repeats_threshold)
        repeats = int(self.cfg.second_stage_verify_repeats_high) if remaining >= thr else int(self.cfg.second_stage_verify_repeats_low)
        max_workers = min(int(self.cfg.second_stage_verify_workers_cap), max(1, self.cfg.workers))

        stop_event = threading.Event()
        supports: dict[int, int] = {c: 0 for c in candidates}

        tasks: list[tuple[str, int, int]] = []
        base = int(self.cfg.second_stage_verify_attempt_base)
        marker = str(getattr(self.cfg, "second_stage_verify_marker", "VERIFIED_OK"))
        require_marker = bool(getattr(self.cfg, "second_stage_verify_require_marker", True))
        for ci, cand in enumerate(candidates):
            verify_problem = (
                f"{user_input}\n\n"
                f"Second-stage verification. Candidate answer: {cand}.\n"
                "Task: verify the candidate using rigorous reasoning and the Python tool if needed.\n"
                f"If and ONLY if you are fully convinced the correct final answer equals {cand}, do BOTH:\n"
                f"- include the marker line: {marker}\n"
                f"- then output exactly one boxed line: \\boxed{{{cand}}}\n"
                "Otherwise output NOBOX (do NOT output any \\boxed{...})."
            )
            for rep in range(repeats):
                tasks.append((verify_problem, cand, base + ci * 10 + rep))

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(self._process_attempt, vp, TIR_PROMPT_VERIFICATION, attempt_idx, stop_event, verify_deadline)
                for (vp, _cand, attempt_idx) in tasks
            ]

            for fut, (_vp, cand, _attempt_idx) in zip(futures, tasks):
                if time.time() > verify_deadline:
                    break
                with contextlib.suppress(Exception):
                    r: AttemptResult = fut.result(timeout=max(0.0, verify_deadline - time.time()))
                    if r.answer == cand and r.stats.tool_verified:
                        if require_marker and not self._has_verification_marker(r.output_text, marker):
                            continue
                        supports[cand] = supports.get(cand, 0) + 1

        winners = sorted(supports.items(), key=lambda kv: kv[1], reverse=True)
        if not winners:
            return None
        best_cand, best_support = winners[0]
        if best_support <= 0:
            return None
        if len(winners) >= 2 and winners[1][1] == best_support:
            return None
        return best_cand

    def solve_problem(self, problem: str) -> int:
        problem_start_time = time.time()
        user_input = f"{problem} {self.cfg.preference_prompt}"

        elapsed_global = time.time() - self.notebook_start_time
        time_left = float(self.cfg.notebook_limit) - elapsed_global
        problems_left_others = max(0, int(self.problems_remaining) - 1)
        reserved = problems_left_others * float(self.cfg.base_problem_timeout)

        slack = max(0.0, time_left - reserved - float(self.cfg.base_problem_timeout))
        extra = min(slack * 0.50, float(self.cfg.high_problem_timeout) - float(self.cfg.base_problem_timeout))
        budget = float(self.cfg.base_problem_timeout) + extra
        budget = min(budget, float(self.cfg.high_problem_timeout))
        budget = max(budget, float(self.cfg.base_problem_timeout))

        now = time.time()
        overall_deadline = now + budget

        # Keep time for verification by using an earlier deadline for attempt generation.
        attempt_deadline, deadline = compute_attempt_and_verify_deadlines(
            now=now,
            overall_deadline=overall_deadline,
            reserve_fraction=float(self.cfg.verification_reserve_fraction),
            reserve_cap_s=float(self.cfg.verification_reserve_cap),
            reserve_min_s=float(self.cfg.verification_reserve_min),
        )

        tasks: list[tuple[str, int]] = []
        for attempt_index in range(int(self.cfg.attempts)):
            base = TIR_PROMPTS[attempt_index % len(TIR_PROMPTS)]
            sys_prompt = (
                augment_system_prompt(base, attempt_index=attempt_index)
                if bool(self.cfg.wickelgren_strategies_enabled)
                else base
            )
            if bool(self.cfg.protocol_enabled):
                sys_prompt = with_protocol(sys_prompt)
            tasks.append((sys_prompt, attempt_index))

        detailed: list[AttemptResult] = []
        valid: list[int] = []
        stop_event = threading.Event()

        with ThreadPoolExecutor(max_workers=int(self.cfg.workers)) as ex:
            futures = [
                ex.submit(self._process_attempt, user_input, sys_prompt, attempt_idx, stop_event, attempt_deadline)
                for (sys_prompt, attempt_idx) in tasks
            ]
            for fut in as_completed(futures):
                with contextlib.suppress(Exception):
                    r: AttemptResult = fut.result()
                    detailed.append(r)
                    if r.answer is not None and isinstance(r.answer, int):
                        valid.append(r.answer)

                    if self._should_early_stop(detailed):
                        stop_event.set()
                        break

        self.problems_remaining = max(0, int(self.problems_remaining) - 1)

        # Retry if no valid answers.
        if not valid:
            retry_budget = min(60.0, max(10.0, budget * 0.25))
            retry_deadline = time.time() + retry_budget
            retry_tasks = [
                (
                    augment_system_prompt(TIR_PROMPT_VERIFICATION, attempt_index=int(self.cfg.attempts) + 0)
                    if bool(self.cfg.wickelgren_strategies_enabled)
                    else TIR_PROMPT_VERIFICATION,
                    int(self.cfg.attempts) + 0,
                ),
                (
                    augment_system_prompt(TIR_PROMPT_ANALYTIC, attempt_index=int(self.cfg.attempts) + 1)
                    if bool(self.cfg.wickelgren_strategies_enabled)
                    else TIR_PROMPT_ANALYTIC,
                    int(self.cfg.attempts) + 1,
                ),
                (
                    augment_system_prompt(TIR_PROMPT_CODE_FIRST, attempt_index=int(self.cfg.attempts) + 2)
                    if bool(self.cfg.wickelgren_strategies_enabled)
                    else TIR_PROMPT_CODE_FIRST,
                    int(self.cfg.attempts) + 2,
                ),
                (
                    augment_system_prompt(TIR_PROMPT_STANDARD, attempt_index=int(self.cfg.attempts) + 3)
                    if bool(self.cfg.wickelgren_strategies_enabled)
                    else TIR_PROMPT_STANDARD,
                    int(self.cfg.attempts) + 3,
                ),
            ]

            # Apply protocol to retry prompts too.
            if bool(self.cfg.protocol_enabled):
                retry_tasks = [(with_protocol(p), idx) for (p, idx) in retry_tasks]

            with ThreadPoolExecutor(max_workers=min(4, int(self.cfg.workers))) as ex:
                futures = [
                    ex.submit(self._process_attempt, user_input, sys_prompt, attempt_idx, stop_event, retry_deadline)
                    for (sys_prompt, attempt_idx) in retry_tasks
                ]
                for fut in as_completed(futures):
                    with contextlib.suppress(Exception):
                        r: AttemptResult = fut.result()
                        detailed.append(r)
                        if r.answer is not None and isinstance(r.answer, int):
                            valid.append(r.answer)

        if not valid:
            return 0

        # Show candidate attempts (best-effort, notebook-friendly).
        self._display_candidates(detailed)

        ranked = self._rank_answers(detailed)
        if not ranked:
            return 0

        top_ans, top_d = ranked[0]
        chosen = top_ans
        if len(ranked) >= 2:
            runner_ans, runner_d = ranked[1]
            votes_gap = int(top_d["votes"]) - int(runner_d["votes"])

            need_verify = False
            if bool(self.cfg.second_stage_verify_enabled):
                if bool(self.cfg.second_stage_verify_trigger_if_no_verified) and int(top_d["verified"]) == 0:
                    need_verify = True
                if votes_gap <= int(self.cfg.second_stage_verify_trigger_votes_gap):
                    need_verify = True

            remaining = deadline - time.time()
            if need_verify and remaining >= float(self.cfg.second_stage_verify_min_remaining):
                verify_budget = min(
                    float(self.cfg.second_stage_verify_budget_cap),
                    remaining * float(self.cfg.second_stage_verify_budget_fraction),
                )
                verify_deadline = time.time() + verify_budget
                top_k = max(2, int(self.cfg.second_stage_verify_top_k))
                candidates = [ans for (ans, _d) in ranked[:top_k]]
                verified_choice = self._second_stage_verify(user_input, candidates, verify_deadline)
                if verified_choice is not None:
                    chosen = verified_choice

        _ = time.time() - problem_start_time  # for optional logging
        return int(chosen)
