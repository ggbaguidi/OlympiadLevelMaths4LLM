from __future__ import annotations

"""AIMO-3 multi-attempt solver (ported and modularized).

This module intentionally keeps imports *lazy* so that the base project can be
installed without the heavy AIMO-3 stack.
"""

import contextlib
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Optional

from .config import AIMO3Config
from .errors import OptionalDependencyError
from .prompts import TIR_PROMPT_ANALYTIC, TIR_PROMPT_CODE_FIRST, TIR_PROMPT_STANDARD, TIR_PROMPT_VERIFICATION, TIR_PROMPT_SMALL_CASES, TIR_PROMPT_SANITY
from .sandbox import AIMO3Sandbox
from .vllm_server import VLLMServer
from .wickelgren import augment_system_prompt_with_meta
from .protocol import with_protocol
from .ranking import rank_candidates
from .budget import adaptive_verify_budget, compute_attempt_and_verify_deadlines, reserve_fraction_for_budget, TimeBudgetTracker

from .answer_extraction import AnswerExtractor
from .attempts import AttemptResult, AttemptStats
from .trace import TraceRecorder, stable_problem_id
from .decoding import temperature_for_attempt
from .recovery import (
    ToolRecoveryPolicy,
    should_abort_attempt,
    should_recycle_sandbox,
    should_schedule_recovery_attempt,
    tool_call_cap_for_attempt,
    should_schedule_format_recovery_attempt,
)
from .tool_drain import iter_tool_calls
from .python_timeouts import parse_timeout_directive, parse_timeout_error
from .python_rewrite import rewrite_python_tool_code


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

    # DeveloperContent was introduced in newer openai_harmony releases.
    DeveloperContent = None
    with contextlib.suppress(Exception):
        from openai_harmony import DeveloperContent as _DeveloperContent  # type: ignore

        DeveloperContent = _DeveloperContent

    return {
        "HarmonyEncodingName": HarmonyEncodingName,
        "load_harmony_encoding": load_harmony_encoding,
        "SystemContent": SystemContent,
        "DeveloperContent": DeveloperContent,
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

    @staticmethod
    def _default_model_identity() -> str:
        # Per the Harmony prompt-format guidance, the system message identity should remain stable.
        return "You are ChatGPT, a large language model trained by OpenAI."

    def get_system_content(self, tool_config):
        SystemContent = self._h["SystemContent"]
        ReasoningEffort = self._h["ReasoningEffort"]
        return (
            SystemContent.new()
            .with_model_identity(self._default_model_identity())
            .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
            .with_tools(tool_config)
        )

    def apply_chat_template(self, developer_prompt: str, user_prompt: str, tool_config):
        Message = self._h["Message"]
        Role = self._h["Role"]
        system_content = self.get_system_content(tool_config)
        system_message = Message.from_role_and_content(Role.SYSTEM, system_content)

        # The project prompts ("system prompts" in older code) are the DEVELOPER instructions
        # in Harmony terms.
        developer_content = None
        if "DeveloperContent" in self._h:
            DeveloperContent = self._h["DeveloperContent"]
            with contextlib.suppress(Exception):
                developer_content = DeveloperContent.new().with_instructions(developer_prompt)

        if developer_content is not None:
            developer_message = Message.from_role_and_content(Role.DEVELOPER, developer_content)
        else:
            # Compatibility fallback for older openai_harmony builds.
            developer_message = Message.from_role_and_content(Role.DEVELOPER, developer_prompt)

        user_message = Message.from_role_and_content(Role.USER, user_prompt)
        return [system_message, developer_message, user_message]


class AIMO3Tool:
    """Bridges Harmony tool-call messages to a sandboxed Jupyter kernel."""

    def __init__(
        self,
        local_jupyter_timeout: float,
        tool_prompt: str,
        sandbox: AIMO3Sandbox | None = None,
        tool_timeout_cap_s: float | None = None,
    ):
        self._h = _require_harmony()
        self._local_jupyter_timeout = float(local_jupyter_timeout)
        self._tool_prompt = tool_prompt
        self._tool_timeout_cap_s = None if tool_timeout_cap_s is None else float(tool_timeout_cap_s)
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
        # Best-effort UX: if the user ends their tool snippet with a simple expression,
        # auto-wrap it in print(...). This helps avoid the common warning:
        #   [WARN] No output. Use print() to see results.
        #
        # Safety: do NOT rewrite multi-line blocks (function/class defs, loops, returns,
        # indented code, etc.) because that can change semantics or introduce SyntaxError.
        src = str(code or "")
        stripped = src.strip("\n")
        if not stripped.strip():
            return src

        lines = stripped.split("\n")
        if not lines:
            return src

        raw_last_line = lines[-1]
        last = raw_last_line.strip()
        if not last or last.startswith("#"):
            return src

        # If the last line is indented, it's almost certainly inside a block.
        indent = raw_last_line[: len(raw_last_line) - len(raw_last_line.lstrip())]
        if indent:
            return src

        # Heuristic: avoid rewriting statements/headers.
        statement_prefixes = (
            "return",
            "def ",
            "class ",
            "for ",
            "while ",
            "if ",
            "elif ",
            "else",
            "try",
            "except",
            "finally",
            "with ",
            "import ",
            "from ",
            "raise",
            "assert",
            "pass",
            "break",
            "continue",
            "yield",
            "del ",
            "global ",
            "nonlocal ",
            "@",
        )
        lower_last = last.lower()
        if lower_last.endswith(":") or lower_last.startswith(statement_prefixes):
            return src

        # If it already prints, do nothing.
        if last.startswith("print(") or "print" in last:
            return src

        # Only apply to single-line snippets or when the last line is a top-level expression.
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

    def process_sync_plus(self, message, timeout_override_s: float | None = None):
        self._ensure_session()
        raw_script = message.content[0].text
        # Apply best-effort rewrites for known API mismatches before execution.
        rewritten_script = rewrite_python_tool_code(str(raw_script or ""))
        final_script = self._ensure_last_print(rewritten_script)

        timeout_s: float | None = None
        if timeout_override_s is not None:
            timeout_s = float(timeout_override_s)
        else:
            # Optional per-call directive: first non-empty line '# timeout: N'
            with contextlib.suppress(Exception):
                timeout_s = parse_timeout_directive(str(raw_script or ""))

        if timeout_s is not None and self._tool_timeout_cap_s is not None:
            timeout_s = min(float(timeout_s), float(self._tool_timeout_cap_s))

        with self._execution_lock:
            output = self._jupyter_session.execute(final_script, timeout=timeout_s)
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
            sb = self.sandbox_pool.get(timeout=float(getattr(self.cfg, "sandbox_timeout", 0.0) or 0.0) or 0.5)
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
            out = sb.execute(code, timeout=min(2.0, float(getattr(self.cfg, "jupyter_timeout", 10.0) or 10.0)))

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

    @staticmethod
    def _has_verification_marker(text: str | None, marker: str) -> bool:
        if not marker:
            return False
        return marker in (text or "")

    @staticmethod
    def _compute_mean_entropy(logprobs_buffer: list[object]) -> float:
        """Best-effort mean entropy from top-k logprobs.

        We only see top-k probabilities, so this underestimates true entropy, but it's
        still a useful confidence proxy to break ties.
        """

        if not logprobs_buffer:
            return float("inf")

        total_entropy = 0.0
        token_count = 0
        for top_dict in logprobs_buffer:
            if not isinstance(top_dict, dict):
                continue
            if not top_dict:
                continue

            ent = 0.0
            # top_dict maps token_str -> logprob
            for _tok, lp in top_dict.items():
                try:
                    prob = math.exp(float(lp))
                except Exception:  # noqa: BLE001
                    continue
                if prob > 0.0:
                    ent -= prob * math.log2(prob)

            total_entropy += ent
            token_count += 1

        if token_count <= 0:
            return float("inf")
        return total_entropy / float(token_count)

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

        # Optional: install Lean toolchain from an offline archive.
        # Do this BEFORE initializing kernels so that PATH is inherited.
        if bool(getattr(self.cfg, "lean_toolchain_enabled", False)):
            try:
                from .lean_toolchain import ensure_lean_toolchain
            except Exception:  # noqa: BLE001
                # Optional module; fail gracefully if something is wrong with packaging.
                ensure_lean_toolchain = None

            if ensure_lean_toolchain is not None:
                ensure_lean_toolchain(
                    enabled=True,
                    dataset_dir=str(getattr(self.cfg, "lean_toolchain_dataset_dir", "") or "") or None,
                    archive_path=str(getattr(self.cfg, "lean_toolchain_archive_path", "") or "") or None,
                    archive_name=str(getattr(self.cfg, "lean_toolchain_archive_name", "") or "") or None,
                    work_dir=str(getattr(self.cfg, "lean_toolchain_work_dir", "") or "") or None,
                    prefer_existing=True,
                    check_versions=False,
                    verbose=bool(getattr(self.cfg, "lean_toolchain_verbose", False)),
                    strict=True,
                )

        self._initialize_kernels()
        self.notebook_start_time = time.time()
        self.problems_remaining = int(self.cfg.problems_total)
        
        # Dynamic time budgeting: track actual solve times to adjust per-problem budgets
        self._budget_tracker = TimeBudgetTracker(
            total_budget_s=float(self.cfg.notebook_limit),
            total_problems=int(self.cfg.problems_total),
            base_timeout_s=float(self.cfg.base_problem_timeout),
            high_timeout_s=float(self.cfg.high_problem_timeout),
        )

        # Notebook-friendly tracing behavior: optionally reset the trace file at startup.
        if bool(getattr(self.cfg, "trace_enabled", False)) and bool(getattr(self.cfg, "trace_reset_on_start", False)):
            with contextlib.suppress(Exception):
                p = str(getattr(self.cfg, "trace_path", "aimo3_trace.jsonl") or "aimo3_trace.jsonl")
                if p and os.path.exists(p):
                    os.remove(p)

        self._trace = TraceRecorder(
            enabled=bool(getattr(self.cfg, "trace_enabled", False)),
            path=str(getattr(self.cfg, "trace_path", "aimo3_trace.jsonl")),
            include_problem_text=bool(getattr(self.cfg, "trace_include_problem_text", False)),
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
        developer_prompt: str,
        attempt_index: int,
        attempt_tag: str | None,
        stop_event: threading.Event,
        deadline: float,
        problem_id: str | None = None,
    ) -> AttemptResult:
        if stop_event.is_set() or time.time() > deadline:
            return AttemptResult(
                attempt=attempt_index + 1,
                answer=None,
                stats=AttemptStats(token_count=0, python_calls=0, python_errors=0, lean_calls=0),
                output_text=None,
            )

        local_tool: AIMO3Tool | None = None
        sandbox: AIMO3Sandbox | None = None
        borrowed_from_pool = False
        python_calls = 0
        python_errors = 0
        lean_calls = 0
        consecutive_python_errors = 0
        timeout_count = 0
        total_tokens = 0
        final_answer: int | None = None
        # Keep a tail buffer of the assistant text so we can display candidate solutions.
        text_tail: str = ""
        cap = int(self.cfg.capture_attempt_text_chars)

        # Per-attempt transcript capture (safe):
        # - store user-visible assistant channels (final/commentary)
        # - store python tool calls + outputs
        # - do NOT store analysis/CoT
        transcript_assistant_final: list[str] = []
        transcript_assistant_commentary: list[str] = []
        transcript_python_calls: list[str] = []
        transcript_python_outputs: list[str] = []

        logprobs_buffer: list[object] = []

        attempt_seed = int(math.pow(self.cfg.seed + attempt_index, 2))

        policy = ToolRecoveryPolicy(
            abort_after_python_errors=int(getattr(self.cfg, "abort_attempt_after_python_errors", 0) or 0),
            abort_after_consecutive_python_errors=int(
                getattr(self.cfg, "abort_attempt_after_consecutive_python_errors", 0) or 0
            ),
            recycle_sandbox_after_python_errors=int(getattr(self.cfg, "recycle_sandbox_after_python_errors", 0) or 0),
        )

        had_exception = False
        had_timeout = False
        aborted_for_tool_errors = False

        tool_call_cap = tool_call_cap_for_attempt(
            attempt_tag=attempt_tag,
            recovery_micro_cap=int(getattr(self.cfg, "recovery_micro_tool_call_cap", 0) or 0),
        )

        conversation = None

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

            local_tool = AIMO3Tool(
                local_jupyter_timeout=self.cfg.jupyter_timeout,
                tool_prompt=self.cfg.tool_prompt,
                sandbox=sandbox,
                tool_timeout_cap_s=float(getattr(self.cfg, "python_tool_timeout_cap_s", 0.0) or 0.0) or None,
            )

            messages = self.template.apply_chat_template(developer_prompt, problem, local_tool.tool_config)
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
                    temperature=temperature_for_attempt(cfg=self.cfg, attempt_index=attempt_index, attempt_tag=attempt_tag),
                    logprobs=(int(self.cfg.top_logprobs) if (bool(getattr(self.cfg, "entropy_weighting_enabled", False)) and int(getattr(self.cfg, "top_logprobs", 0) or 0) > 0) else None),
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

                            # Optional: collect top-k logprobs for entropy.
                            if bool(getattr(self.cfg, "entropy_weighting_enabled", False)):
                                with contextlib.suppress(Exception):
                                    lp = chunk.choices[0].logprobs
                                    tlp = getattr(lp, "top_logprobs", None)
                                    if tlp:
                                        # vLLM returns a list[dict[token->logprob]]
                                        logprobs_buffer.extend(list(tlp))
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

                # Capture non-analysis assistant messages.
                for m in new_messages:
                    ch = getattr(m, "channel", None)
                    if ch == "analysis":
                        continue
                    # Some messages may have multiple content chunks.
                    parts: list[str] = []
                    with contextlib.suppress(Exception):
                        for c in (m.content or []):
                            t = getattr(c, "text", None)
                            if t:
                                parts.append(str(t))
                    msg_text = "".join(parts).strip()
                    if not msg_text:
                        continue
                    if ch == "final":
                        transcript_assistant_final.append(msg_text)
                    else:
                        transcript_assistant_commentary.append(msg_text)

                # IMPORTANT: the model may emit multiple python tool calls in a single completion.
                # We must execute ALL of them sequentially (in message order) and append their
                # outputs before sampling again. Otherwise later calls may assume state from
                # earlier calls and crash (NameError / missing imports).
                had_python_calls_in_batch = False
                for call in iter_tool_calls(new_messages, recipient="python"):
                    had_python_calls_in_batch = True

                    # Enforce tool-call caps for recovery variants.
                    if tool_call_cap is not None and (python_calls + 1) > int(tool_call_cap):
                        aborted_for_tool_errors = True
                        break

                    if call.text:
                        transcript_python_calls.append(str(call.text))

                    # Best-effort: count tool calls that likely invoke Lean/Lake.
                    with contextlib.suppress(Exception):
                        from .lean_toolchain import detect_lean_invocation

                        if detect_lean_invocation(call.text):
                            lean_calls += 1

                    python_calls += 1
                    tool_responses = local_tool.process_sync_plus(call.message)
                    # tool_responses is typically a list[Message]
                    with contextlib.suppress(Exception):
                        resp_text = tool_responses[0].content[0].text
                        if resp_text:
                            transcript_python_outputs.append(str(resp_text))

                        # If we timed out, optionally retry once with a longer timeout.
                        timed_out_s = parse_timeout_error(str(resp_text))
                        if (
                            timed_out_s is not None
                            and bool(getattr(self.cfg, "python_tool_timeout_retry_enabled", True))
                            and (deadline - time.time()) >= float(
                                getattr(self.cfg, "python_tool_timeout_retry_min_remaining_s", 0.0) or 0.0
                            )
                        ):
                            mult = float(getattr(self.cfg, "python_tool_timeout_retry_multiplier", 2.0) or 2.0)
                            cap_s = float(getattr(self.cfg, "python_tool_timeout_cap_s", 0.0) or 0.0) or None
                            new_timeout = float(timed_out_s) * max(1.0, mult)
                            if cap_s is not None:
                                new_timeout = min(new_timeout, float(cap_s))

                            # Only retry if it meaningfully increases the timeout.
                            if new_timeout > float(timed_out_s) + 1e-6:
                                # Ensure retry doesn't violate the tool-call cap.
                                if tool_call_cap is not None and (python_calls + 1) > int(tool_call_cap):
                                    pass
                                else:
                                    python_calls += 1
                                    tool_responses_retry = local_tool.process_sync_plus(
                                        call.message, timeout_override_s=new_timeout
                                    )
                                    with contextlib.suppress(Exception):
                                        resp_text2 = tool_responses_retry[0].content[0].text
                                        if resp_text2:
                                            transcript_python_outputs.append(str(resp_text2))
                                        resp_text = resp_text2
                                    # Replace tool_responses to reflect what we append.
                                    tool_responses = tool_responses_retry

                                    # Update timeout detection based on the retry output.
                                    timed_out_s = parse_timeout_error(str(resp_text))

                        # If we still timed out after any retry, fail fast and optionally recycle.
                        if timed_out_s is not None:
                            had_timeout = True
                            timeout_count += 1
                            if bool(getattr(self.cfg, "abort_attempt_on_python_timeout", True)):
                                aborted_for_tool_errors = True
                            if bool(getattr(self.cfg, "recycle_sandbox_on_python_timeout", True)):
                                had_exception = True

                        if str(resp_text).startswith("[ERROR]") or "Traceback" in str(resp_text) or "Error:" in str(resp_text):
                            python_errors += 1
                            consecutive_python_errors += 1
                        else:
                            consecutive_python_errors = 0

                    conversation.messages.extend(tool_responses)

                    if aborted_for_tool_errors:
                        break

                    if should_abort_attempt(
                        python_errors=python_errors,
                        consecutive_python_errors=consecutive_python_errors,
                        timeout_count=timeout_count,
                        policy=policy,
                    ):
                        aborted_for_tool_errors = True
                        break

                if aborted_for_tool_errors:
                    break

                # If we executed any tool calls, ignore any final message that might have been
                # emitted in the same batch (it was generated without tool outputs). Instead,
                # sample again with the tool outputs appended.
                if had_python_calls_in_batch:
                    continue

                if last.channel == "final":
                    answer_text = last.content[0].text
                    if answer_text:
                        text_tail = (text_tail + answer_text)
                        if cap > 0 and len(text_tail) > cap:
                            text_tail = text_tail[-cap:]
                    final_answer = self._extractor.extract_boxed_int(answer_text)
                    if final_answer is None:
                        final_answer = self._extractor.extract_int_fallback(answer_text)
                    break

                # NOTE: python tool calls are handled above by draining all calls in the batch.

            # Best-effort: attempt might have generated a plausible answer somewhere in the
            # visible text but never emitted a final channel with boxing.
            if final_answer is None:
                with contextlib.suppress(Exception):
                    ans = self._extractor.extract_boxed_int(text_tail)
                    if ans is None:
                        ans = self._extractor.extract_int_fallback(text_tail)
                    if ans is not None:
                        final_answer = ans

            # Last-chance synthesis: if we did work (possibly including tools) but never
            # produced a clean boxed integer, do one short completion asking ONLY for \boxed{N}.
            if (
                final_answer is None
                and conversation is not None
                and bool(getattr(self.cfg, "finalize_answer_enabled", True))
                and not bool(stop_event.is_set())
                and not bool(aborted_for_tool_errors)
            ):
                remaining = float(deadline - time.time())
                min_remaining = float(getattr(self.cfg, "finalize_answer_min_remaining_s", 0.0) or 0.0)
                if remaining >= min_remaining and min_remaining >= 0.0:
                    # Add an explicit user instruction to force a single-line boxed integer.
                    h = _require_harmony()
                    Message = h["Message"]
                    Role = h["Role"]
                    conversation.messages.append(
                        Message.from_role_and_content(
                            Role.USER,
                            "Finalization: output ONLY one line of the form \\\\boxed{N} where N is the final integer answer. "
                            "Do NOT call the python tool. If you cannot determine the answer, output NOBOX.",
                        )
                    )

                    prompt_ids = self.encoding.render_conversation_for_completion(conversation, self.Role.ASSISTANT)
                    max_tokens = self.cfg.context_tokens - len(prompt_ids)
                    max_tokens = min(max_tokens, int(getattr(self.cfg, "finalize_answer_max_tokens", 0) or 0))
                    # This is intentionally a *short* completion; do not apply the usual
                    # buffer-tokens gate (which is tuned for long generations).
                    if max_tokens > 0:
                        resp = self.client.completions.create(
                            model=self.cfg.served_model_name,
                            temperature=float(getattr(self.cfg, "temperature_formatting", 0.10) or 0.10),
                            max_tokens=max_tokens,
                            prompt=prompt_ids,
                            seed=attempt_seed + 11,
                            stream=False,
                            extra_body={
                                "min_p": self.cfg.min_p,
                                "stop_token_ids": self.stop_token_ids,
                                "return_token_ids": False,
                            },
                            timeout=max(0.0, deadline - time.time()),
                        )
                        fin_text = None
                        with contextlib.suppress(Exception):
                            fin_text = resp.choices[0].text

                        if fin_text:
                            text_tail = (text_tail + str(fin_text))
                            if cap > 0 and len(text_tail) > cap:
                                text_tail = text_tail[-cap:]
                            transcript_assistant_final.append(str(fin_text).strip())
                            with contextlib.suppress(Exception):
                                final_answer = self._extractor.extract_boxed_int(str(fin_text))
                                if final_answer is None:
                                    final_answer = self._extractor.extract_int_fallback(str(fin_text))

        except Exception:
            had_exception = True
            python_errors += 1
        finally:
            if local_tool is not None:
                local_tool.close()
            if sandbox is not None:
                if borrowed_from_pool:
                    # If the sandbox appears poisoned (many tool errors / exceptions), recycle it.
                    recycle = should_recycle_sandbox(
                        python_errors=python_errors,
                        had_exception=had_exception,
                        policy=policy,
                    )
                    if recycle:
                        with contextlib.suppress(Exception):
                            sandbox.close()
                        # Replace with a fresh sandbox to keep the pool healthy.
                        with contextlib.suppress(Exception):
                            self.sandbox_pool.put(AIMO3Sandbox(timeout=self.cfg.jupyter_timeout))
                    else:
                        try:
                            if bool(getattr(self.cfg, "sandbox_reset_between_attempts", True)):
                                sandbox.reset()
                            self.sandbox_pool.put(sandbox)
                        except Exception:  # noqa: BLE001
                            # If reset fails, recycle.
                            with contextlib.suppress(Exception):
                                sandbox.close()
                            with contextlib.suppress(Exception):
                                self.sandbox_pool.put(AIMO3Sandbox(timeout=self.cfg.jupyter_timeout))
                else:
                    with contextlib.suppress(Exception):
                        sandbox.close()

        result = AttemptResult(
            attempt=attempt_index + 1,
            answer=final_answer,
            stats=AttemptStats(
                token_count=total_tokens,
                python_calls=python_calls,
                python_errors=python_errors,
                lean_calls=lean_calls,
                timeout_count=timeout_count,
                verification_marker_found=(
                    any(self._has_verification_marker(out, str(getattr(self.cfg, "python_tool_verify_marker", "") or ""))
                        for out in transcript_python_outputs)
                    if bool(getattr(self.cfg, "python_tool_verify_require_marker", False))
                    and bool(str(getattr(self.cfg, "python_tool_verify_marker", "") or ""))
                    else None
                ),
                mean_entropy=(
                    self._compute_mean_entropy(logprobs_buffer)
                    if bool(getattr(self.cfg, "entropy_weighting_enabled", False))
                    else float("inf")
                ),
            ),
            output_text=text_tail,
            tag=(
                (str(attempt_tag) + "|tool_abort")
                if (attempt_tag and aborted_for_tool_errors and "tool_abort" not in str(attempt_tag))
                else attempt_tag
            ),
        )

        # Optional: record a per-attempt transcript for post-run inspection.
        if bool(getattr(self.cfg, "trace_attempts_enabled", False)) and bool(getattr(self.cfg, "trace_enabled", False)):
            max_chars = int(getattr(self.cfg, "trace_attempts_max_chars", 0) or 0)

            def _cap_list(items: list[str], per_item_chars: int = 4000, max_items: int = 20) -> list[str]:
                # Keep attempt payload bounded.
                per_item_chars = max(0, int(per_item_chars))
                max_items = max(0, int(max_items))
                if max_items and len(items) > max_items:
                    items = items[-max_items:]
                if per_item_chars > 0:
                    return [self._truncate(s, per_item_chars) for s in items]
                return items

            payload = {
                "event": "attempt_end",
                "problem_id": problem_id,
                "attempt": int(result.attempt),
                "tag": result.tag,
                "answer": (int(result.answer) if isinstance(result.answer, int) else None),
                "token_count": int(result.stats.token_count),
                "python_calls": int(result.stats.python_calls),
                "lean_calls": int(getattr(result.stats, "lean_calls", 0) or 0),
                "python_errors": int(result.stats.python_errors),
                "timeout_count": int(getattr(result.stats, "timeout_count", 0) or 0),
                "aborted_for_tool_errors": bool(aborted_for_tool_errors),
                "had_exception": bool(had_exception),
                "assistant_final": self._truncate("\n\n".join(transcript_assistant_final).strip(), max_chars) if max_chars > 0 else "\n\n".join(transcript_assistant_final).strip(),
                "assistant_commentary": self._truncate("\n\n".join(transcript_assistant_commentary).strip(), max_chars) if max_chars > 0 else "\n\n".join(transcript_assistant_commentary).strip(),
                "python_calls_text": _cap_list(transcript_python_calls),
                "python_outputs_text": _cap_list(transcript_python_outputs),
            }
            # Avoid dumping empty huge fields.
            if not payload.get("assistant_final"):
                payload.pop("assistant_final", None)
            if not payload.get("assistant_commentary"):
                payload.pop("assistant_commentary", None)
            if not payload.get("python_calls_text"):
                payload.pop("python_calls_text", None)
            if not payload.get("python_outputs_text"):
                payload.pop("python_outputs_text", None)

            with contextlib.suppress(Exception):
                self._trace.record(payload)

        return result

    @staticmethod
    def _rank_answers(detailed_results: list[dict]) -> list[tuple[int, dict]]:
        # Compatibility wrapper around the dedicated module.
        return rank_candidates(detailed_results, filter_to_verified_if_any=True)

    def _second_stage_verify(
        self, user_input: str, candidates: list[int], verify_deadline: float, problem_id: str | None = None
    ) -> int | None:
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
                ex.submit(
                    self._process_attempt,
                    vp,
                    TIR_PROMPT_VERIFICATION,
                    attempt_idx,
                    f"second_stage_verify:cand={_cand}",
                    stop_event,
                    verify_deadline,
                    problem_id,
                )
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

    @staticmethod
    def _enabled_prompt_specs(cfg: AIMO3Config) -> list[tuple[str, str]]:
        """Return enabled (name, prompt) pairs for first-stage rotation.

        Controlled by cfg.disabled_prompts (comma-separated list).
        Always returns at least one spec (falls back to standard).
        """

        disabled_raw = str(getattr(cfg, "disabled_prompts", "") or "")
        disabled = {t.strip().lower() for t in disabled_raw.split(",") if t.strip()}

        specs: list[tuple[str, str]] = [
            ("standard", TIR_PROMPT_STANDARD),
            ("code_first", TIR_PROMPT_CODE_FIRST),
            ("verification", TIR_PROMPT_VERIFICATION),
            ("small_cases", TIR_PROMPT_SMALL_CASES),  # Novel: solve small n first
            ("sanity", TIR_PROMPT_SANITY),  # Novel: check bounds/properties
            ("analytic", TIR_PROMPT_ANALYTIC),  # Last (slowest)
        ]
        enabled = [(name, prompt) for (name, prompt) in specs if name not in disabled]
        if not enabled:
            enabled = [("standard", TIR_PROMPT_STANDARD)]
        return enabled

    def solve_problem(self, problem: str) -> int:
        problem_start_time = time.time()
        user_input = f"{problem} {self.cfg.preference_prompt}"

        pid = stable_problem_id(problem)

        # Dynamic budget: use tracker if available, fallback to legacy calculation
        if hasattr(self, '_budget_tracker'):
            # Sync tracker's view of elapsed time
            self._budget_tracker.total_time_used_s = time.time() - self.notebook_start_time
            budget = self._budget_tracker.compute_budget()
        else:
            # Legacy static calculation
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
        reserve_fraction_eff = reserve_fraction_for_budget(
            budget_s=budget,
            base_fraction=float(self.cfg.verification_reserve_fraction),
        )
        attempt_deadline, deadline = compute_attempt_and_verify_deadlines(
            now=now,
            overall_deadline=overall_deadline,
            reserve_fraction=float(reserve_fraction_eff),
            reserve_cap_s=float(self.cfg.verification_reserve_cap),
            reserve_min_s=float(self.cfg.verification_reserve_min),
        )

        prompt_specs = self._enabled_prompt_specs(self.cfg)
        enabled_names = {name for (name, _p) in prompt_specs}
        tasks: list[tuple[str, int, str]] = []
        for attempt_index in range(int(self.cfg.attempts)):
            base_name, base = prompt_specs[attempt_index % len(prompt_specs)]
            meta_pack = "none"
            meta_card = "none"
            if bool(self.cfg.wickelgren_strategies_enabled):
                sys_prompt, meta = augment_system_prompt_with_meta(
                    base,
                    attempt_index=attempt_index,
                    problem_text=problem,
                    mode=str(getattr(self.cfg, "strategy_pack_mode", "round_robin")),
                    enabled_packs=str(getattr(self.cfg, "strategy_packs", "generic")),
                    shuffle_cards=bool(getattr(self.cfg, "shuffle_cards", True)),
                )
                meta_pack = str(meta.get("pack", "none"))
                meta_card = str(meta.get("card", "none"))
            else:
                sys_prompt = base
            if bool(self.cfg.protocol_enabled):
                sys_prompt = with_protocol(sys_prompt)
            tasks.append((sys_prompt, attempt_index, f"{base_name}|pack={meta_pack}|card={meta_card}"))

        detailed: list[AttemptResult] = []
        valid: list[int] = []
        stop_event = threading.Event()

        solve_start_payload = {
            "event": "solve_start",
            "problem_id": pid,
            "problem_len": len(problem or ""),
            "problem": (problem if self._trace.include_problem_text else None),
            "budget_s": float(budget),
            "attempt_deadline_in_s": float(max(0.0, attempt_deadline - time.time())),
            "overall_deadline_in_s": float(max(0.0, deadline - time.time())),
            "attempts": int(self.cfg.attempts),
            "workers": int(self.cfg.workers),
            "sandbox_pool_size": int(getattr(self.cfg, "sandbox_pool_size", 0) or 0),
        }
        # Add dynamic budget tracker info if available
        if hasattr(self, '_budget_tracker'):
            bt = self._budget_tracker
            solve_start_payload["budget_tracker"] = {
                "problems_solved": bt.problems_solved,
                "problems_remaining": bt.problems_remaining,
                "time_banked_s": round(bt.time_banked_s, 1),
                "avg_solve_time_s": round(bt.avg_solve_time_s, 1),
            }
        if bool(getattr(self.cfg, "trace_env_enabled", False)) and bool(getattr(self.cfg, "trace_enabled", False)):
            with contextlib.suppress(Exception):
                env = self._sandbox_env_snapshot()
                if env is not None:
                    solve_start_payload["sandbox_env"] = env
        self._trace.record(solve_start_payload)

        with ThreadPoolExecutor(max_workers=int(self.cfg.workers)) as ex:
            # Dynamic scheduling allows recovery attempts to reuse freed worker capacity.
            base_tasks = list(tasks)

            # Optional phase-1 policy: for an initial window (or until we have at least one
            # extracted integer), only run code-first / verification prompts.
            phase_s = float(getattr(self.cfg, "code_first_phase_s", 0.0) or 0.0)
            phase_end = problem_start_time + max(0.0, phase_s)

            phase1_names = {"code_first", "verification"} & enabled_names
            phase1_tasks: deque[tuple[str, int, str]] = deque()
            phase2_tasks: deque[tuple[str, int, str]] = deque()
            for _p, _idx, _tag in base_tasks:
                base_name = str(_tag).split("|", 1)[0]
                if base_name in phase1_names:
                    phase1_tasks.append((_p, _idx, _tag))
                else:
                    phase2_tasks.append((_p, _idx, _tag))

            futures = set()

            def _submit_one(sys_prompt: str, attempt_idx: int, attempt_tag: str) -> None:
                futures.add(
                    ex.submit(
                        self._process_attempt,
                        user_input,
                        sys_prompt,
                        attempt_idx,
                        attempt_tag,
                        stop_event,
                        attempt_deadline,
                        pid,
                    )
                )

            def _phase1_active() -> bool:
                if phase_s <= 0.0:
                    return False
                if not phase1_names:
                    return False
                if valid:
                    return False
                # Don't keep phase-1 alive past the attempt-generation deadline.
                return time.time() < min(phase_end, attempt_deadline)

            def _next_base_task() -> tuple[str, int, str] | None:
                if _phase1_active():
                    # During phase-1: only schedule tool-heavy prompts.
                    if phase1_tasks:
                        return phase1_tasks.popleft()
                    return None

                # Phase-2: schedule remaining proof prompts first.
                if phase2_tasks:
                    return phase2_tasks.popleft()
                if phase1_tasks:
                    return phase1_tasks.popleft()
                return None

            def _fill_executor() -> None:
                # Try to keep workers busy, subject to phase gating.
                while (not stop_event.is_set()) and time.time() <= attempt_deadline and len(futures) < int(self.cfg.workers):
                    nxt = _next_base_task()
                    if nxt is None:
                        break
                    sys_prompt, attempt_idx, attempt_tag = nxt
                    _submit_one(sys_prompt, attempt_idx, attempt_tag)

            # Seed the executor.
            _fill_executor()

            recovery_left = int(getattr(self.cfg, "recovery_attempts_cap", 0) or 0)
            format_recovery_left = int(getattr(self.cfg, "format_recovery_cap", 0) or 0)

            while futures or phase1_tasks or phase2_tasks:
                # Keep the pool topped up when possible.
                _fill_executor()

                if not futures:
                    # Nothing running right now (e.g., phase-1 gating exhausted its queue).
                    if stop_event.is_set() or time.time() > attempt_deadline:
                        break
                    # Wait briefly for phase-1 to expire (or for external stop), then retry.
                    time.sleep(0.05)
                    continue

                try:
                    # Wait for one future to complete, but don't block forever so we can
                    # react to phase transitions and deadlines.
                    done = next(as_completed(futures, timeout=0.25))
                except FuturesTimeoutError:
                    continue
                except Exception:  # noqa: BLE001
                    continue

                with contextlib.suppress(Exception):
                    futures.remove(done)
                    r: AttemptResult = done.result()
                    detailed.append(r)
                    if r.answer is not None and isinstance(r.answer, int):
                        valid.append(r.answer)

                    if self._should_early_stop(detailed):
                        stop_event.set()
                        # Best-effort cancellation of remaining work.
                        for f in list(futures):
                            with contextlib.suppress(Exception):
                                f.cancel()
                        break

                    # If we saw tool instability, spend a little extra budget on a recovery attempt.
                    if (
                        bool(getattr(self.cfg, "recovery_attempts_enabled", True))
                        and recovery_left > 0
                        and (not stop_event.is_set())
                    ):
                        remaining_s = float(attempt_deadline - time.time())
                        if should_schedule_recovery_attempt(
                            result=r,
                            remaining_s=remaining_s,
                            recovery_trigger_python_errors=int(getattr(self.cfg, "recovery_trigger_python_errors", 0) or 0),
                            recovery_min_remaining_s=float(getattr(self.cfg, "recovery_min_remaining_s", 0.0) or 0.0),
                        ):
                            recovery_left -= 1
                            mode = str(getattr(self.cfg, "recovery_mode", "auto") or "auto").strip().lower()
                            # Auto: if we had to abort for tool issues, go no-tool; else allow micro-tool.
                            if mode == "auto":
                                mode = "no_tool" if (r.tag and "tool_abort" in str(r.tag)) else "micro_tool"

                            if mode == "no_tool":
                                recovery_prompt = (
                                    TIR_PROMPT_ANALYTIC
                                    + "\n\nRecovery mode: DO NOT use the python tool. Solve by reasoning only."
                                )
                                variant = "no_tool"
                            else:
                                recovery_prompt = (
                                    TIR_PROMPT_ANALYTIC
                                    + "\n\nRecovery mode: The python tool may be unstable. Prefer reasoning-first. "
                                    "If you use python, use at most a couple very small snippets."
                                )
                                variant = "micro_tool"
                            if bool(self.cfg.protocol_enabled):
                                recovery_prompt = with_protocol(recovery_prompt)
                            attempt_idx = int(self.cfg.attempts) + (int(self.cfg.recovery_attempts_cap) - recovery_left)
                            attempt_tag = f"recovery|variant={variant}|pack=recovery|card=tool_instability"
                            _submit_one(recovery_prompt, attempt_idx, attempt_tag)

                    # If extraction failed (many tokens but no extracted answer), schedule a tiny
                    # formatting-focused attempt that outputs ONLY the boxed integer.
                    if (
                        bool(getattr(self.cfg, "format_recovery_enabled", True))
                        and format_recovery_left > 0
                        and (not stop_event.is_set())
                    ):
                        remaining_s = float(attempt_deadline - time.time())
                        if should_schedule_format_recovery_attempt(
                            result=r,
                            remaining_s=remaining_s,
                            trigger_tokens=int(getattr(self.cfg, "format_recovery_trigger_tokens", 0) or 0),
                            min_remaining_s=float(getattr(self.cfg, "format_recovery_min_remaining_s", 0.0) or 0.0),
                        ):
                            format_recovery_left -= 1
                            fmt_prompt = (
                                TIR_PROMPT_STANDARD
                                + "\n\nFormatting recovery: Your job is to output ONLY the final integer answer as a single line \\boxed{N}. "
                                "No extra text. If unsure, output NOBOX."
                            )
                            if bool(self.cfg.protocol_enabled):
                                fmt_prompt = with_protocol(fmt_prompt)
                            attempt_idx = int(self.cfg.attempts) + 50_000 + (int(self.cfg.format_recovery_cap) - format_recovery_left)
                            attempt_tag = "recovery|variant=no_tool|pack=recovery|card=format_only"
                            _submit_one(fmt_prompt, attempt_idx, attempt_tag)

        self.problems_remaining = max(0, int(self.problems_remaining) - 1)
        
        # Record actual solve time for dynamic budgeting
        problem_elapsed = time.time() - problem_start_time
        if hasattr(self, '_budget_tracker'):
            self._budget_tracker.record_solve(problem_elapsed)

        # Retry if no valid answers.
        if not valid:
            remaining_overall = max(0.0, deadline - time.time())
            # Use a meaningful slice of remaining time, but cap it to avoid monopolizing the notebook.
            retry_budget = min(90.0, max(10.0, remaining_overall * 0.50))
            retry_deadline = min(deadline, time.time() + retry_budget)
            retry_order: list[tuple[str, str]] = [
                ("verification", TIR_PROMPT_VERIFICATION),
                ("analytic", TIR_PROMPT_ANALYTIC),
                ("code_first", TIR_PROMPT_CODE_FIRST),
                ("standard", TIR_PROMPT_STANDARD),
            ]
            retry_tasks: list[tuple[str, int, str]] = []
            for name, base_prompt in retry_order:
                if name not in enabled_names:
                    continue
                attempt_idx = int(self.cfg.attempts) + len(retry_tasks)
                sys_prompt = (
                    augment_system_prompt_with_meta(
                        base_prompt,
                        attempt_index=attempt_idx,
                        problem_text=problem,
                        mode=str(getattr(self.cfg, "strategy_pack_mode", "round_robin")),
                        enabled_packs=str(getattr(self.cfg, "strategy_packs", "generic")),
                    )[0]
                    if bool(self.cfg.wickelgren_strategies_enabled)
                    else base_prompt
                )
                retry_tasks.append((sys_prompt, attempt_idx, f"{name}|pack=retry|card=retry"))

            # Apply protocol to retry prompts too.
            if bool(self.cfg.protocol_enabled):
                retry_tasks = [(with_protocol(p), idx, tag) for (p, idx, tag) in retry_tasks]

            with ThreadPoolExecutor(max_workers=min(4, int(self.cfg.workers))) as ex:
                futures = [
                    ex.submit(
                        self._process_attempt,
                        user_input,
                        sys_prompt,
                        attempt_idx,
                        attempt_tag,
                        stop_event,
                        retry_deadline,
                        pid,
                    )
                    for (sys_prompt, attempt_idx, attempt_tag) in retry_tasks
                ]
                for fut in as_completed(futures):
                    with contextlib.suppress(Exception):
                        r: AttemptResult = fut.result()
                        detailed.append(r)
                        if r.answer is not None and isinstance(r.answer, int):
                            valid.append(r.answer)

        if not valid:
            self._trace.record(
                {
                    "event": "solve_end",
                    "problem_id": pid,
                    "status": "no_valid",
                    "elapsed_s": float(time.time() - problem_start_time),
                    "chosen": 0,
                    "n_attempts": len(detailed),
                }
            )
            return 0

        # Show candidate attempts (best-effort, notebook-friendly).
        self._display_candidates(detailed)

        ranked = self._rank_answers(detailed)
        if not ranked:
            self._trace.record(
                {
                    "event": "solve_end",
                    "problem_id": pid,
                    "status": "no_ranked",
                    "elapsed_s": float(time.time() - problem_start_time),
                    "chosen": 0,
                    "n_attempts": len(detailed),
                }
            )
            return 0

        top_ans, top_d = ranked[0]
        chosen = top_ans

        decision: dict[str, object] = {
            "ranked": [{"answer": int(a), **d} for (a, d) in ranked[:10]],
            "second_stage": None,
            "tiebreak": None,
            "contradiction_retry": None,
        }
        tiebreak_used = False
        runner_ans = None
        runner_d = None
        votes_gap = None
        if len(ranked) >= 2:
            runner_ans, runner_d = ranked[1]
            votes_gap = int(top_d["votes"]) - int(runner_d["votes"])

        # Contradiction-driven retry: when answers are wildly different, re-read the problem
        n_distinct = len(ranked)
        top_votes = int(top_d.get("votes", 1))
        remaining_for_contradiction = deadline - time.time()
        
        if (
            bool(getattr(self.cfg, "contradiction_retry_enabled", True))
            and n_distinct >= int(getattr(self.cfg, "contradiction_retry_min_distinct_answers", 3))
            and top_votes <= int(getattr(self.cfg, "contradiction_retry_max_top_votes", 2))
            and remaining_for_contradiction >= float(getattr(self.cfg, "contradiction_retry_min_remaining_s", 45.0))
        ):
            # Build the contradiction prompt
            distinct_answers = [int(a) for (a, _) in ranked[:5]]
            contradiction_prompt = (
                TIR_PROMPT_VERIFICATION
                + f"\n\n**CRITICAL**: Previous attempts produced CONFLICTING answers: {distinct_answers}\n"
                + "This suggests a FUNDAMENTAL MISUNDERSTANDING of the problem.\n\n"
                + "STOP and carefully:\n"
                + "1. Re-read EVERY word of the problem statement\n"
                + "2. List ALL constraints explicitly (you likely missed one)\n"
                + "3. Define ALL terms precisely as stated (don't paraphrase)\n"
                + "4. Solve from scratch with extreme care\n\n"
                + "The disagreement means something was misinterpreted. Find it."
            )
            if bool(self.cfg.protocol_enabled):
                contradiction_prompt = with_protocol(contradiction_prompt)
            
            cr_budget = min(
                float(getattr(self.cfg, "contradiction_retry_budget_cap_s", 90.0)),
                remaining_for_contradiction * 0.5
            )
            cr_deadline = time.time() + cr_budget
            
            cr_result = self._process_attempt(
                user_input,
                contradiction_prompt,
                attempt_index=88_888,
                attempt_tag="contradiction_retry|variant=reread|pack=recovery|card=contradiction",
                stop_event=threading.Event(),
                deadline=min(cr_deadline, deadline),
                problem_id=pid,
            )
            detailed.append(cr_result)
            
            decision["contradiction_retry"] = {
                "triggered": True,
                "distinct_answers": distinct_answers,
                "budget_s": float(cr_budget),
                "result_answer": (int(cr_result.answer) if isinstance(cr_result.answer, int) else None),
            }
            
            # If contradiction retry found an answer, re-rank
            if isinstance(cr_result.answer, int):
                valid.append(cr_result.answer)
                ranked = self._rank_answers(detailed)
                if ranked:
                    top_ans, top_d = ranked[0]
                    chosen = top_ans
                    decision["ranked"] = [{"answer": int(a), **d} for (a, d) in ranked[:10]]
                    if len(ranked) >= 2:
                        runner_ans, runner_d = ranked[1]
                        votes_gap = int(top_d["votes"]) - int(runner_d["votes"])

        # Second-stage verification can be valuable even when there is only one unique candidate,
        # especially if it lacks any clean tool support.
        need_verify = False
        if bool(self.cfg.second_stage_verify_enabled):
            if bool(self.cfg.second_stage_verify_trigger_if_no_verified) and int(top_d.get("verified", 0)) == 0:
                need_verify = True
            if votes_gap is not None and votes_gap <= int(self.cfg.second_stage_verify_trigger_votes_gap):
                need_verify = True

        verified_choice = None
        remaining = deadline - time.time()
        if need_verify and remaining >= float(self.cfg.second_stage_verify_min_remaining):
            mult = 1.0
            if int(top_d.get("verified", 0)) <= 0:
                mult *= 1.50
            if votes_gap is not None and votes_gap <= int(self.cfg.second_stage_verify_trigger_votes_gap):
                mult *= 1.25
            if votes_gap is not None and int(top_d.get("verified", 0)) > 0 and votes_gap >= 3:
                mult *= 0.80

            verify_budget = adaptive_verify_budget(
                remaining_s=remaining,
                base_fraction=float(self.cfg.second_stage_verify_budget_fraction),
                cap_s=float(self.cfg.second_stage_verify_budget_cap),
                multiplier=mult,
                min_s=float(self.cfg.second_stage_verify_min_effective_time),
            )
            verify_deadline = time.time() + verify_budget
            desired_top_k = max(1, int(self.cfg.second_stage_verify_top_k))
            top_k = max(1, min(desired_top_k, len(ranked)))
            candidates = [ans for (ans, _d) in ranked[:top_k]]
            verified_choice = self._second_stage_verify(user_input, candidates, verify_deadline, problem_id=pid)
            if verified_choice is not None:
                chosen = verified_choice

            decision["second_stage"] = {
                "need_verify": bool(need_verify),
                "votes_gap": (int(votes_gap) if votes_gap is not None else None),
                "budget_s": float(verify_budget),
                "candidates": [int(c) for c in candidates],
                "choice": (int(verified_choice) if verified_choice is not None else None),
            }

        # If second-stage verification can't decide, run one short tie-break attempt when:
        # - we actually have two candidates, and
        # - either the leader has no verified support OR it's a vote tie.
        remaining2 = deadline - time.time()
        if (
            bool(getattr(self.cfg, "tiebreak_enabled", True))
            and verified_choice is None
            and runner_ans is not None
            and runner_d is not None
            and remaining2 >= float(getattr(self.cfg, "tiebreak_min_remaining_s", 0.0) or 0.0)
            and (int(top_d.get("verified", 0)) <= 0 or (votes_gap is not None and votes_gap <= 0))
        ):
            tb_budget = min(float(getattr(self.cfg, "tiebreak_budget_cap_s", 35.0) or 35.0), remaining2 * 0.60)
            tb_deadline = time.time() + max(3.0, tb_budget)

            mode = str(getattr(self.cfg, "recovery_mode", "auto") or "auto").strip().lower()
            # Auto: avoid tool if we already saw many errors overall.
            total_errs = sum(int(r.stats.python_errors) for r in detailed)
            if mode == "auto":
                mode = "no_tool" if total_errs >= 8 else "micro_tool"

            variant = "no_tool" if mode == "no_tool" else "micro_tool"
            tb_prompt = (
                TIR_PROMPT_VERIFICATION
                + "\n\nTie-break task: Only choose between the candidate answers below. "
                "Do quick consistency checks. Output EXACTLY one line: \\boxed{CAND}. "
                "If you cannot decide, output NOBOX (do not output any boxed answer).\n"
                f"Candidates: {int(top_ans)}, {int(runner_ans)}."
            )
            if variant == "no_tool":
                tb_prompt += "\nDo NOT use the python tool."
            else:
                tb_prompt += "\nIf you use python, use at most a couple short snippets."
            if bool(self.cfg.protocol_enabled):
                tb_prompt = with_protocol(tb_prompt)

            tb_res = self._process_attempt(
                user_input,
                tb_prompt,
                attempt_index=99_999,
                attempt_tag=f"tiebreak|variant={variant}|pack=recovery|card=top2",
                stop_event=threading.Event(),
                deadline=min(tb_deadline, deadline),
                problem_id=pid,
            )
            detailed.append(tb_res)
            if isinstance(tb_res.answer, int) and tb_res.answer in {int(top_ans), int(runner_ans)}:
                chosen = int(tb_res.answer)
                tiebreak_used = True

            decision["tiebreak"] = {
                "enabled": True,
                "variant": variant,
                "budget_s": float(tb_budget),
                "candidates": [int(top_ans), int(runner_ans)],
                "choice": (int(tb_res.answer) if isinstance(tb_res.answer, int) else None),
                "used": bool(tiebreak_used),
            }

        self._trace.record(
            {
                "event": "solve_end",
                "problem_id": pid,
                "status": "ok",
                "elapsed_s": float(time.time() - problem_start_time),
                "chosen": int(chosen),
                "n_attempts": len(detailed),
                "attempts": [
                    {
                        "attempt": int(r.attempt),
                        "answer": (int(r.answer) if isinstance(r.answer, int) else None),
                        "tag": r.tag,
                        "temperature": float(
                            temperature_for_attempt(cfg=self.cfg, attempt_index=int(r.attempt) - 1, attempt_tag=r.tag)
                        ),
                        "token_count": int(r.stats.token_count),
                        "python_calls": int(r.stats.python_calls),
                        "python_errors": int(r.stats.python_errors),
                    }
                    for r in detailed
                ],
                "decision": decision,
            }
        )

        _ = time.time() - problem_start_time  # for optional logging
        return int(chosen)
