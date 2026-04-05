# OlympiadLevelMaths4LLM

Framework scaffold to help an LLM solve (and *verify*) hard olympiad-style math problems.

## Goals

- Turn a raw problem statement into a structured representation.
- Run an iterative **plan → solve → verify → revise** loop.
- Add lightweight, automatable checks (SymPy + numeric spot checks) before accepting a final answer.
- Keep the LLM-provider layer swappable (mock / OpenAI / Anthropic / local).
