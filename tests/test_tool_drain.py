from dataclasses import dataclass

from olympiad_llm.aimo3.tool_drain import drain_tool_calls


@dataclass
class FakeContent:
    text: str


@dataclass
class FakeMsg:
    recipient: str | None
    content: list[FakeContent]


def test_drain_tool_calls_executes_in_order():
    msgs = [
        FakeMsg(recipient=None, content=[FakeContent("hello")]),
        FakeMsg(recipient="python", content=[FakeContent("x = 1")]),
        FakeMsg(recipient="python", content=[FakeContent("y = x + 1")]),
        FakeMsg(recipient=None, content=[FakeContent("done")]),
    ]

    executed: list[str] = []

    def _exec(m: FakeMsg):
        executed.append(m.content[0].text)
        return f"out:{m.content[0].text}"

    outs = drain_tool_calls(msgs, recipient="python", execute=_exec)

    assert executed == ["x = 1", "y = x + 1"]
    assert outs == ["out:x = 1", "out:y = x + 1"]


def test_drain_tool_calls_respects_cap():
    msgs = [
        FakeMsg(recipient="python", content=[FakeContent("a")]),
        FakeMsg(recipient="python", content=[FakeContent("b")]),
    ]

    def _exec(_m: FakeMsg):
        return "ok"

    try:
        drain_tool_calls(msgs, recipient="python", execute=_exec, call_cap=1)
        assert False, "expected cap exception"
    except RuntimeError as e:
        assert "cap" in str(e) or "tool_call_cap_exceeded" in str(e)
