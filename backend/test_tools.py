"""Self-check for the execution gate. No network, no keys, no real writes
outside a temp dir.

Run: python test_tools.py
"""

import json
import tempfile
from pathlib import Path

import tools
from tools import Risk, execute, in_scope

ran: list[str] = []


@tools.tool(Risk.SAFE, "test")
def _safe() -> str:
    ran.append("safe")
    return "did safe"


@tools.tool(Risk.EXPLICIT, "test")
def _explicit() -> str:
    ran.append("explicit")
    return "did explicit"


@tools.tool(Risk.CONFIRM, "test")
def _confirm() -> str:
    ran.append("confirm")
    return "did confirm"


yes = lambda t, a: True
no = lambda t, a: False


def test_safe_runs_without_asking():
    ran.clear()
    assert execute("_safe") == "did safe"
    assert ran == ["safe"]


def test_explicit_needs_a_direct_command():
    ran.clear()
    out = execute("_explicit")  # model-initiated: default is user_initiated=False
    assert ran == [], "an EXPLICIT tool ran without a direct command"
    assert "direct spoken command" in out

    assert execute("_explicit", user_initiated=True) == "did explicit"
    assert ran == ["explicit"]


def test_confirm_asks_even_when_i_asked_for_it():
    """The point of CONFIRM: user_initiated does not substitute for consent."""
    ran.clear()
    out = execute("_confirm", user_initiated=True, confirm=no)
    assert ran == [], "a CONFIRM tool ran after being declined"
    assert "declined" in out

    assert execute("_confirm", confirm=yes) == "did confirm"
    assert ran == ["confirm"]


def test_unknown_tool_does_not_raise():
    assert "No such tool" in execute("_nope")


def test_errors_are_returned_not_raised():
    @tools.tool(Risk.SAFE, "test")
    def _boom() -> str:
        raise ValueError("kaboom")

    assert "ValueError: kaboom" in execute("_boom")


def test_scope_survives_traversal():
    inside = tools.SCOPE[0] / "MyndOS" / "backend" / "tools.py"
    assert in_scope(inside)
    assert not in_scope("C:/Windows/System32/drivers/etc/hosts")
    # ../ must be resolved before the check, not after.
    assert not in_scope(tools.SCOPE[0] / ".." / ".." / "secrets.txt")


def test_every_call_is_logged_including_refusals():
    with tempfile.TemporaryDirectory() as d:
        tools.LOG_PATH = Path(d) / "actions.jsonl"
        execute("_safe")
        execute("_explicit")  # refused
        execute("_confirm", confirm=no)  # denied
        execute("_nope")  # error
        rows = [json.loads(l) for l in tools.LOG_PATH.read_text().splitlines()]

    assert [r["outcome"] for r in rows] == ["ok", "refused", "denied", "error"]
    assert all("ts" in r and "args" in r for r in rows)


def test_write_file_is_confirm_tier():
    assert tools.REGISTRY["write_file"].risk is Risk.CONFIRM
    assert tools.REGISTRY["read_file"].risk is Risk.SAFE

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "sub" / "note.txt"
        assert "declined" in execute("write_file", {"path": str(p)}, confirm=no)
        assert not p.exists()
        execute("write_file", {"path": str(p), "content": "hi"}, confirm=yes)
        assert p.read_text() == "hi"


def test_schemas_cover_every_tool():
    names = {s["function"]["name"] for s in tools.schemas()}
    assert names == set(tools.REGISTRY)
    write = next(s for s in tools.schemas() if s["function"]["name"] == "write_file")
    assert set(write["function"]["parameters"]["required"]) == {"path", "content"}


if __name__ == "__main__":
    real_log = tools.LOG_PATH
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            tools.LOG_PATH = Path(tempfile.gettempdir()) / "myndos_test_actions.jsonl"
            fn()
    tools.LOG_PATH = real_log
    print("ok")
