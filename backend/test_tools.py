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


@tools.tool(Risk.EXPLICIT, "test", unlock=("show me the magic",))
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


def test_only_my_words_unlock_an_explicit_tool():
    assert tools.unlocked_by("show me the magic please") == {"_explicit"}
    assert tools.unlocked_by("what's the weather") == frozenset()
    # Folded both ways: a transcript won't match my capitalisation, and a phrase
    # declared with capitals still has to match a transcript.
    assert "_explicit" in tools.unlocked_by("SHOW ME THE MAGIC")

    @tools.tool(Risk.EXPLICIT, "test", unlock=("Loud Noises",))
    def _shouty() -> str:
        return "ok"

    assert "_shouty" in tools.unlocked_by("make loud noises")


def test_an_explicit_tool_with_no_unlock_words_is_rejected_at_registration():
    """The hole this closes: such a tool sits in the registry looking available
    and refuses every call, because nothing can ever authorise it."""
    try:

        @tools.tool(Risk.EXPLICIT, "unreachable")
        def _no_words() -> str:
            return "never"

    except ValueError as e:
        assert "unlock words" in str(e)
    else:
        raise AssertionError("an unreachable EXPLICIT tool was accepted")


def test_unlocking_one_tool_does_not_unlock_another():
    """The grant is per tool, not a blanket "he spoke" flag -- asking for a
    screenshot must not also switch the microphone on."""

    @tools.tool(Risk.EXPLICIT, "test", unlock=("record audio",))
    def _mic() -> str:
        return "recorded"

    assert tools.unlocked_by("show me the magic") == {"_explicit"}
    assert tools.unlocked_by("record audio now") == {"_mic"}


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


def test_a_dialog_that_cannot_open_is_a_no():
    """Fail-closed, which is the whole reason the dialog exists.

    `ask` blocks on stdin, so at login with no terminal a CONFIRM tool hung the
    voice loop forever. A dialog that can't open must refuse -- falling back to
    input() would reintroduce exactly that hang. Stubbed rather than real: a
    self-check must not pop a window and wait for a human.
    """
    import sys
    import types

    def boom():
        raise RuntimeError("no display")

    stub = types.ModuleType("tkinter")
    stub.Tk = boom
    real = sys.modules.get("tkinter")
    sys.modules["tkinter"] = stub
    try:
        assert tools.ask_dialog(tools.REGISTRY["_confirm"], {}) is False
    finally:
        if real is None:
            del sys.modules["tkinter"]
        else:
            sys.modules["tkinter"] = real


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
