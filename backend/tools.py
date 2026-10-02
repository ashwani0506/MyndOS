"""Trusted execution layer: the boundary between what the model proposes and
what actually runs.

Three risk tiers:

  SAFE      Execute. Reading the clipboard, reading a file inside a scoped
            project folder.
  EXPLICIT  Only when my own words asked for it. Each tool declares the words
            that unlock it, and the grant is computed from my utterance alone --
            so screen and audio capture cannot be switched on by the model, or
            by anything the model read.
  CONFIRM   Stop and ask, every single time. Sending anything, deleting or
            overwriting, spending money, installing, writing outside scope.
            There is deliberately no "remember this choice".

Two invariants this file exists to hold:

  1. Nothing the agent *read* can trigger a consequential call. An EXPLICIT
     tool is authorised only by words in my own utterance (see unlocked_by),
     which is not a channel the model or a poisoned file can write to; CONFIRM
     tools stop for a human regardless of who asked.
  2. Every call is logged -- name, arguments, outcome -- including refusals and
     denials, which are the interesting ones.

Registering a tool is a decorator. Adding one should be boring.
"""

import json
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

HERE = Path(__file__).parent
LOG_PATH = HERE / "logs" / "actions.jsonl"

# Folders the agent may read from without asking. Anything outside is refused
# for reads and needs confirmation for writes.
SCOPE = [Path.home() / "OneDrive" / "Desktop" / "Projects"]


class Risk(Enum):
    SAFE = "safe"
    EXPLICIT = "explicit"
    CONFIRM = "confirm"


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    risk: Risk
    params: dict
    fn: Callable
    unlock: tuple[str, ...] = ()  # EXPLICIT only: phrases in my words that authorise it


REGISTRY: dict[str, Tool] = {}


def tool(
    risk: Risk,
    description: str,
    params: dict | None = None,
    unlock: tuple[str, ...] = (),
):
    def register(fn):
        # An EXPLICIT tool with no unlock words can never be authorised by
        # anything, so it would sit in the registry looking available and refuse
        # every call. Fail at import rather than at 3am.
        if risk is Risk.EXPLICIT and not unlock:
            raise ValueError(
                f"{fn.__name__} is EXPLICIT but declares no unlock words, "
                f"so nothing could ever authorise it"
            )
        REGISTRY[fn.__name__] = Tool(
            # Folded here, once, so a phrase declared with capitals still
            # matches a lower-cased transcript.
            fn.__name__, description, risk, params or {}, fn,
            tuple(w.lower() for w in unlock),
        )
        return fn

    return register


def unlocked_by(utterance: str) -> frozenset[str]:
    """Names of the EXPLICIT tools this utterance authorises, for this turn only.

    The grant is derived from what the human said and nothing else -- not the
    model's reasoning, not a file it read, not a web page. That asymmetry is the
    whole tier: injected text can ask for a screenshot all it likes, but it
    cannot put the word in my mouth.

    ponytail: crude substring match. Swap in the intent classifier when the
    command router lands -- the call site doesn't change.
    """
    low = utterance.lower()
    return frozenset(
        t.name
        for t in REGISTRY.values()
        if t.risk is Risk.EXPLICIT and any(w in low for w in t.unlock)
    )


def in_scope(path: str | Path) -> bool:
    """True if `path` sits inside a scoped folder. Resolves first, so ../ can't
    walk out of scope."""
    p = Path(path).expanduser().resolve()
    return any(p.is_relative_to(d.expanduser().resolve()) for d in SCOPE)


def _log(record: dict) -> None:
    LOG_PATH.parent.mkdir(exist_ok=True)
    record["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def ask(t: Tool, args: dict) -> bool:
    """Default confirmation prompt. Anything but an explicit yes is a no.

    Public because the voice loop wraps it to speak before it blocks.
    """
    print(f"\n  {t.name}({', '.join(f'{k}={v!r}' for k, v in args.items())})")
    return input("  Allow? [y/N] ").strip().lower() in ("y", "yes")


CONFIRM_TIMEOUT_SEC = 60


def ask_dialog(t: Tool, args: dict) -> bool:
    """Confirmation as a window instead of a terminal prompt.

    `ask` blocks on stdin, which works in the REPL and nowhere else. Started
    at login there is no terminal attached, so a CONFIRM tool would block
    forever on input that can never arrive -- wedging the voice loop and
    taking the confirmation gate down with it. A gate that hangs is a gate
    that gets removed.

    Fails closed in every direction. Closing the window, Escape, the timeout
    and Tk failing to open at all are each a no; the only yes is a click on
    Allow. There is deliberately no keyboard default -- Enter on a dialog you
    did not read should not be able to send an email.

    Tk is not thread-safe, so this must run on the main thread. It does: the
    voice loop drives the agent from main(), and the audio callback never
    reaches the gate.
    """
    import tkinter as tk

    try:
        root = tk.Tk()
    except Exception:
        # No display, no Tk. Refusing is the only safe answer -- falling back
        # to input() would reintroduce the exact hang this exists to avoid.
        return False

    allowed = False

    def allow():
        nonlocal allowed
        allowed = True
        root.destroy()

    root.title("MyndOS needs confirmation")
    root.attributes("-topmost", True)
    root.resizable(False, False)
    root.protocol("WM_DELETE_WINDOW", root.destroy)  # the X is a no
    root.bind("<Escape>", lambda _: root.destroy())

    detail = "\n".join(f"{k} = {v!r}" for k, v in args.items()) or "(no arguments)"
    tk.Label(
        root, text=t.description, wraplength=440, justify="left",
        font=("", 10, "bold"),
    ).pack(padx=16, pady=(16, 6), anchor="w")
    tk.Label(
        root, text=f"{t.name}\n{detail}", wraplength=440, justify="left", fg="#555",
    ).pack(padx=16, pady=(0, 14), anchor="w")

    row = tk.Frame(root)
    row.pack(padx=16, pady=(0, 16), anchor="e")
    tk.Button(row, text="Deny", width=10, command=root.destroy).pack(side="right")
    tk.Button(row, text="Allow", width=10, command=allow).pack(side="right", padx=8)

    # Unanswered is a no. Without this, a dialog raised while he is away wedges
    # the loop as thoroughly as the stdin prompt did -- just visibly.
    root.after(int(CONFIRM_TIMEOUT_SEC * 1000), root.destroy)

    root.lift()
    root.focus_force()
    root.mainloop()
    return allowed


def execute(
    name: str,
    args: dict | None = None,
    *,
    user_initiated: bool = False,
    confirm: Callable[[Tool, dict], bool] | None = None,
) -> str:
    """Run a registered tool through the gate. Returns a string for the model.

    `user_initiated` must come from unlocked_by() on my own utterance -- never
    from the model asserting that a call was my idea.
    """
    args = args or {}
    entry = {"tool": name, "args": args, "user_initiated": user_initiated}

    t = REGISTRY.get(name)
    if t is None:
        _log({**entry, "outcome": "error", "detail": "no such tool"})
        return f"No such tool: {name}"

    entry["risk"] = t.risk.value

    if t.risk is Risk.EXPLICIT and not user_initiated:
        _log({**entry, "outcome": "refused", "detail": "not user-initiated"})
        return (
            f"{name} needs a direct spoken command from the user. "
            f"Ask them to invoke it; do not retry."
        )

    if t.risk is Risk.CONFIRM and not (confirm or ask)(t, args):
        _log({**entry, "outcome": "denied"})
        return f"{name} was declined by the user. Do not retry."

    try:
        result = t.fn(**args)
    except Exception as e:
        _log({**entry, "outcome": "error", "detail": f"{type(e).__name__}: {e}"})
        return f"{name} failed: {type(e).__name__}: {e}"

    _log({**entry, "outcome": "ok", "result": str(result)[:500]})
    return str(result)


def schemas() -> list[dict]:
    """OpenAI-format tool definitions for the model."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": {
                    "type": "object",
                    "properties": t.params,
                    "required": list(t.params),
                },
            },
        }
        for t in REGISTRY.values()
    ]


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------


@tool(Risk.SAFE, "Read the text currently on the clipboard.")
def read_clipboard() -> str:
    import tkinter

    root = tkinter.Tk()
    root.withdraw()
    try:
        return root.clipboard_get()
    except tkinter.TclError:
        return ""  # empty, or holding something that isn't text
    finally:
        root.destroy()


@tool(
    Risk.SAFE,
    "Read a UTF-8 text file from inside a scoped project folder.",
    {"path": {"type": "string", "description": "Absolute path to the file."}},
)
def read_file(path: str) -> str:
    # Raise rather than return a polite string, so the action log records this
    # as an error and a scope violation is visible when I read the log back.
    if not in_scope(path):
        raise PermissionError(f"{path} is outside the scoped folders")
    text = Path(path).expanduser().resolve().read_text(encoding="utf-8", errors="replace")
    # Labelled so the model treats it as data. persona.md says the same thing;
    # this is the belt to that pair of braces.
    return f"<file path={path!r}>\n{text}\n</file>"


@tool(
    Risk.CONFIRM,
    "Write text to a file, creating or overwriting it.",
    {
        "path": {"type": "string", "description": "Absolute path to write."},
        "content": {"type": "string", "description": "Full new contents."},
    },
)
def write_file(path: str, content: str) -> str:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    existed = p.exists()
    p.write_text(content, encoding="utf-8")
    return f"{'Overwrote' if existed else 'Wrote'} {p} ({len(content)} chars)"
