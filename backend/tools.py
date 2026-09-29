"""Trusted execution layer: the boundary between what the model proposes and
what actually runs.

Three risk tiers:

  SAFE      Execute. Reading the clipboard, reading a file inside a scoped
            project folder.
  EXPLICIT  Only on a direct command from me -- never model-initiated. Screen
            and audio capture live here, so no plan the model writes can switch
            them on by itself.
  CONFIRM   Stop and ask, every single time. Sending anything, deleting or
            overwriting, spending money, installing, writing outside scope.
            There is deliberately no "remember this choice".

Two invariants this file exists to hold:

  1. Nothing the agent *read* can trigger a consequential call. Model-initiated
     calls carry user_initiated=False, which can never reach an EXPLICIT tool;
     CONFIRM tools still stop for a human regardless of who asked.
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


REGISTRY: dict[str, Tool] = {}


def tool(risk: Risk, description: str, params: dict | None = None):
    def register(fn):
        REGISTRY[fn.__name__] = Tool(fn.__name__, description, risk, params or {}, fn)
        return fn

    return register


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


def execute(
    name: str,
    args: dict | None = None,
    *,
    user_initiated: bool = False,
    confirm: Callable[[Tool, dict], bool] | None = None,
) -> str:
    """Run a registered tool through the gate. Returns a string for the model.

    `user_initiated` must be True only for something I said directly -- never
    for a call the model produced while reasoning over content it read.
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
