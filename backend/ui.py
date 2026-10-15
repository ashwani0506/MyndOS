"""The one Tk root, and the status panel that lives on it.

Two reasons this is a module rather than a widget tucked into main.py.

**A process gets one Tk root.** `tools.ask_dialog` and `tools.read_clipboard`
each created their own, which worked only because neither outlived the other.
A panel that stays up for the whole session makes that a conflict, so nothing
calls `Tk()` directly any more -- everything borrows `root()`, which is hidden,
and puts its own window on it as a `Toplevel`. That also keeps the text REPL
working, where there is no panel but the clipboard tool still needs a root.

**At login there is no terminal.** Every `[vad]`, `[wakeword]` and
`[transcriber]` line that made this thing debuggable at all is invisible to the
person actually using it -- and the failures those lines describe are precisely
the ones that look identical from outside: not hearing, mishearing, hearing and
failing to answer. The panel is where that information goes when stdout isn't
being read by anyone.

**No animation, anywhere, deliberately.** A voice turn is synchronous on this
thread -- the model call blocks it -- so Tk cannot repaint mid-turn. A spinner
that is supposed to be spinning and isn't reads as a crash, which is worse than
no spinner. The pattern throughout is: set the state, pump once, then do the
slow thing. A static label saying "Thinking" is honest about exactly as much as
it knows.

ponytail: tkinter, because it is stdlib and already a dependency for the
confirmation dialog. It will not win any awards. Swap in a real overlay
toolkit when the panel needs to do something Tk genuinely can't -- drawing on
the screen (README roadmap) is the first thing on that list.
"""

import tkinter as tk

WIDTH, HEIGHT = 460, 96
TOP_MARGIN = 24

# State name -> (dot colour, label). The dot is the only colour in here, so it
# carries the whole "what is it doing" signal at a glance.
STATES = {
    "idle": ("#4b5563", "Idle"),
    "listening": ("#22c55e", "Listening"),
    "thinking": ("#eab308", "Thinking"),
    "speaking": ("#3b82f6", "Speaking"),
    "error": ("#ef4444", "Problem"),
}

BG = "#111317"
FG = "#e5e7eb"
DIM = "#8b93a1"

# Quoted so it isn't evaluated at import. test_tools swaps in a stub tkinter
# whose Tk is a function, to prove the confirmation dialog fails closed with no
# display -- and an annotation that touches tk.Tk at module level turns that
# into an import error instead of the "no display" path it is testing.
_root: "tk.Tk | None" = None


def root() -> tk.Tk:
    """The process's single Tk root, hidden. Created on first use.

    Withdrawn rather than shown: it exists to own the toplevels, not to be
    looked at. Nothing destroys it -- it lives as long as the process, because
    the next clipboard read or confirmation dialog will want it.
    """
    global _root
    if _root is None:
        _root = tk.Tk()
        _root.withdraw()
    return _root


def _meter(rms: float, blocks: int = 14, full: float = 3000.0) -> str:
    """A level bar, as text.

    Square-rooted because loudness isn't linear to the eye any more than to the
    ear: a linear bar sits at nothing for ordinary speech and then slams to the
    end on one loud word. `full` is a calibration knob, set near the level
    speech actually reaches on this hardware rather than at int16 full scale,
    which nothing ever gets near.
    """
    filled = round(blocks * min(1.0, (max(rms, 0.0) / full) ** 0.5))
    return "█" * filled + "·" * (blocks - filled)


class Panel:
    """A small always-on-top strip: what it's doing, and what was said.

    Deliberately not interactive. There is nothing to click, so it needs no
    focus, and taking focus from whatever he is actually working in would make
    it a nuisance rather than a readout.
    """

    def __init__(self):
        self.win = tk.Toplevel(root())
        self.win.overrideredirect(True)      # no title bar, no taskbar entry
        self.win.attributes("-topmost", True)
        self.win.attributes("-alpha", 0.94)
        self.win.configure(bg=BG)

        screen_w = self.win.winfo_screenwidth()
        self.win.geometry(
            f"{WIDTH}x{HEIGHT}+{(screen_w - WIDTH) // 2}+{TOP_MARGIN}"
        )

        pad = {"padx": 14, "anchor": "w"}
        top = tk.Frame(self.win, bg=BG)
        top.pack(fill="x", pady=(12, 2), **pad)

        self._dot = tk.Label(top, text="●", bg=BG, fg=DIM, font=("Segoe UI", 10))
        self._dot.pack(side="left")
        self._state = tk.Label(
            top, text="Idle", bg=BG, fg=FG, font=("Segoe UI Semibold", 10)
        )
        self._state.pack(side="left", padx=(6, 10))
        self._level = tk.Label(
            top, text=_meter(0), bg=BG, fg=DIM, font=("Consolas", 9)
        )
        self._level.pack(side="left")

        self._heard = tk.Label(
            self.win, text="", bg=BG, fg=FG, font=("Segoe UI", 10),
            wraplength=WIDTH - 28, justify="left",
        )
        self._heard.pack(fill="x", pady=(2, 0), **pad)
        self._reply = tk.Label(
            self.win, text="", bg=BG, fg=DIM, font=("Segoe UI", 10),
            wraplength=WIDTH - 28, justify="left",
        )
        self._reply.pack(fill="x", pady=(0, 10), **pad)

        self.state("idle")

    # -- updates. Each pumps, so a caller can set a state and then block. ----

    def state(self, name: str, detail: str = "") -> None:
        colour, label = STATES.get(name, STATES["idle"])
        self._dot.configure(fg=colour)
        self._state.configure(text=f"{label}{f' · {detail}' if detail else ''}")
        self.pump()

    def level(self, rms: float) -> None:
        self._level.configure(text=_meter(rms))
        self.pump()

    def heard(self, text: str) -> None:
        self._heard.configure(text=f"You: {text}" if text else "")
        self.pump()

    def replied(self, text: str) -> None:
        # Trimmed to two lines' worth. The panel is a readout, not a transcript;
        # a long answer that resized the window every turn would be worse than
        # a truncated one, and it is being spoken aloud anyway.
        short = text if len(text) <= 150 else text[:149] + "…"
        self._reply.configure(text=short)
        self.pump()

    def pump(self) -> None:
        """Let Tk redraw and handle events, without entering a mainloop.

        `update()` rather than `mainloop()` because the voice loop owns this
        thread; Tk only gets it between audio blocks. Wrapped because a panel
        that has been destroyed must not take the assistant with it -- the
        readout is a convenience and the assistant is not.
        """
        try:
            self.win.update()
        except tk.TclError:
            pass

    def close(self) -> None:
        try:
            self.win.destroy()
        except tk.TclError:
            pass
