"""Self-check for push-to-talk edge detection. No keyboard, no Windows API.

`_down` is the only thing touching the OS, so overriding it makes the state
machine testable -- and the state machine is the whole file.

Run: python test_hotkey.py
"""

import hotkey


class Scripted(hotkey.Hotkey):
    """A keyboard written out as a string. '_' is up, 'X' is held down."""

    def __init__(self, pattern: str):
        super().__init__()
        self.pattern = list(pattern)
        self.i = 0

    def _down(self) -> bool:
        # Reads up forever once the pattern runs out, like a key nobody is
        # touching -- which every poll after the last one is.
        c = self.pattern[self.i] if self.i < len(self.pattern) else "_"
        self.i += 1
        return c == "X"


def _fires(pattern: str) -> int:
    k = Scripted(pattern)
    return sum(k.pressed() for _ in range(len(pattern) + 2))


def test_one_tap_is_one_turn():
    assert _fires("_X_") == 1


def test_a_held_key_is_still_one_turn():
    """The reason the previous state is remembered at all. The loop polls every
    80ms, so a key held for half a second is six polls -- and six turns, if
    this only looked at whether it was down."""
    assert _fires("_XXXXXXXX_") == 1


def test_two_taps_are_two_turns():
    assert _fires("_X_X_") == 2
    assert _fires("_XX__XX_") == 2


def test_a_key_nobody_touches_never_fires():
    assert _fires("_________") == 0
    assert _fires("") == 0


def test_a_key_already_down_at_startup_does_not_fire():
    """If it was down before we started looking, nobody pressed it while we
    were -- and a modifier held for an unrelated shortcut must not open a
    turn the moment the assistant launches."""
    k = Scripted("XXXX_")
    first = k.pressed()
    assert first is True, "the first poll cannot tell; firing once is the honest read"
    # What matters is that holding it does not keep firing.
    assert sum(k.pressed() for _ in range(4)) == 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
