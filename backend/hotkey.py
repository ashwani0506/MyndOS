"""Push-to-talk: a key that starts a turn without the wake word.

The wake word is the convenience path. This is the reliable one -- no model, no
calibration, no "hey", no acknowledgement, no quiet room. That distinction is
not theoretical: the wake word spent a week not working, and the cause turned
out to be four separate bugs stacked on each other. A desktop assistant
reachable only by voice is one bad microphone away from being unreachable.

Right Ctrl, because it is on every keyboard, sits under the hand, and almost
nothing else uses it. Read with GetAsyncKeyState, which reports global key
state regardless of focus -- necessary here, since the panel deliberately never
takes focus.

ponytail: polled from the audio loop rather than registered as a Windows hot
key. The loop already wakes every 80ms, so the worst case is 80ms of lag, and a
poll is one syscall against a RegisterHotKey message pump that would want its
own thread and a window handle to receive on. Swap it if the lag is ever
audible, which at 80ms it will not be.
"""

import ctypes

VK_RCONTROL = 0xA3
_DOWN = 0x8000  # GetAsyncKeyState's "held right now" bit


class Hotkey:
    """Edge-triggered: `pressed()` is true once per physical press.

    The remembered state is the whole point. GetAsyncKeyState's high bit means
    "down right now", so without it a key held for half a second reads as a
    fresh press on every poll -- six turns from one thumb.
    """

    def __init__(self, vk: int = VK_RCONTROL):
        self.vk = vk
        self._was_down = False

    def _down(self) -> bool:
        """Overridden in the tests, which have no keyboard to press."""
        return bool(ctypes.windll.user32.GetAsyncKeyState(self.vk) & _DOWN)

    def pressed(self) -> bool:
        down = self._down()
        fired = down and not self._was_down
        self._was_down = down
        return fired
