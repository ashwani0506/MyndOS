"""Self-check for the voice-activity detector. No microphone, no audio device.

`capture()` takes its frame reader as an argument precisely so this file can
script a microphone, and the two tests that matter are the two failure modes:
cutting him off mid-sentence, and recording an empty room for fifteen seconds.

Run: python test_vad.py
"""

import numpy as np

import vad

FRAME = 480  # samples per 30ms frame at 16kHz
LOUD = 3000  # comfortably above the test threshold, as speech is above a room
LEVEL = 500

# Short enough to write the pattern out: 3 quiet frames to stop, 5 to give up.
SHORT = dict(hang=0.09, lead_in=0.15, max_sec=0.6)


def _pcm(amp: int, frames: int = 1) -> bytes:
    return np.full(FRAME * frames, amp, dtype=np.int16).tobytes()


def _mic(pattern: str):
    """A scripted microphone. '.' is a silent frame, '!' a loud one.

    Goes silent forever once the pattern runs out, which is what a real mic
    does and what every one of the exit conditions has to cope with.
    """
    it = iter(pattern)
    return lambda: _pcm(LOUD if next(it, ".") == "!" else 0)


def _frames(audio: bytes) -> int:
    return len(audio) // (FRAME * 2)


def test_rms_reads_loudness():
    assert vad.rms(b"") == 0.0
    assert vad.rms(_pcm(0)) == 0.0
    assert round(vad.rms(_pcm(3000))) == 3000


def test_rms_does_not_overflow_on_loud_audio():
    """int16 squared wraps negative, which would make a shout read as silence.
    The float32 cast in rms() is the only thing standing between those."""
    assert round(vad.rms(_pcm(30000))) == 30000


def test_the_noise_floor_steps_over_speech_in_the_buffer():
    """The calibration audio is the rolling buffer, which ends with the wake
    word -- so it is not pure ambience. A mean would be dragged up by the
    speech and set a threshold that then ignores the next sentence."""
    room = _pcm(50, 80) + _pcm(5000, 20)
    floor = vad.noise_floor(room)
    assert floor < 100, f"speech leaked into the floor: {floor}"
    assert floor < 1040, "that's the mean -- the percentile is the point"


def test_too_little_audio_to_calibrate_is_not_an_error():
    assert vad.noise_floor(b"") == 0.0
    assert vad.noise_floor(_pcm(500)[:100]) == 0.0


def test_a_silent_room_still_gets_a_usable_threshold():
    """A floor near zero would otherwise scale to a threshold near zero and
    turn mic hiss into speech, holding the capture open to max_sec."""
    assert vad.threshold(0.0) == vad.MIN_RMS
    assert vad.threshold(1000.0) == 3000.0


def test_capture_ends_after_a_run_of_quiet():
    audio = vad.capture(_mic("..!!!!"), LEVEL, **SHORT)
    assert _frames(audio) == 9  # 2 lead-in + 4 speech + 3 hang


def test_a_pause_for_breath_does_not_end_the_capture():
    """The one that decides whether this is usable. A two-frame pause inside a
    sentence is shorter than hang, so the second half must survive -- stopping
    at the pause would have returned 5 frames and lost the rest."""
    audio = vad.capture(_mic("!!..!!!"), LEVEL, **SHORT)
    assert _frames(audio) == 10, "cut off at a mid-sentence pause"


def test_a_misfired_wake_word_gives_up_quickly():
    """Nothing said at all. Without this the capture would sit there for the
    full max_sec, which is the fixed-window problem with extra steps."""
    audio = vad.capture(_mic(""), LEVEL, **SHORT)
    assert _frames(audio) == 5  # lead_in, not max_sec


def test_lead_in_is_the_deadline_for_starting_to_speak():
    """Speech that starts after lead_in is missed -- so LEAD_IN_SEC has to be
    longer than the gap between the spoken acknowledgement ending and him
    starting. 2s in production; this pins the boundary, it doesn't excuse it."""
    audio = vad.capture(_mic(".....!!!"), LEVEL, **SHORT)
    assert _frames(audio) == 5, "gave up before the late start, as designed"


def test_someone_who_will_not_stop_talking_is_capped():
    audio = vad.capture(_mic("!" * 50), LEVEL, **SHORT)
    assert _frames(audio) == 20  # max_sec 0.6 / 0.03


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
