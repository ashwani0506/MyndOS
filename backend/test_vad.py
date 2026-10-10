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


def _cap(pattern: str):
    """Run a scripted capture. Returns (frames recorded, seconds spoken)."""
    audio, spoke = vad.capture(_mic(pattern), LEVEL, **SHORT)
    return len(audio) // (FRAME * 2), round(spoke, 3)


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
    assert vad.threshold(0.0, 0.0) == vad.MIN_RMS
    assert vad.threshold(0.0, 4000.0) == vad.MIN_RMS


def test_the_threshold_lands_between_the_room_and_the_voice():
    """Every level pair actually measured on this hardware.

    The middle row is why the formula changed. `floor * 3` gave 14032 on this
    laptop's mic array with Windows boost on, where speech peaked at 11446 --
    a bar louder than the person talking, so the capture could never start.
    Turning the microphone up had made the assistant deaf, which is not a
    direction anyone would think to debug.

    Both numbers move together, because gain raises the floor as much as the
    voice. That is why a geometric mean holds where a fixed multiple doesn't.
    """
    for floor, loud in ((79, 259), (4677, 11446), (5, 3996)):
        level = vad.threshold(float(floor), float(loud))
        assert level < loud, f"floor={floor} loud={loud}: bar above the voice"
        assert level >= vad.MIN_RMS, f"floor={floor} loud={loud}: below the hiss"


def test_levels_reads_the_room_and_the_voice_off_one_buffer():
    """The rolling buffer holds both: seconds of room ending in the wake word.
    The quiet fifth is the room and the loud top is him, which is the pair
    needed to put a threshold between them."""
    floor, loud = vad.levels(_pcm(100, 80) + _pcm(4000, 20))
    assert floor < 500, f"speech leaked into the floor: {floor}"
    assert loud > 3000, f"the voice was averaged away: {loud}"
    assert vad.threshold(floor, loud) < 4000, "the bar must sit under the voice"
    # Nothing to measure is not a quiet room; it is no answer at all.
    assert vad.levels(b"") == (0.0, 0.0)


def test_capture_ends_after_a_run_of_quiet():
    frames, spoke = _cap("..!!!!")
    assert frames == 9  # 2 lead-in + 4 speech + 3 hang
    assert spoke == 0.12, "4 speech frames, and neither silence counted"


def test_a_pause_for_breath_does_not_end_the_capture():
    """The one that decides whether this is usable. A two-frame pause inside a
    sentence is shorter than hang, so the second half must survive -- stopping
    at the pause would have returned 5 frames and lost the rest."""
    frames, spoke = _cap("!!..!!!")
    assert frames == 10, "cut off at a mid-sentence pause"
    # First speech frame to last, pause included: he was mid-sentence, not
    # waiting on the machine, so that pause belongs to him.
    assert spoke == 0.21


def test_a_misfired_wake_word_gives_up_quickly():
    """Nothing said at all. Without this the capture would sit there for the
    full max_sec, which is the fixed-window problem with extra steps."""
    frames, spoke = _cap("")
    assert frames == 5  # lead_in, not max_sec
    assert spoke == 0.0, "silence must not read as speech"


def test_lead_in_is_the_deadline_for_starting_to_speak():
    """Speech that starts after lead_in is missed -- so LEAD_IN_SEC has to be
    longer than the gap between the spoken acknowledgement ending and him
    starting. 2s in production; this pins the boundary, it doesn't excuse it."""
    frames, spoke = _cap(".....!!!")
    assert frames == 5, "gave up before the late start, as designed"
    assert spoke == 0.0


def test_someone_who_will_not_stop_talking_is_capped():
    frames, spoke = _cap("!" * 50)
    assert frames == 20  # max_sec 0.6 / 0.03
    assert spoke == 0.6, "no hang to subtract -- it hit the cap mid-sentence"


def test_the_silence_he_sat_through_is_not_counted_as_talking():
    """The headline in measure.py is total minus spoken, so anything misfiled
    as speech here shortens the reported wait -- in the flattering direction.

    Two silences bracket every command: the gap before he starts, and the hang
    the VAD deliberately waits out to be sure he's done. Both are time he spent
    staring at the machine.
    """
    frames, spoke = _cap("..!!!!")
    assert frames == 9                       # 0.27s of audio
    assert round(frames * 0.03 - spoke, 3) == 0.15  # 2 frames before, 3 after


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
