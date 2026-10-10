"""Stop recording when he stops talking, instead of after a fixed three seconds.

The old loop recorded for exactly EXTRA_RECORD_SEC after the wake word, so
"what time is it" and a twelve-word question cost the same wall clock, and the
short one -- which is most of them -- paid for silence. That fixed window is
larger than the entire fast/deep gap the router was built to save.

This is energy-based: a frame counts as speech when its RMS is above a
threshold, and the capture ends after HANG_SEC of quiet. No new dependency,
since numpy is already here for the audio path.

The threshold is calibrated per command rather than hard-coded, because a
laptop fan, an air conditioner and a quiet room are three different rooms.
`noise_floor()` reads it off the audio already sitting in the rolling buffer,
which is ambient by definition -- it's what the mic heard before the wake word.

ponytail: RMS can't tell speech from a slammed door, so a loud noise can hold
the capture open until MAX_SEC. Acceptable for a desk mic in a room with one
person; swap `rms()` for webrtcvad (a 30ms frame classifier, same shape) if
that stops being true.
"""

import numpy as np

FRAME_MS = 30            # webrtcvad's frame size too, so the upgrade path fits
HANG_SEC = 0.7           # quiet before the capture ends
LEAD_IN_SEC = 2.0        # give up if nothing was said at all
MAX_SEC = 15.0           # hard ceiling, so a stuck mic can't record all night
QUIET_PCT = 20           # "the room": the quiet fifth of the buffer
LOUD_PCT = 95            # "someone talking": the loud top of it

# Calibration knob. A room silent enough that the measured floor is near zero
# would otherwise set a near-zero threshold and treat hiss as speech. int16
# full scale is 32768; a quiet desk mic idles near 30, speech runs 1000+.
MIN_RMS = 180.0


def rms(frame: bytes) -> float:
    """Loudness of one frame of int16 PCM."""
    a = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
    # float32 first: squaring int16 overflows and silently wraps negative.
    return float(np.sqrt(np.mean(a * a))) if a.size else 0.0


def _frames(audio: bytes, samplerate: int) -> np.ndarray:
    """Per-frame RMS of a stretch of audio. Empty when there isn't a frame."""
    a = np.frombuffer(audio, dtype=np.int16)
    size = int(samplerate * FRAME_MS / 1000)
    n = len(a) // size
    if n == 0:
        return np.empty(0, dtype=np.float32)
    f = a[: n * size].reshape(n, size).astype(np.float32)
    return np.sqrt((f * f).mean(axis=1))


def noise_floor(audio: bytes, samplerate: int = 16000, pct: int = QUIET_PCT) -> float:
    """Ambient level of a stretch of audio, as the quietest fifth of its frames.

    A percentile rather than a mean because this audio isn't pure ambience --
    it ends with the wake word, and may contain speech. Speech is the loud
    minority of a ten-second buffer, so the 20th percentile steps over it.
    """
    f = _frames(audio, samplerate)
    return float(np.percentile(f, pct)) if f.size else 0.0


def levels(audio: bytes, samplerate: int = 16000) -> tuple[float, float]:
    """(the room, someone talking) for this stretch of audio.

    Both come from the same buffer because that buffer holds both: ten seconds
    of room ending in the wake word. The quiet fifth is the room and the loud
    top is him, which is exactly the pair needed to put a threshold between
    them.
    """
    f = _frames(audio, samplerate)
    if not f.size:
        return 0.0, 0.0
    return float(np.percentile(f, QUIET_PCT)), float(np.percentile(f, LOUD_PCT))


def threshold(floor: float, loud: float) -> float:
    """Where speech starts, for this room and this microphone.

    The geometric mean of the two, which lands between them wherever they sit.
    That property is the whole point, because they move together: turning up
    the mic raises the floor as much as the voice.

    This replaced `floor * 3`, which was unbounded above and failed exactly
    when it mattered. Measured on this laptop's mic array with Windows boost
    on: the floor came up to 4677 and speech peaked at 11446 -- only 2.4x
    apart -- so three times the floor was 14032, a bar louder than the person
    talking. The capture could never start. Boosting the microphone had made
    the assistant deaf, which is not a direction anyone would think to debug.

    The three levels actually measured, all of which this now clears:

        floor   loud    old     new
           79    259    238*    180    quiet mic, barely cleared
         4677  11446  14032     7316   boosted mic, impossible
            5   3996    180      180   headset, fine either way

    MIN_RMS still floors it, for a room so quiet that the geometric mean lands
    in the hiss.
    """
    return max((floor * loud) ** 0.5, MIN_RMS)


def capture(
    read_frame,
    level: float,
    hang: float = HANG_SEC,
    lead_in: float = LEAD_IN_SEC,
    max_sec: float = MAX_SEC,
) -> tuple[bytes, float]:
    """Collect frames until he stops talking. Returns (audio, seconds spoken).

    `read_frame` is a callable returning one FRAME_MS frame of int16 PCM --
    injected rather than opening a stream here, so the whole decision can be
    tested against a scripted microphone.

    Three ways out, and the order matters: silence after speech (the normal
    one), nothing said at all within `lead_in`, and the hard `max_sec` cap.
    Without the second, a misfired wake word costs fifteen seconds of staring.

    The second return value is first speech frame to last, which is not the
    length of the audio: that also holds the gap before he started and the
    `hang` of quiet at the end. Both of those are silence he waited through, so
    counting them as speech would file his dead time on the wrong side of the
    ledger -- and the measurement they feed exists to separate exactly those
    two. A pause mid-sentence does count as speaking, because it is.
    """
    frame_sec = FRAME_MS / 1000
    frames, quiet, elapsed = [], 0.0, 0.0
    # Also serves as "has he said anything yet", which is why there's no
    # separate flag: two variables holding one fact is one of them going stale.
    first = last = None

    while elapsed < max_sec:
        frame = read_frame()
        frames.append(frame)
        elapsed += frame_sec

        if rms(frame) > level:
            if first is None:
                first = elapsed - frame_sec  # the start of this frame
            last = elapsed
            quiet = 0.0
            continue

        quiet += frame_sec
        # Only the run of quiet *after* speech ends it. A pause for breath
        # mid-sentence is shorter than HANG_SEC, which is the whole reason
        # that constant isn't smaller.
        if first is not None and quiet >= hang:
            break
        if first is None and elapsed >= lead_in:
            break

    return b"".join(frames), (last - first if first is not None else 0.0)
