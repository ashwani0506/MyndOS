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
MARGIN = 3.0             # speech must be this many times the noise floor

# Calibration knob. A room silent enough that the measured floor is near zero
# would otherwise set a near-zero threshold and treat hiss as speech. int16
# full scale is 32768; a quiet desk mic idles near 30, speech runs 1000+.
MIN_RMS = 180.0


def rms(frame: bytes) -> float:
    """Loudness of one frame of int16 PCM."""
    a = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
    # float32 first: squaring int16 overflows and silently wraps negative.
    return float(np.sqrt(np.mean(a * a))) if a.size else 0.0


def noise_floor(audio: bytes, samplerate: int = 16000, pct: int = 20) -> float:
    """Ambient level of a stretch of audio, as the quietest fifth of its frames.

    A percentile rather than a mean because this audio isn't pure ambience --
    it ends with the wake word, and may contain speech. Speech is the loud
    minority of a ten-second buffer, so the 20th percentile steps over it.
    """
    a = np.frombuffer(audio, dtype=np.int16)
    size = int(samplerate * FRAME_MS / 1000)
    n = len(a) // size
    if n == 0:
        return 0.0
    frames = a[: n * size].reshape(n, size).astype(np.float32)
    return float(np.percentile(np.sqrt((frames * frames).mean(axis=1)), pct))


def threshold(floor: float) -> float:
    return max(floor * MARGIN, MIN_RMS)


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
