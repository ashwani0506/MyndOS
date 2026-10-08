"""Self-check for the speech cache. No speech engine, no sound, no mic.

The WAVs here are built by hand rather than rendered, so this runs anywhere and
tests the trimming logic rather than SAPI5's mood.

Run: python test_tts.py
"""

import tempfile
import wave
from pathlib import Path

import numpy as np

import tts

SR = 22050


def _wav(path: Path, lead: float, speech: float, tail: float) -> None:
    """A phrase-shaped WAV: digital silence, a loud tone, more silence."""
    tone = (np.sin(np.linspace(0, speech * 440 * 2 * np.pi, int(SR * speech))) * 8000)
    audio = np.concatenate([
        np.zeros(int(SR * lead)), tone, np.zeros(int(SR * tail)),
    ]).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(audio.tobytes())


def _seconds(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def test_the_padding_comes_off_and_the_phrase_does_not():
    """Measured on real output: 0.10s of silence before the first word and
    0.70s after the last. The trailing three quarters of a second is the one
    that matters -- on the voice path it sits between the acknowledgement and
    the microphone reopening, so it is dead air he waits through every time."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.wav"
        _wav(p, lead=0.10, speech=1.00, tail=0.70)
        assert round(_seconds(p), 1) == 1.8

        tts._trim(p)
        # The speech survives, plus the tail deliberately left on so a final
        # consonant isn't clipped. Both ends of the padding are gone.
        assert 1.0 <= _seconds(p) <= 1.0 + tts.KEEP_TAIL_SEC + tts.FRAME_SEC * 2


def test_trimming_is_idempotent():
    """warm() only renders missing phrases, but a cache rebuilt by hand or a
    retried render must not shave the phrase a little shorter each pass."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.wav"
        _wav(p, lead=0.1, speech=1.0, tail=0.7)
        tts._trim(p)
        once = _seconds(p)
        tts._trim(p)
        tts._trim(p)
        assert _seconds(p) == once


def test_a_silent_file_is_left_alone():
    """Shortening a file is an optimisation. Writing an empty one would be a
    phrase the assistant can no longer say, so silence throughout bails."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.wav"
        _wav(p, lead=0.5, speech=0.0, tail=0.5)
        before = _seconds(p)
        tts._trim(p)
        assert _seconds(p) == before


def test_a_format_it_cannot_measure_is_left_alone():
    """Stereo, or any width but 16-bit, is not something the frame maths above
    is right about -- so it declines rather than mangles the file."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "stereo.wav"
        with wave.open(str(p), "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(np.zeros(SR, dtype=np.int16).tobytes())
        before = _seconds(p)
        tts._trim(p)
        assert _seconds(p) == before


def test_the_cache_path_is_stable_and_collision_free():
    """The cache is keyed by exact text, which is the whole reason callers hold
    their phrases in named constants: "Yes, sir" and "Yes sir" are different
    files, and a drifted string is a cache that silently never hits."""
    assert tts._path("Yes, sir") == tts._path("Yes, sir")
    assert tts._path("Yes, sir") != tts._path("Yes sir")
    # Punctuation and apostrophes must not reach the filename.
    assert tts._path("I didn't catch that.").name.endswith(".wav")
    assert tts._path("I didn't catch that.").stem.isalnum()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
