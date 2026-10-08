"""Speech, with the fixed lines pre-rendered to disk.

Two things this file exists to work around, both measured here rather than
inferred:

**pyttsx3 only honours the first `say()` of an engine's life.** Three
consecutive utterances on one engine took 3711ms, 174ms and 114ms -- the second
and third returned without making a sound. That is not long enough to have
spoken anything, and it is the whole reason an engine is now built per utterance
and thrown away. It was a live bug before any of this: the assistant would
answer the first wake word and then be mute until restarted, which from the
outside looks like the model failing rather than the speech engine.

**Most of what it says is one of a handful of phrases.** "Yes, sir" on every
wake word, "I didn't catch that" on every misfire. Those are rendered to WAV
once and played from disk, which skips building an engine at all -- borrowed
from hey-jev, which pre-renders its scripted lines for the same reason. Replies
a model wrote still go through an engine live, because there is no second time
for those.

The cache is keyed by the exact text, which is why callers keep their phrases in
named constants: pre-rendering "Yes, sir" and then speaking "Yes sir" is a cache
that never hits and never says so.

ponytail: an engine per utterance costs roughly half a second of setup before
the first word. Acceptable against not speaking at all, and the pre-rendered
path avoids it entirely for the phrases said most often. The real fix is Piper
(README roadmap), which this file's shape is deliberately ready for -- `speak`
and `warm` are the only two things anything calls.
"""

import gc
import hashlib
import threading
import wave
import winsound
from pathlib import Path

import numpy as np
import pyttsx3

HERE = Path(__file__).parent
CACHE = HERE / "cache" / "tts"
RATE = 180

# SAPI5 pads every rendered phrase with silence: measured at 0.10s before the
# first word and 0.70s after the last. Live speech doesn't do this, so it is
# purely an artefact of rendering -- and on the voice path that trailing 0.7s
# sits between the acknowledgement and the microphone reopening, which makes it
# 0.7s of him waiting on nothing, on every single wake word.
SILENCE_RMS = 30.0    # int16 full scale is 32768; rendered digital silence is ~0
KEEP_TAIL_SEC = 0.06  # left on deliberately, so a final consonant isn't clipped
FRAME_SEC = 0.02


def _path(text: str) -> Path:
    """Where this phrase's WAV lives.

    Hashed rather than slugged from the words: these phrases carry apostrophes
    and full stops, and a filename built from the text is one that eventually
    collides or meets a character Windows refuses.
    """
    return CACHE / f"{hashlib.md5(text.encode('utf-8')).hexdigest()[:12]}.wav"


def _trim(path: Path) -> None:
    """Strip the silence a rendered phrase is padded with, in place.

    Bails rather than guesses, in every direction: a format it can't measure, a
    file with no frames, or audio that is silent throughout all leave the file
    exactly as it was. Writing a shorter file is an optimisation; writing an
    empty one would be a phrase the assistant can no longer say.
    """
    with wave.open(str(path)) as r:
        params, frames = r.getparams(), r.readframes(r.getnframes())

    audio = np.frombuffer(frames, dtype=np.int16)
    if params.nchannels != 1 or params.sampwidth != 2 or not audio.size:
        return

    size = max(1, int(params.framerate * FRAME_SEC))
    count = len(audio) // size
    if not count:
        return
    blocks = audio[: count * size].reshape(count, size).astype(np.float32)
    loud = np.flatnonzero(np.sqrt((blocks * blocks).mean(axis=1)) > SILENCE_RMS)
    if not loud.size:
        return

    start = loud[0] * size
    end = min(len(audio), (loud[-1] + 1) * size + int(params.framerate * KEEP_TAIL_SEC))
    with wave.open(str(path), "wb") as w:
        w.setparams(params)
        w.writeframes(audio[start:end].tobytes())


def _spend(engine) -> None:
    """Dispose of an engine that has been used.

    The `gc.collect()` is load-bearing, not hygiene. pyttsx3 keeps its engines
    in a registry behind a weak reference and hands the same one back from every
    `init()`, so without dropping the last reference *and* collecting it, the
    next `init()` returns the spent engine and the next utterance is silent.
    """
    try:
        engine.stop()
    except Exception:
        pass
    del engine
    gc.collect()


class TTS:
    def __init__(self, rate=RATE, voice=None):
        self.rate = rate
        self.voice = voice
        # Serialises speech. Two utterances at once is SAPI5 talking over
        # itself at best and deadlocking at worst.
        self._lock = threading.Lock()

    def _engine(self):
        """A fresh engine, configured. Caller must pass it to `_spend`."""
        engine = pyttsx3.init()
        engine.setProperty("rate", self.rate)
        if self.voice:
            for v in engine.getProperty("voices"):
                if self.voice.lower() in v.name.lower():
                    engine.setProperty("voice", v.id)
                    break
        return engine

    def warm(self, *phrases: str) -> None:
        """Render any phrase not already on disk. Call once at startup.

        Only the missing ones, so this is free on every launch after the first
        and still repairs a cache that was deleted.

        Wrapped: a speech engine that cannot write a file is not a reason the
        assistant fails to start. Every phrase here is still speakable live, so
        a failure costs the saving, not the sentence.
        """
        try:
            missing = [p for p in phrases if not _path(p).exists()]
            if not missing:
                return
            CACHE.mkdir(parents=True, exist_ok=True)
            with self._lock:
                engine = self._engine()
                for phrase in missing:
                    engine.save_to_file(phrase, str(_path(phrase)))
                engine.runAndWait()
                _spend(engine)
            done = 0
            for phrase in missing:
                if not _path(phrase).exists():
                    continue
                done += 1
                # Per phrase and wrapped: one unmeasurable WAV costs its own
                # 0.7s of padding, not the whole warm-up.
                try:
                    _trim(_path(phrase))
                except Exception as e:
                    print(f"[tts] Trim skipped for {phrase!r}: {type(e).__name__}: {e}")
            print(f"[tts] Pre-rendered {done}/{len(missing)} phrase(s) to {CACHE}")
        except Exception as e:
            print(f"[tts] Pre-render skipped: {type(e).__name__}: {e}")

    def speak(self, text: str) -> None:
        """Say `text`, from disk when it's one of the pre-rendered lines.

        Synchronous either way: callers rely on this returning only once the
        words are actually out, or the microphone reopens over the top of them.
        """
        with self._lock:
            cached = _path(text)
            if cached.exists():
                try:
                    winsound.PlaySound(str(cached), winsound.SND_FILENAME)
                    return
                except RuntimeError:
                    # A truncated or half-written WAV. Saying it live is always
                    # available, so a bad cache entry degrades to the slow path
                    # rather than to silence.
                    print(f"[tts] Unplayable cache entry {cached.name}, speaking live")
            engine = self._engine()
            engine.say(text)
            engine.runAndWait()
            _spend(engine)


# Shared instance. The lock inside it is what serialises speech, so every
# module speaks through this one.
tts = TTS()
