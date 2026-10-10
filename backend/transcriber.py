"""faster-whisper wrapper, with a confidence gate in front of the agent.

Whisper always returns its best guess, and on a bad capture its best guess is a
plausible sentence made out of a cough and a fan. The scores that would have
told you so come back on every segment and were being thrown away -- so a
misheard command went to the model, which answered it confidently, out loud.

Borrowed from hey-jev, which re-asks below a confidence threshold rather than
acting on a guess. Here a rejected transcript comes back as the empty string,
which is already the shape main.py handles: it says "I didn't catch that" and
waits, which is the right answer to not having heard.

ponytail: two thresholds and a duration-weighted mean, not a model. They are
calibration knobs and the measured values are printed on every rejection, so
they can be tuned against this microphone in this room rather than guessed at.
Start by watching what a *good* capture scores before moving either.
"""

from faster_whisper import WhisperModel
import numpy as np

# Whisper's average log-probability per token: around -0.2 on a clean short
# command, below -1.0 when it is reaching. Not a percentage -- it is the model's
# own confidence in the tokens it chose, and 0.0 would be certainty.
MIN_AVG_LOGPROB = -1.0
# Its own estimate that a segment is not speech at all. This is the one that
# catches a door, a cough, or the fan holding the capture open.
MAX_NO_SPEECH_PROB = 0.6


def confidence(segments) -> tuple[float, float]:
    """(mean log-probability, worst no-speech probability) across segments.

    The mean is weighted by segment duration, so a half-second fragment can't
    outvote a four-second sentence. Empty input scores as unusable rather than
    as perfect, because "nothing to measure" must not read as "nothing wrong".
    """
    if not segments:
        return float("-inf"), 1.0
    spans = [max(s.end - s.start, 1e-6) for s in segments]
    total = sum(spans)
    mean = sum(s.avg_logprob * w for s, w in zip(segments, spans)) / total
    return mean, max(s.no_speech_prob for s in segments)


class Transcriber:
    def __init__(self,
                 model_size="base.en",
                 compute_type="int8",
                 device="cpu"):
        """CPU and int8 by design, not as a fallback.

        `device="auto"` picked the GPU here and then raised "Library
        cublas64_12.dll is not found" on the first encode -- so every
        transcription failed, which on the voice path took the whole loop with
        it. CUDA for this needs a cuBLAS the machine doesn't have.

        Worth it even once that's installed: the local LLM is pinned resident
        in VRAM (brain.py's keep_alive), and on a 4GB card there isn't room for
        both. base.en in int8 reads a short command on CPU in well under a
        second and leaves the GPU to the model that actually needs it -- which
        is what the README has claimed all along.
        """
        print(f"[transcriber] Loading Whisper model ({model_size}, {device}/{compute_type})...")
        self.model = WhisperModel(
            model_size,
            compute_type=compute_type,
            device=device
        )

    def transcribe(self,
                   audio_data: np.ndarray,
                   language="en",
                   beam_size=1) -> str:
        """
        Transcribes raw PCM audio samples (int16) using Faster-Whisper.

        Returns "" when the transcription is too unreliable to act on, which the
        caller already treats as "didn't hear that".

        Args:
            audio_data (np.ndarray): Raw PCM audio samples (int16).
            language (str): Language code.
            beam_size (int): Beam size for decoding.

        Returns:
            str: Transcribed text, or "" if empty or low-confidence.
        """
        if audio_data is None or len(audio_data) == 0:
            print("[transcriber] Empty audio received.")
            return ""

        # Convert int16 PCM to float32 in range [-1, 1]
        audio_float32 = audio_data.astype(np.float32) / 32768.0

        segments, info = self.model.transcribe(
            audio_float32,
            language=language,
            beam_size=beam_size
        )
        # Materialised because transcribe() returns a generator and the scores
        # have to be read off the same segments the text comes from.
        segments = [s for s in segments if s.text.strip()]

        transcript = " ".join(seg.text.strip() for seg in segments)
        if not transcript:
            return ""

        logprob, no_speech = confidence(segments)
        if logprob < MIN_AVG_LOGPROB or no_speech > MAX_NO_SPEECH_PROB:
            # Printed with the text and both numbers: "it ignored me" and "it
            # misheard me" look identical from the outside, and this is the one
            # place that can tell them apart -- and the only way to know whether
            # the thresholds above are set right for this room.
            print(
                f"[transcriber] Discarded (logprob {logprob:.2f}, "
                f"no_speech {no_speech:.2f}): {transcript!r}"
            )
            return ""

        print(f"[transcriber] Transcript: {transcript}")
        return transcript

    def transcribe_bytes(self,
                         audio_bytes: bytes,
                         language="en",
                         beam_size=1) -> str:
        """
        Converts raw bytes to numpy array and transcribes them.

        Args:
            audio_bytes (bytes): Raw PCM audio bytes.
            language (str): Language code.
            beam_size (int): Beam size for decoding.

        Returns:
            str: Transcribed text.
        """
        if audio_bytes is None or len(audio_bytes) == 0:
            print("[transcriber] Empty audio bytes received.")
            return ""

        # Convert bytes → numpy array of int16
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        return self.transcribe(audio_array, language=language, beam_size=beam_size)
