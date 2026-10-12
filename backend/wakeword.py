"""Wake word: one small model trained on one phrase.

Vosk ran a full speech recogniser continuously in order to spot a single word,
and on real audio it never once managed it. "Jarvis", said into this laptop's
microphone, came back as "journalists", "chat with", "this your" and "huh" --
across three different microphone levels, including one with clean levels and
a near-silent noise floor. Whisper, on the same recordings, managed it once in
three. The word is in Vosk's lexicon and it transcribes synthesised speech
perfectly, so this was never a vocabulary problem: a general recogniser
listening for one rare proper noun is the wrong instrument, like hiring a
translator to listen for a doorbell.

Measured on the same five seconds of audio:

    openWakeWord, phrase it was trained on   0.998
    openWakeWord, recording missing "hey"    0.487
    Vosk                                     "chat with what time is it"

0.487 is below the firing threshold, but it is a model getting most of the way
there on a phrase with its first word absent -- which is the whole argument.
It also costs a fraction of the CPU, because it runs one 1.2MB classifier
instead of decoding every word anyone says all day.

The phrase is "hey jarvis" rather than "jarvis", which is what the pretrained
model is for and is better anyway: two syllables is far harder to trigger by
accident than one common-ish name.
"""

import numpy as np

FRAME = 1280        # 80ms at 16kHz -- the window the model expects
THRESHOLD = 0.5     # openWakeWord's own suggested default
PHRASE = "hey jarvis"
MODEL = "hey_jarvis"


def _windows(tail: np.ndarray, pcm: bytes) -> tuple[list, np.ndarray]:
    """Split buffered audio into model-sized windows, plus what's left over.

    Separate from the class, and pure, so the buffering can be tested without
    loading a model or owning a microphone. The leftover matters: audio arrives
    in whatever block size the caller chose, the model wants exactly 80ms, and
    a wake word that straddles two blocks must not be dropped on the seam.
    """
    audio = np.concatenate([tail, np.frombuffer(pcm, dtype=np.int16)])
    n = len(audio) // FRAME
    return [audio[i * FRAME:(i + 1) * FRAME] for i in range(n)], audio[n * FRAME:]


class WakeWord:
    def __init__(self, model: str = MODEL, threshold: float = THRESHOLD):
        # Imported here rather than at module scope: it pulls in onnxruntime,
        # and test_wakeword.py tests the buffering without either.
        from openwakeword.model import Model

        self.model = Model(wakeword_models=[model], inference_framework="onnx")
        self.threshold = threshold
        self._tail = np.empty(0, dtype=np.int16)
        self.last_score = 0.0

    def heard(self, pcm: bytes) -> bool:
        """True the moment the wake phrase completes in this audio.

        Returns on the first window over the threshold and resets, rather than
        scoring the rest of the block: the audio after the phrase is the phrase
        trailing off, and letting it score again would fire twice on one
        "hey jarvis".
        """
        windows, self._tail = _windows(self._tail, pcm)
        for window in windows:
            self.last_score = max(self.model.predict(window).values())
            if self.last_score >= self.threshold:
                self.reset()
                return True
        return False

    def reset(self) -> None:
        """Forget what it has heard so far.

        Needed after a turn as well as after a hit. The model scores a rolling
        window of recent audio, so the tail of the last wake word is still in
        there when the microphone reopens -- and without this it can fire again
        on audio from before the command was even answered.
        """
        self.model.reset()
        self._tail = np.empty(0, dtype=np.int16)
