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
# Measured, not guessed, and the two numbers that set it are the only two that
# matter: how high unrelated speech scores, and how low a real attempt scores.
#
#   213s of recorded speech containing no wake phrase   worst 0.359
#   his own attempts, weakest to strongest              0.45 .. 0.82
#
# openWakeWord suggests 0.5. At 0.5 he was saying it three and four times per
# turn -- 0.49, 0.39, 0.45, then 0.61 -- and the repeated attempts were what
# read as a delay between turns. 0.40 sits near the midpoint of the gap, clear
# of the false-positive ceiling and under the weakest genuine attempt.
#
# The asymmetry says lean this way too. A missed phrase costs saying it again.
# A false wake costs attention and nothing else: it cannot act, because an
# EXPLICIT tool is unlocked only by words in his own utterance and a CONFIRM
# tool stops for a click. There is also a hotkey now, so voice is no longer
# the only way in.
#
# ponytail: a single threshold on a single window. If false wakes ever show up
# in practice, requiring two consecutive windows over the line is the next
# move -- it rejects a one-frame phonetic coincidence while keeping the lower
# bar, at the cost of 80ms and some state. Not before there is one to fix.
THRESHOLD = 0.40
# Worth mentioning but not worth waking for. "Nothing happened" is the least
# useful thing a wake word can report: it covers a dead microphone, a wrong
# device, a mangled signal and a phrase that merely came close, and those need
# opposite fixes. A score above this gets printed so the difference is visible.
NOTICE = 0.2
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
