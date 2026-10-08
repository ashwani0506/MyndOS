"""Self-check for the transcription confidence gate. No model, no audio, no mic.

`confidence()` takes anything with the three attributes faster-whisper puts on
a segment, so the scoring is testable without loading Whisper or recording
anything.

Run: python test_transcriber.py
"""

from types import SimpleNamespace

import transcriber as T


def seg(start, end, avg_logprob, no_speech_prob=0.1, text="x"):
    return SimpleNamespace(
        start=start, end=end, avg_logprob=avg_logprob,
        no_speech_prob=no_speech_prob, text=text,
    )


def test_a_clean_command_passes():
    """Around -0.2 is what a good short capture actually scores. If this starts
    failing, the thresholds have drifted away from the microphone."""
    logprob, no_speech = T.confidence([seg(0.0, 1.5, -0.2)])
    assert logprob > T.MIN_AVG_LOGPROB
    assert no_speech < T.MAX_NO_SPEECH_PROB


def test_a_long_sentence_outvotes_a_short_fragment():
    """Duration-weighted, so half a second of noise can't drag down four
    seconds of a perfectly clear sentence -- or rescue the reverse."""
    good_long, bad_short = seg(0.0, 4.0, -0.2), seg(4.0, 4.5, -2.0)
    weighted, _ = T.confidence([good_long, bad_short])
    unweighted = (-0.2 + -2.0) / 2
    assert weighted > unweighted, "the short fragment is being given equal say"
    assert weighted > T.MIN_AVG_LOGPROB, "a clear sentence must survive a blip"


def test_a_guess_made_out_of_noise_is_rejected():
    """The case this exists for: Whisper returns a plausible sentence built from
    a cough and a fan, and without this it went to the model and was answered
    confidently, out loud."""
    logprob, _ = T.confidence([seg(0.0, 2.0, -1.8)])
    assert logprob < T.MIN_AVG_LOGPROB


def test_a_segment_that_is_not_speech_is_rejected_on_its_own():
    """no_speech_prob is the worst across segments, not an average: one segment
    confidently flagged as a door is enough, and averaging would dilute exactly
    the signal worth acting on."""
    _, no_speech = T.confidence([seg(0.0, 2.0, -0.2, no_speech_prob=0.95)])
    assert no_speech > T.MAX_NO_SPEECH_PROB


def test_nothing_to_measure_is_not_a_pass():
    """"No segments" must not read as "nothing wrong" -- the gate has to fail
    closed, or an empty capture sails through as perfectly confident."""
    logprob, no_speech = T.confidence([])
    assert logprob < T.MIN_AVG_LOGPROB
    assert no_speech > T.MAX_NO_SPEECH_PROB


def test_a_zero_length_segment_does_not_divide_by_zero():
    """Whisper occasionally reports start == end. A crash here would take the
    whole voice turn with it, for a segment carrying no information."""
    logprob, _ = T.confidence([seg(1.0, 1.0, -0.3)])
    assert logprob == -0.3


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
