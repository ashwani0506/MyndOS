"""Self-check for the wake-word buffering. No model, no onnxruntime, no mic.

The scoring is openWakeWord's job and is not re-tested here. What is tested is
the seam between the caller's block size and the model's fixed 80ms window,
which is this file's own logic and the place a wake word would get silently
dropped.

Run: python test_wakeword.py
"""

import numpy as np

import wakeword


def _pcm(n: int, value: int = 1000) -> bytes:
    return np.full(n, value, dtype=np.int16).tobytes()


def test_a_block_splits_into_whole_windows():
    """main.py reads 80ms blocks, so the common case is one window per block
    and nothing left over."""
    windows, tail = wakeword._windows(np.empty(0, dtype=np.int16), _pcm(wakeword.FRAME))
    assert len(windows) == 1
    assert len(windows[0]) == wakeword.FRAME
    assert tail.size == 0


def test_the_remainder_is_carried_to_the_next_call():
    """A block that isn't a multiple of the window leaves audio over, and that
    audio is the start of a phrase as often as not."""
    size = np.empty(0, dtype=np.int16)
    windows, tail = wakeword._windows(size, _pcm(wakeword.FRAME + 500))
    assert len(windows) == 1
    assert tail.size == 500

    # The carried 500 join the next block rather than being dropped.
    windows, tail = wakeword._windows(tail, _pcm(wakeword.FRAME - 500))
    assert len(windows) == 1
    assert tail.size == 0


def test_a_phrase_split_across_two_blocks_is_not_lost_on_the_seam():
    """The failure this function exists to prevent. Audio arrives in whatever
    size the caller chose and the model wants exactly 1280 samples, so without
    carrying the remainder a wake word landing on a block boundary would be
    scored as two halves and recognised as neither."""
    half = wakeword.FRAME // 2
    tail = np.empty(0, dtype=np.int16)
    collected = []
    for _ in range(4):
        windows, tail = wakeword._windows(tail, _pcm(half))
        collected += windows
    # Four half-windows in, two whole ones out, and every sample accounted for.
    assert len(collected) == 2
    assert sum(len(w) for w in collected) + tail.size == 4 * half


def test_a_block_too_short_to_score_yields_nothing_yet():
    """Not an error -- just not enough audio. It has to accumulate rather than
    score a padded window, which would be scoring silence the mic never heard."""
    windows, tail = wakeword._windows(np.empty(0, dtype=np.int16), _pcm(100))
    assert windows == []
    assert tail.size == 100


def test_a_large_block_yields_every_window_in_it():
    """The old block size was 8000 samples. If this only ever scored the first
    window of a block, a wake word later in it would be missed."""
    windows, tail = wakeword._windows(np.empty(0, dtype=np.int16), _pcm(8000))
    assert len(windows) == 8000 // wakeword.FRAME
    assert tail.size == 8000 % wakeword.FRAME


def test_the_threshold_sits_in_the_measured_gap():
    """The threshold is set from two measurements, and this is where they live
    so that moving it has to argue with them.

        213s of recorded speech with no wake phrase   worst 0.359
        his own attempts, weakest to strongest        0.45 .. 0.82

    Above the first and below the second, or it is the wrong number. At
    openWakeWord's suggested 0.5 he was saying the phrase three and four times
    a turn, and those repeats were what read as a delay between turns.
    """
    worst_false_positive = 0.359
    weakest_real_attempt = 0.45

    assert wakeword.THRESHOLD > worst_false_positive, "will wake on other speech"
    assert wakeword.THRESHOLD < weakest_real_attempt, "will miss his quiet attempts"
    # And NOTICE has to stay below the line, or a near miss is never reported.
    assert wakeword.NOTICE < wakeword.THRESHOLD


def test_the_phrase_has_a_hey_on_the_front():
    """The pretrained model is "hey jarvis", and handlers.py strips both words
    from a transcript. If this constant ever disagrees with that, the wake
    phrase ends up inside the command the model is asked to answer."""
    assert wakeword.PHRASE == "hey jarvis"
    import handlers
    assert handlers._normalise("hey jarvis what time is it") == "what time is it"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
