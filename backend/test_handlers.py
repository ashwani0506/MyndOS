"""Self-check for the no-model handler layer. No network, no model, no mic.

Run: python test_handlers.py
"""

import re
from datetime import datetime

import handlers


def test_the_wake_word_does_not_stop_a_match():
    """main.py hands Whisper three seconds of pre-roll plus the command, so the
    wake word lands in the transcript -- and on either end of it, depending on
    whether he said "jarvis, what time is it" or "what time is it, jarvis".
    Both are the same command and neither may miss."""
    for said in (
        "what time is it",
        "Jarvis, what time is it?",
        "what time is it, jarvis",
        "hey jarvis what time is it",
        "OK Jarvis, what's the time?",
        "  what   time  is  it  ",
    ):
        assert handlers.answer(said), said


def test_both_spellings_of_a_contraction_match():
    """Whisper writes "what's" with an apostrophe or without, depending on the
    audio. A pattern list that only spells one is a pattern list with a hole."""
    assert handlers.answer("what's the time")
    assert handlers.answer("whats the time")
    assert handlers.answer("what is the time")


def test_a_real_question_is_not_hijacked():
    """The failure mode worth designing against. Each of these *contains* a
    handler phrase and must still reach the model, which is what fullmatch
    rather than search buys."""
    for said in (
        "what do you think about the date on this contract",
        "what time is it best to send a job application",
        "remind me what time the interview is",
        "is the time complexity of this loop quadratic",
        "what's the date format in this file",
        "tell me the time complexity",
    ):
        assert handlers.answer(said) is None, said


def test_nothing_matches_an_empty_or_wake_word_only_utterance():
    """A misfired wake word transcribes to "jarvis" and nothing else. That is
    not a command, and must not be answered as one."""
    for said in ("", "   ", "jarvis", "hey jarvis", "ok", "..."):
        assert handlers.answer(said) is None, repr(said)


def test_the_replies_are_written_for_the_ear():
    """These go to a speech engine, which reads "05:03" as "oh five oh three"
    and a day of "08" as "zero eight".

    Only the *leading* number matters. Padded minutes are correct and wanted --
    "5:01" is read "five oh one", which is how the time is actually said -- so
    this checks the hour and the day number, not every digit in the string.
    """
    name, reply = handlers.answer("what time is it")
    assert name == "time"
    assert re.match(r"^It's [1-9]\d?:\d\d (AM|PM)\.$", reply), reply

    name, reply = handlers.answer("what's the date")
    assert name == "date"
    now = datetime.now()
    # The weekday and month spelled out, the day bare -- not 2026-10-08.
    assert reply == f"It's {now.strftime('%A, %B')} {now.day}.", reply
    assert not re.search(r"\b0\d", reply), reply


def test_the_handler_name_comes_back_for_the_log():
    """Without it the log says a turn was handled locally but not what by, and
    the first question asked of this layer is which patterns earn their place."""
    assert handlers.answer("what time is it")[0] == "time"
    assert handlers.answer("what day is it today")[0] == "date"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
