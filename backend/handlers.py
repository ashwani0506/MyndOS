"""Commands answered without a model at all.

The fastest model call is the one that never happens. "What time is it" has
exactly one right answer, and the standard library already knows it. Sending it
to a language model costs a round trip to be told something `datetime` could
have said instantly -- and, on a small local model, to be told it slightly
wrong, confidently, out loud.

This sits in front of the agent loop: `answer()` returns a reply when the
utterance is one of these, and `None` the rest of the time, which is the normal
case. Nothing here is a tool call. A tool is something the *model* decides to
use; this is the layer that decides the model is not needed.

Two rules keep it from doing harm:

  Every pattern is a full match on the whole utterance, never a substring. The
  failure mode worth designing against is hijacking a turn that wanted real
  thought -- "what do you think about the date on this contract" contains "the
  date" and must still reach the model. `fullmatch` is what makes that safe.

  It only answers what is genuinely unambiguous. "Read my clipboard" is
  deliberately *not* here: nine times in ten that means "tell me what this says",
  which is interpretation, and interpretation is the model's job. The test for
  belonging here is not "can I write a regex for it" but "is there exactly one
  right answer, and does a library already know it".

ponytail: a hand-written pattern list, which is fine at this size and would not
be at fifty. The upgrade is an intent classifier, and `answer()` is the only
call site that would change.
"""

import re
from datetime import datetime

# Stripped before matching: the wake word survives into the transcript, because
# main.py hands Whisper three seconds of pre-roll and the command together, so
# "jarvis what time is it" and "what time is it jarvis" are both routine.
_NOISE = re.compile(r"\b(hey|ok|okay|please|jarvis)\b")
# Removed outright rather than replaced with a space, which would turn "what's"
# into "what s" and match nothing. Whisper writes the curly one about as often
# as the straight one, so both are here.
_APOSTROPHE = re.compile(r"['’]")
_PUNCT = re.compile(r"[^a-z0-9 ]+")
_SPACE = re.compile(r"\s+")


def _normalise(utterance: str) -> str:
    """Lower-case, punctuation gone, wake word gone, single spaces.

    Apostrophes first and separately: Whisper writes "what's" with one and
    "whats" without, depending on the audio, and a pattern list that has to
    spell both is a pattern list with a hole in it.
    """
    s = _APOSTROPHE.sub("", utterance.lower())
    s = _PUNCT.sub(" ", s)
    return _SPACE.sub(" ", _NOISE.sub(" ", s)).strip()


def _time() -> str:
    # %I is zero-padded and Windows has no %-I, so 05:03 PM would be read aloud
    # as "oh five oh three". lstrip is the portable fix.
    return f"It's {datetime.now().strftime('%I:%M %p').lstrip('0')}."


def _date() -> str:
    now = datetime.now()
    # Written for the ear: the day number bare, not "08", which a speech engine
    # reads as "zero eight".
    return f"It's {now.strftime('%A, %B')} {now.day}."


HANDLERS = (
    ("time", re.compile(r"what(s| is)? the time|what time is it( now)?|tell me the time"), _time),
    ("date", re.compile(r"what(s| is)? (the|todays) date|what day is it( today)?"), _date),
)


def answer(utterance: str) -> tuple[str, str] | None:
    """`(handler name, reply)` when no model is needed, else None.

    The name comes back so the caller can log which one fired. Without it the
    log says a turn was handled locally but not what by, and the first question
    anyone asks of this layer is which patterns actually earn their place.
    """
    text = _normalise(utterance)
    for name, pattern, fn in HANDLERS:
        if pattern.fullmatch(text):
            return name, fn()
    return None
