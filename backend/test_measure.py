"""Self-check for the log reader. No network, no real log touched.

The check that matters most here is that classification calls stay out of the
fast-answer latency. If they leaked in, every report would show the fast tier
as quicker than it is -- and the whole point of measuring was to stop trusting
the argument and look at the number.

Run: python test_measure.py
"""

import json
import tempfile
from pathlib import Path

import measure


def _log(*records) -> str:
    """Write records to a temp log and return the report built from it."""
    p = Path(tempfile.mkdtemp()) / "brain.jsonl"
    p.write_text(
        "".join(
            (r if isinstance(r, str) else json.dumps(r)) + "\n" for r in records
        ),
        encoding="utf-8",
    )
    return measure.report(*measure.load(p))


def _n(out: str, label: str) -> str:
    """The last column of a latency row, robust to the column widths moving."""
    return next(l for l in out.splitlines() if label in l).split()[-1]


def _call(tier, ms, purpose="answer", provider="ollama", ok=True):
    return {
        "tier": tier,
        "purpose": purpose,
        "provider": provider,
        "ms": ms,
        "ok": ok,
        "ts": "2026-10-01T12:00:00",
    }


def test_an_empty_log_is_not_an_error():
    assert "Nothing logged yet" in _log()
    assert measure.load(Path(tempfile.mkdtemp()) / "nope.jsonl") == ([], 0)


def test_a_half_written_line_does_not_lose_the_rest():
    """A crash mid-write leaves a truncated final line. One bad line must not
    be the reason the other ten thousand are unreadable."""
    out = _log(_call("fast", 100), '{"tier": "fast", "ms": 1', _call("fast", 200))
    assert "1 unreadable" in out
    assert _n(out, "fast answer") == "2"


def test_classification_is_not_counted_as_a_fast_answer():
    out = _log(
        _call("fast", 150, purpose="classify"),
        _call("fast", 150, purpose="classify"),
        _call("fast", 150, purpose="classify"),
        _call("fast", 400),
        _call("fast", 500),
    )
    assert _n(out, "fast answer") == "2", "classify calls leaked into the tier"


def test_calls_from_before_the_router_still_count_as_answers():
    """Lines logged before `purpose` existed have no such key, and every one of
    them was an answer. Dropping them would throw away the baseline."""
    out = _log({"tier": "deep", "provider": "groq", "ms": 2000, "ok": True})
    assert _n(out, "deep answer") == "1"


def test_a_failed_call_is_counted_but_not_timed():
    out = _log(_call("deep", 300), _call("deep", 50, provider="groq", ok=False))
    assert _n(out, "deep answer") == "1", "a failed call is not an answer latency"
    assert next(l for l in out.splitlines() if "groq" in l).split()[-1] == "1"


def test_percentiles_survive_a_single_sample():
    assert measure._pct([], 50) is None
    assert measure._pct([7], 50) == 7
    assert measure._pct([7], 95) == 7
    assert measure._pct([1, 2, 3, 4, 5], 50) == 3


def test_a_router_that_costs_more_than_it_saves_says_so():
    """The report has to be able to return a verdict against the router.
    One that can only confirm the decision isn't a measurement."""
    out = _log(
        {"router": "fast", "ms": 500, "ts": "2026-10-01T12:00:00"},
        {"router": "deep", "ms": 500, "ts": "2026-10-01T12:00:00"},
        _call("fast", 100),
        _call("deep", 200),
    )
    assert "costing more than it saves" in out
    assert "-450ms per turn" in out


def test_a_worthwhile_router_reports_the_saving():
    out = _log(
        {"router": "fast", "ms": 100, "ts": "2026-10-01T12:00:00"},
        {"router": "fast", "ms": 100, "ts": "2026-10-01T12:00:00"},
        _call("fast", 300),
        _call("deep", 2300),
    )
    assert "+1900ms per turn" in out  # 1.0 * (2300 - 300) - 100
    assert "costing more" not in out


def test_a_skipped_router_is_reported_with_its_reason():
    """A report full of skips means the local model is down, not that the
    assistant kept choosing deep on the merits. Those read identically in the
    tier counts, so the reason has to show up separately."""
    out = _log(
        {"router": "deep", "why": "fast chain does not start local", "ts": "x"},
        {"router": "deep", "why": "fast chain does not start local", "ts": "x"},
    )
    assert "skipped: fast chain does not start local" in out
    assert next(l for l in out.splitlines() if "skipped" in l).split()[-1] == "2"


def test_not_enough_data_refuses_to_give_a_verdict():
    out = _log({"router": "fast", "ms": 100, "ts": "x"}, _call("fast", 300))
    assert "Not enough turns" in out


# --------------------------------------------------------------------------
# The voice path. Separate log, separate report, deliberately no overlap.
# --------------------------------------------------------------------------


def _turn(total, spoke=1000, at="2026-10-02T12:00:00", transcript=None, **stages):
    """One voice.jsonl line. `transcript` is named explicitly because it lives
    beside `stages` rather than inside it -- passing it via **stages would put
    a string where the stage table expects milliseconds."""
    r = {"stages": stages, "total_ms": total, "spoke_ms": spoke, "ts": at}
    if transcript is not None:
        r["transcript"] = transcript
    return r


def _voice(*records) -> str:
    p = Path(tempfile.mkdtemp()) / "voice.jsonl"
    p.write_text(
        "".join((r if isinstance(r, str) else json.dumps(r)) + "\n" for r in records),
        encoding="utf-8",
    )
    return measure.voice_report(*measure.load(p))


def test_no_voice_turns_is_not_an_error():
    assert "No voice turns logged yet" in _voice()


def test_a_voice_stage_is_not_a_model_call():
    """The two logs exist apart so a stage can never land in a tier percentile.
    If voice lines ever reached report(), every latency number in it would be
    describing something that isn't a model call."""
    turn = _turn(4000, at="2026-10-02T12:00:00", capture=2500, action=900)
    out = measure.report([turn])
    assert "0 model calls" in out
    # It did see the record -- it just correctly declined to treat it as a call.
    assert "Nothing logged yet" not in out


def test_the_first_turn_of_a_burst_is_dropped_as_cold():
    """Whisper loading and pyttsx3 waking are paid on the first turn of a
    sitting and never again. In a 20-turn sample that one turn sits in the p95
    and describes a machine that doesn't exist."""
    out = _voice(
        # Sitting one: cold 6s transcribe, then warm.
        _turn(9000, at="2026-10-02T12:00:00", transcribe=6000),
        _turn(2000, at="2026-10-02T12:00:20", transcribe=400),
        # Sitting two, half an hour later: cold again, then warm.
        _turn(9000, at="2026-10-02T12:30:00", transcribe=6000),
        _turn(2000, at="2026-10-02T12:30:20", transcribe=400),
    )
    assert "2 first-turn outliers excluded" in out
    assert "2 voice turns" in out
    # Both 6000ms cold turns gone, so the p95 describes a warm machine.
    assert _n(out, "transcribe") == "2"


def test_a_sparse_log_is_left_alone_rather_than_gutted():
    """Three commands spread across a day is three bursts of one turn each.
    Filtering would leave a single sample -- trading a real measurement for a
    clean one, which is the wrong trade for a report whose whole job is honesty
    about noise."""
    out = _voice(
        _turn(5000, at="2026-10-02T09:00:00", transcribe=3000),
        _turn(5000, at="2026-10-02T13:00:00", transcribe=3000),
        _turn(5000, at="2026-10-02T19:00:00", transcribe=3000),
    )
    assert "3 voice turns" in out
    assert "excluded" not in out
    assert _n(out, "transcribe") == "3"


def test_an_unparseable_timestamp_is_kept_but_does_not_start_a_burst():
    """A line with a broken ts is still a turn, so it stays. It can't be placed
    on the clock either, so it doesn't become the reference the next turn is
    measured against."""
    out = _voice(
        _turn(2000, at="not-a-date", capture=1000),
        _turn(2000, at="2026-10-02T12:00:00", capture=1000),
    )
    assert "2 voice turns" in out


def test_the_headline_separates_waiting_from_talking():
    """The stage table says which part is slow. This says how long he waited in
    silence, which is the question the person standing there actually has."""
    out = _voice(
        _turn(5000, spoke=2000, at="2026-10-02T12:00:00", capture=2700),
        _turn(5000, spoke=2000, at="2026-10-02T12:00:10", capture=2700),
        _turn(5000, spoke=2000, at="2026-10-02T12:00:20", capture=2700),
    )
    assert _n(out, "to a spoken reply") == "2", "the burst leader is excluded"
    # 5000 total - 2000 spoken = 3000 of silence, 60% of the turn.
    assert "3000ms above is silence" in out
    assert "60% of a 5000ms turn" in out


def test_an_older_line_without_a_spoken_time_suppresses_the_headline():
    """Computing it from the turns that do have it would report a number from a
    subset while labelling it as all of them."""
    out = _voice(
        _turn(5000, spoke=2000, at="2026-10-02T12:00:00", capture=1000),
        _turn(5000, spoke=2000, at="2026-10-02T12:00:10", capture=1000),
        {"stages": {"capture": 1000}, "total_ms": 5000, "ts": "2026-10-02T12:00:20"},
    )
    assert "can't be computed" in out
    assert "silence" not in out


def test_a_stage_that_predates_the_log_is_a_missing_sample_not_a_zero():
    """A turn logged before a stage existed has no entry for it. Counting that
    as 0ms would drag the stage's p50 towards a latency never observed.

    The samples sit on kept turns on purpose: the burst leader is dropped, so a
    lone `speak` on the first turn would test nothing but the filter."""
    out = _voice(
        _turn(2000, at="2026-10-02T12:00:00", capture=1000),  # leader, dropped
        _turn(2000, at="2026-10-02T12:00:10", capture=1000, speak=800),
        _turn(2000, at="2026-10-02T12:00:20", capture=1000),
        _turn(2000, at="2026-10-02T12:00:30", capture=1000, speak=800),
        _turn(2000, at="2026-10-02T12:00:40", capture=1000),  # no speak
    )
    assert _n(out, "capture") == "4", "every kept turn has a capture"
    assert _n(out, "speak") == "2", "only the turns that have it are counted"


def test_a_turn_that_heard_nothing_is_counted_separately():
    out = _voice(
        _turn(2000, at="2026-10-02T12:00:00", transcript=""),  # leader, dropped
        _turn(2000, at="2026-10-02T12:00:10", transcript="what time is it"),
        _turn(2000, at="2026-10-02T12:00:20", transcript=""),
        _turn(2000, at="2026-10-02T12:00:30", transcript="and this"),
    )
    assert "2 of 3 turns produced words, 1 heard nothing recognisable" in out
    assert "Nothing was transcribed at all" not in out


def test_a_log_where_nothing_was_ever_heard_says_so():
    """Guarding this on `if any words` would hide the one log most worth
    seeing: the mic that never picked anything up. Every turn empty reads as a
    working report of a broken pipeline unless it's called out."""
    out = _voice(
        *[
            _turn(2000, at=f"2026-10-02T12:00:{s:02d}", transcript="")
            for s in (0, 10, 20)
        ]
    )
    assert "0 of 2 turns produced words" in out
    assert "Nothing was transcribed at all" in out


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
