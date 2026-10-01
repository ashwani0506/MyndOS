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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
