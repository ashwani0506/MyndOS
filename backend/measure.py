"""Reads the logs back and says whether the router -- and now the voice loop --
actually pay off.

The router was built on an argument -- classifying cheaply beats paying deep
latency on every lookup. An argument isn't a number, and the log already has
the raw material, so this turns it into one.

What it can honestly measure: what classification costs, what each tier costs,
and how often each is chosen. What it cannot: whether a routing decision was
*correct*. Turns routed `fast` are easier turns, so comparing their latency to
`deep` turns is observational, not a controlled comparison -- the report says
so rather than quietly presenting it as an experiment.

Two logs, read separately and never mixed:

  brain.jsonl  model calls, routing decisions, and turns answered with no
               model at all -- this file's original job.
  voice.jsonl  the stages the model calls sit inside: ack, record window,
               transcription, speech. Written by main.py.

They are kept apart on purpose. A voice stage is not a model call, and giving
it a fake `tier` to squeeze it into the tables below would bend every
percentile in this report around data that isn't a model call. The two meet
only in the voice report's headline, where a whole turn is the sum of both.

ponytail: one pass, whole file in memory. A year of heavy use is a few MB, so
the day that stops being fine is a long way off; stream it then.

Run: python measure.py
"""

import json
import time
from collections import Counter
from pathlib import Path

import brain

HERE = Path(__file__).parent
VOICE_LOG = HERE / "logs" / "voice.jsonl"

# Fixed order, so the report reads top-to-bottom in the order the turn happens
# rather than in whatever order the log happens to hold.
VOICE_STAGES = (
    "teardown",   # closing the Vosk stream
    "calibrate",  # noise floor off the rolling buffer
    "ack",        # "Yes, sir" -- guessed at before this existed
    "capture",    # mic open until the VAD ends it
    "transcribe", # Whisper
    "action",     # the agent: routing, model, tools (brain.jsonl has the detail)
    "speak",      # TTS of the reply
)


def load(path=None) -> tuple[list[dict], int]:
    """Parse the log. Returns (records, unparseable_lines).

    Bad lines are counted rather than raised on: a half-written final line
    after a crash shouldn't be the reason you can't read the other 10,000.
    """
    path = path or brain.LOG_PATH
    if not path.exists():
        return [], 0

    records, bad = [], 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            bad += 1
    return records, bad


def log(record: dict) -> None:
    """Append one turn's voice timings. Same shape as brain._log, same reason:
    appending has to be independent of anyone remembering to read it back."""
    VOICE_LOG.parent.mkdir(exist_ok=True)
    record["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with VOICE_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _pct(xs: list[int], p: int) -> int | None:
    """Nearest-rank percentile, None when there's no data.

    Percentiles rather than means throughout: one cold model load or one
    provider timeout drags a mean somewhere no real turn ever was.
    """
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(p / 100 * len(s)))]


def _row(label: str, xs: list[int]) -> str:
    p50, p95 = _pct(xs, 50), _pct(xs, 95)
    if p50 is None:
        return f"  {label:<16}{'--':>7}{'--':>8}{0:>7}"
    return f"  {label:<16}{p50:>7}{p95:>8}{len(xs):>7}"


def report(records: list[dict], bad: int = 0) -> str:
    calls = [r for r in records if "tier" in r]
    routes = [r for r in records if "router" in r]

    def ms(tier: str) -> list[int]:
        # Missing `purpose` means a call logged before the router existed, and
        # every one of those was an answer.
        return [
            r["ms"]
            for r in calls
            if r.get("ok")
            and r["tier"] == tier
            and r.get("purpose", "answer") == "answer"
            and "ms" in r
        ]

    stamps = [r["ts"] for r in records if "ts" in r]
    out = [
        f"{len(calls)} model calls, {len(routes)} routing decisions"
        + (f", {bad} unreadable lines" if bad else "")
    ]
    if stamps:
        out.append(f"{min(stamps)[:10]} to {max(stamps)[:10]}")
    if not records:
        return "Nothing logged yet. Talk to it first: python agent.py"

    decided = Counter(r["router"] for r in routes)
    total = sum(decided.values())
    out.append("\nRouting")
    for tier in ("fast", "deep"):
        n = decided[tier]
        share = f"{round(100 * n / total)}%" if total else "--"
        out.append(f"  {tier:<16}{n:>7}{share:>8}")

    # A skip is the router declining to run at all. Worth surfacing separately:
    # a report full of these means the local model is down, not that the
    # assistant is choosing deep on the merits.
    skips = Counter(r["why"] for r in routes if "why" in r)
    for why, n in skips.most_common():
        out.append(f"  skipped: {why[:40]:<7}{n:>7}")

    # Turns that never reached a model at all. Counted apart from the fast/deep
    # split for the same reason the voice stages live in another file: a turn
    # with no model call is not a routing decision, and folding these into the
    # percentages above would describe a classifier that never ran on them.
    #
    # This is the cheapest latency in the system and the easiest to overstate,
    # so it reports a count and not a saving. What a handled turn costs is a
    # regex; what it would have cost is whatever tier it would have been routed
    # to, which is not observable precisely because it wasn't.
    handled = Counter(r["handled"] for r in records if "handled" in r)
    if handled:
        out.append(f"\nAnswered with no model{sum(handled.values()):>7}")
        for name, n in handled.most_common():
            out.append(f"  {name:<16}{n:>7}")

    classify = [r["ms"] for r in routes if "ms" in r]
    fast, deep = ms("fast"), ms("deep")
    out.append(f"\nLatency (ms){'p50':>11}{'p95':>8}{'n':>7}")
    out.append(_row("classify", classify))
    out.append(_row("fast answer", fast))
    out.append(_row("deep answer", deep))

    out.append("\nVerdict")
    c50, f50, d50 = _pct(classify, 50), _pct(fast, 50), _pct(deep, 50)
    if None in (c50, f50, d50) or not total:
        out.append("  Not enough turns on both tiers yet to compare.")
    else:
        gap = d50 - f50
        saved = round(decided["fast"] / total * gap - c50)
        out.append(f"  Classifying costs {c50}ms on every turn.")
        out.append(f"  A fast answer lands {gap}ms sooner than a deep one.")
        out.append(
            f"  At {round(100 * decided['fast'] / total)}% routed fast, that nets "
            f"{saved:+}ms per turn."
        )
        if saved <= 0:
            out.append("  Negative: the router is costing more than it saves.")
        out.append(
            "  Observational -- fast turns are easier turns, so this is an\n"
            "  estimate of the saving, not a measured one."
        )

    out.append(f"\nProviders{'calls':>10}{'failed':>8}")
    by_provider = Counter(r["provider"] for r in calls if "provider" in r)
    failed = Counter(r["provider"] for r in calls if not r.get("ok") and "provider" in r)
    for name, n in by_provider.most_common():
        out.append(f"  {name:<16}{n:>7}{failed[name]:>8}")

    return "\n".join(out)


def _stage(turns: list[dict], name: str) -> list[int]:
    """Every turn's value for one stage. `stages` is a dict per turn, so a turn
    that predates a stage simply has no entry and contributes a missing sample
    rather than a wrong one."""
    return [r["stages"][name] for r in turns if name in r.get("stages", {})]


def _drop_warmup(records: list[dict], gap_ms: int = 30_000, least: int = 2) -> tuple[list[dict], int]:
    """Drop the first turn of each burst, and count them.

    A burst is turns less than 30s apart -- that's one sitting, and it is the
    *first* turn of it that pays for Whisper loading into memory and pyttsx3
    waking its device. Every turn after that is the warm machine, which is the
    one worth measuring. Excluding the first of each burst removes the cold
    outlier and keeps everything else, which is the opposite of dropping turns
    for being close together.

    A line with an unparseable timestamp can't be placed on the clock, so it is
    kept and treated as starting fresh -- the next turn is measured against
    nothing rather than against a moment that never existed.

    Refuses when it would leave fewer than `least` turns. A log with three
    commands spread across a day is three bursts of one turn each, and cutting
    that to one sample would trade a real measurement for a clean one. When it
    declines, the count says so and nothing is hidden.

    ponytail: a gap is a guess at "was this cold", not a measurement of it.
    Log a `cold` flag from the loop if warm/cold ever needs to be exact.
    """
    kept, prev_t = [], None
    for r in records:
        try:
            t = time.mktime(time.strptime(r["ts"], "%Y-%m-%dT%H:%M:%S"))
        except (KeyError, ValueError):
            kept.append(r)  # unreadable timestamp is not grounds for exclusion
            prev_t = None   # nor is it a moment the next turn can be timed from
            continue
        if prev_t is None or (t - prev_t) * 1000 >= gap_ms:
            prev_t = t  # first of a burst, and therefore the cold one
            continue
        prev_t = t
        kept.append(r)

    cold = len(records) - len(kept)
    if len(kept) < least:
        return records, 0  # too little left to be a measurement; keep it all
    return kept, cold


def voice_report(records: list[dict], bad: int = 0) -> str:
    """The voice loop's half: where a spoken turn's wall clock actually goes.

    The router report answers "is this faster than always using the deep tier".
    This answers the question the VAD commit argued in prose: how much of the
    wait is the machine, and how much is the hang it sits through after I stop
    talking. Same standard -- an argument isn't a number.
    """
    turns, warmup = _drop_warmup(records)
    if not turns:
        return (
            "No voice turns logged yet. Talk to it first: python main.py\n"
            "(Nothing appears here until the loop has run once with a wake word.)"
        )

    out = [
        f"{len(turns)} voice turns"
        + (f", {bad} unreadable lines" if bad else "")
        + (f", {warmup} first-turn outliers excluded" if warmup else "")
    ]

    out.append(f"\nStage (ms){'p50':>11}{'p95':>8}{'n':>7}")
    for stage in VOICE_STAGES:
        out.append(_row(stage, _stage(turns, stage)))

    # The headline. A stage table answers "which part is slow"; this answers
    # "how long after I stopped talking did it answer", which is the question
    # the person waiting actually has. Sum-of-stages would be the same number
    # minus whatever went unmeasured, and quietly reading low.
    spoke = [r["spoke_ms"] for r in turns if "spoke_ms" in r]
    if spoke and len(spoke) == len(turns):
        quiet = [r["total_ms"] - r["spoke_ms"] for r in turns]
        t50 = _pct([r["total_ms"] for r in turns], 50)
        p50, p95 = _pct(quiet, 50), _pct(quiet, 95)
        out.append("\nAfter he stops talking")
        out.append(f"  {'to a spoken reply':<16}{p50:>7}{p95:>8}{len(quiet):>7}")
        if t50:
            out.append(
                f"\n  The {p50}ms above is silence spent waiting on the machine,\n"
                f"  not on him speaking -- {round(100 * p50 / t50)}% of a "
                f"{t50}ms turn. The rest is him talking."
            )
    else:
        out.append(
            "\n  No spoke_ms on some turns, so the silent gap can't be computed.\n"
            "  (Older lines, logged before the split existed.)"
        )

    # Counted whenever the field is present, not only when something was heard.
    # A log where every turn came back empty is the one worth seeing, and it is
    # exactly the case a `if any words` guard would hide.
    logged = [r for r in turns if "transcript" in r]
    if logged:
        words = [r for r in logged if r["transcript"].strip()]
        empty = len(logged) - len(words)
        line = f"\n  {len(words)} of {len(logged)} turns produced words"
        if empty:
            line += f", {empty} heard nothing recognisable"
        if not words:
            line += "\n  Nothing was transcribed at all -- check the mic and the\n" \
                    "  VAD threshold before reading anything else here."
        out.append(line)
    return "\n".join(out)


if __name__ == "__main__":
    print(report(*load()))
    print()
    print(voice_report(*load(VOICE_LOG)))
