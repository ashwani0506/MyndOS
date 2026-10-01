"""Reads logs/brain.jsonl back and says whether the router actually pays off.

The router was built on an argument -- classifying cheaply beats paying deep
latency on every lookup. An argument isn't a number, and the log already has
the raw material, so this turns it into one.

What it can honestly measure: what classification costs, what each tier costs,
and how often each is chosen. What it cannot: whether a routing decision was
*correct*. Turns routed `fast` are easier turns, so comparing their latency to
`deep` turns is observational, not a controlled comparison -- the report says
so rather than quietly presenting it as an experiment.

ponytail: one pass, whole file in memory. A year of heavy use is a few MB, so
the day that stops being fine is a long way off; stream it then.

Run: python measure.py
"""

import json
from collections import Counter

import brain


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


if __name__ == "__main__":
    print(report(*load()))
