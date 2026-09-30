"""Tiered LLM routing with graceful degradation.

Every provider below is OpenAI-compatible, so one client shape covers all of
them -- only base_url, key and model name change. That is the entire router;
no gateway process, no extra daemon to be alive before login.

Ordering is deliberate:
  fast  -- starts local. No network round trip, no quota, no failure mode.
  deep  -- starts with the best model currently reachable and ends local, so
           the assistant degrades instead of going dark when quotas run out.

A provider that fails (429, auth, network) is skipped for COOLDOWN_SEC rather
than retried on every call. A provider with a known service `window` is skipped
outside it on the clock alone. The two compose: the window covers the hours a
key is reliably dead, the cooldown covers it draining early inside the window.

classify() picks the tier for a turn, so a lookup doesn't pay reasoning
latency. It only runs when the fast chain starts local, and every unclear
outcome falls back to `deep` -- see its docstring for why that direction.
"""

import datetime
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

HERE = Path(__file__).parent
LOG_PATH = HERE / "logs" / "brain.jsonl"
COOLDOWN_SEC = 600

# Repo root, explicitly. A bare load_dotenv() walks up the directory tree and
# would pick up an unrelated project's .env from a parent folder.
load_dotenv(HERE.parent / ".env")


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    key_env: str
    fast: str | None = None
    deep: str | None = None
    window: tuple[str, str] | None = None  # local-time range this key actually serves

    def model_for(self, tier: str) -> str | None:
        return self.fast if tier == "fast" else self.deep

    @property
    def local(self) -> bool:
        return self.base_url.startswith("http://localhost")

    def api_key(self) -> str | None:
        # Local servers need no key but the OpenAI SDK insists on a non-empty one.
        return os.getenv(self.key_env) or ("local" if self.local else None)


def _in_window(window: tuple[str, str], now: datetime.time | None = None) -> bool:
    """True if `now` falls inside a HH:MM-HH:MM local-time range.

    ponytail: same-day ranges only. A window crossing midnight ("22:00","02:00")
    reads as always-false; split it into two providers if you ever need one.
    """
    start, end = (datetime.time.fromisoformat(t) for t in window)
    return start <= (now or datetime.datetime.now().time()) <= end


PROVIDERS = {
    p.name: p
    for p in [
        Provider(
            "ollama",
            "http://localhost:11434/v1",
            "OLLAMA_API_KEY",
            fast="qwen3:4b",
            deep="qwen3:4b",
        ),
        Provider(
            "groq",
            "https://api.groq.com/openai/v1",
            "GROQ_API_KEY",
            fast="openai/gpt-oss-20b",
            deep="openai/gpt-oss-120b",
        ),
        Provider(
            "gemini",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
            "GEMINI_API_KEY",
            fast="gemini-2.5-flash-lite",
            deep="gemini-2.5-flash",
        ),
        Provider(
            "openrouter",
            "https://openrouter.ai/api/v1",
            "OPENROUTER_API_KEY",
            deep="deepseek/deepseek-chat-v3.1:free",
        ),
        # Third-party reseller key: serves a known 2-hour window, so the clock
        # skips it outright for the other 22 and the cooldown handles the pool
        # draining early inside it. Flip to ("04:30","06:30") if it's mornings.
        Provider(
            "claude",
            os.getenv("CLAUDE_BASE_URL", "https://api.anthropic.com/v1/"),
            "CLAUDE_API_KEY",
            deep="claude-sonnet-5",
            window=("16:30", "18:30"),
        ),
    ]
}

CHAINS = {
    "fast": ["ollama", "groq", "gemini"],
    "deep": ["claude", "groq", "gemini", "openrouter", "ollama"],
}

_cooldown: dict[str, float] = {}


class NoProviderAvailable(RuntimeError):
    pass


def _log(record: dict) -> None:
    LOG_PATH.parent.mkdir(exist_ok=True)
    record["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _chain(tier: str) -> list[Provider]:
    """Providers for this tier with a model, a key, in-window, not cooling down."""
    now = time.time()
    out = []
    for name in CHAINS[tier]:
        p = PROVIDERS[name]
        if not (p.model_for(tier) and p.api_key()):
            continue
        if p.window and not _in_window(p.window):
            continue
        if _cooldown.get(name, 0) < now:
            out.append(p)
    return out


def complete(messages: list[dict], tier: str = "deep", **kwargs):
    """Send `messages` to the first provider in `tier`'s chain that answers.

    Returns the raw completion so callers can read tool_calls. Use think() if
    you only want text.

    ponytail: a provider that rejects a kwarg (an older endpoint with no tool
    support, say) fails like any other error and eats a cooldown. Acceptable
    while every provider in CHAINS handles tools; revisit if one stops.
    """
    candidates = _chain(tier)
    if not candidates:
        raise NoProviderAvailable(
            f"No provider available for tier {tier!r}. Set a key in .env "
            f"(see .env.example) or start Ollama for the always-on local floor."
        )

    errors = []
    for p in candidates:
        started = time.time()
        try:
            client = OpenAI(base_url=p.base_url, api_key=p.api_key(), timeout=30.0)
            resp = client.chat.completions.create(
                model=p.model_for(tier), messages=messages, **kwargs
            )
            _log(
                {
                    "tier": tier,
                    "provider": p.name,
                    "model": p.model_for(tier),
                    "ms": round((time.time() - started) * 1000),
                    "tokens": getattr(resp.usage, "total_tokens", None),
                    "ok": True,
                }
            )
            return resp
        except Exception as e:
            _cooldown[p.name] = time.time() + COOLDOWN_SEC
            errors.append(f"{p.name}: {type(e).__name__}: {e}")
            _log({"tier": tier, "provider": p.name, "ok": False, "error": str(e)})

    raise NoProviderAvailable(
        f"Every provider for tier {tier!r} failed:\n  " + "\n  ".join(errors)
    )


def think(prompt: str, tier: str = "deep", system: str = "", **kwargs) -> str:
    """One-shot text in, text out."""
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    return complete(messages, tier=tier, **kwargs).choices[0].message.content or ""


# --------------------------------------------------------------------------
# Intent routing
# --------------------------------------------------------------------------

CLASSIFY_SYSTEM = """You route requests for a desktop voice assistant.

FAST: a short factual answer, a lookup, a direct instruction, reading or
acting on something. One obvious right answer, and little cost to being wrong.

DEEP: an opinion, a judgement, a comparison, a plan, a design or code
question, anything open-ended or multi-step, anything where a wrong answer
would mislead.

Reply with exactly one word: FAST or DEEP. When in doubt, reply DEEP."""

_THINK = re.compile(r"<think>.*?</think>", re.S)


def classify(utterance: str, context: str = "") -> str:
    """Pick a tier for one turn: "fast" for reflexive, "deep" for reasoning.

    The asymmetry that sets every default here: routing to `deep` when `fast`
    would have done costs a second of latency, while routing to `fast` when it
    wouldn't costs a confidently wrong answer spoken aloud. So an unparseable
    reply, a dead provider, or any exception lands on `deep` -- which is
    exactly what the assistant did before this function existed.
    """
    chain = _chain("fast")
    if not chain or not chain[0].local:
        # Classifying over the network costs a round trip to save one, which is
        # a coin flip at best. ponytail: relax this once brain.jsonl can show
        # whether a remote classify + fast answer really beats a deep answer.
        _log({"router": "deep", "why": "fast chain does not start local"})
        return "deep"

    prompt = f"Request: {utterance}"
    if context:
        prompt = f"Your previous reply was: {context}\n\n{prompt}"

    started = time.time()
    try:
        # /no_think is Qwen3's switch for skipping its reasoning block, which on
        # a one-word answer is pure latency. It's also why max_tokens is 32 and
        # not 2: the template still emits an empty <think></think> pair, and a
        # reply truncated before the word would fall through to deep every time.
        answer = think(
            f"{prompt}\n/no_think",
            tier="fast",
            system=CLASSIFY_SYSTEM,
            max_tokens=32,
            temperature=0,
        )
    except Exception as e:
        _log({"router": "deep", "why": f"{type(e).__name__}: {e}"})
        return "deep"

    # Last word wins, so "this isn't FAST, it's DEEP" reads correctly. Neither
    # word present leaves both at -1, which is not greater than itself: deep.
    low = _THINK.sub("", answer).lower()
    tier = "fast" if low.rfind("fast") > low.rfind("deep") else "deep"
    _log(
        {
            "router": tier,
            "ms": round((time.time() - started) * 1000),
            "said": low.strip()[:40],
        }
    )
    return tier


def status() -> str:
    """Human-readable view of which providers are configured and live."""
    now = time.time()
    lines = []
    for name, p in PROVIDERS.items():
        if not p.api_key():
            state = f"no key ({p.key_env})"
        elif p.window and not _in_window(p.window):
            state = f"outside window {p.window[0]}-{p.window[1]}"
        elif _cooldown.get(name, 0) > now:
            state = f"cooling down {int(_cooldown[name] - now)}s"
        else:
            state = "ready"
        tiers = ",".join(t for t in ("fast", "deep") if p.model_for(t))
        lines.append(f"  {name:<11} [{tiers:<9}] {state}")
    return "\n".join(lines)
