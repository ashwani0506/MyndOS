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
import socket
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import OpenAI

HERE = Path(__file__).parent
LOG_PATH = HERE / "logs" / "brain.jsonl"
COOLDOWN_SEC = 600
# A local failure is not a cloud failure. See Provider.cooldown.
LOCAL_COOLDOWN_SEC = 30

# Repo root, explicitly. A bare load_dotenv() walks up the directory tree and
# would pick up an unrelated project's .env from a parent folder.
ENV_PATH = HERE.parent / ".env"
load_dotenv(ENV_PATH)


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    key_env: str
    fast: str | None = None
    deep: str | None = None
    window: tuple[str, str] | None = None  # local-time range this key actually serves
    # Seconds to wait on one call. Generous for a local model, which has to
    # read itself off disk into VRAM on the first call of a session -- a slow
    # path, not a failure. A warm cloud endpoint that hasn't answered in 30s
    # isn't going to.
    timeout: float = 30.0
    # Sent on fast-tier calls only: thinking is the point of the deep tier and
    # pure latency on a reflexive one. "none" switches a reasoning model's
    # monologue off.
    #
    # This is the standard OpenAI parameter, not a vendor hack, which is why it
    # can live on the one client shape every provider shares. Three things that
    # look like they should do this and do not, on ollama 0.35 + qwen3:1.7b:
    # a literal "/no_think" in the prompt (the chat template stopped reading
    # it), `think: false` through /v1 (the compat layer drops it; it works only
    # on the native /api/chat), and `chat_template_kwargs`. Each returned 249
    # characters of reasoning and an empty `content`.
    reasoning_effort: str | None = None
    # Ollama only: how long to keep the model in VRAM, -1 being "never unload".
    # Sent in the request body rather than relied on from OLLAMA_KEEP_ALIVE,
    # because the env var has to be set for whichever process launched the
    # server -- on Windows the desktop app auto-starts, so a `setx` after that
    # does nothing until a reboot, and a fresh clone inherits none of it. The
    # latency decision belongs next to the latency, not in a shell profile.
    #
    # Measured here: cold 154,630ms, warm 431ms. Same model, same prompt.
    keep_alive: int | None = None

    def model_for(self, tier: str) -> str | None:
        return self.fast if tier == "fast" else self.deep

    @property
    def local(self) -> bool:
        return self.base_url.startswith("http://localhost")

    @property
    def cooldown(self) -> float:
        """How long to bench this provider after a failure.

        A cloud failure is quota or auth. It persists, and each retry burns a
        real request, so ten minutes is right. A local failure is one of two
        other things: not running, which costs under a millisecond to retest,
        or still loading the model, which will succeed shortly. Ten minutes is
        wrong for both.

        It is worse than wrong when local is the only provider configured --
        which is the state of a fresh clone with no keys, and was the state
        this was found in. There, benching ollama once means ten minutes of no
        assistant at all, triggered by a cold model load that was going to
        finish in twenty seconds.
        """
        return LOCAL_COOLDOWN_SEC if self.local else COOLDOWN_SEC

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


def _listening(base_url: str, timeout: float = 0.2) -> bool:
    """Is anything actually accepting connections there?

    Only asked of local providers, and only by status(). A local server needs
    no key, so api_key() hands one out unconditionally -- which means every
    other check passes for a provider that isn't installed, and status()
    reports a dead port as "ready". That is a lie in the one place whose whole
    job is telling you what is live.

    Not wired into _chain(): a refused connection to localhost costs under a
    millisecond and the cooldown absorbs it, so paying a probe on every call to
    save that would be the more expensive mistake.
    """
    url = urlparse(base_url)
    try:
        with socket.create_connection((url.hostname, url.port or 80), timeout):
            return True
    except OSError:
        return False


PROVIDERS = {
    p.name: p
    for p in [
        Provider(
            "ollama",
            "http://localhost:11434/v1",
            "OLLAMA_API_KEY",
            fast="qwen3:1.7b",
            deep="qwen3:1.7b",
            # A cold model coming off disk into VRAM is slow once per load --
            # measured at 70s here, on a 4GB card. ponytail: a flat ceiling,
            # not a cold/warm distinction -- a genuinely stuck local server now
            # costs two minutes instead of thirty seconds. Worth it while local
            # is the only offline floor; split it if a warm call ever
            # legitimately needs this long.
            #
            # The ceiling is not the fix for the cold load, only the thing that
            # stops it being recorded as a failure. The fix is keeping the model
            # resident: see OLLAMA_KEEP_ALIVE in the README.
            timeout=120.0,
            reasoning_effort="none",
            keep_alive=-1,
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
        # Anthropic direct. No `window`: that belonged to a third-party reseller
        # key that served a fixed two-hour slot, and keeping its schedule on a
        # real key would skip Claude for 22 hours a day for no reason. The
        # mechanism stays (see _in_window) -- it just has no user right now.
        Provider(
            "claude",
            os.getenv("CLAUDE_BASE_URL", "https://api.anthropic.com/v1/"),
            "CLAUDE_API_KEY",
            deep="claude-sonnet-5",
        ),
    ]
}

CHAINS = {
    "fast": ["ollama", "groq", "gemini"],
    "deep": ["claude", "groq", "gemini", "openrouter", "ollama"],
}

_cooldown: dict[str, float] = {}

# A reasoning block, as qwen3 and friends emit it inline in `content`.
_THINK = re.compile(r"<think>.*?</think>", re.S)
# The same thing cut off by max_tokens: an opening tag with no close.
_THINK_OPEN = re.compile(r"<think>.*$", re.S)


def _spoken(text: str) -> str:
    """The part of a reply meant for a human.

    Belt to `reasoning_effort`'s braces. Ollama 0.35 returns the monologue in a
    separate `reasoning` field, so nothing inline survives to be stripped here
    -- but an older ollama, and openrouter's deepseek, put it in `content` with
    the tags still on. Two regexes is a cheap insurance premium against a model
    reading its own notes out loud.

    Done here because this is the one point every reply passes through. At the
    call sites it is one `or ""` away from being forgotten again, and the cost
    of forgetting is paid out loud.

    The unterminated case is truncation: max_tokens can cut a reply off
    mid-thought, leaving `<think>` with no close, and the tag-to-end strip is
    right because whatever followed was never going to be the answer.
    """
    return _THINK_OPEN.sub("", _THINK.sub("", text)).strip()


def _extra_body(p: Provider, tier: str) -> dict:
    """Provider-specific request fields. Empty dict for a plain cloud endpoint,
    which is the point -- `keep_alive` sent to Groq is an unknown field.

    `reasoning_effort` is fast-tier only: a deep turn is exactly when the
    monologue is worth paying for.
    """
    body = {}
    if p.keep_alive is not None:
        body["keep_alive"] = p.keep_alive
    if tier == "fast" and p.reasoning_effort:
        body["reasoning_effort"] = p.reasoning_effort
    return body


class NoProviderAvailable(RuntimeError):
    pass


@lru_cache(maxsize=None)
def _client(base_url: str, api_key: str, timeout: float) -> OpenAI:
    """One client per provider, reused for the life of the process.

    Building a fresh `OpenAI()` per call cost 2.2 seconds before the model saw a
    single token; the same call on a client that already existed took 45ms. The
    cost is establishing the connection, not constructing the object, so it was
    paid once per *call* rather than once per provider -- and a voice turn makes
    two calls, which put four and a half seconds of handshake inside a turn that
    was being optimised in milliseconds elsewhere.

    Keyed on the key as well as the URL, so rotating a credential builds a new
    client rather than reusing one authenticated with the old one. Five
    providers means at most five live clients.
    """
    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)


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


def complete(messages: list[dict], tier: str = "deep", purpose: str = "answer", **kwargs):
    """Send `messages` to the first provider in `tier`'s chain that answers.

    Returns the raw completion so callers can read tool_calls. Use think() if
    you only want text.

    `purpose` only reaches the log. It's there because a classification is a
    fast-tier call that isn't a fast-tier *answer*, and measuring them together
    would make the router look like it pays for itself when it might not.

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
            client = _client(p.base_url, p.api_key(), p.timeout)
            resp = client.chat.completions.create(
                model=p.model_for(tier),
                messages=messages,
                **({"extra_body": extra} if (extra := _extra_body(p, tier)) else {}),
                **kwargs,
            )
            # Before anything downstream can read it, speak it, or store it in
            # the conversation history as if the model had said it.
            msg = resp.choices[0].message
            if msg.content:
                msg.content = _spoken(msg.content)
            _log(
                {
                    "tier": tier,
                    "purpose": purpose,
                    "provider": p.name,
                    "model": p.model_for(tier),
                    "ms": round((time.time() - started) * 1000),
                    "tokens": getattr(resp.usage, "total_tokens", None),
                    "ok": True,
                }
            )
            return resp
        except Exception as e:
            _cooldown[p.name] = time.time() + p.cooldown
            errors.append(f"{p.name}: {type(e).__name__}: {e}")
            _log(
                {
                    "tier": tier,
                    "purpose": purpose,
                    "provider": p.name,
                    "ok": False,
                    "error": str(e),
                }
            )

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
        # 32 and not 2 for headroom: a reply truncated before the word falls
        # through to deep every time, which is how this spent a while looking
        # like a routing decision instead of an empty string. The switch that
        # makes 32 enough is the provider's `reasoning_effort` -- without it the
        # budget goes entirely to the monologue and `content` arrives blank.
        answer = think(
            prompt,
            tier="fast",
            system=CLASSIFY_SYSTEM,
            purpose="classify",
            max_tokens=32,
            temperature=0,
        )
    except Exception as e:
        _log({"router": "deep", "why": f"{type(e).__name__}: {e}"})
        return "deep"

    # Last word wins, so "this isn't FAST, it's DEEP" reads correctly. Neither
    # word present leaves both at -1, which is not greater than itself: deep.
    low = answer.lower()
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
        elif p.local and not _listening(p.base_url):
            # Checked before the cooldown, because "not running" is the cause
            # and "cooling down" is only its symptom.
            state = f"NOT RUNNING -- nothing listening on {p.base_url}"
        elif p.window and not _in_window(p.window):
            state = f"outside window {p.window[0]}-{p.window[1]}"
        elif _cooldown.get(name, 0) > now:
            state = f"cooling down {int(_cooldown[name] - now)}s"
        else:
            state = "ready"
        tiers = ",".join(t for t in ("fast", "deep") if p.model_for(t))
        lines.append(f"  {name:<11} [{tiers:<9}] {state}")

    # Five "no key" lines are five symptoms of one cause, and the cause is
    # almost never five missing keys. Say which file wasn't there, and -- the
    # one that actually happened -- say when the keys exist a directory below
    # and are being ignored for it.
    if not ENV_PATH.exists() and any(not p.api_key() for p in PROVIDERS.values()):
        lines.append(f"\n  No .env at {ENV_PATH}")
        stray = HERE / ".env"
        if stray.exists():
            lines.append(
                f"  There is one at {stray},\n"
                f"  which is not read. Move it up one level."
            )

    # The fast chain starting local is the precondition for routing at all, so
    # when it isn't met, say so here rather than leaving it to be inferred from
    # a log full of skips.
    chain = _chain("fast")
    if not chain or not chain[0].local:
        lines.append(
            "\n  Intent routing is OFF: the fast chain does not start local,\n"
            "  so every turn pays deep latency. /stats has the damage."
        )
    return "\n".join(lines)
