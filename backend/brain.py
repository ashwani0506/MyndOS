"""Tiered LLM routing with graceful degradation.

Every provider below is OpenAI-compatible, so one client shape covers all of
them -- only base_url, key and model name change. That is the entire router;
no gateway process, no extra daemon to be alive before login.

Ordering is deliberate:
  fast  -- starts local. No network round trip, no quota, no failure mode.
  deep  -- starts with the best model currently reachable and ends local, so
           the assistant degrades instead of going dark when quotas run out.

A provider that fails (429, auth, network) is skipped for COOLDOWN_SEC rather
than retried on every call. This is what makes an intermittently-live key
usable: we pay one failed request every 10 minutes, not one per command.
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

HERE = Path(__file__).parent
LOG_PATH = HERE / "logs" / "brain.jsonl"
COOLDOWN_SEC = 600


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    key_env: str
    fast: str | None = None
    deep: str | None = None

    def model_for(self, tier: str) -> str | None:
        return self.fast if tier == "fast" else self.deep

    def api_key(self) -> str | None:
        # Local servers need no key but the OpenAI SDK insists on a non-empty one.
        return os.getenv(self.key_env) or ("local" if self.base_url.startswith("http://localhost") else None)


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
        # Third-party reseller key: alive only part of the day, so it sits
        # first in `deep` and the cooldown absorbs the dead hours.
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


class NoProviderAvailable(RuntimeError):
    pass


def _log(record: dict) -> None:
    LOG_PATH.parent.mkdir(exist_ok=True)
    record["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _chain(tier: str) -> list[Provider]:
    """Providers for this tier that have a model, a key, and aren't cooling down."""
    now = time.time()
    out = []
    for name in CHAINS[tier]:
        p = PROVIDERS[name]
        if p.model_for(tier) and p.api_key() and _cooldown.get(name, 0) < now:
            out.append(p)
    return out


def think(prompt: str, tier: str = "deep", system: str = "", **kwargs) -> str:
    """Send `prompt` to the first provider in `tier`'s chain that answers."""
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]

    candidates = _chain(tier)
    if not candidates:
        raise NoProviderAvailable(
            f"No provider available for tier {tier!r}. Set a key in backend/.env "
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
            text = resp.choices[0].message.content or ""
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
            return text
        except Exception as e:
            _cooldown[p.name] = time.time() + COOLDOWN_SEC
            errors.append(f"{p.name}: {type(e).__name__}: {e}")
            _log({"tier": tier, "provider": p.name, "ok": False, "error": str(e)})

    raise NoProviderAvailable(
        f"Every provider for tier {tier!r} failed:\n  " + "\n  ".join(errors)
    )


def status() -> str:
    """Human-readable view of which providers are configured and live."""
    now = time.time()
    lines = []
    for name, p in PROVIDERS.items():
        if not p.api_key():
            state = f"no key ({p.key_env})"
        elif _cooldown.get(name, 0) > now:
            state = f"cooling down {int(_cooldown[name] - now)}s"
        else:
            state = "ready"
        tiers = ",".join(t for t in ("fast", "deep") if p.model_for(t))
        lines.append(f"  {name:<11} [{tiers:<9}] {state}")
    return "\n".join(lines)
