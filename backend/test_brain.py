"""Self-check for the router's provider selection. No network, no keys needed.

Run: python test_brain.py
"""

import os
import time
from datetime import time as t

import brain


def test_window_bounds_are_inclusive():
    w = ("16:30", "18:30")
    assert not brain._in_window(w, t(16, 29))
    assert brain._in_window(w, t(16, 30))   # inclusive start
    assert brain._in_window(w, t(17, 0))
    assert brain._in_window(w, t(18, 30))   # inclusive end
    assert not brain._in_window(w, t(18, 31))
    assert not brain._in_window(w, t(4, 30))  # am/pm confusion would show here


def test_window_drops_provider_from_chain_outside_hours():
    os.environ["CLAUDE_API_KEY"] = "x"
    os.environ["GROQ_API_KEY"] = "x"
    brain._cooldown.clear()

    claude = brain.PROVIDERS["claude"]
    assert claude.window, "claude is expected to carry a service window"

    inside = brain._in_window(claude.window)
    named = "claude" in [p.name for p in brain._chain("deep")]
    assert named == inside, (
        f"claude in chain={named} but in-window={inside}; the clock check isn't wired up"
    )


def test_chain_respects_tier_and_order():
    os.environ["GROQ_API_KEY"] = "x"
    os.environ["GEMINI_API_KEY"] = "x"
    os.environ.pop("OPENROUTER_API_KEY", None)
    os.environ.pop("CLAUDE_API_KEY", None)
    brain._cooldown.clear()

    # ollama needs no key, so it's always in the fast chain -- and first.
    fast = [p.name for p in brain._chain("fast")]
    assert fast == ["ollama", "groq", "gemini"], fast

    # claude/openrouter have no key set, so they drop out of deep entirely.
    deep = [p.name for p in brain._chain("deep")]
    assert deep == ["groq", "gemini", "ollama"], deep

    # openrouter has no `fast` model, so it never appears there even with a key.
    os.environ["OPENROUTER_API_KEY"] = "x"
    assert "openrouter" not in [p.name for p in brain._chain("fast")]
    assert "openrouter" in [p.name for p in brain._chain("deep")]


def test_cooldown_skips_then_restores():
    os.environ["GROQ_API_KEY"] = "x"
    brain._cooldown.clear()

    brain._cooldown["groq"] = time.time() + brain.COOLDOWN_SEC
    assert "groq" not in [p.name for p in brain._chain("deep")]

    brain._cooldown["groq"] = time.time() - 1  # expired
    assert "groq" in [p.name for p in brain._chain("deep")]


def test_no_providers_raises_rather_than_hanging():
    for key in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY", "CLAUDE_API_KEY"):
        os.environ.pop(key, None)
    brain._cooldown.clear()
    # Only ollama survives (no key needed); cool it down and nothing is left.
    brain._cooldown["ollama"] = time.time() + brain.COOLDOWN_SEC
    assert brain._chain("deep") == []
    try:
        brain.think("hello")
    except brain.NoProviderAvailable:
        pass
    else:
        raise AssertionError("expected NoProviderAvailable")


if __name__ == "__main__":
    test_window_bounds_are_inclusive()
    test_window_drops_provider_from_chain_outside_hours()
    test_chain_respects_tier_and_order()
    test_cooldown_skips_then_restores()
    test_no_providers_raises_rather_than_hanging()
    print("ok")
