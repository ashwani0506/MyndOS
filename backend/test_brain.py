"""Self-check for the router's provider selection. No network, no keys needed.

Run: python test_brain.py
"""

import os
import socket
import tempfile
import time
from datetime import time as t
from pathlib import Path

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


def test_status_does_not_report_a_dead_local_port_as_ready():
    """A local server needs no key, so every other check passes for one that
    isn't installed -- and status() called it "ready" while the router it
    fronts silently did nothing. Loopback only; no network."""
    assert not brain._listening("http://localhost:1/v1")

    with socket.socket() as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        assert brain._listening(f"http://127.0.0.1:{srv.getsockname()[1]}/v1")


def test_status_says_so_when_routing_is_off():
    """With no local model the router returns deep on every turn. That has to
    be visible in /status, not inferred from a log full of skips."""
    os.environ["GROQ_API_KEY"] = "x"
    brain._cooldown.clear()
    brain._cooldown["ollama"] = time.time() + brain.COOLDOWN_SEC
    assert "Intent routing is OFF" in brain.status()


# --------------------------------------------------------------------------
# Intent routing. brain.think is faked; nothing here reaches a model.
# --------------------------------------------------------------------------


def _answers(text, *, boom=False):
    """Fake brain.think. Records the calls it received into the returned list."""
    calls = []

    def fake(prompt, tier="deep", system="", **kwargs):
        calls.append({"prompt": prompt, "tier": tier, "system": system, **kwargs})
        if boom:
            raise RuntimeError("model died")
        return text

    brain.think = fake
    return calls


def _local_fast_chain():
    """Make the fast chain start on ollama, which is what classify requires."""
    brain._cooldown.clear()
    assert brain._chain("fast")[0].local, "expected ollama at the head of fast"


def test_a_reflexive_request_routes_to_fast():
    _local_fast_chain()
    calls = _answers("FAST")
    assert brain.classify("what's on my clipboard") == "fast"
    assert calls[0]["tier"] == "fast", "classification must not use the deep tier"


def test_an_open_ended_request_routes_to_deep():
    _local_fast_chain()
    _answers("DEEP")
    assert brain.classify("what do you think of this architecture") == "deep"


def test_a_thinking_model_reply_is_still_parsed():
    """qwen3 wraps answers in <think> blocks. /no_think asks it not to, but the
    template still emits the tags, and a stray "deep" inside them would flip
    the decision if they weren't stripped."""
    _local_fast_chain()
    _answers("<think>\nis this deep? no.\n</think>\n\nFAST")
    assert brain.classify("what time is it") == "fast"


def test_the_last_word_wins():
    """A chatty model saying "not FAST, this is DEEP" must not read as fast."""
    _local_fast_chain()
    _answers("This isn't FAST, it's DEEP.")
    assert brain.classify("design me a schema") == "deep"


def test_an_unparseable_reply_falls_back_to_deep():
    """Wrong towards deep costs a second. Wrong towards fast costs a confident
    bad answer spoken aloud, so every unclear case has to land here."""
    _local_fast_chain()
    _answers("I'm not sure what you mean?")
    assert brain.classify("anything") == "deep"


def test_a_dead_classifier_falls_back_to_deep():
    _local_fast_chain()
    _answers("FAST", boom=True)
    assert brain.classify("what's on my clipboard") == "deep"


def test_classify_does_not_pay_a_network_hop_to_save_one():
    """With no local model the classify call itself goes over the network, so
    routing would cost a round trip to save one. Bail without calling."""
    os.environ["GROQ_API_KEY"] = "x"
    brain._cooldown.clear()
    brain._cooldown["ollama"] = time.time() + brain.COOLDOWN_SEC
    assert not brain._chain("fast")[0].local

    calls = _answers("FAST")
    assert brain.classify("what's on my clipboard") == "deep"
    assert calls == [], "classified over the network"


def test_previous_reply_is_offered_as_context():
    """"What do you think?" is four words and looks reflexive on its own."""
    _local_fast_chain()
    calls = _answers("DEEP")
    brain.classify("what do you think", "I'd use SQLite over Postgres here.")
    assert "SQLite over Postgres" in calls[0]["prompt"]


if __name__ == "__main__":
    real_think, real_log = brain.think, brain.LOG_PATH
    brain.LOG_PATH = Path(tempfile.gettempdir()) / "myndos_test_brain.jsonl"
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            # Restored per test, not once at the end: the routing tests fake
            # think(), and test_no_providers calls it for real.
            brain.think = real_think
            fn()
    brain.think, brain.LOG_PATH = real_think, real_log
    print("ok")
