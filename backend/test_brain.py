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
    """The clock check, on a provider built here rather than whichever one in
    PROVIDERS happens to carry a window today.

    It used to assert against the real `claude` entry, which carried a
    reseller's two-hour slot. When that key went away so did the window, and a
    test of the mechanism failed for a reason that had nothing to do with the
    mechanism. Config is not a fixture.
    """
    os.environ["GROQ_API_KEY"] = "x"
    brain._cooldown.clear()

    timed = brain.Provider(
        "timed", "https://example.invalid/v1", "GROQ_API_KEY",
        deep="m", window=("16:30", "18:30"),
    )
    providers, chains = brain.PROVIDERS, brain.CHAINS
    brain.PROVIDERS = {**providers, "timed": timed}
    brain.CHAINS = {**chains, "deep": ["timed", "groq"]}
    try:
        named = "timed" in [p.name for p in brain._chain("deep")]
        assert named == brain._in_window(timed.window), (
            f"in chain={named} but in-window={brain._in_window(timed.window)}; "
            f"the clock check isn't wired up"
        )
    finally:
        brain.PROVIDERS, brain.CHAINS = providers, chains


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


def test_a_local_provider_is_not_benched_like_a_cloud_one():
    """A cloud failure is quota or auth: it persists, and each retry burns a
    real request. A local one is "not running" -- sub-millisecond to retest --
    or "still loading the model", which will succeed shortly.

    Found the hard way on a machine with no cloud keys, where ollama is the
    only provider: one cold model load slower than the timeout benched it for
    ten minutes, and ten minutes of the only provider is no assistant at all.
    """
    assert brain.PROVIDERS["ollama"].cooldown == brain.LOCAL_COOLDOWN_SEC
    assert brain.PROVIDERS["groq"].cooldown == brain.COOLDOWN_SEC
    assert brain.LOCAL_COOLDOWN_SEC * 10 <= brain.COOLDOWN_SEC


def test_a_local_model_gets_time_to_load():
    """A cold 2.6GB model reaching VRAM is a slow path, not a failure. The
    cloud default has to stay short -- a warm endpoint silent for 30s is
    down."""
    assert brain.PROVIDERS["ollama"].timeout >= 120.0
    assert brain.PROVIDERS["groq"].timeout == 30.0


def test_the_local_model_is_pinned_in_vram():
    """Cold 154,630ms, warm 431ms -- the same model answering the same prompt.
    That 358x is the whole difference between an assistant and a wait, and
    ollama drops an idle model after 5 minutes by default, so an assistant used
    a few times an hour pays the cold load on *every* command.

    Sent in the request body rather than read from OLLAMA_KEEP_ALIVE: on Windows
    the desktop app auto-starts the server, so a `setx` only lands after a
    reboot, and a fresh clone inherits nothing. `keep_alive` must not go to a
    cloud endpoint, which would reject the unknown field.
    """
    assert brain.PROVIDERS["ollama"].keep_alive == -1
    for name in ("groq", "gemini", "openrouter", "claude"):
        assert brain.PROVIDERS[name].keep_alive is None, name



    """qwen3 emits <think> inline in `content`. classify() stripped it for its
    own use and nothing else did, so every answer carried it -- and on the
    voice path the whole monologue went to the speech engine. A 4B model
    arguing with itself about how many words are in "hi", read aloud."""
    assert brain._spoken("<think>a ramble</think>Hi there.") == "Hi there."
    assert brain._spoken("<think>\nmulti\nline\n</think>\n\nHi.") == "Hi."
    # Truncated mid-thought by max_tokens: an opening tag with no close.
    # Whatever followed was never going to be the answer.
    assert brain._spoken("<think>and then I thought, what if") == ""
    # Nothing to strip is the common case and must come back untouched.
    assert brain._spoken("Just an answer.") == "Just an answer."


def test_one_client_per_provider_is_reused():
    """Measured: 2.2s to make a call on a freshly built client, 45ms on one that
    already existed. complete() built a new one every call, so every model call
    in the system paid a connection handshake -- two of them per voice turn,
    inside a pipeline being tuned in milliseconds elsewhere.

    Same arguments must hand back the same object, and a rotated key must not.
    """
    a = brain._client("http://localhost:11434/v1", "local", 30.0)
    assert brain._client("http://localhost:11434/v1", "local", 30.0) is a
    assert brain._client("http://localhost:11434/v1", "rotated", 30.0) is not a



    """Thinking is the point of the deep tier and pure latency on the fast one.

    Measured, warm, same model and prompt: 48ms with it off, and with it on the
    32-token budget went entirely to the monologue and `content` came back
    empty -- so every turn fell through to deep and the router had never once
    routed anything. `reasoning_effort` is the standard OpenAI parameter;
    "/no_think" in the prompt and `think: false` through /v1 both silently did
    nothing on ollama 0.35.
    """
    ollama, groq = brain.PROVIDERS["ollama"], brain.PROVIDERS["groq"]
    assert ollama.reasoning_effort == "none"
    assert brain._extra_body(ollama, "fast")["reasoning_effort"] == "none"
    # Deep is exactly when the monologue is worth paying for.
    assert "reasoning_effort" not in brain._extra_body(ollama, "deep")
    # ...but keep_alive is not tier-specific: a deep local call wants the model
    # resident just as much.
    assert brain._extra_body(ollama, "deep")["keep_alive"] == -1
    assert groq.reasoning_effort is None


def test_a_local_only_field_never_reaches_a_cloud_endpoint():
    """`keep_alive` is Ollama's. Sent to Groq it is an unknown field, so the
    empty dict is load-bearing rather than merely tidy."""
    for name in ("groq", "gemini", "openrouter", "claude"):
        p = brain.PROVIDERS[name]
        assert brain._extra_body(p, "fast") == {}, name
        assert brain._extra_body(p, "deep") == {}, name


def test_status_names_the_env_file_it_could_not_find():
    """Five "no key" lines are five symptoms of one cause, and the cause is
    almost never five missing keys. This exact confusion cost a run: the file
    existed one directory below, and nothing said so."""
    for key in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY", "CLAUDE_API_KEY"):
        os.environ.pop(key, None)
    brain._cooldown.clear()

    real = brain.ENV_PATH
    brain.ENV_PATH = Path(tempfile.mkdtemp()) / "nope.env"
    try:
        assert "No .env at" in brain.status()
    finally:
        brain.ENV_PATH = real


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
