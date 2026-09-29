# MyndOS

A voice-first desktop assistant that runs locally on Windows, starts at login,
and is built so that **the model can propose an action but never executes it directly** —
a separate execution layer decides what's permitted.

Single-user by design. This is my daily driver and my portfolio piece, so the
status table below is honest about what is built and what isn't.

---

## Status

| Component | State |
|---|---|
| Wake word → STT → TTS loop | **Working.** Vosk wake word, faster-whisper transcription, pyttsx3 speech. |
| Model router (`brain.py`) | **Working.** Tiered, multi-provider, cooldown-aware fallback. |
| Trusted execution layer (`tools.py`) | **Working.** Risk-tiered registry, confirmation gate, action log. |
| Tools | Clipboard read, scoped file read, confirmed file write. |
| Persona + user profile | **Working.** Plain markdown, re-read per request. |
| Text REPL (`agent.py`) | **Working.** Full tool-calling loop. |
| Voice → agent | **Working.** `main.py` drives the same `Agent` as the REPL, so every voice command goes through the gate and the log. |
| Long-term memory | Not built. |
| Intent router / VAD / local TTS | Not built — voice loop is a rebuild, see Roadmap. |

---

## The model layer

The constraint that shaped this: **it has to work all day on a student budget,
including offline, without a paid API.**

`brain.py` is ~120 lines and routes across providers that are all
OpenAI-compatible, so one client shape covers every one of them — only
`base_url`, key and model name change. No gateway, no second daemon that has to
be alive before a login-start assistant can work.

Two tiers, each an ordered chain:

```
fast  →  ollama (local) → groq → gemini
deep  →  claude → groq → gemini → openrouter → ollama (local)
```

- **`fast` starts local.** No network round trip, no quota, no failure mode.
  Intent classification and reflexive answers never leave the machine.
- **`deep` ends local.** When every free quota is exhausted the assistant
  degrades instead of going dark.
- **Failures cool down, they don't retry.** A provider that returns 429 or fails
  auth is skipped for 10 minutes, so one wasted request replaces one per command.
- **Known-dead hours are skipped on the clock.** A provider can declare a
  `window` — the local-time range its key actually serves. Outside it the
  provider is never tried at all. The two mechanisms compose: the window covers
  hours that are reliably dead, the cooldown covers a key draining early inside
  its window.

The window is what makes a part-time key worth having in first position. A
reseller key that only serves 16:30–18:30 gives me Sonnet during those two hours
and costs exactly nothing — not even one failed request — for the other 22.

Every provider uses my own key on its own published free tier. No credential
pooling, no free-tier aggregation across throwaway accounts, no TLS
interception — all of which are the reason I did not adopt an off-the-shelf
routing gateway for this.

Adding a provider is one entry in `PROVIDERS` and one line in `CHAINS`.

---

## Security model

The assistant runs as a normal user account and never requests elevation.
Capabilities are tiered rather than granted wholesale — `tools.py` enforces
this, and no tool reaches the model except through it:

| Tier | Examples | Handling |
|---|---|---|
| `SAFE` | Clipboard read, scoped project file read | Execute |
| `EXPLICIT` | Screen capture, audio beyond wake-word detection | Only on a direct command from me. A model-initiated call is refused, so nothing the agent *read* can switch these on. |
| `CONFIRM` | Send a message, submit a form, delete/overwrite, spend money, install software, write outside a scoped folder | Prompt every time. No "remember this choice", and `user_initiated` does not substitute for consent. |

Two rules that constrain the whole design:

1. **Anything the agent reads is data, not instructions.** Tool calls the model
   produces carry `user_initiated=False` — the flag is set at the call site in
   `agent.py`, not by the model — so a web page or file cannot reach an
   `EXPLICIT` tool, and a `CONFIRM` tool still stops for a human.
2. **Every tool call is logged** to `backend/logs/actions.jsonl`: name,
   arguments, outcome. Refusals and denials are logged too — those are the
   interesting ones.

Scoped paths are resolved before the check, so `../` cannot walk out of scope.
`python backend/test_tools.py` exercises the gate: every tier, traversal, and
the log. `python backend/test_agent.py` covers the loop that drives it, with the
model faked — including that a confirmation callback actually reaches the gate.

---

## Setup

Requires Python 3.12+. Optional but recommended: [Ollama](https://ollama.com)
for the local tier.

```bash
pip install -r backend/requirements.txt
cp .env.example .env    # fill in whichever keys you have
```

All keys are optional — the router uses what's present and skips the rest.
For the local floor:

```bash
ollama pull qwen3:4b
```

Text mode:

```bash
python backend/agent.py
```

`/status` shows which providers are configured and live. `/tools` lists the
registry with each tool's risk tier. `/fast <message>` forces the cheap tier.

Voice mode (wake word "Jarvis"):

```bash
python backend/main.py
```

On a 4GB GPU, keep the LLM on CUDA and let Whisper run on CPU — `base.en` in
int8 transcribes a short command in well under a second and leaves the VRAM free.

---

## Roadmap

1. **Memory** — SQLite FTS5 over markdown notes. Hand-editable and inspectable;
   embeddings only if keyword recall demonstrably falls short.
2. **More tools** — calendar read, scoped screen capture, web search
   (read-only). Screen capture is the first `EXPLICIT` tool, and nothing sets
   `user_initiated=True` yet, so it needs that path built with it.
3. **Voice rebuild** — the current loop takes ~6–8s to first action and runs full
   ASR continuously. Replacing with openWakeWord (near-zero idle compute) →
   VAD-terminated capture instead of a fixed 3s window → Piper for local TTS.
   An intent router belongs here too: every command currently pays `deep`-tier
   latency because there's nothing classifying them.
4. Persona tuning last.

---

## Layout

```
backend/
  brain.py        model router — tiers, fallback chain, cooldown, service windows
  tools.py        execution layer — risk tiers, confirmation gate, action log
  agent.py        conversation loop + text REPL
  persona.md      voice and behaviour (edit freely)
  profile.md      who I am, current projects (edit freely)
  main.py         voice loop: wake word → STT → agent → speech
  transcriber.py  faster-whisper wrapper
  tts.py          shared pyttsx3 engine
  rolling_buffer.py
  test_brain.py   router self-check (no network, no keys needed)
  test_tools.py   gate self-check (no network, no real writes)
  test_agent.py   tool-loop self-check (brain faked, no network)
  logs/           brain.jsonl, actions.jsonl — gitignored
```

## License

MIT
