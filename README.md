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
| Persona + user profile | **Working.** Plain markdown, re-read per request. |
| Text REPL (`agent.py`) | **Working.** |
| Command dispatch | **Placeholder.** Two hardcoded keywords. Replaced by the planner + execution layer next. |
| Trusted execution layer | Not built — next up. |
| Action log | Partial: LLM calls are logged, tool calls don't exist yet. |
| Long-term memory | Not built. |
| Intent router / VAD / local TTS | Not built — voice loop is a rebuild, see Roadmap. |
| Tauri UI | Empty shell, parked until there's a memory/log inspector to put in it. |

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
  auth is skipped for 10 minutes. This is what makes an intermittently-available
  key usable: one wasted request every 10 minutes instead of one per command.

Every provider uses my own key on its own published free tier. No credential
pooling, no free-tier aggregation across throwaway accounts, no TLS
interception — all of which are the reason I did not adopt an off-the-shelf
routing gateway for this.

Adding a provider is one entry in `PROVIDERS` and one line in `CHAINS`.

---

## Security model

The assistant runs as a normal user account and never requests elevation.
Capabilities are tiered rather than granted wholesale:

| Tier | Examples | Handling |
|---|---|---|
| Always-on | Clipboard/selection read, scoped project file read | Execute |
| Explicit invocation | Screen capture, audio beyond wake-word detection | Only on a direct command, never continuous |
| Always confirmed | Send a message, submit a form, delete/overwrite, spend money, install software, write outside a scoped folder | Confirmation every time, no exceptions |

Two rules that constrain the whole design:

1. **Anything the agent reads is data, not instructions.** Web pages, files and
   screen contents cannot trigger a consequential tool call — only I can.
2. **Every tool call is logged** to human-readable JSONL: what was called, with
   what input, what came back.

The tiering and the confirmation gate are specified but **not yet implemented** —
that is the next piece of work, and it lands before any consequential tool does.

---

## Setup

Requires Python 3.12+. Optional but recommended: [Ollama](https://ollama.com)
for the local tier.

```bash
pip install -r backend/requirements.txt
cp backend/.env.example backend/.env    # fill in whichever keys you have
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

`/status` shows which providers are configured and live. `/fast <message>`
forces the cheap tier.

Voice mode (wake word "Jarvis"):

```bash
python backend/main.py
```

On a 4GB GPU, keep the LLM on CUDA and let Whisper run on CPU — `base.en` in
int8 transcribes a short command in well under a second and leaves the VRAM free.

---

## Roadmap

1. **Trusted execution layer** — tool registry with a risk tier per tool, the
   confirmation gate, JSONL action log. Before any real tool ships.
2. **Memory** — SQLite FTS5 over markdown notes. Hand-editable and inspectable;
   embeddings only if keyword recall demonstrably falls short.
3. **First real tools** — clipboard read, then scoped file read/write.
4. **Voice rebuild** — the current loop takes ~6–8s to first action and runs full
   ASR continuously. Replacing with openWakeWord (near-zero idle compute) →
   VAD-terminated capture instead of a fixed 3s window → Piper for local TTS.
5. Remaining skills one at a time; persona tuning last.

---

## Layout

```
backend/
  brain.py        model router — tiers, fallback chain, cooldown
  agent.py        conversation loop + text REPL
  persona.md      voice and behaviour (edit freely)
  profile.md      who I am, current projects (edit freely)
  main.py         voice loop: wake word → STT → dispatch
  transcriber.py  faster-whisper wrapper
  tts.py          shared pyttsx3 engine
  rolling_buffer.py
  commands.py     placeholder dispatch, being replaced
frontend/MyndOS/  Tauri shell, parked
```

## License

MIT
