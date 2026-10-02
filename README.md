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
| Model router (`brain.py`) | **Working.** Tiered, multi-provider, cooldown-aware fallback, plus intent routing per turn. |
| Trusted execution layer (`tools.py`) | **Working.** Risk-tiered registry, confirmation gate, action log. |
| Tools | Clipboard read, scoped file read, confirmed file write, remember a fact. |
| Persona + user profile | **Working.** Plain markdown, re-read per request. |
| Text REPL (`agent.py`) | **Working.** Full tool-calling loop. |
| Voice → agent | **Working.** `main.py` drives the same `Agent` as the REPL, so every voice command goes through the gate and the log. |
| Long-term memory | **Working.** Markdown notes on disk, recalled into the prompt before the model sees the turn. |
| Measurement (`measure.py`) | **Working.** Reads the brain log back: tier latency, routing split, and whether the router nets out positive. |
| Intent router / VAD / local TTS | Intent router and VAD **working**; wake word and TTS are still Vosk and pyttsx3 — see Roadmap. |

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

### Which tier a turn gets

Every command used to pay `deep`-tier latency, so "what's on my clipboard"
cost the same as "what do you think of this design". `brain.classify()` now
picks the tier per turn with a one-word call to the local model, given the
utterance and the previous reply as context — the context matters, because
"what do you think?" is four words and looks reflexive on its own.

Two decisions shape it:

- **It only runs when the fast chain starts local.** Classifying over the
  network costs a round trip to save one, which is a coin flip. Without Ollama
  the router is a no-op and logs why.
- **Every unclear outcome is `deep`.** An unparseable reply, a dead provider,
  an exception — all land on `deep`, which is what the assistant did before the
  router existed. Wrong towards `deep` costs a second of latency; wrong towards
  `fast` costs a confidently wrong answer spoken aloud. Those are not the same
  mistake, so the defaults aren't symmetric.

Each decision is logged to `logs/brain.jsonl` with its latency. `/fast` and
`/deep` in the REPL override it.

### Does the router pay for itself

`measure.py` reads the log back and answers that, because the router was built
on an argument and an argument isn't a number:

```bash
python backend/measure.py      # or /stats in the REPL
```

It reports p50/p95 latency per tier, what classification costs, the fast/deep
split, and a per-turn net. Three things it does deliberately:

- **Classification is excluded from fast-tier answer latency.** It's a
  fast-tier call that isn't a fast-tier answer, and counting them together
  would make the router look like it pays for itself whether or not it does.
- **Percentiles, not means.** One cold model load drags a mean somewhere no
  real turn ever was.
- **It can return a verdict against the router.** If classifying costs more
  than the tier gap saves, the report says so in those words. A measurement
  that can only confirm the decision isn't a measurement.

What it can't measure is whether a given decision was *correct* — turns routed
`fast` are easier turns, so the comparison is observational, and the report
labels itself as an estimate rather than an experiment.

---

## Knowing when he stopped talking

The loop used to record for exactly three seconds after the wake word, so
*"what time is it"* and a twelve-word question cost the same wall clock — and
the short one, which is most of them, paid for silence. That fixed window is
larger than the entire tier gap the router exists to save, which is why this
came before any further work on the model layer.

`vad.py` ends the capture after 0.7s of quiet instead. Energy-based, ~40 lines,
no new dependency — numpy was already here for the audio path.

Two things make it work rather than merely exist:

- **The threshold is calibrated per command, not hard-coded.** A laptop fan, an
  air conditioner and a quiet room are three different rooms. The level is read
  off the audio already sitting in the rolling buffer — ambient by definition,
  since it is what the mic heard *before* the wake word. It's the 20th
  percentile of frame loudness rather than the mean, because that buffer ends
  with the wake word and so isn't pure ambience; a mean would be dragged up by
  the speech in it and set a threshold that then ignores the next sentence.
- **Only quiet *after* speech ends the capture.** A pause for breath is shorter
  than the hang time, which is the whole reason that constant isn't smaller.
  Cutting someone off mid-sentence is a worse failure than half a second of
  latency, so the two aren't tuned symmetrically.

`capture()` takes its frame reader as an argument, so `test_vad.py` scripts a
microphone out of a string — `"!!..!!!"` is a sentence with a pause in it — and
the whole decision is testable with no audio device. The two tests that matter
are the two failure modes: cutting him off at a pause, and recording an empty
room until the ceiling.

Whisper now sees 3s of pre-roll plus the command instead of a flat 13s. The
pre-roll is load-bearing rather than padding: Vosk only reports an utterance
once it has ended, so by the time *"what time is it, jarvis"* fires the wake
word, the command has already been spoken.

ponytail: RMS can't tell speech from a slammed door, so a loud noise can hold
the capture open to its ceiling. Fine for a desk mic in a room with one person;
`webrtcvad` classifies frames of the same size if that stops being true.

---

## Security model

The assistant runs as a normal user account and never requests elevation.
Capabilities are tiered rather than granted wholesale — `tools.py` enforces
this, and no tool reaches the model except through it:

| Tier | Examples | Handling |
|---|---|---|
| `SAFE` | Clipboard read, scoped project file read | Execute |
| `EXPLICIT` | Screen capture, audio beyond wake-word detection | Only when my own words asked for it. Each tool declares the phrases that unlock it; the grant is per turn and per tool, so asking for a screenshot doesn't also switch the microphone on. |
| `CONFIRM` | Send a message, submit a form, delete/overwrite, spend money, install software, write outside a scoped folder | Prompt every time. No "remember this choice", and `user_initiated` does not substitute for consent. |

Two rules that constrain the whole design:

1. **Anything the agent reads is data, not instructions.** An `EXPLICIT` tool is
   authorised only by words in my own utterance: `tools.unlocked_by()` computes
   the grant from the raw transcript before the model sees it, and the model's
   tool calls are checked against that set. A file or web page can ask for a
   screenshot all it likes — it cannot put the word in my mouth, because read
   content is not an input to the grant. `CONFIRM` tools stop for a human
   regardless. Registering an `EXPLICIT` tool with no unlock phrases raises at
   import, so the tier can't quietly contain something unreachable.
2. **Every tool call is logged** to `backend/logs/actions.jsonl`: name,
   arguments, outcome. Refusals and denials are logged too — those are the
   interesting ones.

### Asking in a place I can answer

The confirmation prompt is a terminal question in the REPL and a window
(`ask_dialog`, stdlib tkinter) on the voice path. That split isn't cosmetic:
this is meant to start at login, where there is no terminal attached, so a
`stdin` prompt would block forever on input that can never arrive — hanging
the assistant and taking the confirmation gate down with it. **A gate that
hangs is a gate that gets removed**, which is the actual failure mode.

The dialog fails closed in every direction. Closing it, Escape, a 60s timeout
and Tk failing to open at all are each a *no*; the only yes is a click on
Allow. There's deliberately no keyboard default — Enter on a dialog I didn't
read shouldn't be able to send an email — and no fallback to `input()` when Tk
is unavailable, since that would reintroduce the hang this exists to remove.

Scoped paths are resolved before the check, so `../` cannot walk out of scope.
`python backend/test_tools.py` exercises the gate: every tier, traversal, and
the log. `python backend/test_agent.py` covers the loop that drives it, with the
model faked — including that content the agent *reads* cannot unlock an
`EXPLICIT` tool, which is the prompt-injection case run end to end.

---

## Memory

One fact per markdown file in `backend/memory/`, named by date and first few
words. The store is the folder: inspecting what it remembers is opening a
directory, correcting a fact is editing a file, forgetting one is deleting it.
No database to reconcile against, no export step.

Recall happens **before the model sees the turn**, not as a tool call — the
utterance is scored against every note and the best three are appended to the
system prompt. That costs no extra round trip, and it works with a small local
model that's unreliable at deciding to search.

Scoring is inverse document frequency over the words: a term that appears in
every note is worth almost nothing, so *"what did I say about the router"*
isn't dragged around by *"what"* and *"the"*. That's a stopword list I never
have to maintain, and it adapts to whatever I actually write about.

No index, deliberately. A scan of a few hundred short notes is well under a
millisecond from page cache, and the version with an index has to handle
staleness, a database file, and — the one that decided it — escaping. Voice
transcripts arrive full of apostrophes and dashes, every one of which is a
syntax error to SQLite's `MATCH`. There is nothing to escape here because
nothing is a query language.

Recalled notes are wrapped and labelled as recollection, the same way a file
read is labelled as data. A note can't authorise anything either way — that
comes from my utterance — but it doesn't get to pose as a system rule.

---

## Setup

Requires Python 3.12+, and [Ollama](https://ollama.com) if you want the parts
of this that depend on a local model. It's optional in the sense that nothing
crashes without it, but two things quietly switch off: there is no offline
floor, so an exhausted quota means no assistant rather than a slower one, and
**intent routing stops entirely** — `classify()` refuses to classify over the
network, so every turn pays deep latency. `/status` says so in those words
when it happens, rather than leaving you to infer it.

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
registry with each tool's risk tier. `/stats` reports routing and latency.
`/fast <message>` and `/deep <message>` override the router for one turn.

Voice mode (wake word "Jarvis"):

```bash
python backend/main.py
```

On a 4GB GPU, keep the LLM on CUDA and let Whisper run on CPU — `base.en` in
int8 transcribes a short command in well under a second and leaves the VRAM free.

---

## Roadmap

1. **More tools** — calendar read, scoped screen capture, web search
   (read-only). Calendar goes in over a published secret ICS URL rather than
   OAuth: read-only is all the permission tiers allow unprompted anyway, so an
   Azure app registration and a token refresh held by a login-start process buy
   nothing until something needs to *write*. Screen capture will be the first
   real `EXPLICIT` tool; the authorisation path it needs is already built and
   tested.
2. **Voice rebuild** — VAD-terminated capture is in. Still to go: openWakeWord
   replacing Vosk (which runs full ASR continuously just to hear one word),
   Piper replacing pyttsx3, and a spoken filler on `deep` turns so a 2s think
   doesn't read as a hang.
3. **Recall on demand** — memory is auto-injected only. A `recall` tool would
   let the model search with a better query than the raw utterance. Worth it
   once auto-recall measurably misses; not before.
4. Persona tuning last.

---

## Layout

```
backend/
  brain.py        model router — tiers, fallback chain, cooldown, service windows
  tools.py        execution layer — risk tiers, confirmation gate, action log
  memory.py       long-term memory — markdown notes, scored recall, remember tool
  measure.py      reads brain.jsonl back — tier latency, routing split, net
  agent.py        conversation loop + text REPL
  persona.md      voice and behaviour (edit freely)
  profile.md      who I am, current projects (edit freely)
  main.py         voice loop: wake word → STT → agent → speech
  vad.py          ends a capture when he stops talking, on a calibrated level
  transcriber.py  faster-whisper wrapper
  tts.py          shared pyttsx3 engine
  rolling_buffer.py
  test_brain.py   router self-check (no network, no keys needed)
  test_tools.py   gate self-check (no network, no real writes)
  test_agent.py   tool-loop self-check (brain faked, no network)
  test_memory.py  recall self-check (notes in a temp dir)
  test_measure.py report self-check (synthetic log in a temp dir)
  test_vad.py     capture self-check (microphone scripted from a string)
  memory/         one markdown file per remembered fact — gitignored
  logs/           brain.jsonl, actions.jsonl — gitignored
```

## License

MIT
