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
| No-model handlers (`handlers.py`) | **Working.** Time and date answered from the standard library, with no model call at all. |
| Speech (`tts.py`) | **Working.** Fixed phrases pre-rendered to WAV and played from disk; an engine per utterance, which is what makes repeated speech work at all. |
| Transcription confidence gate | **Working.** A capture Whisper isn't confident about is discarded rather than answered. |
| Persona + user profile | **Working.** Plain markdown, re-read per request. |
| Text REPL (`agent.py`) | **Working.** Full tool-calling loop. |
| Voice → agent | **Built, not yet verified end to end.** `main.py` drives the same `Agent` as the REPL, so every voice command goes through the gate and the log — but no complete spoken turn has been recorded yet (`logs/voice.jsonl` is still empty). Promoted to *working* when it has turns in it. |
| Long-term memory | **Working.** Markdown notes on disk, recalled into the prompt before the model sees the turn. |
| Measurement (`measure.py`) | **Working.** Reads the brain log back: tier latency, routing split, and whether the router nets out positive. |
| Intent router / VAD / local TTS | Intent router and VAD **working**; wake word is still Vosk — see Roadmap. |

---

## The model layer

The constraint that shaped this: **it has to work all day on a student budget,
including offline, without a paid API.**

`brain.py` routes across providers that are all
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

The window is what makes a part-time key worth having in first position: a key
that only serves a fixed slot costs exactly nothing — not even one failed
request — for the hours it's dead. No provider currently declares one. It was
built for a third-party reseller key that served a two-hour slot; that key is
gone (see below), and its schedule went with it rather than being left on a
provider it no longer describes.

Every provider uses my own key on its own published free tier. That rules out
the reseller this originally shipped with: a two-hour daily slot onto a pool
that drained early is pooled capacity, not a licensed quota, and it contradicted
the sentence above. Groq's free tier already covers what it was there for.
No credential pooling, no free-tier aggregation across throwaway accounts, no
TLS interception — all of which are the reason I did not adopt an off-the-shelf
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

### The cheapest turn never reaches a model

Routing a turn to the fast tier makes it cheaper. Not routing it to a model at
all makes it free. *"What time is it"* has exactly one right answer and
`datetime` already knows it, so sending it to a language model costs a round
trip to be told something the standard library could have said instantly — and,
on a small local model, to occasionally be told it wrong, confidently, aloud.

`handlers.py` sits in front of the agent loop and answers those directly. Two
rules keep it from doing harm:

- **Every pattern is a full match on the whole utterance, never a substring.**
  The failure worth designing against is hijacking a turn that wanted real
  thought: *"what do you think about the date on this contract"* contains "the
  date" and must still reach the model. `fullmatch` is what makes that safe, and
  it's the case the tests spend most of their time on.
- **It only answers what is genuinely unambiguous.** *"Read my clipboard"* is
  deliberately absent — nine times in ten that means "tell me what this says",
  which is interpretation, and interpretation is the model's job. The test for
  belonging here isn't "can I write a regex for it" but "is there exactly one
  right answer, and does a library already know it".

An explicit `/fast` or `/deep` stands the handlers down, which doubles as the
escape hatch when one of them is wrong about a sentence.

These are counted in `brain.jsonl` under `handled` and reported separately from
the fast/deep split, for the same reason the voice stages live in their own
file: a turn with no model call is not a routing decision, and folding it into
those percentages would describe a classifier that never ran. The report gives a
count and not a saving — what a handled turn costs is a regex, and what it
*would* have cost is whatever tier it would have been routed to, which isn't
observable precisely because it wasn't.

ponytail: a hand-written pattern list, which is right at this size and wouldn't
be at fifty. The upgrade is an intent classifier, and `answer()` is the only
call site that changes.

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

### Where a spoken turn's time actually goes

Everything above is an argument. `measure.voice_report()` is the number, read
back from a second log the voice loop writes per turn — seven stages, p50/p95,
plus a headline. The shape, with the figures left out because the ones from my
machine aren't yours:

```
Stage (ms)        p50     p95      n
  teardown        ...     ...     ...
  calibrate       ...
  ack             ...        <- the spoken "Yes, sir"
  capture         ...        <- mic open, him talking
  transcribe      ...        <- Whisper
  action          ...        <- the agent; brain.jsonl has the breakdown
  speak           ...        <- TTS of the reply
```

Four decisions shape it, and each one is a way the report could have lied:

- **A separate log from `brain.jsonl`.** A voice stage is not a model call.
  Giving it a `tier` field to fit the router's tables would have bent every
  percentile there around data that isn't a model call.
- **The stages tile the turn.** `lap` closes one as it opens the next, so the
  seven rows add up to the total and unmeasured time has nowhere to hide.
- **The first turn of each sitting is excluded**, and the report says so.
  Whisper loading and pyttsx3 waking its device are paid once and never again;
  left in, that single turn *is* the p95 and describes a machine nobody uses.
  Below two surviving turns it declines to filter at all and says that instead
  — three commands across a day is three sittings of one, and trading a real
  measurement for a clean one is the wrong trade here.
- **A stage with no entry is a missing sample, not a zero.** Scoring `0ms` for
  a stage that a log line predates would halve its median and retire a problem
  that hadn't been fixed.

The headline isn't in the table, because the table answers *which stage is
slow* and the person standing there is asking something else: **how long after
I stopped talking did it answer.** That needs to know how long he talked, which
is why `capture()` returns first speech frame to last rather than the length of
the recording. The recording also holds the gap before he started and the 0.7s
hang at the end, and both of those are silence he sat through. Filing them as
speech would have shortened the reported wait by about a second — in the
flattering direction, in the one number the whole measurement exists to produce.

### Saying the same four things faster

Most of what this says back is one of a handful of fixed lines: *"Yes, sir"* on
every wake word, *"I didn't catch that"* on every misfire. Synthesising those
live means the speech engine does identical work every time, and on the voice
path that work is `ack` — a measured stage sitting between him finishing the
wake word and the microphone reopening.

So they're rendered to WAV once at startup and played from disk afterwards,
which skips building an engine at all. Taken from hey-jev, which pre-renders its
scripted lines for the same reason. Replies a model wrote still go through the
engine live, because there is no second time for those.

Two details that turned out to matter more than the caching:

- **pyttsx3 only honours the first `say()` of an engine's life.** Three
  consecutive utterances on one engine measured 3711ms, 174ms and 114ms — the
  second and third returned without making a sound. That was a live bug, not a
  theoretical one: the assistant would answer the first wake word and then be
  mute until restarted, which from the outside looks like the *model* failing.
  An engine is now built per utterance and disposed of, and the `gc.collect()`
  that does the disposing is load-bearing — pyttsx3 keeps engines behind a weak
  reference and hands the same one back from every `init()`, so without
  collecting it the next utterance gets the spent engine and is silent.
- **SAPI5 pads every rendered phrase with silence**, measured at 0.10s before
  the first word and 0.70s after the last. Live speech doesn't do this, so it's
  an artefact of rendering — and that trailing 0.7s is dead air he waits through
  on every single wake word. `_trim()` strips it, which took the acknowledgement
  from ~2300ms to ~1200ms. It bails rather than guesses in every direction: a
  format it can't measure, a file with no frames, or audio that's silent
  throughout all leave the file untouched. Shortening a file is an optimisation;
  writing an empty one would be a phrase the assistant can no longer say.

The cache is keyed by exact text, which is why the callers hold their phrases in
named constants — pre-rendering *"Yes, sir"* and then speaking *"Yes sir"* is a
cache that never hits and never says so.

### Not answering what it didn't hear

Whisper always returns its best guess, and on a bad capture its best guess is a
plausible sentence made out of a cough and a fan. The scores that would have
said so come back on every segment and were being thrown away — so a misheard
command reached the model, which answered it confidently, out loud.

`transcriber.py` now gates on two of them: the duration-weighted mean
`avg_logprob` (around -0.2 on a clean short command, below -1.0 when Whisper is
reaching) and the worst `no_speech_prob` across segments. A rejected transcript
comes back as the empty string, which is already the shape `main.py` handles —
it says *"I didn't catch that"* and waits, which is the right answer to not
having heard. Also from hey-jev, which re-asks below a confidence threshold
rather than acting on a guess.

The mean is weighted by duration so half a second of noise can't outvote four
seconds of a clear sentence, and `no_speech_prob` is the worst rather than the
average because one segment confidently flagged as a door is enough — averaging
would dilute exactly the signal worth acting on. Both thresholds are calibration
knobs and both measured values are printed on every rejection, because *"it
ignored me"* and *"it misheard me"* are one symptom from the outside and
opposite numbers here.

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
ollama pull qwen3:1.7b
```

Keep it resident, or the first command of every sitting pays for the model
loading off disk — measured at **70 seconds** on a 4GB card, which the router
then records as a classification that took 70s and returned nothing:

```bash
setx OLLAMA_KEEP_ALIVE -1
```

Ollama unloads an idle model after 5 minutes by default, so an assistant used
a few times an hour reloads it on *every* command. That default is for a server
sharing a GPU between models; this is one small model that needs to answer now.

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
2. **Voice rebuild** — VAD-terminated capture, a pre-rendered phrase cache and a
   transcription confidence gate are in. Still to go: openWakeWord replacing
   Vosk (which runs full ASR continuously just to hear one word), Piper
   replacing pyttsx3, and a spoken filler on `deep` turns so a 2s think
   doesn't read as a hang. Piper moved up the list after pyttsx3 turned out to
   need a fresh engine per utterance to speak more than once — half a second of
   setup before every unscripted reply, which a real TTS library doesn't charge.
3. **Recall on demand** — memory is auto-injected only. A `recall` tool would
   let the model search with a better query than the raw utterance. Worth it
   once auto-recall measurably misses; not before.
4. Persona tuning last.

---

## Layout

```
backend/
  brain.py        model router — tiers, fallback chain, cooldown, service windows
  handlers.py     commands answered with no model at all — time, date
  tools.py        execution layer — risk tiers, confirmation gate, action log
  memory.py       long-term memory — markdown notes, scored recall, remember tool
  measure.py      reads brain.jsonl back — tier latency, routing split, net
  agent.py        conversation loop + text REPL
  persona.md      voice and behaviour (edit freely)
  profile.md      who I am, current projects (edit freely)
  main.py         voice loop: wake word → STT → agent → speech
  vad.py          ends a capture when he stops talking, on a calibrated level
  transcriber.py  faster-whisper wrapper + confidence gate
  tts.py          speech — pre-rendered fixed phrases, engine per utterance
  rolling_buffer.py
  test_brain.py   router self-check (no network, no keys needed)
  test_tools.py   gate self-check (no network, no real writes)
  test_agent.py   tool-loop self-check (brain faked, no network)
  test_memory.py  recall self-check (notes in a temp dir)
  test_measure.py report self-check (synthetic log in a temp dir)
  test_vad.py     capture self-check (microphone scripted from a string)
  test_handlers.py no-model handlers (no model, no mic)
  test_tts.py     speech cache trimming (WAVs built by hand, no engine)
  test_transcriber.py confidence gate (segments faked, no Whisper)
  memory/         one markdown file per remembered fact — gitignored
  cache/tts/      pre-rendered WAVs for the fixed phrases — gitignored
  logs/           brain.jsonl, actions.jsonl, voice.jsonl — gitignored
```

## License

MIT
