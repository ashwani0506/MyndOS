import sounddevice as sd
import numpy as np
import json
import queue
import sys
import time

from vosk import Model, KaldiRecognizer
from rolling_buffer import RollingBuffer
from transcriber import Transcriber
from tts import tts
import brain
import measure
import tools
import vad
from agent import Agent

# Config
DEVICE = None                  # Default input device
SAMPLERATE = 16000
CHANNELS = 1
BLOCKSIZE = 8000               # 0.5 seconds per chunk
ROLLING_DURATION_SEC = 10      # store last N seconds of audio
PRE_ROLL_SEC = 3               # how much of the buffer reaches Whisper

WAKE_WORD = "jarvis"

# The lines it says back verbatim, held as constants because tts.py caches by
# exact text: pre-rendering "Yes, sir" and then speaking "Yes sir" is a cache
# that never hits and nothing that ever says so. Anything interpolated is
# deliberately absent -- a phrase with a tool name in it is a different string
# every time and could never be rendered ahead.
ACK = "Yes, sir"
NOT_HEARD = "I didn't catch that."
NO_MODEL = "I can't reach a model right now."
NOTHING_TO_SAY = "I've got nothing useful to say to that."
SCRIPTED = (ACK, NOT_HEARD, NO_MODEL, NOTHING_TO_SAY)


class Stopwatch:
    """Wall clock for one voice turn, split into named stages.

    `lap` closes the stage that was running and starts the next, so the stages
    tile the turn exactly rather than being seven independent timers with gaps
    between them that nothing accounts for. Milliseconds, ints: a sub-ms
    difference in a turn that takes seconds is noise, and formatting floats
    in a report costs more than it tells anyone.
    """

    def __init__(self):
        self.t0 = self.mark = time.perf_counter()
        self.stages: dict[str, int] = {}

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.stages[name] = round((now - self.mark) * 1000)
        self.mark = now

    def total(self) -> int:
        return round((time.perf_counter() - self.t0) * 1000)


def confirm_aloud(t, args) -> bool:
    """Speak, then put the question somewhere he can actually answer it.

    A voice turn has no terminal in front of it -- started at login there is no
    terminal at all -- so `tools.ask` would block on stdin forever and take the
    whole loop with it. `ask_dialog` is a window, and fails closed.
    """
    tts.speak(f"I need your confirmation to {t.name.replace('_', ' ')}.")
    return tools.ask_dialog(t, args)


def reply(agent: Agent, transcription: str) -> str:
    """One agent turn. Nothing here may raise -- this loop is meant to survive
    from login to shutdown, so a bad turn costs a turn, not the assistant.

    The tier is left to the router inside Agent.say, so a lookup no longer pays
    reasoning latency.
    """
    try:
        return agent.say(transcription) or NOTHING_TO_SAY
    except brain.NoProviderAvailable as e:
        print(f"[main] {e}", file=sys.stderr)
        return NO_MODEL
    except Exception as e:
        print(f"[main] {type(e).__name__}: {e}", file=sys.stderr)
        return f"Something broke: {type(e).__name__}. It's in the terminal."


def main():
    print("[main] Loading Vosk model...")
    model = Model(lang="en-us")
    rec = KaldiRecognizer(model, SAMPLERATE)

    # Before the first wake word, so the acknowledgement is never the thing
    # being rendered while he waits for it.
    tts.warm(*SCRIPTED)

    buffer = RollingBuffer(max_duration=ROLLING_DURATION_SEC, samplerate=SAMPLERATE)
    transcriber = Transcriber()
    # One Agent for the whole session, so it remembers across wake words.
    agent = Agent(confirm=confirm_aloud)

    q_in = queue.Queue()

    # Define a stream variable outside so we can start/stop it
    stream = None

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        # Convert to mono int16 bytes
        data = indata.copy().flatten()
        data_bytes = (data * 32767).astype(np.int16).tobytes()

        buffer.add_chunk(data_bytes)
        q_in.put(data_bytes)

    def start_stream():
        nonlocal stream
        stream = sd.InputStream(
            device=DEVICE,
            channels=CHANNELS,
            samplerate=SAMPLERATE,
            blocksize=BLOCKSIZE,
            dtype='float32',
            callback=callback
        )
        stream.start()
        print("[main] Microphone stream started.")

    def stop_stream():
        nonlocal stream
        if stream:
            stream.stop()
            stream.close()
            stream = None
            print("[main] Microphone stream stopped.")

    # Start the mic initially
    start_stream()

    try:
        while True:
            data_bytes = q_in.get()

            if rec.AcceptWaveform(data_bytes):
                result_json = json.loads(rec.Result())
                text = result_json.get("text", "").lower()
                if WAKE_WORD in text:
                    print(f"[wakeword] Detected wake word: {WAKE_WORD}")
                    clock = Stopwatch()

                    # Stop mic so TTS doesn't leak into audio
                    stop_stream()
                    clock.lap("teardown")

                    # Measured before the acknowledgement, off the whole
                    # buffer: ten seconds of the actual room is a better
                    # sample of it than anything measurable after. Both levels
                    # come from the same audio because it holds both -- room,
                    # ending in the wake word.
                    floor, loud = vad.levels(buffer.get_audio(), SAMPLERATE)
                    level = vad.threshold(floor, loud)
                    clock.lap("calibrate")
                    # Printed because "it didn't hear me" and "it never stopped
                    # recording" are one symptom from the outside and opposite
                    # numbers here. "(floor)" means the room measured quieter
                    # than MIN_RMS, so the threshold is the hard minimum rather
                    # than anything this room told us.
                    print(
                        f"[vad] room {floor:.0f}, voice {loud:.0f}, "
                        f"speech above {level:.0f}"
                        + (" (floor)" if level == vad.MIN_RMS else "")
                    )

                    tts.speak(ACK)
                    clock.lap("ack")

                    command, spoke = record_command(level)
                    clock.lap("capture")
                    recorded = len(command) / (SAMPLERATE * 2)
                    print(f"[vad] recorded {recorded:.1f}s, {spoke:.1f}s of speech")

                    # Pre-roll, because Vosk only fires once an utterance ends
                    # -- "what time is it, jarvis" is already spoken by then.
                    # Three seconds covers that; the other seven are the room.
                    audio_to_transcribe = buffer.tail(PRE_ROLL_SEC) + command

                    # Transcribe. Wrapped for the same reason reply() is: this
                    # loop is meant to survive from login to shutdown, so a
                    # bad turn costs a turn. It was the one unwrapped stage,
                    # and a Whisper that raises on every call -- a missing CUDA
                    # library, as it turned out -- took the whole assistant
                    # down on the first wake word rather than one command.
                    try:
                        transcription = transcriber.transcribe_bytes(audio_to_transcribe)
                    except Exception as e:
                        print(f"[main] transcribe failed: {type(e).__name__}: {e}",
                              file=sys.stderr)
                        transcription = ""
                    clock.lap("transcribe")
                    print(f"[main] User said: {transcription}")

                    if transcription.strip():
                        answer = reply(agent, transcription)
                        clock.lap("action")
                        tts.speak(answer)
                    else:
                        clock.lap("action")
                        tts.speak(NOT_HEARD)
                    clock.lap("speak")

                    # Last thing in the turn, and wrapped: a broken log must
                    # cost the measurement, never the assistant. Everything
                    # above has already happened by now -- he has his answer.
                    try:
                        measure.log(
                            {
                                "stages": clock.stages,
                                "total_ms": clock.total(),
                                # First word to last -- not the length of the
                                # recording, which also holds the gap before he
                                # started and the hang at the end. Both of those
                                # are silence he waited through, and counting
                                # them here shortened the reported wait by about
                                # a second, in the flattering direction. Named
                                # for what it is, because "heard" read as
                                # "recorded" once already and cost exactly that.
                                "spoke_ms": round(spoke * 1000),
                                "transcript": transcription.strip()[:200],
                            }
                        )
                    except Exception as e:
                        print(f"[main] voice log failed: {type(e).__name__}: {e}",
                              file=sys.stderr)

                    # Clear rolling buffer
                    buffer.clear()

                    # Restart mic after all TTS is done
                    start_stream()

    except KeyboardInterrupt:
        print("[main] Exiting.")
    finally:
        stop_stream()


def record_command(level):
    """Record until he stops talking. Returns (audio, seconds he spoke).

    A dedicated int16 stream, read a frame at a time: the resident stream is
    float32 with half-second blocks for Vosk's benefit, and neither suits a
    30ms decision. Reading int16 directly also drops the rescaling the
    callback has to do.
    """
    n = int(SAMPLERATE * vad.FRAME_MS / 1000)
    stream = sd.InputStream(
        device=DEVICE, channels=CHANNELS, samplerate=SAMPLERATE, dtype='int16'
    )
    stream.start()
    try:
        return vad.capture(lambda: stream.read(n)[0].tobytes(), level)
    finally:
        stream.stop()
        stream.close()


if __name__ == "__main__":
    main()
