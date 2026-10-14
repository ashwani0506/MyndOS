import sounddevice as sd
import queue
import sys
import time

from rolling_buffer import RollingBuffer
from transcriber import Transcriber
from tts import tts
from wakeword import WakeWord
import brain
import measure
import tools
import vad
import wakeword
from agent import Agent

# Config
DEVICE = None                  # Default input device
SAMPLERATE = 16000
CHANNELS = 1
# 80ms, which is exactly the window the wake-word model scores. Vosk wanted
# half-second blocks and the loop inherited them, so detection waited for a
# block to fill before it could react; this reacts six times sooner and the
# extra callbacks are 12 a second instead of 2.
BLOCKSIZE = 1280
ROLLING_DURATION_SEC = 10      # store last N seconds of audio
# Whisper sees this much audio before the command. It used to be 3 seconds for
# a reason that no longer holds: Vosk only reported an utterance once it had
# ended, so by the time it fired, the command had already been spoken and the
# pre-roll was the only place it existed. openWakeWord fires on the phrase
# itself, so the command is still ahead of us. What's left is a short guard in
# case he starts talking before the acknowledgement finishes.
PRE_ROLL_SEC = 1

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
    print("[main] Loading wake-word model...")
    detector = WakeWord()

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
        # int16 straight from the driver, no rescaling. It used to arrive as
        # float32 and get multiplied by 32767, which silently wrapped: any
        # sample the driver delivered above full scale -- routine with Windows
        # mic boost on -- came out as a large *negative* number. That shreds
        # exactly the loudest syllables, which are the ones carrying the wake
        # word, and it is why mic_check scored a near miss on the same voice
        # that this loop heard nothing from: mic_check asks the driver for
        # int16 and never touched this path.
        data_bytes = indata.copy().flatten().tobytes()

        buffer.add_chunk(data_bytes)
        q_in.put(data_bytes)

    def start_stream():
        nonlocal stream
        stream = sd.InputStream(
            device=DEVICE,
            channels=CHANNELS,
            samplerate=SAMPLERATE,
            blocksize=BLOCKSIZE,
            dtype='int16',
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

    # Peak score of whatever he just said, while it stays above NOTICE. Printed
    # once when it falls back, so a near miss is one line per attempt rather
    # than twelve a second -- and so "it didn't react" comes with a number.
    near_peak = 0.0

    try:
        while True:
            data_bytes = q_in.get()

            if detector.heard(data_bytes):
                near_peak = 0.0
                print(f"[wakeword] heard {wakeword.PHRASE!r} "
                      f"(score {detector.last_score:.2f})")
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

                # A short pre-roll, in case he ran into the command before the
                # acknowledgement finished. It is no longer where the command
                # lives -- see PRE_ROLL_SEC -- and the wake phrase itself lands
                # in here, which handlers._normalise strips back out.
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
                # And drop whatever the callback queued between the wake word
                # firing and the stream closing. That audio is the tail of
                # "hey jarvis" plus the command itself, and handing it back to
                # the detector can start a second turn on words already
                # answered. The detector resets itself on a hit; this is the
                # queue in front of it.
                while not q_in.empty():
                    q_in.get_nowait()

                # Restart mic after all TTS is done
                start_stream()

            elif detector.last_score >= wakeword.NOTICE:
                near_peak = max(near_peak, detector.last_score)
            elif near_peak:
                print(f"[wakeword] heard something at {near_peak:.2f}, under "
                      f"the {wakeword.THRESHOLD} line -- say {wakeword.PHRASE!r}")
                near_peak = 0.0

    except KeyboardInterrupt:
        print("[main] Exiting.")
    finally:
        stop_stream()


def record_command(level):
    """Record until he stops talking. Returns (audio, seconds he spoke).

    A dedicated int16 stream, read a frame at a time: the resident stream uses
    80ms blocks to suit the wake-word model, which doesn't suit a 30ms
    decision. Both read int16 directly from the driver -- nothing in this
    pipeline wants float.
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
