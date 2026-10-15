import sounddevice as sd
import queue
import sys
import time

from rolling_buffer import RollingBuffer
from transcriber import Transcriber
from tts import tts
from hotkey import Hotkey
from wakeword import WakeWord
import brain
import measure
import tools
import ui
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


FRAME_BYTES = int(SAMPLERATE * vad.FRAME_MS / 1000) * 2  # 30ms of int16


def drain(q_in: queue.Queue) -> None:
    """Throw away whatever is queued.

    Used twice a turn, for the same reason both times: the microphone is never
    closed any more, so it hears the assistant's own acknowledgement and its
    own reply. Those must not be read as the command, and must not be handed
    back to the wake-word detector afterwards.
    """
    while not q_in.empty():
        try:
            q_in.get_nowait()
        except queue.Empty:
            return


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

    # One stream, opened once, closed on the way out. Everything -- the wake
    # word and the command capture both -- reads the queue it feeds. It used to
    # be stopped and restarted around every turn, plus a second stream opened
    # for the capture, and the second turn hung inside PortAudio's read()
    # hard enough to need the terminal killed.
    stream = sd.InputStream(
        device=DEVICE,
        channels=CHANNELS,
        samplerate=SAMPLERATE,
        blocksize=BLOCKSIZE,
        dtype='int16',
        callback=callback,
    )

    panel = ui.Panel()
    key = Hotkey()

    def take_turn(trigger: str) -> None:
        """One voice turn, however it was started.

        The trigger is not cosmetic. A hotkey turn skips the spoken
        acknowledgement: if he pressed a key he is at the keyboard looking at
        the panel, which reads "Listening" the instant this is called, so
        1200ms of "Yes, sir" buys nothing. A wake word may have come from
        across the room, where the panel can't be read and the only
        acknowledgement worth anything is an audible one.

        `clock.lap("ack")` happens either way, so the stages still tile the
        turn -- it just measures roughly zero on the hotkey path, which is the
        point of having it there.
        """
        clock = Stopwatch()
        # The stream is deliberately *not* closed here any more. It used to be,
        # so the acknowledgement couldn't leak into the microphone, and reopened
        # at the end of the turn -- two open/close cycles per turn on the same
        # device. That cost 290ms, and on the second turn PortAudio's read()
        # never returned: a hard hang, uninterruptible by Ctrl-C because the GIL
        # was released inside it. The leak is dealt with by draining the queue,
        # which is cheaper and cannot wedge a device.
        clock.lap("teardown")

        # Measured before the acknowledgement, off the whole buffer: ten
        # seconds of the actual room is a better sample of it than anything
        # measurable after. Both levels come from the same audio because it
        # holds both -- room, ending in the wake word.
        floor, loud = vad.levels(buffer.get_audio(), SAMPLERATE)
        level = vad.threshold(floor, loud)
        clock.lap("calibrate")
        # Printed because "it didn't hear me" and "it never stopped recording"
        # are one symptom from the outside and opposite numbers here.
        # "(floor)" means the room measured quieter than MIN_RMS, so the
        # threshold is the hard minimum rather than anything this room told us.
        print(
            f"[vad] room {floor:.0f}, voice {loud:.0f}, "
            f"speech above {level:.0f}"
            + (" (floor)" if level == vad.MIN_RMS else "")
        )

        panel.state("listening", trigger)
        if trigger == "wake word":
            tts.speak(ACK)
            # Everything the microphone heard while that played -- which
            # includes the acknowledgement itself, now that the stream stays
            # open. Without this the capture reads "Yes, sir" as the command.
            drain(q_in)
            buffer.clear()
        clock.lap("ack")

        # A fresh reader per turn, so no half-frame is carried over from the
        # last one. Wrapped so the meter keeps moving while he talks: the main
        # loop isn't running during a capture, and that is the one moment the
        # readout most needs to show the microphone is live. Every third frame,
        # because Tk redrawing 33 times a second to animate a bar of blocks
        # would be the capture's own biggest cost.
        read = vad.queue_reader(q_in, SAMPLERATE)
        seen = 0

        def watched() -> bytes:
            nonlocal seen
            frame = read()
            seen += 1
            if seen % 3 == 0:
                panel.level(vad.rms(frame))
            return frame

        command, spoke = vad.capture(watched, level)
        clock.lap("capture")
        recorded = len(command) / (SAMPLERATE * 2)
        print(f"[vad] recorded {recorded:.1f}s, {spoke:.1f}s of speech")

        panel.state("thinking")

        # Pre-roll on the hotkey path only. Nothing was spoken there, so the
        # buffer holds the room plus any command he ran into early -- worth
        # having. On the wake-word path the buffer was just cleared of the
        # acknowledgement, so there is nothing in it but the acknowledgement's
        # own silence.
        pre_roll = buffer.tail(PRE_ROLL_SEC) if trigger == "hotkey" else b""
        audio_to_transcribe = pre_roll + command

        # Transcribe. Wrapped for the same reason reply() is: this loop is
        # meant to survive from login to shutdown, so a bad turn costs a turn.
        # It was the one unwrapped stage, and a Whisper that raises on every
        # call -- a missing CUDA library, as it turned out -- took the whole
        # assistant down on the first wake word rather than one command.
        try:
            transcription = transcriber.transcribe_bytes(audio_to_transcribe)
        except Exception as e:
            print(f"[main] transcribe failed: {type(e).__name__}: {e}",
                  file=sys.stderr)
            transcription = ""
        clock.lap("transcribe")
        print(f"[main] User said: {transcription}")
        panel.heard(transcription)

        if transcription.strip():
            answer = reply(agent, transcription)
            clock.lap("action")
            panel.state("speaking")
            panel.replied(answer)
            tts.speak(answer)
        else:
            clock.lap("action")
            panel.state("speaking")
            panel.replied(NOT_HEARD)
            tts.speak(NOT_HEARD)
        clock.lap("speak")

        # Last thing in the turn, and wrapped: a broken log must cost the
        # measurement, never the assistant. Everything above has already
        # happened by now -- he has his answer.
        try:
            measure.log(
                {
                    "trigger": trigger,
                    "stages": clock.stages,
                    "total_ms": clock.total(),
                    # First word to last -- not the length of the recording,
                    # which also holds the gap before he started and the hang
                    # at the end. Both of those are silence he waited through,
                    # and counting them here shortened the reported wait by
                    # about a second, in the flattering direction. Named for
                    # what it is, because "heard" read as "recorded" once
                    # already and cost exactly that.
                    "spoke_ms": round(spoke * 1000),
                    "transcript": transcription.strip()[:200],
                }
            )
        except Exception as e:
            print(f"[main] voice log failed: {type(e).__name__}: {e}",
                  file=sys.stderr)

        # Clear rolling buffer
        buffer.clear()
        # The detector scores a rolling window, so the tail of the phrase is
        # still inside it; on the hotkey path nothing reset it at all.
        detector.reset()
        # And drop everything the microphone heard during the turn -- the reply
        # it just spoke, most of all. Handing that back to the detector is how
        # an assistant wakes itself up.
        drain(q_in)

        panel.state("idle")

    stream.start()
    print("[main] Microphone stream started.")
    panel.state("idle")
    print(f"[main] Say {wakeword.PHRASE!r}, or press Right Ctrl.")

    # Peak score of whatever he just said, while it stays above NOTICE. Printed
    # once when it falls back, so a near miss is one line per attempt rather
    # than twelve a second -- and so "it didn't react" comes with a number.
    near_peak = 0.0

    try:
        while True:
            # A timeout rather than a blocking get, so Tk gets this thread
            # between audio blocks and the hotkey is polled even in a silent
            # room. Blocking here froze the panel until somebody spoke, which
            # is the one moment a status readout has to be alive.
            try:
                data_bytes = q_in.get(timeout=0.05)
            except queue.Empty:
                panel.pump()
                if key.pressed():
                    take_turn("hotkey")
                continue

            panel.level(vad.rms(data_bytes))

            # The key is checked first on purpose: it is the deliberate signal,
            # and it should win over a wake word that happens to score on the
            # same block.
            if key.pressed():
                take_turn("hotkey")
            elif detector.heard(data_bytes):
                near_peak = 0.0
                print(f"[wakeword] heard {wakeword.PHRASE!r} "
                      f"(score {detector.last_score:.2f})")
                take_turn("wake word")
            elif detector.last_score >= wakeword.NOTICE:
                near_peak = max(near_peak, detector.last_score)
            elif near_peak:
                print(f"[wakeword] heard something at {near_peak:.2f}, under "
                      f"the {wakeword.THRESHOLD} line -- say {wakeword.PHRASE!r}")
                near_peak = 0.0

    except KeyboardInterrupt:
        print("[main] Exiting.")
    finally:
        stream.stop()
        stream.close()
        print("[main] Microphone stream stopped.")
        panel.close()


if __name__ == "__main__":
    main()
