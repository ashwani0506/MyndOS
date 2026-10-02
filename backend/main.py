import sounddevice as sd
import numpy as np
import json
import queue
import sys

from vosk import Model, KaldiRecognizer
from rolling_buffer import RollingBuffer
from transcriber import Transcriber
from tts import tts
import brain
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
        return agent.say(transcription) or "I've got nothing useful to say to that."
    except brain.NoProviderAvailable as e:
        print(f"[main] {e}", file=sys.stderr)
        return "I can't reach a model right now."
    except Exception as e:
        print(f"[main] {type(e).__name__}: {e}", file=sys.stderr)
        return f"Something broke: {type(e).__name__}. It's in the terminal."


def main():
    print("[main] Loading Vosk model...")
    model = Model(lang="en-us")
    rec = KaldiRecognizer(model, SAMPLERATE)

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

                    # Stop mic so TTS doesn't leak into audio
                    stop_stream()

                    # Measured before the acknowledgement, off the whole
                    # buffer: ten seconds of the actual room is a better
                    # sample of it than anything measurable after.
                    level = vad.threshold(
                        vad.noise_floor(buffer.get_audio(), SAMPLERATE)
                    )

                    tts.speak("Yes, sir")

                    command = record_command(level)
                    print(f"[vad] recorded {len(command) / (SAMPLERATE * 2):.1f}s")

                    # Pre-roll, because Vosk only fires once an utterance ends
                    # -- "what time is it, jarvis" is already spoken by then.
                    # Three seconds covers that; the other seven are the room.
                    audio_to_transcribe = buffer.tail(PRE_ROLL_SEC) + command

                    # Transcribe
                    transcription = transcriber.transcribe_bytes(audio_to_transcribe)
                    print(f"[main] User said: {transcription}")

                    if transcription.strip():
                        tts.speak(reply(agent, transcription))
                    else:
                        tts.speak("I didn't catch that.")

                    # Clear rolling buffer
                    buffer.clear()

                    # Restart mic after all TTS is done
                    start_stream()

    except KeyboardInterrupt:
        print("[main] Exiting.")
    finally:
        stop_stream()


def record_command(level):
    """Record until he stops talking, rather than for a fixed three seconds.

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
