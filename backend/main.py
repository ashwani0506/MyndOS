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
from agent import Agent

# Config
DEVICE = None                  # Default input device
SAMPLERATE = 16000
CHANNELS = 1
BLOCKSIZE = 8000               # 0.5 seconds per chunk
ROLLING_DURATION_SEC = 10      # store last N seconds of audio
EXTRA_RECORD_SEC = 3           # extra time after wakeword

WAKE_WORD = "jarvis"


def confirm_aloud(t, args) -> bool:
    """Speak before blocking on stdin, so a CONFIRM-tier prompt during a voice
    turn isn't a silent hang with the mic already closed."""
    tts.speak(f"I need your confirmation to {t.name.replace('_', ' ')}. Check the terminal.")
    return tools.ask(t, args)


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

                    tts.speak("Yes, sir")

                    # Capture extra audio after wake word
                    additional_audio = capture_extra_audio(EXTRA_RECORD_SEC)

                    # Combine rolling buffer + new audio
                    audio_to_transcribe = buffer.get_audio() + additional_audio

                    # Transcribe
                    transcription = run_transcription(transcriber, audio_to_transcribe)
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


def capture_extra_audio(duration_sec):
    """Record a few extra seconds after wake word is detected."""
    print(f"[main] Capturing additional {duration_sec} seconds of audio...")
    frames = int(duration_sec * SAMPLERATE)
    audio = sd.rec(frames, samplerate=SAMPLERATE, channels=CHANNELS, dtype='float32')
    sd.wait()
    data = audio.flatten()
    data_bytes = (data * 32767).astype(np.int16).tobytes()
    return data_bytes


def run_transcription(transcriber, audio_bytes):
    """
    Helper to run asynchronous transcription synchronously for main.py context.
    """
    import asyncio

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(transcriber.transcribe_bytes(audio_bytes))
    loop.close()
    return result


if __name__ == "__main__":
    main()
