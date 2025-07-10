import asyncio
import sounddevice as sd
import numpy as np
import io
import wave

from transcriber import Transcriber
from commands import dispatch_intent
from wakeword import WakeWordDetector
from llm_client import call_gemini
from tts import TTS

# Initialize components
transcriber = Transcriber(model_size="base.en", compute_type="float16", device="cuda")
wakeword_detector = WakeWordDetector(wake_words=["jarvis", "computer", "assistant"])
tts = TTS()

# --------- FIX: record_audio now safe for asyncio -----------

def record_audio_sync(duration=5, sample_rate=16000):
    print(f"[recorder] Recording (sync) for {duration} seconds...")
    recording = sd.rec(int(duration * sample_rate),
                       samplerate=sample_rate,
                       channels=1,
                       dtype='int16')
    sd.wait()
    print("[recorder] Finished recording.")

    # Convert numpy array to WAV bytes buffer
    audio_flat = recording.flatten()
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(audio_flat.tobytes())
    buffer.seek(0)
    return buffer

async def record_audio(duration=5, sample_rate=16000):
    """
    Runs the blocking record_audio_sync() in a separate thread
    so it doesn't freeze the asyncio event loop.
    """
    loop = asyncio.get_running_loop()
    buffer = await loop.run_in_executor(None, record_audio_sync, duration, sample_rate)
    return buffer

async def listen_for_command():
    audio_buffer = await record_audio()
    transcript = await transcriber.transcribe(audio_buffer)
    return transcript

async def main_loop():
    print("[main] Inside main_loop...")

    try:
        while True:
            # Start the wake word detector stream
            wakeword_detector.start_stream()
            print("[wakeword] Listening for wake word...")

            # Wait asynchronously for wake word detection
            while not wakeword_detector.detect():
                await asyncio.sleep(0.1)

            print("[wakeword] Wake word detected.")

            # Stop the wake word stream while we handle the command
            wakeword_detector.stop_stream()

            # Speak response
            await tts.speak_async("Yes Sir")

            # Record + transcribe user command
            user_text = await listen_for_command()
            print(f"[transcription] User said: {user_text}")

            if user_text:
                # Call Gemini
                result_json = call_gemini(user_text)
                print(f"[llm] Gemini result: {result_json}")

                # Dispatch intent
                await dispatch_intent(result_json)

            # Go back to listening
            print("[wakeword] Ready for next wake word...")

    except KeyboardInterrupt:
        print("\n[main] Keyboard interrupt. Exiting...")
    finally:
        wakeword_detector.stop_stream()

if __name__ == "__main__":
    print("[main] Starting assistant...")
    asyncio.run(main_loop())
