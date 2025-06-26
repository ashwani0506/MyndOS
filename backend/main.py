import asyncio
import sounddevice as sd
import numpy as np

from transcriber import Transcriber
from commands import execute_intent
from wakeword import WakeWordDetector
from beep_player import play_beep

transcriber = Transcriber(model_size="base.en", compute_type="float16", device="cuda")
wakeword_detector = WakeWordDetector()

async def record_audio(duration=4, sample_rate=16000):
    print(f"[recorder] Recording for {duration} seconds...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    await asyncio.sleep(duration)
    sd.stop()
    print("[recorder] Finished recording.")
    return recording.flatten()

async def listen_for_command():
    audio_data = await record_audio()
    return await transcriber.transcribe(audio_data)

async def main_loop():
    print("[main] Inside main_loop...")
    with sd.InputStream(callback=lambda *a: None, channels=1, samplerate=16000, dtype='int16'):
        print("[wakeword] Listening for wake word...")
        while True:
            await asyncio.sleep(0.1)
            if wakeword_detector.detect():
                print("[wakeword] Wake word detected.")
                play_beep()
                command = await listen_for_command()
                if command:
                    await execute_intent(command)
                print("[wakeword] Listening for wake word...")

if __name__ == "__main__":
    print("[main] Starting assistant...")
    asyncio.run(main_loop())
