from faster_whisper import WhisperModel
import numpy as np

class Transcriber:
    def __init__(self, model_size="base.en", compute_type="float16", device="cuda"):
        print("[transcriber] Loading Whisper model...")
        self.model = WhisperModel(model_size, compute_type=compute_type, device=device)

    async def transcribe(self, audio_data):
        print("[transcriber] Transcribing...")
        audio_float32 = audio_data.astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(audio_float32, language="en", beam_size=1)
        transcript = " ".join([seg.text for seg in segments])
        print(f"[transcriber] Transcript: {transcript}")
        return transcript
