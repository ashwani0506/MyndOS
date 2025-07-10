from faster_whisper import WhisperModel
import numpy as np

class Transcriber:
    def __init__(
        self,
        model_size="base.en",
        compute_type="float16",
        device="cuda"
    ):
        print("[transcriber] Loading Whisper model...")
        self.model = WhisperModel(
            model_size,
            compute_type=compute_type,
            device=device
        )

    async def transcribe(self, audio_data, language="en", beam_size=1):
        """
        Transcribes audio data using Faster-Whisper.
        
        Args:
            audio_data (np.ndarray): Raw PCM audio samples (int16).
            language (str): Language code (e.g. "en").
            beam_size (int): Beam size for decoding.
        
        Returns:
            str: Transcribed text.
        """
        if audio_data is None or len(audio_data) == 0:
            print("[transcriber] Empty audio received.")
            return ""

        # Convert int16 PCM to float32 [-1, 1]
        audio_float32 = audio_data.astype(np.float32) / 32768.0

        # Transcribe entire chunk
        segments, info = self.model.transcribe(
            audio_float32,
            language=language,
            beam_size=beam_size
        )

        transcript = " ".join([seg.text.strip() for seg in segments if seg.text.strip()])
        
        print(f"[transcriber] Transcript: {transcript}")
        return transcript
