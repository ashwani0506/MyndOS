from faster_whisper import WhisperModel
import numpy as np

class Transcriber:
    def __init__(self,
                 model_size="base.en",
                 compute_type="auto",
                 device="auto"):  # cuda when available, else cpu — don't hard-crash on a machine without it
        print("[transcriber] Loading Whisper model...")
        self.model = WhisperModel(
            model_size,
            compute_type=compute_type,
            device=device
        )

    def transcribe(self,
                   audio_data: np.ndarray,
                   language="en",
                   beam_size=1) -> str:
        """
        Transcribes raw PCM audio samples (int16) using Faster-Whisper.

        Args:
            audio_data (np.ndarray): Raw PCM audio samples (int16).
            language (str): Language code.
            beam_size (int): Beam size for decoding.

        Returns:
            str: Transcribed text.
        """
        if audio_data is None or len(audio_data) == 0:
            print("[transcriber] Empty audio received.")
            return ""

        # Convert int16 PCM to float32 in range [-1, 1]
        audio_float32 = audio_data.astype(np.float32) / 32768.0

        segments, info = self.model.transcribe(
            audio_float32,
            language=language,
            beam_size=beam_size
        )

        transcript = " ".join(
            seg.text.strip() for seg in segments if seg.text.strip()
        )
        
        print(f"[transcriber] Transcript: {transcript}")
        return transcript

    def transcribe_bytes(self,
                         audio_bytes: bytes,
                         language="en",
                         beam_size=1) -> str:
        """
        Converts raw bytes to numpy array and transcribes them.

        Args:
            audio_bytes (bytes): Raw PCM audio bytes.
            language (str): Language code.
            beam_size (int): Beam size for decoding.

        Returns:
            str: Transcribed text.
        """
        if audio_bytes is None or len(audio_bytes) == 0:
            print("[transcriber] Empty audio bytes received.")
            return ""

        # Convert bytes → numpy array of int16
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        return self.transcribe(audio_array, language=language, beam_size=beam_size)
