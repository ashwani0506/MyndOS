import webrtcvad
import numpy as np

class VAD:
    def __init__(self, aggressiveness=2):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = 16000
        self.frame_duration_ms = 30
        self.frame_size = int(self.sample_rate * self.frame_duration_ms / 1000)
        self.bytes_per_sample = 2  # 16-bit

    def is_speech(self, pcm_frame):
        try:
            return self.vad.is_speech(pcm_frame, self.sample_rate)
        except Exception as e:
            print(f"[VAD] Error: {e}")
            return False

    def process(self, audio_chunk):
        if audio_chunk.dtype != np.int16:
            audio_chunk = (audio_chunk * 32768).astype(np.int16)

        if audio_chunk.ndim > 1:
            audio_chunk = np.mean(audio_chunk, axis=1).astype(np.int16)

        audio_bytes = audio_chunk.tobytes()
        speech_detected = False

        for i in range(0, len(audio_bytes) - self.frame_size * self.bytes_per_sample + 1,
                       self.frame_size * self.bytes_per_sample):
            frame = audio_bytes[i:i + self.frame_size * self.bytes_per_sample]
            if self.is_speech(frame):
                speech_detected = True
                break

        return speech_detected
