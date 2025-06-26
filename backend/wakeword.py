import queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import json

model = Model("models/vosk-model-small-en-us-0.15")
recognizer = KaldiRecognizer(model, 16000, '["jarvis", "[unk]"]')

class WakeWordDetector:
    def __init__(self):
        self.q = queue.Queue()
        self.stream = sd.InputStream(callback=self.callback, channels=1, samplerate=16000, dtype='int16')
        self.stream.start()

    def callback(self, indata, frames, time, status):
        if status:
            print(f"[wakeword] Error: {status}")
        self.q.put(indata.copy())

    def detect(self):
        if not self.q.empty():
            data = self.q.get()
            if recognizer.AcceptWaveform(data.tobytes()):
                result = json.loads(recognizer.Result())
                if "jarvis" in result.get("text", "").lower():
                    return True
        return False
