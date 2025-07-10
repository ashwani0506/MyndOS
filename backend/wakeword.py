import sounddevice as sd
import queue
import threading
from vosk import Model, KaldiRecognizer
import json

class WakeWordDetector:
    def __init__(self, wake_words, model_path="models/vosk-model-small-en-us-0.15", sample_rate=16000):
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, sample_rate)
        self.sample_rate = sample_rate
        self.wake_words = [w.lower() for w in wake_words]

        self.q = queue.Queue()
        self.stream = None
        self._detected = False
        self.running = False
        self.worker_thread = None

    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(f"[wakeword] Stream status: {status}")
        self.q.put(bytes(indata))

    def _worker(self):
        print("[wakeword] Worker thread started.")
        while self.running:
            try:
                data = self.q.get(timeout=0.1)
            except queue.Empty:
                continue

            if self.recognizer.AcceptWaveform(data):
                result = json.loads(self.recognizer.Result())
                text = result.get("text", "").lower()
                if any(word in text for word in self.wake_words):
                    print(f"[wakeword] Detected wake word in text: {text}")
                    self._detected = True
            else:
                # partial = json.loads(self.recognizer.PartialResult())
                # could check partials here if you want fast triggers
                pass

    def start_stream(self):
        if self.stream is None:
            self.stream = sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize = 8000,
                device = None,
                dtype='int16',
                channels=1,
                callback=self._audio_callback
            )
            self.stream.start()
            print("[wakeword] Stream started.")

        self.running = True
        self._detected = False

        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def stop_stream(self):
        self.running = False

        if self.worker_thread is not None:
            self.worker_thread.join()
            self.worker_thread = None

        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            print("[wakeword] Stream stopped.")

    def detect(self):
        if self._detected:
            self._detected = False
            return True
        else:
            return False
