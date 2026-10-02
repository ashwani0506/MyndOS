import pyttsx3
import threading

class TTS:
    def __init__(self, rate=180, voice=None):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        if voice:
            voices = self.engine.getProperty("voices")
            for v in voices:
                if voice.lower() in v.name.lower():
                    self.engine.setProperty("voice", v.id)
                    print(f"[tts] Voice set to: {v.name}")
                    break
        # Lock for thread safety
        self._lock = threading.Lock()

    def speak(self, text):
        """
        Synchronously speak text using pyttsx3.
        """
        with self._lock:
            self.engine.say(text)
            self.engine.runAndWait()


# Shared engine. pyttsx3/SAPI5 deadlocks if two engines run concurrently,
# so every module speaks through this one.
tts = TTS()
