
import collections

class RollingBuffer:
    def __init__(self, max_duration, samplerate):
        self.samplerate = samplerate
        self.max_bytes = int(max_duration * samplerate * 2) # 16-bit PCM
        self.buffer = collections.deque()
        self.size = 0

    def add_chunk(self, chunk):
        self.buffer.append(chunk)
        self.size += len(chunk)
        while self.size > self.max_bytes:
            old = self.buffer.popleft()
            self.size -= len(old)

    def get_audio(self):
        return b''.join(self.buffer)

    def tail(self, seconds):
        """The last N seconds.

        The full buffer is mostly ambience, and it is worth keeping that way --
        it is what the noise floor is measured from. But only the end of it can
        contain a command, so that is all Whisper needs to read.
        """
        n = int(seconds * self.samplerate * 2)
        return self.get_audio()[-n:] if n else b''

    def clear(self):
        self.buffer.clear()
        self.size = 0
