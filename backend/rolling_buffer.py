
import collections

class RollingBuffer:
    def __init__(self, max_duration, samplerate):
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

    def clear(self):
        self.buffer.clear()
        self.size = 0
