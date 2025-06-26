import numpy as np
from collections import deque

class RollingBuffer:
    def __init__(self, max_duration=8, sample_rate=16000):
        self.sample_rate = sample_rate
        self.max_samples = max_duration * sample_rate
        self.buffer = deque()

    def extend(self, data):
        self.buffer.extend(data)
        while len(self.buffer) > self.max_samples:
            self.buffer.popleft()

    def get_audio(self):
        if len(self.buffer) == 0:
            return None
        return np.array(self.buffer, dtype=np.int16)
