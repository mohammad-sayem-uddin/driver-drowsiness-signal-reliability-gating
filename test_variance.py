from collections import deque

class EMASmoother:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self._value = None
    def update(self, raw):
        if self._value is None: self._value = raw
        else: self._value = self.alpha * raw + (1.0 - self.alpha) * self._value
        return self._value

def analyze(mars):
    buffer = deque(maxlen=15)
    smoother = EMASmoother(0.3)
    for i, m in enumerate(mars):
        smoothed = smoother.update(m)
        buffer.append(smoothed)
        if len(buffer) < 2:
            continue
        deltas = [abs(buffer[j] - buffer[j-1]) for j in range(1, len(buffer))]
        jitter = sum(deltas) / len(deltas)
        variance = sum((x - (sum(buffer)/len(buffer)))**2 for x in buffer) / len(buffer)
        print(f"MAR: {m:.2f}, Jitter: {jitter:.4f}, Variance: {variance:.4f}")

print("Speech:")
analyze([0.1, 0.5, 0.1, 0.6, 0.2, 0.5])
print("\nYawn:")
analyze([0.1, 0.1, 0.1, 0.2, 0.4, 0.6, 0.7, 0.8, 0.8, 0.8, 0.8, 0.8])
