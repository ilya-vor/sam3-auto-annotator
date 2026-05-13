import random
import colorsys

# Generate distinct random colors with fixed seed for consistency
class Colors:
    def __init__(self, n=100):
        self.n = n
        self.palette = {}
        
    def __call__(self, idx):
        if idx not in self.palette:
            # Use deterministic generation based on idx to ensure same color for same ID
            state = random.getstate()
            random.seed(idx)
            h = idx * (1.0 / self.n)
            s = 0.9 - random.random() * 0.5
            v = 0.95 + random.random() * 0.1
            s = max(0.0, min(1.0, s))
            v = max(0.0, min(1.0, v))
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            r = int(r * 255)
            g = int(g * 255)
            b = int(b * 255)
            self.palette[idx] = (r, g, b)
            random.setstate(state)
        # Ensure colors are in valid range
        col = self.palette[idx]
        return (max(0, min(255, col[0])), max(0, min(255, col[1])), max(0, min(255, col[2])))

colors = Colors()