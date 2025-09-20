from dataclasses import dataclass


@dataclass
class WarmupCosine:
    warmup: int
    total: int
    base_lr: float
    min_lr: float

    def get_lr(self, step: int) -> float:
        if step < self.warmup:
            return self.base_lr * (step + 1) / self.warmup
        import math
        t = (step - self.warmup) / max(1, self.total - self.warmup)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * t))

