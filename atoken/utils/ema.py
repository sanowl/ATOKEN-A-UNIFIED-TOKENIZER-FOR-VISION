import copy
from typing import Iterable

import torch


class EMAModel:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = float(decay)
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        d = self.decay
        ema_params = dict(self.ema.named_parameters())
        for n, p in model.named_parameters():
            if n in ema_params:
                ema_params[n].mul_(d).add_(p.detach(), alpha=1 - d)
        ema_buffs = dict(self.ema.named_buffers())
        for n, b in model.named_buffers():
            if n in ema_buffs:
                ema_buffs[n].copy_(b)

