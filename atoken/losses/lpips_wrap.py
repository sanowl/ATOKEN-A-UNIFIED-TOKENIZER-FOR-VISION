import torch
import torch.nn as nn


class LPIPSLoss(nn.Module):
    def __init__(self, net: str = "alex", weight: float = 1.0):
        super().__init__()
        self.weight = float(weight)
        try:
            import lpips  # type: ignore
            self.lpips = lpips.LPIPS(net=net)
            for p in self.lpips.parameters():
                p.requires_grad = False
        except Exception:
            self.lpips = None

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.lpips is None:
            return x.new_zeros(())
        # LPIPS expects [-1,1]
        x_n = x * 2 - 1
        y_n = y * 2 - 1
        return self.weight * self.lpips(x_n, y_n).mean()

