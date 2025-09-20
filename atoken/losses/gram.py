from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _imagenet_norm(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


class VGGFeatures(nn.Module):
    def __init__(self, layers: List[int] = [3, 8, 17, 26]):
        super().__init__()
        try:
            from torchvision.models import vgg19, VGG19_Weights
            vgg = vgg19(weights=VGG19_Weights.DEFAULT)
        except Exception:
            # fallback without weights
            from torchvision.models import vgg19
            vgg = vgg19(weights=None)
        self.slices = nn.ModuleList()
        last = 0
        for li in layers:
            self.slices.append(nn.Sequential(*list(vgg.features.children())[last:li]))
            last = li
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        out = []
        h = x
        for s in self.slices:
            h = s(h)
            out.append(h)
        return out


class GramLoss(nn.Module):
    def __init__(self, layers: List[int] = [3, 8, 17, 26], weight: float = 1.0):
        super().__init__()
        self.vgg = VGGFeatures(layers)
        self.weight = float(weight)

    @staticmethod
    def gram_matrix(f: torch.Tensor) -> torch.Tensor:
        B, C, H, W = f.shape
        v = f.view(B, C, H * W)
        G = torch.bmm(v, v.transpose(1, 2)) / (C * H * W)
        return G

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_n = _imagenet_norm(x)
        y_n = _imagenet_norm(y)
        xf = self.vgg(x_n)
        yf = self.vgg(y_n)
        loss = x.new_zeros(())
        for a, b in zip(xf, yf):
            Gx = self.gram_matrix(a)
            Gy = self.gram_matrix(b)
            loss = loss + F.mse_loss(Gx, Gy)
        return self.weight * loss

