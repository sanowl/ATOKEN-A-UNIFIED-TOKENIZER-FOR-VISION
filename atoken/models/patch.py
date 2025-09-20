from typing import Tuple

import torch
import torch.nn as nn


class PatchEmbed2D(nn.Module):
    def __init__(self, in_ch: int, dim: int, patch: int = 16):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        x = self.proj(x)
        B, C, H, W = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        return tokens, (H, W)


class PatchDeEmbed2D(nn.Module):
    def __init__(self, out_ch: int, dim: int, patch: int = 16):
        super().__init__()
        self.patch = patch
        self.deproj = nn.ConvTranspose2d(dim, out_ch, kernel_size=patch, stride=patch)

    def forward(self, tokens: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        B, N, C = tokens.shape
        H, W = hw
        x = tokens.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x = self.deproj(x)
        return torch.sigmoid(x)

