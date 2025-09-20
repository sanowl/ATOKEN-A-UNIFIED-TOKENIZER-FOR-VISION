from typing import Tuple

import torch
import torch.nn as nn


class PatchEmbed3D(nn.Module):
    def __init__(self, in_ch: int, dim: int, patch: Tuple[int, int, int] = (4, 16, 16)):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv3d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor):
        # x: (B, C, T, H, W)
        x = self.proj(x)  # (B, D, T', H', W')
        B, D, T, H, W = x.shape
        tokens = x.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, D)
        return tokens, (T, H, W)


class PatchDeEmbed3D(nn.Module):
    def __init__(self, out_ch: int, dim: int, patch: Tuple[int, int, int] = (4, 16, 16)):
        super().__init__()
        self.patch = patch
        self.deproj = nn.ConvTranspose3d(dim, out_ch, kernel_size=patch, stride=patch)

    def forward(self, tokens: torch.Tensor, thw: Tuple[int, int, int]) -> torch.Tensor:
        B, N, D = tokens.shape
        T, H, W = thw
        x = tokens.view(B, T, H, W, D).permute(0, 4, 1, 2, 3).contiguous()
        x = self.deproj(x)
        return torch.sigmoid(x)

