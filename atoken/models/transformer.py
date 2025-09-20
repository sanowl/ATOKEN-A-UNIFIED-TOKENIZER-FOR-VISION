from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from atoken.ops import MultiAxisRoPE, apply_rope


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, act_layer=nn.GELU, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SelfAttentionRoPE(nn.Module):
    def __init__(self, dim: int, heads: int, rope: Optional[MultiAxisRoPE] = None, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.dim = dim
        self.heads = heads
        self.dh = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: torch.Tensor, sincos) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)  # 3,B,H,N,Dh
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.rope is not None:
            q, k = apply_rope(q, k, sincos)  # (B,H,N,Dh)

        attn = (q @ k.transpose(-2, -1)) * (1.0 / (self.dh ** 0.5))  # (B,H,N,N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v  # (B,H,N,Dh)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, rope: Optional[MultiAxisRoPE] = None, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttentionRoPE(dim, heads, rope)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, sincos) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), sincos))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class TransformerStack(nn.Module):
    def __init__(self, depth: int, dim: int, heads: int, rope: Optional[MultiAxisRoPE] = None):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads, rope=rope) for _ in range(depth)])

    def forward(self, x: torch.Tensor, sincos) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, sincos)
        return x

