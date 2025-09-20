from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FSQOutput:
    z_q: torch.Tensor          # (B, N, D)
    tokens: torch.Tensor       # (B, N, G) integer codes per group
    q_loss: torch.Tensor       # scalar (optional)


class FSQQuantizer(nn.Module):
    """
    Finite Scalar Quantization for D=G*Dg (e.g., G=8, Dg=6 -> D=48).

    - Each dimension quantized to `levels_per_dim` discrete levels (default 4)
    - We use a learned per-dimension scale; base levels are [-3, -1, 1, 3]
    - Group Dg dims into a single code with base-K digits (K=levels_per_dim):
        code = sum_{i=0..Dg-1} digit_i * K^i  -> in [0, K^Dg)
    - Straight-through estimator for gradients through quantization.
    """

    def __init__(
        self,
        dim: int = 48,
        groups: int = 8,
        dims_per_group: int = 6,
        levels_per_dim: int = 4,
        quant_loss_weight: float = 0.0,
    ) -> None:
        super().__init__()
        assert groups * dims_per_group == dim, "dim must equal groups*dims_per_group"
        self.dim = dim
        self.G = groups
        self.Dg = dims_per_group
        self.K = levels_per_dim
        self.qw = float(quant_loss_weight)

        # learned positive scales per dimension
        self.log_scales = nn.Parameter(torch.zeros(self.G, self.Dg))

        # base levels (K=4)
        base = torch.tensor([-3.0, -1.0, 1.0, 3.0])
        if self.K != 4:
            # fallback to uniform levels around zero
            base = torch.linspace(-1.0, 1.0, steps=self.K)
        self.register_buffer("base_levels", base, persistent=False)

        # powers for code index mapping (Dg digits base K)
        powers = (self.K ** torch.arange(self.Dg)).long()  # (Dg,)
        self.register_buffer("digit_powers", powers, persistent=False)

    def forward(self, z: torch.Tensor) -> FSQOutput:
        """
        z: (B, N, D)
        returns: quantized z_q, integer tokens (B,N,G)
        """
        assert z.dim() == 3 and z.shape[-1] == self.dim
        B, N, D = z.shape
        zg = z.view(B, N, self.G, self.Dg)  # (B,N,G,Dg)
        scales = self.log_scales.exp()  # (G,Dg)

        # Quantization per scalar dim
        # Compute distances to scaled levels and choose nearest
        # levels: (1,1,G,Dg,K)
        levels = (self.base_levels.view(1, 1, 1, 1, self.K) * scales.view(1, 1, self.G, self.Dg, 1))
        x = zg.unsqueeze(-1)  # (B,N,G,Dg,1)
        dists = (x - levels) ** 2  # (B,N,G,Dg,K)
        idx = dists.argmin(dim=-1)  # (B,N,G,Dg), digits in [0..K-1]

        # Quantized values
        levels_exp = levels.expand(B, N, self.G, self.Dg, self.K)
        q = torch.gather(levels_exp, -1, idx.unsqueeze(-1)).squeeze(-1)  # (B,N,G,Dg)

        # Straight-through: z_q = z + (q - z).detach()
        z_q = z + (q.view(B, N, D) - z).detach()

        # Pack digits per group to single integer code
        codes = (idx.long() * self.digit_powers.view(1, 1, 1, self.Dg)).sum(dim=-1)  # (B,N,G)

        q_loss = z.new_zeros(())
        if self.qw > 0:
            q_loss = self.qw * F.mse_loss(z, q.view(B, N, D))

        return FSQOutput(z_q=z_q, tokens=codes, q_loss=q_loss)

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """
        codes: (B, N, G) integers
        returns z_q: (B, N, D)
        """
        B, N, G = codes.shape
        assert G == self.G
        scales = self.log_scales.exp()  # (G,Dg)
        # Unpack base-K digits
        x = codes.view(B, N, G, 1)
        digits = []
        tmp = x
        for i in range(self.Dg):
            digit = tmp % self.K
            digits.append(digit)
            tmp = tmp // self.K
        digits = torch.cat(digits, dim=-1)  # (B,N,G,Dg)
        levels = (self.base_levels.view(1, 1, 1, 1, self.K) * scales.view(1, 1, G, self.Dg, 1))
        levels_exp = levels.expand(B, N, G, self.Dg, self.K)
        q = torch.gather(levels_exp, -1, digits.unsqueeze(-1)).squeeze(-1)  # (B,N,G,Dg)
        return q.view(B, N, self.dim)
