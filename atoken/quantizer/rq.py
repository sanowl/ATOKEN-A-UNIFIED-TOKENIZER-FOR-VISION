from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RVQOutput:
    z_q: torch.Tensor                 # (B, C, H, W)
    codes: torch.Tensor               # (B, L, H, W), dtype long
    vq_loss: torch.Tensor             # scalar
    perplexity: torch.Tensor          # scalar


class ResidualVectorQuantizer(nn.Module):
    """
    Residual Vector Quantizer (RVQ) with L codebooks.

    - Input z: (B, C, H, W)
    - Flattens to (N, D) where D=C and N=B*H*W
    - Iteratively quantizes residual using L codebooks of size K
    - Returns quantized z_q, codes per level, VQ loss, and perplexity.

    Loss: Standard codebook + commitment formulation per level:
        L_vq = ||sg[r] - e||^2 + beta * ||r - sg[e]||^2
    where r is the current residual and e is the chosen embedding.
    """

    def __init__(
        self,
        dim: int,
        num_levels: int = 4,
        codebook_size: int = 1024,
        commitment_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.beta = float(commitment_weight)

        # Learnable codebooks: L x (K, D)
        embeds = []
        for _ in range(num_levels):
            e = nn.Parameter(torch.randn(codebook_size, dim))
            nn.init.normal_(e, mean=0.0, std=1.0 / dim**0.5)
            embeds.append(e)
        self.embeddings = nn.ParameterList(embeds)

    @torch.no_grad()
    def _compute_perplexity(self, codes: torch.Tensor) -> torch.Tensor:
        """Compute average codebook perplexity across levels.

        codes: (B, L, H, W)
        """
        B, L, H, W = codes.shape
        Ks = self.codebook_size
        levels = []
        for l in range(L):
            idx = codes[:, l].reshape(-1)
            hist = torch.bincount(idx, minlength=Ks).float()
            probs = hist / (hist.sum() + 1e-9)
            entropy = -(probs * (probs + 1e-9).log()).sum()
            perp = torch.exp(entropy)
            levels.append(perp)
        return torch.stack(levels).mean()

    def forward(self, z: torch.Tensor) -> RVQOutput:
        assert z.dim() == 4, "z must be (B, C, H, W)"
        B, C, H, W = z.shape
        D = C
        x = z.permute(0, 2, 3, 1).contiguous().view(-1, D)  # (N, D)

        residual = x
        sum_q = torch.zeros_like(x)
        code_indices: List[torch.Tensor] = []
        vq_loss = x.new_zeros(())

        for level, emb in enumerate(self.embeddings):
            # distances: (N, K)
            # ||r - e||^2 = ||r||^2 - 2 r·e + ||e||^2
            r2 = (residual ** 2).sum(dim=1, keepdim=True)  # (N, 1)
            e2 = (emb ** 2).sum(dim=1)  # (K,)
            logits = - (r2 - 2 * residual @ emb.t() + e2)  # higher is better
            indices = torch.argmax(logits, dim=1)  # (N,)
            q = F.embedding(indices, emb)  # (N, D)

            # VQ Loss per level
            # codebook loss (grad to emb): ||sg[r] - e||^2
            codebook_loss = F.mse_loss(residual.detach(), q)
            # commitment loss (grad to residual): ||r - sg[e]||^2
            commitment_loss = F.mse_loss(residual, q.detach())
            vq_loss = vq_loss + codebook_loss + self.beta * commitment_loss

            sum_q = sum_q + q
            # Update residual for the next level; stop gradient through q in residual path
            residual = residual - q.detach()
            code_indices.append(indices.view(B, H, W))

        # Straight‑through estimator for quantized output
        z_q_flat = x + (sum_q - x).detach()
        z_q = z_q_flat.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

        codes = torch.stack(code_indices, dim=1)  # (B, L, H, W)
        with torch.no_grad():
            perplexity = self._compute_perplexity(codes)

        return RVQOutput(z_q=z_q, codes=codes, vq_loss=vq_loss, perplexity=perplexity)

    def decode_codes(self, codes: torch.Tensor, shape_hw: Tuple[int, int]) -> torch.Tensor:
        """Decode integer codes back to latent z_q.

        codes: (B, L, H, W) or list of length L with (B, H, W)
        shape_hw: (H, W) to confirm target latent shape
        returns z_q: (B, C, H, W) where C==dim
        """
        if isinstance(codes, list):
            codes = torch.stack(codes, dim=1)
        assert codes.dim() == 4, "codes must be (B, L, H, W)"
        B, L, H, W = codes.shape
        assert L == self.num_levels, "num levels mismatch"
        assert (H, W) == shape_hw, "latent HW mismatch"

        device = codes.device
        sum_q = None
        for l in range(L):
            emb = self.embeddings[l]
            q = F.embedding(codes[:, l].reshape(-1), emb).view(B, H, W, self.dim)
            q = q.permute(0, 3, 1, 2).contiguous()
            sum_q = q if sum_q is None else (sum_q + q)
        return sum_q

