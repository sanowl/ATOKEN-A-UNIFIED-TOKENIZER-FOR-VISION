from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from atoken.quantizer import ResidualVectorQuantizer


class ConvBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, dim),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(x) + x)
        d


class Encoder(nn.Module):
    def __init__(self, in_ch: int = 3, dim: int = 256, depth: int = 2, patch: int = 8):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        self.blocks = nn.Sequential(*[ConvBlock(dim) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.blocks(x)
        return x


class Decoder(nn.Module):
    def __init__(self, out_ch: int = 3, dim: int = 256, depth: int = 2, patch: int = 8):
        super().__init__()
        self.blocks = nn.Sequential(*[ConvBlock(dim) for _ in range(depth)])
        self.deproj = nn.ConvTranspose2d(dim, out_ch, kernel_size=patch, stride=patch)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.blocks(z)
        x = self.deproj(x)
        x = torch.sigmoid(x)
        return x


@dataclass
class ATokenOutput:
    recon: torch.Tensor
    tokens: torch.Tensor
    vq_loss: torch.Tensor
    recon_loss: torch.Tensor
    total_loss: torch.Tensor
    stats: Dict[str, torch.Tensor]


class AToken(nn.Module):
    """AToken: minimal residual VQ tokenizer for images.

    - Encoder downsamples images by `patch` and produces latent (B, D, H', W')
    - RVQ quantizes the latent into L codebooks (discrete tokens)
    - Decoder reconstructs the image from quantized latent
    - Exposes encode() and decode() for tokenization
    """

    def __init__(
        self,
        in_ch: int = 3,
        dim: int = 256,
        depth_enc: int = 2,
        depth_dec: int = 2,
        patch: int = 8,
        num_levels: int = 4,
        codebook_size: int = 1024,
        commitment_weight: float = 0.25,
        recon_loss_type: str = "l1",
    ) -> None:
        super().__init__()
        self.patch = patch
        self.encoder = Encoder(in_ch=in_ch, dim=dim, depth=depth_enc, patch=patch)
        self.quantizer = ResidualVectorQuantizer(
            dim=dim,
            num_levels=num_levels,
            codebook_size=codebook_size,
            commitment_weight=commitment_weight,
        )
        self.decoder = Decoder(out_ch=in_ch, dim=dim, depth=depth_dec, patch=patch)

        if recon_loss_type == "l1":
            self.recon_loss_fn = nn.L1Loss()
        elif recon_loss_type == "mse":
            self.recon_loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported recon_loss_type: {recon_loss_type}")

    def forward(self, images: torch.Tensor) -> ATokenOutput:
        z = self.encoder(images)
        q_out = self.quantizer(z)
        recon = self.decoder(q_out.z_q)
        recon_loss = self.recon_loss_fn(recon, images)
        total = recon_loss + q_out.vq_loss
        stats = {
            "perplexity": q_out.perplexity.detach(),
            "vq_loss": q_out.vq_loss.detach(),
            "recon_loss": recon_loss.detach(),
            "latent_h": torch.tensor(q_out.codes.shape[2], device=images.device),
            "latent_w": torch.tensor(q_out.codes.shape[3], device=images.device),
        }
        return ATokenOutput(
            recon=recon,
            tokens=q_out.codes,
            vq_loss=q_out.vq_loss,
            recon_loss=recon_loss,
            total_loss=total,
            stats=stats,
        )

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to discrete tokens: returns (B, L, H', W')."""
        z = self.encoder(images)
        q_out = self.quantizer(z)
        return q_out.codes

    @torch.no_grad()
    def decode(self, tokens: Union[torch.Tensor, list], target_hw: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Decode tokens back to images.

        tokens: (B, L, H', W') or list of length L with (B, H', W')
        target_hw: optional latent spatial size; if None, infer from tokens
        """
        if isinstance(tokens, list):
            B, H, W = tokens[0].shape
            L = len(tokens)
        else:
            assert tokens.dim() == 4
            B, L, H, W = tokens.shape

        if target_hw is None:
            target_hw = (H, W)
        z_q = self.quantizer.decode_codes(tokens, target_hw)
        recon = self.decoder(z_q)
        return recon

    @torch.no_grad()
    def tokens_to_sequence(self, tokens: torch.Tensor) -> torch.Tensor:
        """Flatten (B, L, H, W) -> (B, H*W, L) as an example sequence format."""
        B, L, H, W = tokens.shape
        return tokens.permute(0, 2, 3, 1).reshape(B, H * W, L)

