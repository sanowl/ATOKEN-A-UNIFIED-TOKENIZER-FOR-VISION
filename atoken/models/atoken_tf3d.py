from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from atoken.models.patch3d import PatchEmbed3D, PatchDeEmbed3D
from atoken.models.transformer import TransformerStack
from atoken.ops import MultiAxisRoPE
from atoken.quantizer.fsq import FSQQuantizer, FSQOutput


@dataclass
class ATokenTF3DOutput:
    recon: torch.Tensor
    tokens: Optional[torch.Tensor]
    losses: Dict[str, torch.Tensor]
    video_embed: Optional[torch.Tensor]


class ATokenTransformer3D(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        patch: Tuple[int, int, int] = (4, 16, 16),
        dim: int = 384,
        depth_enc: int = 6,
        depth_dec: int = 6,
        heads: int = 6,
        latent_dim: int = 48,
        use_fsq: bool = True,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.patch = patch
        self.use_fsq = use_fsq
        self.in_ch = in_ch

        self.pe = PatchEmbed3D(in_ch, dim, patch=patch)
        self.pd = PatchDeEmbed3D(in_ch, dim, patch=patch)

        rope = MultiAxisRoPE(head_dim=dim // heads, num_axes=3)
        self.enc = TransformerStack(depth=depth_enc, dim=dim, heads=heads, rope=rope)
        self.dec = TransformerStack(depth=depth_dec, dim=dim, heads=heads, rope=rope)

        self.to_latent = nn.Linear(dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, dim)

        self.pool_norm = nn.LayerNorm(dim)
        self.proj_head = nn.Linear(dim, embed_dim)

        if use_fsq:
            self.fsq = FSQQuantizer(dim=latent_dim, groups=8, dims_per_group=6, levels_per_dim=4)
        else:
            self.fsq = None

    def forward(self, videos: torch.Tensor) -> ATokenTF3DOutput:
        tokens, thw = self.pe(videos)  # (B,N,C)
        B, N, C = tokens.shape
        T, H, W = thw
        rope_in = MultiAxisRoPE(head_dim=C // (self.enc.blocks[0].attn.heads), num_axes=3)
        sincos = rope_in.build_3d_sincos((T, H, W), tokens.device)
        h = self.enc(tokens, sincos)
        vid_emb = self.proj_head(self.pool_norm(h.mean(dim=1)))
        vid_emb = torch.nn.functional.normalize(vid_emb, dim=-1)
        z = self.to_latent(h)

        q_loss = videos.new_zeros(())
        toks = None
        if self.fsq is not None:
            fsq_out: FSQOutput = self.fsq(z)
            z_q = fsq_out.z_q
            toks = fsq_out.tokens
            q_loss = fsq_out.q_loss
        else:
            z_q = z

        h_dec = self.from_latent(z_q)
        h_dec = self.dec(h_dec, sincos)
        recon = self.pd(h_dec, thw)
        return ATokenTF3DOutput(recon=recon, tokens=toks, losses={"q_loss": q_loss}, video_embed=vid_emb)

