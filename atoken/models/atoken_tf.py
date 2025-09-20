from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from atoken.models.patch import PatchEmbed2D, PatchDeEmbed2D
from atoken.models.transformer import TransformerStack
from atoken.ops import MultiAxisRoPE
from atoken.quantizer.fsq import FSQQuantizer, FSQOutput


@dataclass
class ATokenTFOutput:
    recon: torch.Tensor
    tokens: Optional[torch.Tensor]
    losses: Dict[str, torch.Tensor]
    image_embed: Optional[torch.Tensor]


class ATokenTransformer(nn.Module):
    """
    Transformer-based AToken (image-focused) with optional FSQ discrete tokens.

    - Patch16 embed -> Transformer encoder -> latent head (D_lat=48)
    - Optional FSQ quantization into 8 tokens/patch (4096 vocab each)
    - Transformer decoder -> Patch de-embed to image
    """

    def __init__(
        self,
        in_ch: int = 3,
        patch: int = 16,
        dim: int = 384,  # smaller default than paper to fit single GPU
        depth_enc: int = 6,
        depth_dec: int = 6,
        heads: int = 6,
        latent_dim: int = 48,  # FSQ expects 48
        use_fsq: bool = True,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.patch = patch
        self.use_fsq = use_fsq
        self.in_ch = in_ch

        # Patch embed / de-embed
        self.pe = PatchEmbed2D(in_ch, dim, patch=patch)
        self.pd = PatchDeEmbed2D(in_ch, dim, patch=patch)

        # RoPE + stacks
        rope = MultiAxisRoPE(head_dim=dim // heads, num_axes=2)
        self.enc = TransformerStack(depth=depth_enc, dim=dim, heads=heads, rope=rope)
        self.dec = TransformerStack(depth=depth_dec, dim=dim, heads=heads, rope=rope)

        # Heads: latent projection and image head
        self.to_latent = nn.Linear(dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, dim)

        # Image understanding head (pooled embedding)
        self.pool_norm = nn.LayerNorm(dim)
        self.proj_head = nn.Linear(dim, embed_dim)

        if use_fsq:
            self.fsq = FSQQuantizer(dim=latent_dim, groups=8, dims_per_group=6, levels_per_dim=4)
        else:
            self.fsq = None

    def forward(self, images: torch.Tensor) -> ATokenTFOutput:
        tokens, hw = self.pe(images)  # (B,N,C)

        # build 2D sincos
        B, N, C = tokens.shape
        H, W = hw
        rope_in = MultiAxisRoPE(head_dim=C // (self.enc.blocks[0].attn.heads), num_axes=2)
        sincos = rope_in.build_2d_sincos((H, W), tokens.device)

        h = self.enc(tokens, sincos)
        # pooled embedding for understanding
        img_emb = self.proj_head(self.pool_norm(h.mean(dim=1)))  # (B, E)
        img_emb = torch.nn.functional.normalize(img_emb, dim=-1)
        z = self.to_latent(h)  # (B,N,D_lat)

        q_loss = images.new_zeros(())
        toks = None
        if self.fsq is not None:
            fsq_out: FSQOutput = self.fsq(z)
            z_q = fsq_out.z_q
            toks = fsq_out.tokens  # (B,N,G)
            q_loss = fsq_out.q_loss
        else:
            z_q = z

        h_dec = self.from_latent(z_q)
        h_dec = self.dec(h_dec, sincos)
        recon = self.pd(h_dec, hw)

        return ATokenTFOutput(recon=recon, tokens=toks, losses={"q_loss": q_loss}, image_embed=img_emb)

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        assert self.fsq is not None, "FSQ not enabled"
        tokens, hw = self.pe(images)
        B, N, C = tokens.shape
        H, W = hw
        rope_in = MultiAxisRoPE(head_dim=C // (self.enc.blocks[0].attn.heads), num_axes=2)
        sincos = rope_in.build_2d_sincos((H, W), tokens.device)
        h = self.enc(tokens, sincos)
        z = self.to_latent(h)
        fsq_out: FSQOutput = self.fsq(z)
        return fsq_out.tokens  # (B,N,G)

    @torch.no_grad()
    def decode(self, tokens: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        assert self.fsq is not None, "FSQ not enabled"
        B, N, G = tokens.shape
        z_q = self.fsq.decode_codes(tokens)
        C = self.from_latent.in_features
        rope_in = MultiAxisRoPE(head_dim=C // (self.dec.blocks[0].attn.heads), num_axes=2)
        sincos = rope_in.build_2d_sincos(hw, tokens.device)
        h_dec = self.from_latent(z_q)
        h_dec = self.dec(h_dec, sincos)
        recon = self.pd(h_dec, hw)
        return recon
