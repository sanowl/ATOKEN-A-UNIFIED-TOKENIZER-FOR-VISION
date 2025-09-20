from typing import Optional, Tuple

import torch
import torch.nn as nn


class MultiAxisRoPE(nn.Module):
    """
    Multi-axis Rotary Positional Embeddings.

    Splits the head dimension across provided axes (e.g., x,y for 2D; t,y,x for 3D)
    and applies separate rotary embeddings to each split. Leftover channels (if any)
    are passed through unchanged.

    Usage:
        rope = MultiAxisRoPE(head_dim, axes=(H, W)) or (T, H, W)
        sincos = rope.build_sincos((T?, H, W), device)
        q, k = apply_rope(q, k, sincos)  # q,k: (B, heads, tokens, head_dim)
    """

    def __init__(self, head_dim: int, num_axes: int, rope_theta: float = 10000.0) -> None:
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.head_dim = head_dim
        self.num_axes = num_axes
        self.theta = rope_theta

        # Split head_dim evenly across axes (use floor) and leave remainder as pass-through
        base = (head_dim // 2) // num_axes * 2  # ensure even per-axis
        self.axis_dim = base
        self.pass_dim = head_dim - base * num_axes

        # Frequencies per axis chunk
        if self.axis_dim > 0:
            dim_half = self.axis_dim // 2
            inv_freq = 1.0 / (self.theta ** (torch.arange(0, dim_half, 1).float() / dim_half))
        else:
            inv_freq = torch.zeros(0)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _axis_sin_cos(self, n: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.axis_dim == 0:
            return torch.zeros(0, device=device), torch.zeros(0, device=device)
        t = torch.arange(n, device=device).float()  # (n,)
        freqs = torch.einsum("n,d->nd", t, self.inv_freq)  # (n, dim_half)
        emb = torch.cat([freqs, freqs], dim=-1)  # (n, axis_dim)
        return emb.sin(), emb.cos()  # (n, axis_dim)

    @torch.no_grad()
    def build_2d_sincos(self, hw: Tuple[int, int], device: torch.device):
        H, W = hw
        axis_dim, pass_dim = self.axis_dim, self.pass_dim
        y_s, y_c = self._axis_sin_cos(H, device)
        x_s, x_c = self._axis_sin_cos(W, device)
        # Broadcast to tokens (H*W, axis_dim)
        y_s_tok = y_s[:, None, :].expand(H, W, axis_dim).reshape(H * W, axis_dim)
        y_c_tok = y_c[:, None, :].expand(H, W, axis_dim).reshape(H * W, axis_dim)
        x_s_tok = x_s[None, :, :].expand(H, W, axis_dim).reshape(H * W, axis_dim)
        x_c_tok = x_c[None, :, :].expand(H, W, axis_dim).reshape(H * W, axis_dim)
        return {
            "dims": (axis_dim, pass_dim),
            "y": (y_s_tok, y_c_tok),  # (N, axis_dim)
            "x": (x_s_tok, x_c_tok),
        }

    @torch.no_grad()
    def build_3d_sincos(self, thw: Tuple[int, int, int], device: torch.device):
        T, H, W = thw
        axis_dim, pass_dim = self.axis_dim, self.pass_dim
        t_s, t_c = self._axis_sin_cos(T, device)
        y_s, y_c = self._axis_sin_cos(H, device)
        x_s, x_c = self._axis_sin_cos(W, device)
        # Tokens in (T,H,W) raster order
        t_s_tok = t_s[:, None, None, :].expand(T, H, W, axis_dim).reshape(T * H * W, axis_dim)
        t_c_tok = t_c[:, None, None, :].expand(T, H, W, axis_dim).reshape(T * H * W, axis_dim)
        y_s_tok = y_s[None, :, None, :].expand(T, H, W, axis_dim).reshape(T * H * W, axis_dim)
        y_c_tok = y_c[None, :, None, :].expand(T, H, W, axis_dim).reshape(T * H * W, axis_dim)
        x_s_tok = x_s[None, None, :, :].expand(T, H, W, axis_dim).reshape(T * H * W, axis_dim)
        x_c_tok = x_c[None, None, :, :].expand(T, H, W, axis_dim).reshape(T * H * W, axis_dim)
        return {
            "dims": (axis_dim, pass_dim),
            "t": (t_s_tok, t_c_tok),
            "y": (y_s_tok, y_c_tok),
            "x": (x_s_tok, x_c_tok),
        }


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, sincos):
    """
    Apply multi-axis RoPE to q,k.
    q,k: (B, H, N, Dh)
    sincos: from MultiAxisRoPE.build_sincos
    """
    B, Hh, N, Dh = q.shape
    axis_dim, pass_dim = sincos["dims"]
    if axis_dim == 0:
        return q, k

    # This function assumes sincos carries token-wise sin/cos per axis as built by
    # build_2d_sincos or build_3d_sincos. We support 2D (x,y) or 3D (t,y,x).
    # Split head_dim: [axis(x), axis(y), axis(t?)] + pass
    chunks = []
    has_t = ("t" in sincos)
    if axis_dim > 0:
        if has_t:
            chunks += [axis_dim, axis_dim, axis_dim]  # t,y,x
        else:
            chunks += [axis_dim, axis_dim]  # y,x
    if pass_dim > 0:
        chunks.append(pass_dim)
    q_chunks = list(torch.split(q, chunks, dim=-1))
    k_chunks = list(torch.split(k, chunks, dim=-1))

    out_q_chunks = []
    out_k_chunks = []

    idx = 0
    if has_t and axis_dim > 0:
        ts, tc = sincos["t"]  # (N, axis_dim)
        ts = ts.view(1, 1, -1, axis_dim)
        tc = tc.view(1, 1, -1, axis_dim)
        qt = q_chunks[idx]
        kt = k_chunks[idx]
        idx += 1
        qt = (qt * tc) + (_rotate_half(qt) * ts)
        kt = (kt * tc) + (_rotate_half(kt) * ts)
        out_q_chunks.append(qt)
        out_k_chunks.append(kt)

    if axis_dim > 0:
        ys, yc = sincos["y"]
        ys = ys.view(1, 1, -1, axis_dim)
        yc = yc.view(1, 1, -1, axis_dim)
        qy = q_chunks[idx]
        ky = k_chunks[idx]
        idx += 1
        qy = (qy * yc) + (_rotate_half(qy) * ys)
        ky = (ky * yc) + (_rotate_half(ky) * ys)
        out_q_chunks.append(qy)
        out_k_chunks.append(ky)

        xs, xc = sincos["x"]
        xs = xs.view(1, 1, -1, axis_dim)
        xc = xc.view(1, 1, -1, axis_dim)
        qx = q_chunks[idx]
        kx = k_chunks[idx]
        idx += 1
        qx = (qx * xc) + (_rotate_half(qx) * xs)
        kx = (kx * xc) + (_rotate_half(kx) * xs)
        out_q_chunks.append(qx)
        out_k_chunks.append(kx)

    if pass_dim > 0:
        out_q_chunks.append(q_chunks[idx])
        out_k_chunks.append(k_chunks[idx])

    return torch.cat(out_q_chunks, dim=-1), torch.cat(out_k_chunks, dim=-1)
