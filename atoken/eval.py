from typing import List, Tuple

import torch
import torch.nn.functional as F


def compute_psnr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    """PSNR for images or videos in [0,1]. Shapes must match."""
    mse = F.mse_loss(x, y).item()
    if mse <= eps:
        return 100.0
    return float(10.0 * torch.log10(torch.tensor(1.0 / mse)).item())


def rfid_score(real_imgs: torch.Tensor, fake_imgs: torch.Tensor) -> float:
    """Placeholder rFID (FID in reconstruction). Requires Inception features in practice.
    Returns 0.0 as stub.
    """
    return 0.0


def rfvd_score(real_vids: torch.Tensor, fake_vids: torch.Tensor) -> float:
    """Placeholder rFVD. Requires I3D features in practice. Returns 0.0 as stub."""
    return 0.0


@torch.no_grad()
def zero_shot_accuracy(img_emb: torch.Tensor, class_texts: List[str], teacher) -> float:
    """Zero-shot top-1 accuracy stub using OpenCLIP teacher for text embeddings.
    Returns percent correct if ground-truth labels are provided via class_texts mapping; else 0.
    """
    # Placeholder: requires labels per image to compute accuracy.
    return 0.0

