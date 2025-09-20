from typing import Optional

import torch
import torch.nn as nn


class CLIPDistillLoss(nn.Module):
    """
    Optional CLIP/SigLIP distillation placeholder.
    If open-clip is installed and models are provided, computes KL between similarity
    distributions. Otherwise returns zero.
    """

    def __init__(self, weight: float = 0.0, temperature: float = 1.0):
        super().__init__()
        self.weight = float(weight)
        self.tau = float(temperature)
        try:
            import open_clip  # noqa: F401
            self.has_clip = True
        except Exception:
            self.has_clip = False

    def forward(
        self,
        image_embed_student: Optional[torch.Tensor] = None,
        image_embed_teacher: Optional[torch.Tensor] = None,
        text_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.has_clip or self.weight == 0.0:
            return torch.tensor(0.0, device=image_embed_student.device if image_embed_student is not None else 'cpu')
        if image_embed_student is None or image_embed_teacher is None or text_embed is None:
            return torch.tensor(0.0, device='cpu')
        # cosine similarities
        s_teacher = (image_embed_teacher @ text_embed.t())
        s_student = (image_embed_student @ text_embed.t())
        p = torch.softmax(s_teacher / self.tau, dim=-1)
        q = torch.log_softmax(s_student / self.tau, dim=-1)
        kl = torch.sum(p * (torch.log(p + 1e-9) - q), dim=-1).mean()
        return self.weight * kl

