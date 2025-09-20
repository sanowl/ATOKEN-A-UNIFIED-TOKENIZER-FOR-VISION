from typing import List, Optional, Tuple

import torch


class OpenCLIPTeacher:
    def __init__(self, model_name: str = "ViT-B-16", pretrained: str = "laion2b_s34b_b88k", device: Optional[torch.device] = None):
        try:
            import open_clip
        except Exception as e:
            raise RuntimeError("open-clip-torch not installed") from e
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images (B,3,H,W) in [0,1] using OpenCLIP preprocess.

        Falls back to direct encode if torchvision/PIL not available.
        """
        imgs = images.to(self.device)
        try:
            from torchvision.transforms.functional import to_pil_image
            # Apply preprocess per image
            proc = []
            for i in range(imgs.shape[0]):
                pil = to_pil_image(imgs[i].cpu().clamp(0, 1))
                proc.append(self.preprocess(pil).to(self.device))
            x = torch.stack(proc, dim=0)
            feats = self.model.encode_image(x)
        except Exception:
            # Fallback without preprocess
            feats = self.model.encode_image(imgs)
        return torch.nn.functional.normalize(feats, dim=-1)

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        toks = self.tokenizer(texts).to(self.device)
        return torch.nn.functional.normalize(self.model.encode_text(toks), dim=-1)
