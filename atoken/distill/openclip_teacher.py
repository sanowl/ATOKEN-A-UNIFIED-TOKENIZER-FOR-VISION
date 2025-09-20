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
        # images assumed in [0,1]; preprocess expects PIL or normalized; we approximate with model.visual.forward on normalized tensors
        x = images * 255.0
        # Fallback: rely on model.encode_image which expects transforms; we do minimal compat
        return torch.nn.functional.normalize(self.model.encode_image(images.to(self.device)), dim=-1)

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        toks = self.tokenizer(texts).to(self.device)
        return torch.nn.functional.normalize(self.model.encode_text(toks), dim=-1)

