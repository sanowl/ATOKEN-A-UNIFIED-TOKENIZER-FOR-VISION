from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset


class ImageTextDataset(Dataset):
    """
    ImageFolder with optional captions file.
    captions.txt format: `relative/path/to/image.jpg<TAB>caption text`
    If not provided, uses an empty caption.
    """

    def __init__(self, root: str, captions_file: Optional[str] = None, img_size: int = 256):
        self.root = Path(root)
        self.img_size = img_size
        try:
            from torchvision import datasets, transforms
        except Exception as e:
            raise RuntimeError("torchvision required for ImageTextDataset") from e
        self.tfm = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        self.ds = datasets.ImageFolder(self.root, transform=self.tfm)

        self.captions = {}
        if captions_file and Path(captions_file).exists():
            for line in Path(captions_file).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rel, cap = line.split("\t", 1) if "\t" in line else (line, "")
                self.captions[str(Path(rel).as_posix())] = cap

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, str]:
        img, _ = self.ds[idx]
        path, _ = self.ds.samples[idx]
        # find relative path from root
        rel = str(Path(path).relative_to(self.root).as_posix())
        cap = self.captions.get(rel, "")
        return img, cap

