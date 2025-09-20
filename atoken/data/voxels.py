from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset


class VoxelFolderDataset(Dataset):
    def __init__(self, root: Optional[str] = None, size: int = 64):
        self.size = size
        self.files = []
        if root is not None and Path(root).exists():
            self.files = list(Path(root).rglob("*.npy"))

    def __len__(self) -> int:
        return 512 if not self.files else len(self.files)

    def __getitem__(self, idx):
        if not self.files:
            # Random occupancy grid (C=1)
            return (torch.rand(1, self.size, self.size, self.size) > 0.7).float()
        import numpy as np
        arr = np.load(self.files[idx % len(self.files)])
        x = torch.tensor(arr).float()
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return x

