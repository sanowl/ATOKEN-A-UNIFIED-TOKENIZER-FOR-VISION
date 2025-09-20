from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset


class VideoFolderDataset(Dataset):
    def __init__(self, root: str, clip_len: int = 16, frame_stride: int = 2, size: int = 224, captions_file: Optional[str] = None):
        self.root = Path(root)
        self.clip_len = clip_len
        self.stride = frame_stride
        self.size = size
        self.files = [p for p in self.root.rglob("*.mp4")]
        if not self.files:
            self.files = [p for p in self.root.rglob("*.mov")]
        try:
            from torchvision import transforms
            self.resize = transforms.Resize((size, size))
        except Exception:
            self.resize = None
        try:
            from torchvision.io import read_video
            self.read_video = read_video
        except Exception as e:
            raise RuntimeError("torchvision.io.read_video required for VideoFolderDataset") from e
        # optional captions mapping
        self.captions = {}
        if captions_file and Path(captions_file).exists():
            for line in Path(captions_file).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rel, cap = line.split("\t", 1) if "\t" in line else (line, "")
                self.captions[rel] = cap

    def __len__(self) -> int:
        return max(1, len(self.files))

    def __getitem__(self, idx):
        if not self.files:
            # fallback random clip
            T = self.clip_len
            return torch.rand(3, T, self.size, self.size), ""
        path = self.files[idx % len(self.files)]
        vframes, _, _ = self.read_video(str(path), pts_unit='sec')  # (N, H, W, 3)
        # sample clip
        frames = vframes[:: self.stride]
        if frames.shape[0] < self.clip_len:
            # loop pad
            reps = (self.clip_len + frames.shape[0] - 1) // frames.shape[0]
            frames = frames.repeat(reps, 1, 1, 1)
        frames = frames[: self.clip_len]
        frames = frames.permute(0, 3, 1, 2).contiguous().float() / 255.0  # (T,C,H,W)
        if self.resize is not None:
            frames = torch.stack([self.resize(f) for f in frames], dim=0)
        # return (C, T, H, W)
        clip = frames.permute(1, 0, 2, 3).contiguous()
        rel = str(path.relative_to(self.root).as_posix())
        cap = self.captions.get(rel, "")
        return clip, cap
