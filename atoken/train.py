import argparse
import math
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def try_import_vision():
    try:
        import torchvision  # noqa: F401
        from torchvision import transforms, datasets
        return transforms, datasets
    except Exception:
        return None, None


def build_dataloaders(
    data_dir: Optional[str], batch_size: int, img_size: int
) -> DataLoader:
    transforms, datasets = try_import_vision()

    if transforms is None or datasets is None or (data_dir and not Path(data_dir).exists()):
        # Fallback: random dataset if torchvision or path is unavailable
        class RandomDataset(torch.utils.data.Dataset):
            def __init__(self, n: int = 512, size: int = 256) -> None:
                self.n = n
                self.size = size

            def __len__(self):
                return self.n

            def __getitem__(self, idx):
                img = torch.rand(3, self.size, self.size)
                return img

        ds = RandomDataset(n=512, size=img_size)
        return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    tfm = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    ds = datasets.ImageFolder(data_dir, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)


def main():
    parser = argparse.ArgumentParser(description="Train a minimal AToken model")
    parser.add_argument("--data_dir", type=str, default=None, help="ImageFolder root (optional)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Model hparams
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--depth_enc", type=int, default=2)
    parser.add_argument("--depth_dec", type=int, default=2)
    parser.add_argument("--patch", type=int, default=8)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--codebook_size", type=int, default=1024)
    parser.add_argument("--commit", type=float, default=0.25)
    parser.add_argument("--recon_loss", type=str, default="l1", choices=["l1", "mse"])

    args = parser.parse_args()

    from atoken.models import AToken

    device = torch.device(args.device)
    model = AToken(
        in_ch=3,
        dim=args.dim,
        depth_enc=args.depth_enc,
        depth_dec=args.depth_dec,
        patch=args.patch,
        num_levels=args.levels,
        codebook_size=args.codebook_size,
        commitment_weight=args.commit,
        recon_loss_type=args.recon_loss,
    ).to(device)

    loader = build_dataloaders(args.data_dir, args.batch_size, args.img_size)

    opt = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-4)

    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        for it, batch in enumerate(loader):
            if isinstance(batch, (tuple, list)):
                images = batch[0]
            else:
                images = batch
            images = images.to(device)

            out = model(images)
            loss = out.total_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if global_step % 50 == 0:
                stats = {k: float(v) for k, v in out.stats.items()}
                print(
                    f"epoch {epoch} it {it} step {global_step} | "
                    f"loss {loss.item():.4f} | recon {stats['recon_loss']:.4f} | "
                    f"vq {stats['vq_loss']:.4f} | perp {stats['perplexity']:.1f} | "
                    f"latent {int(stats['latent_h'])}x{int(stats['latent_w'])}"
                )

            global_step += 1

    # Save a small checkpoint
    ckpt = {
        "model": model.state_dict(),
        "hparams": {
            "dim": args.dim,
            "depth_enc": args.depth_enc,
            "depth_dec": args.depth_dec,
            "patch": args.patch,
            "levels": args.levels,
            "codebook_size": args.codebook_size,
            "commit": args.commit,
            "recon_loss": args.recon_loss,
        },
    }
    out_dir = Path("checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_dir / "atoken-minimal.pt")
    print("Saved", out_dir / "atoken-minimal.pt")


if __name__ == "__main__":
    main()

