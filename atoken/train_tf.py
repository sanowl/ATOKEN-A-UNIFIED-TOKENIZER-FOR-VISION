import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from atoken.losses import GramLoss, LPIPSLoss
from atoken.schedule import WarmupCosine
from atoken.utils import EMAModel


def try_import_vision():
    try:
        import torchvision  # noqa: F401
        from torchvision import transforms, datasets
        return transforms, datasets
    except Exception:
        return None, None


def build_dataloader(data_dir, batch_size, img_size):
    transforms, datasets = try_import_vision()
    if transforms is None or datasets is None or (data_dir and not Path(data_dir).exists()):
        class RandomDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 512
            def __getitem__(self, idx):
                return torch.rand(3, img_size, img_size)
        return DataLoader(RandomDataset(), batch_size=batch_size, shuffle=True, num_workers=0)
    tfm = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    ds = datasets.ImageFolder(data_dir, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr_min", type=float, default=3e-5)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--ema", type=float, default=0.9999)
    p.add_argument("--enc_lr_mult", type=float, default=0.1)
    # Model params closer to paper but smaller
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--depth_enc", type=int, default=12)
    p.add_argument("--depth_dec", type=int, default=12)
    p.add_argument("--patch", type=int, default=16)
    p.add_argument("--use_fsq", action="store_true")

    # Loss weights
    p.add_argument("--w_l1", type=float, default=1.0)
    p.add_argument("--w_lpips", type=float, default=0.5)
    p.add_argument("--w_gram", type=float, default=0.5)

    args = p.parse_args()

    from atoken.models.atoken_tf import ATokenTransformer

    device = torch.device(args.device)
    model = ATokenTransformer(
        in_ch=3,
        patch=args.patch,
        dim=args.dim,
        depth_enc=args.depth_enc,
        depth_dec=args.depth_dec,
        heads=args.heads,
        latent_dim=48,
        use_fsq=args.use_fsq,
    ).to(device)

    # Parameter groups: encoder vs rest (for LR multiplier)
    enc_params = list(model.enc.parameters())
    other_params = [p for p in model.parameters() if p not in enc_params]
    opt = optim.AdamW([
        {"params": enc_params, "lr": args.lr * args.enc_lr_mult},
        {"params": other_params, "lr": args.lr},
    ], betas=(0.9, 0.95), weight_decay=0.1)

    sched = WarmupCosine(warmup=args.warmup, total=args.steps, base_lr=args.lr, min_lr=args.lr_min)
    ema = EMAModel(model, decay=args.ema)

    loader = build_dataloader(args.data_dir, args.batch_size, args.img_size)
    l1 = nn.L1Loss()
    lp = LPIPSLoss(weight=args.w_lpips)
    gram = GramLoss(weight=args.w_gram)

    step = 0
    model.train()
    for epoch in range(args.epochs):
        for batch in loader:
            if isinstance(batch, (tuple, list)):
                images = batch[0]
            else:
                images = batch
            images = images.to(device)

            out = model(images)
            recon = out.recon
            loss_l1 = args.w_l1 * l1(recon, images)
            loss_lp = lp(recon, images)
            loss_gram = gram(recon, images)
            loss = loss_l1 + loss_lp + loss_gram + out.losses.get("q_loss", torch.tensor(0.0, device=device))

            # Update LR
            lr_now = sched.get_lr(step)
            for pg in opt.param_groups:
                base = args.lr if pg["lr"] == args.lr else args.lr * args.enc_lr_mult
                # scale proportionally to keep encoder/remainder ratio
                if base == args.lr:
                    pg["lr"] = lr_now
                else:
                    pg["lr"] = lr_now * args.enc_lr_mult

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)

            if step % 50 == 0:
                print(
                    f"step {step} | loss {loss.item():.4f} | l1 {loss_l1.item():.4f} | "
                    f"lp {float(loss_lp):.4f} | gram {float(loss_gram):.4f} | lr {lr_now:.6f}"
                )
            step += 1
            if step >= args.steps:
                break
        if step >= args.steps:
            break

    out_dir = Path("checkpoints")
    out_dir.mkdir(exist_ok=True, parents=True)
    torch.save({"model": model.state_dict()}, out_dir / "atoken_tf.pt")
    torch.save({"model": ema.ema.state_dict()}, out_dir / "atoken_tf_ema.pt")
    print("Saved checkpoints to", out_dir)


if __name__ == "__main__":
    main()

