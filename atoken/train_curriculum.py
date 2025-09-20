import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from atoken.losses import GramLoss, LPIPSLoss
from atoken.schedule import WarmupCosine
from atoken.utils import EMAModel
from atoken.data import ImageTextDataset, VideoFolderDataset, VoxelFolderDataset
from atoken.distill import OpenCLIPTeacher


def build_loaders(args):
    loaders = {}
    if args.img_dir:
        loaders["image_rec"] = DataLoader(ImageTextDataset(args.img_dir, captions_file=args.captions, img_size=args.img_size), batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        loaders["image_und"] = loaders["image_rec"]
    else:
        class RandImg(torch.utils.data.Dataset):
            def __len__(self): return 512
            def __getitem__(self, idx): return torch.rand(3, args.img_size, args.img_size), ""
        loaders["image_rec"] = DataLoader(RandImg(), batch_size=args.batch_size, shuffle=True)
        loaders["image_und"] = loaders["image_rec"]

    if args.video_dir:
        loaders["video_rec"] = DataLoader(
            VideoFolderDataset(
                args.video_dir,
                clip_len=args.clip_len,
                frame_stride=args.frame_stride,
                size=args.vid_size,
                captions_file=args.video_captions,
            ),
            batch_size=max(1, args.batch_size // 2), shuffle=True, num_workers=2, pin_memory=True
        )
        loaders["video_und"] = loaders["video_rec"]
    else:
        class RandVid(torch.utils.data.Dataset):
            def __len__(self): return 256
            def __getitem__(self, idx): return torch.rand(3, args.clip_len, args.vid_size, args.vid_size), ""
        loaders["video_rec"] = DataLoader(RandVid(), batch_size=max(1, args.batch_size // 2), shuffle=True)
        loaders["video_und"] = loaders["video_rec"]
    
    if args.voxel_dir or args.include_voxels:
        loaders["voxel_rec"] = DataLoader(VoxelFolderDataset(args.voxel_dir, size=args.voxel_size), batch_size=max(1,args.batch_size//2), shuffle=True, num_workers=2)
    return loaders


def main():
    p = argparse.ArgumentParser(description="AToken Progressive Curriculum Trainer")
    # Data
    p.add_argument("--img_dir", type=str, default=None)
    p.add_argument("--captions", type=str, default=None)
    p.add_argument("--video_dir", type=str, default=None)
    p.add_argument("--video_captions", type=str, default=None)
    p.add_argument("--voxel_dir", type=str, default=None)
    p.add_argument("--include_voxels", action="store_true")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--vid_size", type=int, default=224)
    p.add_argument("--voxel_size", type=int, default=64)
    p.add_argument("--clip_len", type=int, default=16)
    p.add_argument("--frame_stride", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=4)

    # Training
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr_min", type=float, default=3e-5)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--ema", type=float, default=0.9999)
    p.add_argument("--enc_lr_mult", type=float, default=0.1)

    # Model
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--depth_enc", type=int, default=12)
    p.add_argument("--depth_dec", type=int, default=12)
    p.add_argument("--patch", type=int, default=16)
    p.add_argument("--patch_3d_t", type=int, default=4)
    p.add_argument("--patch_3d_h", type=int, default=16)
    p.add_argument("--patch_3d_w", type=int, default=16)
    p.add_argument("--use_fsq", action="store_true")
    p.add_argument("--paper_preset", action="store_true")

    # Loss weights
    p.add_argument("--w_l1", type=float, default=1.0)
    p.add_argument("--w_lpips", type=float, default=0.5)
    p.add_argument("--w_gram", type=float, default=0.5)
    p.add_argument("--w_sem", type=float, default=0.5)
    p.add_argument("--tau", type=float, default=1.0)

    # Distillation teacher
    p.add_argument("--distill", action="store_true")
    p.add_argument("--teacher_model", type=str, default="ViT-B-16")
    p.add_argument("--teacher_pretrained", type=str, default="laion2b_s34b_b88k")

    # Curriculum ratios per stage
    p.add_argument("--stage", type=int, default=1, help="1:img und+rec, 2:+video, 3:+3D, 4:+FSQ")
    p.add_argument("--ratio_img_rec", type=int, default=2)
    p.add_argument("--ratio_img_und", type=int, default=1)
    p.add_argument("--ratio_video_rec", type=int, default=1)
    p.add_argument("--ratio_video_und", type=int, default=1)
    p.add_argument("--ratio_voxel_rec", type=int, default=1)
    p.add_argument("--ratio_voxel_und", type=int, default=0)
    p.add_argument("--eval_interval", type=int, default=0, help="steps between eval logs (0:disable)")

    args = p.parse_args()

    # Paper preset: scale architecture to reported config
    if args.paper_preset:
        args.dim = 1152
        args.heads = 16
        args.depth_enc = 27
        args.depth_dec = 27
        args.patch = 16
        args.lr = 3e-4
        args.lr_min = 3e-5
        args.warmup = 2000

    from atoken.models.atoken_tf import ATokenTransformer
    from atoken.models.atoken_tf3d import ATokenTransformer3D

    device = torch.device(args.device)
    model2d = ATokenTransformer(
        in_ch=3, patch=args.patch, dim=args.dim, depth_enc=args.depth_enc, depth_dec=args.depth_dec, heads=args.heads, latent_dim=48, use_fsq=args.use_fsq and args.stage>=4,
    ).to(device)
    model3d = None
    if args.stage >= 2:
        model3d = ATokenTransformer3D(
            in_ch=3, patch=(args.patch_3d_t, args.patch_3d_h, args.patch_3d_w), dim=args.dim, depth_enc=args.depth_enc//2, depth_dec=args.depth_dec//2, heads=args.heads, latent_dim=48, use_fsq=args.use_fsq and args.stage>=4,
        ).to(device)

    # Parameter groups: encoder vs rest
    enc_params = list(model2d.enc.parameters()) + ([] if model3d is None else list(model3d.enc.parameters()))
    other_params = [p for p in list(model2d.parameters()) + ([] if model3d is None else list(model3d.parameters())) if p not in enc_params]
    opt = optim.AdamW([
        {"params": enc_params, "lr": args.lr * args.enc_lr_mult},
        {"params": other_params, "lr": args.lr},
    ], betas=(0.9, 0.95), weight_decay=0.1)
    sched = WarmupCosine(warmup=args.warmup, total=args.steps, base_lr=args.lr, min_lr=args.lr_min)

    ema2d = EMAModel(model2d, decay=args.ema)
    ema3d = EMAModel(model3d, decay=args.ema) if model3d is not None else None

    loaders = build_loaders(args)
    iters = {k: iter(v) for k, v in loaders.items()}

    teacher = None
    if args.distill:
        try:
            teacher = OpenCLIPTeacher(args.teacher_model, args.teacher_pretrained, device=device)
        except Exception as e:
            print("Distillation disabled:", e)
            teacher = None

    l1 = nn.L1Loss()
    lp = LPIPSLoss(weight=args.w_lpips)
    gram = GramLoss(weight=args.w_gram)

    def next_batch(name):
        it = iters[name]
        try:
            return next(it)
        except StopIteration:
            iters[name] = iter(loaders[name])
            return next(iters[name])

    # Variable resolution sampler per stage (approximate from paper)
    import torch.nn.functional as F
    def sample_img_size(stage: int) -> int:
        if stage <= 1:
            lo, hi = 64, 512
        elif stage == 2:
            lo, hi = 64, 1024
        else:
            lo, hi = 64, 2048
        s = int(torch.randint(lo//16, hi//16 + 1, (1,)).item() * 16)
        return max(64, s)

    def resize_images(imgs: torch.Tensor, size: int) -> torch.Tensor:
        return F.interpolate(imgs, size=(size, size), mode="bilinear", align_corners=False)

    def resize_videos(vids: torch.Tensor, size: int) -> torch.Tensor:
        # vids: (B, C, T, H, W) -> resize spatial only
        B, C, T, H, W = vids.shape
        x = vids.flatten(2, 3)  # (B, C, T*H, W) not correct; do per-frame
        frames = vids.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
        frames = F.interpolate(frames, size=(size, size), mode="bilinear", align_corners=False)
        return frames.view(B, T, C, size, size).permute(0, 2, 1, 3, 4).contiguous()

    # Build task schedule list per epoch based on ratios and stage
    tasks: List[str] = []
    tasks += ["image_rec"] * args.ratio_img_rec
    tasks += ["image_und"] * args.ratio_img_und
    if args.stage >= 2:
        tasks += ["video_rec"] * args.ratio_video_rec
        tasks += ["video_und"] * args.ratio_video_und
    if args.stage >= 3 and args.include_voxels:
        tasks += ["voxel_rec"] * args.ratio_voxel_rec
        tasks += ["voxel_und"] * args.ratio_voxel_und
    if not tasks:
        tasks = ["image_rec"]

    step = 0
    while step < args.steps:
        for task in tasks:
            if step >= args.steps:
                break
            loss = torch.tensor(0.0, device=device)
            if task == "image_rec":
                imgs, _ = next_batch("image_rec")
                imgs = imgs.to(device)
                # variable resolution
                size = sample_img_size(args.stage)
                imgs = resize_images(imgs, size)
                out = model2d(imgs)
                recon = out.recon
                loss = args.w_l1 * l1(recon, imgs) + gram(recon, imgs) + lp(recon, imgs) + out.losses.get("q_loss", torch.tensor(0.0, device=device))
            elif task == "image_und" and teacher is not None:
                imgs, caps = next_batch("image_und")
                imgs = imgs.to(device)
                size = sample_img_size(args.stage)
                imgs = resize_images(imgs, size)
                out = model2d(imgs)
                z_img = out.image_embed
                with torch.no_grad():
                    zt_img = teacher.encode_images(imgs)
                    zt_txt = teacher.encode_texts(caps)
                s_t = (zt_img @ zt_txt.t())
                s_s = (z_img @ zt_txt.t())
                p = torch.softmax(s_t / args.tau, dim=-1)
                q = torch.log_softmax(s_s / args.tau, dim=-1)
                kl = torch.sum(p * (torch.log(p + 1e-9) - q), dim=-1).mean()
                loss = args.w_sem * kl
            elif task == "video_rec" and model3d is not None:
                vids, _ = next_batch("video_rec")
                vids = vids.to(device)
                size = sample_img_size(max(2, args.stage))  # larger window in later stages
                vids = resize_videos(vids, size)
                out = model3d(vids)
                recon = out.recon
                loss = args.w_l1 * l1(recon, vids) + out.losses.get("q_loss", torch.tensor(0.0, device=device))
            elif task == "video_und" and model3d is not None and teacher is not None:
                vids, caps = next_batch("video_und")
                vids = vids.to(device)
                size = sample_img_size(max(2, args.stage))
                vids = resize_videos(vids, size)
                out = model3d(vids)
                z_vid = out.video_embed  # (B, E)
                with torch.no_grad():
                    zt_txt = teacher.encode_texts(caps)
                # SigLIP-style sigmoid loss on positive pairs only
                sim = (z_vid * zt_txt).sum(dim=-1)
                loss = args.w_sem * torch.nn.functional.softplus(-sim).mean()
            elif task == "voxel_rec" and model3d is not None:
                vox = next_batch("voxel_rec")
                vox = vox.to(device)
                out = model3d(vox)
                recon = out.recon
                loss = args.w_l1 * l1(recon, vox) + out.losses.get("q_loss", torch.tensor(0.0, device=device))
            elif task == "voxel_und" and model3d is not None and teacher is not None:
                # no captions for voxels by default; skip unless provided separately
                pass

            # Update LR
            lr_now = sched.get_lr(step)
            for pg in opt.param_groups:
                # Keep encoder vs others ratio
                if pg["lr"] <= args.lr * args.enc_lr_mult + 1e-12:
                    pg["lr"] = lr_now * args.enc_lr_mult
                else:
                    pg["lr"] = lr_now

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model2d.parameters()) + ([] if model3d is None else list(model3d.parameters())), 1.0)
            opt.step()
            ema2d.update(model2d)
            if ema3d is not None:
                ema3d.update(model3d)

            if step % 50 == 0:
                print(f"step {step} | task {task} | loss {float(loss):.4f} | lr {lr_now:.6f}")

            # Simple image PSNR eval hook
            if args.eval_interval and step % args.eval_interval == 0 and task == "image_rec":
                from atoken.eval import compute_psnr
                psnr = compute_psnr(recon.detach().cpu(), imgs.detach().cpu())
                print(f"eval | PSNR {psnr:.2f} dB")
            step += 1

    out_dir = Path("checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model2d.state_dict()}, out_dir / "atoken_tf2d.pt")
    torch.save({"model": ema2d.ema.state_dict()}, out_dir / "atoken_tf2d_ema.pt")
    if model3d is not None:
        torch.save({"model": model3d.state_dict()}, out_dir / "atoken_tf3d.pt")
        torch.save({"model": ema3d.ema.state_dict()}, out_dir / "atoken_tf3d_ema.pt")
    print("Saved checkpoints to", out_dir)


if __name__ == "__main__":
    main()
