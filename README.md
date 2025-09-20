AToken: A Unified Tokenizer for Vision (Paper‑Aligned)

This repo implements the core ideas from “AToken: A Unified Tokenizer for Vision” (arXiv:2509.14476) in PyTorch:

- Pure transformer tokenizer/decoder with patch embeddings and multi‑axis RoPE
- Continuous latent head (D=48) and optional FSQ discrete tokens (8×6D @ 4 levels → 4096 vocab)
- Adversarial‑free training losses: L1 + Gram (+ optional LPIPS and CLIP/SigLIP‑style distillation)
- Warmup + cosine LR schedule, AdamW(0.9,0.95,wd=0.1), EMA(0.9999), encoder LR multiplier 0.1×
- Progressive curriculum scaffolding across images → video → 3D → FSQ

Quickstart
- Install: `pip install torch torchvision` (optional: `lpips open-clip-torch`)
- Random transformer demo: `python scripts/demo_tf.py`
- Image training (2D):
  `python atoken/train_tf.py --data_dir /path/to/images --epochs 1 --steps 2000 --use_fsq`
- Progressive curriculum (images → video → 3D) with optional distillation:
  `python atoken/train_curriculum.py --img_dir /path/to/images --video_dir /path/to/videos --include_voxels --distill --captions /path/to/captions.txt --stage 2 --steps 5000`

Key Files
- atoken/models/patch.py — 2D patch embed/de‑embed
- atoken/models/patch3d.py — 3D patch embed/de‑embed (video/voxels)
- atoken/models/transformer.py — Transformer stack with multi‑axis RoPE attention
- atoken/models/atoken_tf.py — Transformer AToken (2D) with FSQ and image embed head
- atoken/models/atoken_tf3d.py — Transformer AToken (3D) for video/voxels with embed head
- atoken/ops/rope.py — 2D/3D RoPE builders + application
- atoken/quantizer/fsq.py — Finite Scalar Quantizer (48D → 8 tokens/patch)
- atoken/losses/{gram,lpips_wrap,clip_distill}.py — Adversarial‑free loss suite + distill hook
- atoken/distill/openclip_teacher.py — OpenCLIP teacher wrapper for distillation
- atoken/data/{image_text,video,voxels}.py — Minimal datasets for modalities
- atoken/schedule.py — Warmup + cosine scheduler
- atoken/utils/ema.py — EMA wrapper
- atoken/train_tf.py — Image training with L1+Gram(+LPIPS), EMA, scheduler, FSQ
- atoken/train_curriculum.py — Progressive curriculum across modalities with optional distillation
- paper/AToken.pdf, paper/AToken.txt — Paper and extracted text (for reference)

RVQ Baseline (optional)
- atoken/quantizer/rq.py, atoken/models/atoken.py, atoken/train.py, scripts/demo_random.py
- Kept for comparison; safe to remove if you want a strict paper‑only codebase.

Notes aligned to the paper
- Architecture scale: paper uses 27 blocks, d=1152, 16 heads; defaults here are lighter for single‑GPU.
- Loss recipe: image rec uses L1 + LPIPS + Gram + CLIP distillation. Video/3D rec use L1 by default, with understanding losses via sigmoid/CLIP style when configured.
- Discrete tokens (Stage 4): FSQ on a 48‑D latent split into 8×6D, 4 levels per dim → 4096 vocab each.
- Training: AdamW(0.9,0.95,wd=0.1), warmup 2k to 3e‑4, cosine to 3e‑5, EMA 0.9999, encoder LR = 0.1× base.

