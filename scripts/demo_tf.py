import torch


def main():
    from atoken.models import ATokenTransformer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ATokenTransformer(
        in_ch=3,
        patch=16,
        dim=256,
        depth_enc=4,
        depth_dec=4,
        heads=8,
        latent_dim=48,
        use_fsq=True,
    ).to(device)
    model.eval()

    imgs = torch.rand(2, 3, 128, 128, device=device)
    out = model(imgs)
    print("Recon shape:", tuple(out.recon.shape))
    print("Tokens shape:", None if out.tokens is None else tuple(out.tokens.shape))

    # Encode/Decode roundtrip
    toks = model.encode(imgs)
    H = imgs.shape[2] // model.patch
    W = imgs.shape[3] // model.patch
    rec = model.decode(toks, (H, W))
    print("Roundtrip diff:", float((rec - out.recon).abs().mean()))


if __name__ == "__main__":
    main()

