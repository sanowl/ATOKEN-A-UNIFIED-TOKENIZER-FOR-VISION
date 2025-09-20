import torch


def main():
    from atoken.models import AToken

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AToken(
        in_ch=3,
        dim=128,
        depth_enc=1,
        depth_dec=1,
        patch=8,
        num_levels=3,
        codebook_size=512,
        commitment_weight=0.25,
        recon_loss_type="l1",
    ).to(device)

    imgs = torch.rand(2, 3, 128, 128, device=device)
    out = model(imgs)
    print("Output recon shape:", tuple(out.recon.shape))
    print("Tokens shape:", tuple(out.tokens.shape))
    print(
        "Losses:",
        {
            "total": float(out.total_loss),
            "recon": float(out.recon_loss),
            "vq": float(out.vq_loss),
        },
    )

    # Encode -> Decode
    toks = model.encode(imgs)
    rec = model.decode(toks)
    diff = (rec - out.recon).abs().mean().item()
    print("Encode/Decode recon diff vs forward:", f"{diff:.6f}")

    # Sequence packing example
    seq = model.tokens_to_sequence(toks)
    print("Seq shape (B, H*W, L):", tuple(seq.shape))


if __name__ == "__main__":
    main()

