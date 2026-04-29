# sample_sr.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

import config
from dataset import make_dataloaders
from unet import UNetV2
from diffusion import GaussianDiffusion, center_stride_downsample


def denormalize(x, mean, std):
    return x * std + mean


def plot_one(hr_true, lr_cond, sr, save_path, title=""):
    """
    hr_true: (256, 256)
    lr_cond: (4, 4)
    sr: (256, 256)
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    vmin = min(hr_true.min(), sr.min())
    vmax = max(hr_true.max(), sr.max())

    im0 = axes[0].imshow(hr_true, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[0].set_title("HR truth 256x256")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(lr_cond, origin="lower", cmap="RdBu_r")
    axes[1].set_title("LR condition 4x4")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(sr, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[2].set_title("Diffusion-SR 256x256")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    # 把 LR 条件 nearest 插值到 256x256，仅用于视觉参考
    lr_tensor = torch.tensor(lr_cond)[None, None]
    lr_up = F.interpolate(lr_tensor, size=(256, 256), mode="nearest")[0, 0].numpy()

    im3 = axes[3].imshow(lr_up, origin="lower", cmap="RdBu_r")
    axes[3].set_title("LR nearest upsample")
    axes[3].axis("off")
    plt.colorbar(im3, ax=axes[3], fraction=0.046)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def load_model(device):
    model = UNetV2(
        in_channels=config.IN_CHANNELS,
        base_channels=config.BASE_CHANNELS,
        channel_mults=config.CHANNEL_MULTS,
        time_emb_dim=config.TIME_EMB_DIM,
        dropout=config.DROPOUT,
    ).to(device)

    ckpt = torch.load(config.BEST_CKPT, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"[Model] Loaded checkpoint: {config.BEST_CKPT}")
    print(f"[Model] Epoch: {ckpt.get('epoch')}, best_val: {ckpt.get('best_val')}")

    return model


def sample_sr(num_samples=8, num_ensemble=1):
    device = torch.device(config.DEVICE)
    print(f"[Device] {device}")

    train_loader, val_loader, test_loader, mean, std = make_dataloaders()
    mean = mean.to(device)
    std = std.to(device)

    model = load_model(device)

    diffusion = GaussianDiffusion(
        timesteps=config.TIMESTEPS,
        beta_schedule=config.BETA_SCHEDULE,
        device=device,
    )

    out_dir = os.path.join(config.SAMPLE_DIR, "conditional_sr")
    os.makedirs(out_dir, exist_ok=True)

    batch = next(iter(test_loader))
    lr = batch["lr"][:num_samples].to(device)
    hr = batch["hr"][:num_samples].to(device)

    all_results = []

    for ens in range(num_ensemble):
        print(f"[Sample] Ensemble {ens + 1}/{num_ensemble}")

        sr_norm = diffusion.ddim_sample_conditional_sr(
            model=model,
            lr_cond=lr,
            shape=(lr.shape[0], 1, 256, 256),
            ddim_steps=config.DDIM_STEPS,
            eta=config.DDIM_ETA,
            cond_scale=config.COND_SCALE,
            final_cond_scale=config.FINAL_COND_SCALE,
            final_cond_steps=config.FINAL_COND_STEPS,
            refine_steps=config.REFINE_STEPS,
            refine_lr=config.REFINE_LR,
        )

        with torch.no_grad():
            lr_from_sr_norm = center_stride_downsample(sr_norm)
            cond_mse = F.mse_loss(lr_from_sr_norm, lr).item()
            print(f"[Sample] normalized condition MSE: {cond_mse:.6e}")

            sr = denormalize(sr_norm, mean, std)
            hr_denorm = denormalize(hr, mean, std)
            lr_denorm = denormalize(lr, mean, std)
            lr_from_sr = center_stride_downsample(sr)

        sr_np = sr.detach().cpu().numpy()
        hr_np = hr_denorm.detach().cpu().numpy()
        lr_np = lr_denorm.detach().cpu().numpy()
        lr_from_sr_np = lr_from_sr.detach().cpu().numpy()

        all_results.append(sr_np)

        for i in range(num_samples):
            save_png = os.path.join(out_dir, f"sample_{i:03d}_ens_{ens:02d}.png")
            plot_one(
                hr_true=hr_np[i, 0],
                lr_cond=lr_np[i, 0],
                sr=sr_np[i, 0],
                save_path=save_png,
                title=f"Sample {i}, ensemble {ens}",
            )

            save_npz = os.path.join(out_dir, f"sample_{i:03d}_ens_{ens:02d}.npz")
            np.savez_compressed(
                save_npz,
                hr_true=hr_np[i, 0],
                lr_cond=lr_np[i, 0],
                sr=sr_np[i, 0],
                lr_from_sr=lr_from_sr_np[i, 0],
            )

    all_results = np.stack(all_results, axis=0)
    np.savez_compressed(
        os.path.join(out_dir, "all_ensemble_results.npz"),
        sr_ensemble=all_results,
    )

    print(f"[Done] Results saved to {out_dir}")


if __name__ == "__main__":
    sample_sr(num_samples=8, num_ensemble=4)
