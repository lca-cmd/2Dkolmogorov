# train.py

import os
import time
import torch
from torch.cuda.amp import autocast, GradScaler

import config
from dataset import make_dataloaders
from unet import UNetV2
from diffusion import GaussianDiffusion


def save_checkpoint(path, model, optimizer, epoch, best_val):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "config": {
                "image_size": config.IMAGE_SIZE,
                "in_channels": config.IN_CHANNELS,
                "base_channels": config.BASE_CHANNELS,
                "channel_mults": config.CHANNEL_MULTS,
                "timesteps": config.TIMESTEPS,
            },
        },
        path,
    )


@torch.no_grad()
def validate(model, diffusion, val_loader, device):
    model.eval()
    losses = []

    for batch in val_loader:
        hr = batch["hr"].to(device, non_blocking=True)
        loss = diffusion.p_losses(model, hr)
        losses.append(loss.item())

    return sum(losses) / max(len(losses), 1)


def train():
    device = torch.device(config.DEVICE)
    print(f"[Device] {device}")

    train_loader, val_loader, test_loader, mean, std = make_dataloaders()

    model = UNetV2(
        in_channels=config.IN_CHANNELS,
        base_channels=config.BASE_CHANNELS,
        channel_mults=config.CHANNEL_MULTS,
        time_emb_dim=config.TIME_EMB_DIM,
        dropout=config.DROPOUT,
    ).to(device)

    diffusion = GaussianDiffusion(
        timesteps=config.TIMESTEPS,
        beta_schedule=config.BETA_SCHEDULE,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY,
    )

    scaler = GradScaler(enabled=config.USE_AMP)

    best_val = float("inf")
    global_step = 0

    print("[Train] Start training")

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        epoch_losses = []
        start = time.time()

        for batch in train_loader:
            hr = batch["hr"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=config.USE_AMP):
                loss = diffusion.p_losses(model, hr)

            scaler.scale(loss).backward()

            if config.GRAD_CLIP is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)

            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(loss.item())
            global_step += 1

        train_loss = sum(epoch_losses) / len(epoch_losses)
        val_loss = validate(model, diffusion, val_loader, device)
        elapsed = time.time() - start

        print(
            f"[Epoch {epoch:04d}] "
            f"train_loss={train_loss:.6e} "
            f"val_loss={val_loss:.6e} "
            f"time={elapsed:.1f}s"
        )

        save_checkpoint(config.LAST_CKPT, model, optimizer, epoch, best_val)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(config.BEST_CKPT, model, optimizer, epoch, best_val)
            print(f"[Checkpoint] Saved best: {config.BEST_CKPT}")

        if epoch % config.SAVE_EVERY == 0:
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"epoch_{epoch:04d}.pt")
            save_checkpoint(ckpt_path, model, optimizer, epoch, best_val)

    print("[Train] Finished")


if __name__ == "__main__":
    train()
