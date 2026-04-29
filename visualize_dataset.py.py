# visualize_dataset.py

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

import config


def center_stride_np(x):
    idx = config.CENTER_INDICES
    return x[np.ix_(idx, idx)]


def main():
    files = sorted(glob.glob(os.path.join(config.DATA_DIR, "pair_batch_*.npz")))
    if len(files) == 0:
        raise FileNotFoundError("No npz files found")

    f = files[0]
    data = np.load(f)

    print(f"[File] {f}")
    print("[Keys]", data.files)

    lr = data[config.LR_KEY]
    hr = data[config.HR_KEY]

    print("raw lr shape:", lr.shape, lr.dtype)
    print("raw hr shape:", hr.shape, hr.dtype)

    if lr.ndim == 4:
        lr0 = lr[0, 0]
    else:
        lr0 = lr[0]

    if hr.ndim == 4:
        hr0 = hr[0, 0]
    else:
        hr0 = hr[0]

    sampled = center_stride_np(hr0)
    diff = sampled - lr0

    print("center sampled HR:")
    print(sampled)
    print("stored LR:")
    print(lr0)
    print("max abs difference:", np.max(np.abs(diff)))
    print("mean abs difference:", np.mean(np.abs(diff)))

    os.makedirs(config.SAMPLE_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    im0 = axes[0].imshow(hr0, origin="lower", cmap="RdBu_r")
    axes[0].set_title("HR vort 256x256")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(lr0, origin="lower", cmap="RdBu_r")
    axes[1].set_title("Stored LR 4x4")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(sampled, origin="lower", cmap="RdBu_r")
    axes[2].set_title("Center sampled HR")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    im3 = axes[3].imshow(diff, origin="lower", cmap="RdBu_r")
    axes[3].set_title("Sampled HR - stored LR")
    axes[3].axis("off")
    plt.colorbar(im3, ax=axes[3], fraction=0.046)

    plt.tight_layout()
    save_path = os.path.join(config.SAMPLE_DIR, "check_dataset_center_stride.png")
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"[Saved] {save_path}")


if __name__ == "__main__":
    main()
