# evaluate.py

import os
import glob
import numpy as np
import matplotlib.pyplot as plt


# =========================
# Path
# =========================
EVAL_DIR = "samples/conditional_sr"
OUT_DIR = os.path.join(EVAL_DIR, "eval_results")
os.makedirs(OUT_DIR, exist_ok=True)


# =========================
# Basic utilities
# =========================
def ensure_4d(x):
    """
    Convert:
        (N, H, W) -> (N, 1, H, W)
        (H, W)    -> (1, 1, H, W)
    """
    if x.ndim == 2:
        x = x[None, None, :, :]
    elif x.ndim == 3:
        x = x[:, None, :, :]
    return x


def field_get_err(true, pred):
    """
    Same style as your eval_diffusion.py:

    mean over samples of:
        MSE(field) / energy(field)

    true, pred: (N, C, H, W)
    """
    return np.mean(
        np.mean((true - pred) ** 2, axis=(2, 3))
        / (np.mean(true ** 2, axis=(2, 3)) + 1e-12)
    )


def radial_power_spectrum(batch):
    """
    Same style as your eval_diffusion.py.

    batch: (N, C, H, W) or (N, H, W)

    return:
        log radial spectrum, shape (N, nbins)
    """
    if batch.ndim == 4:
        batch = batch[:, 0]

    n, h, w = batch.shape
    assert h == w, f"Only square fields are supported, got {h}x{w}"

    npix = h
    kfreq = np.fft.fftfreq(npix) * npix
    kx, ky = np.meshgrid(kfreq, kfreq, indexing="ij")
    knrm = np.sqrt(kx ** 2 + ky ** 2).reshape(-1)

    kbins = np.arange(0.5, npix // 2 + 1, 1.0)
    nbins = len(kbins) - 1

    out = np.zeros((n, nbins), dtype=np.float64)
    bin_ids = np.digitize(knrm, kbins)

    for i in range(n):
        fft_img = np.fft.fftn(batch[i])
        amp2 = (np.abs(fft_img) ** 2).reshape(-1)

        Abins = np.zeros(nbins, dtype=np.float64)

        for b in range(1, len(kbins)):
            mask = bin_ids == b
            if np.any(mask):
                Abins[b - 1] = amp2[mask].mean()

        # Shell-area weighted spectrum
        Abins *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)

        # log spectrum, same as your previous test code
        out[i] = np.log(Abins + 1e-12)

    return out


def spec_get_err(true_spec, pred_spec):
    """
    Same style as your eval_diffusion.py:

    mean over samples of:
        MSE(log spectrum) / energy(log spectrum)
    """
    return np.mean(
        np.mean((true_spec - pred_spec) ** 2, axis=1)
        / (np.mean(true_spec ** 2, axis=1) + 1e-12)
    )


def center_stride_np(x):
    """
    x: (256, 256)

    Return the 4x4 center-stride sampled field:
        indices: 32, 96, 160, 224
    """
    idx = [32, 96, 160, 224]
    return x[np.ix_(idx, idx)]


# =========================
# Loading generated samples
# =========================
def load_generated_npz(folder):
    files = sorted(glob.glob(os.path.join(folder, "sample_*_ens_*.npz")))

    if len(files) == 0:
        raise FileNotFoundError(
            f"No sample_*_ens_*.npz files found in {folder}. "
            f"Please run sample_sr.py first."
        )

    all_hr = []
    all_sr = []
    all_lr = []
    all_lr_from_sr = []
    all_file_names = []

    for f in files:
        data = np.load(f)

        required_keys = ["hr_true", "lr_cond", "sr", "lr_from_sr"]
        for k in required_keys:
            if k not in data:
                raise KeyError(f"{f} does not contain key '{k}'. Available keys: {data.files}")

        hr = data["hr_true"].astype(np.float32)
        sr = data["sr"].astype(np.float32)
        lr = data["lr_cond"].astype(np.float32)
        lr_from_sr = data["lr_from_sr"].astype(np.float32)

        # saved format is usually:
        # hr: (256, 256)
        # sr: (256, 256)
        # lr: (4, 4)
        # lr_from_sr: (4, 4)
        all_hr.append(hr)
        all_sr.append(sr)
        all_lr.append(lr)
        all_lr_from_sr.append(lr_from_sr)
        all_file_names.append(os.path.basename(f))

    all_hr = ensure_4d(np.stack(all_hr, axis=0))
    all_sr = ensure_4d(np.stack(all_sr, axis=0))
    all_lr = ensure_4d(np.stack(all_lr, axis=0))
    all_lr_from_sr = ensure_4d(np.stack(all_lr_from_sr, axis=0))

    print(f"[Load] Loaded {len(files)} generated samples")
    print(f"[Load] HR true shape:     {all_hr.shape}")
    print(f"[Load] SR shape:          {all_sr.shape}")
    print(f"[Load] LR condition shape:{all_lr.shape}")
    print(f"[Load] LR from SR shape:  {all_lr_from_sr.shape}")

    return all_hr, all_sr, all_lr, all_lr_from_sr, all_file_names


# =========================
# Metrics
# =========================
def compute_metrics(all_true, all_refined, all_lr, all_lr_from_sr):
    """
    all_true:       (N, 1, 256, 256)
    all_refined:    (N, 1, 256, 256)
    all_lr:         (N, 1, 4, 4)
    all_lr_from_sr: (N, 1, 4, 4)
    """
    eps = 1e-12

    mse_refined = np.mean((all_refined - all_true) ** 2)
    mae_refined = np.mean(np.abs(all_refined - all_true))

    global_nmse_refined = np.sum((all_refined - all_true) ** 2) / (
        np.sum(all_true ** 2) + eps
    )

    sample_nmse_refined = np.sum((all_refined - all_true) ** 2, axis=(1, 2, 3)) / (
        np.sum(all_true ** 2, axis=(1, 2, 3)) + eps
    )

    field_err_refined = field_get_err(all_true, all_refined)

    spec_true = radial_power_spectrum(all_true)
    spec_refined = radial_power_spectrum(all_refined)

    spec_err_refined = spec_get_err(spec_true, spec_refined)

    condition_mse = np.mean((all_lr_from_sr - all_lr) ** 2)
    sample_condition_mse = np.mean((all_lr_from_sr - all_lr) ** 2, axis=(1, 2, 3))

    condition_mae = np.mean(np.abs(all_lr_from_sr - all_lr))

    condition_nmse = np.sum((all_lr_from_sr - all_lr) ** 2) / (
        np.sum(all_lr ** 2) + eps
    )

    metrics = {
        "mse_refined": float(mse_refined),
        "mae_refined": float(mae_refined),
        "global_nmse_refined": float(global_nmse_refined),
        "mean_nmse_refined": float(np.mean(sample_nmse_refined)),
        "std_nmse_refined": float(np.std(sample_nmse_refined)),
        "field_err_refined": float(field_err_refined),
        "spec_err_refined": float(spec_err_refined),
        "condition_mse": float(condition_mse),
        "condition_mae": float(condition_mae),
        "condition_nmse": float(condition_nmse),
        "mean_condition_mse": float(np.mean(sample_condition_mse)),
        "std_condition_mse": float(np.std(sample_condition_mse)),
    }

    return metrics, sample_nmse_refined, sample_condition_mse, spec_true, spec_refined


# =========================
# Figures
# =========================
def save_sample_figure(lr_cond, lr_from_sr, refined, true, save_path, title_prefix):
    """
    lr_cond:    (4, 4)
    lr_from_sr: (4, 4)
    refined:    (1, 256, 256)
    true:       (1, 256, 256)
    """
    refined_2d = refined[0]
    true_2d = true[0]

    vmin = min(true_2d.min(), refined_2d.min())
    vmax = max(true_2d.max(), refined_2d.max())

    diff = refined_2d - true_2d
    lr_diff = lr_from_sr - lr_cond

    fig, axes = plt.subplots(1, 6, figsize=(30, 5))

    im0 = axes[0].imshow(lr_cond, origin="lower", cmap="twilight", interpolation="nearest")
    axes[0].set_title(f"{title_prefix} | Input LR 4x4")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(lr_from_sr, origin="lower", cmap="twilight", interpolation="nearest")
    axes[1].set_title("Center sampled SR 4x4")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(lr_diff, origin="lower", cmap="twilight", interpolation="nearest")
    axes[2].set_title("SR center - LR")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    im3 = axes[3].imshow(refined_2d, origin="lower", cmap="twilight", vmin=vmin, vmax=vmax)
    axes[3].set_title("Diffusion-SR 256x256")
    axes[3].axis("off")
    plt.colorbar(im3, ax=axes[3], fraction=0.046)

    im4 = axes[4].imshow(true_2d, origin="lower", cmap="twilight", vmin=vmin, vmax=vmax)
    axes[4].set_title("Ground Truth 256x256")
    axes[4].axis("off")
    plt.colorbar(im4, ax=axes[4], fraction=0.046)

    im5 = axes[5].imshow(diff, origin="lower", cmap="twilight")
    axes[5].set_title("SR - Ground Truth")
    axes[5].axis("off")
    plt.colorbar(im5, ax=axes[5], fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_spectrum_figure(spec_true, spec_refined, save_path):
    """
    spec_true/spec_refined are log spectra, shape (N, nbins)
    """
    mean_true = spec_true.mean(axis=0)
    mean_refined = spec_refined.mean(axis=0)

    std_true = spec_true.std(axis=0)
    std_refined = spec_refined.std(axis=0)

    k = np.arange(1, len(mean_true) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(k, mean_true, label="Ground Truth")
    plt.fill_between(k, mean_true - std_true, mean_true + std_true, alpha=0.2)

    plt.plot(k, mean_refined, label="Diffusion-SR")
    plt.fill_between(k, mean_refined - std_refined, mean_refined + std_refined, alpha=0.2)

    plt.xlabel("Wavenumber k")
    plt.ylabel("Log radial power spectrum")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_pdf_figure(all_true, all_refined, save_path):
    true_flat = all_true.reshape(-1)
    refined_flat = all_refined.reshape(-1)

    plt.figure(figsize=(7, 5))
    plt.hist(true_flat, bins=120, density=True, alpha=0.5, label="Ground Truth")
    plt.hist(refined_flat, bins=120, density=True, alpha=0.5, label="Diffusion-SR")
    plt.xlabel("Vorticity")
    plt.ylabel("PDF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_nmse_hist(sample_nmse, save_path):
    plt.figure(figsize=(7, 5))
    plt.hist(sample_nmse, bins=50, density=True, alpha=0.75)
    plt.xlabel("Sample NMSE")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_condition_hist(sample_condition_mse, save_path):
    plt.figure(figsize=(7, 5))
    plt.hist(sample_condition_mse, bins=50, density=True, alpha=0.75)
    plt.xlabel("Sample condition MSE")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# =========================
# Main evaluation
# =========================
def evaluate_npz_folder(folder=EVAL_DIR):
    all_true, all_refined, all_lr, all_lr_from_sr, file_names = load_generated_npz(folder)

    metrics, sample_nmse, sample_condition_mse, spec_true, spec_refined = compute_metrics(
        all_true=all_true,
        all_refined=all_refined,
        all_lr=all_lr,
        all_lr_from_sr=all_lr_from_sr,
    )

    print("\n========== Diffusion-SR Metrics ==========")
    print(f"MSE refined:              {metrics['mse_refined']:.8e}")
    print(f"MAE refined:              {metrics['mae_refined']:.8e}")
    print(f"Global NMSE refined:      {metrics['global_nmse_refined']:.8e}")
    print(
        f"Mean NMSE refined:        {metrics['mean_nmse_refined']:.8e} "
        f"+- {metrics['std_nmse_refined']:.8e}"
    )
    print(f"Field Error refined:      {metrics['field_err_refined']:.8e}")
    print(f"Spec Error refined:       {metrics['spec_err_refined']:.8e}")
    print(f"Condition MSE:            {metrics['condition_mse']:.8e}")
    print(f"Condition MAE:            {metrics['condition_mae']:.8e}")
    print(f"Condition NMSE:           {metrics['condition_nmse']:.8e}")
    print(
        f"Mean condition MSE:       {metrics['mean_condition_mse']:.8e} "
        f"+- {metrics['std_condition_mse']:.8e}"
    )
    print("==========================================\n")

    # Save arrays
    np.save(os.path.join(OUT_DIR, "sample_nmse_refined.npy"), sample_nmse)
    np.save(os.path.join(OUT_DIR, "sample_condition_mse.npy"), sample_condition_mse)
    np.save(os.path.join(OUT_DIR, "spec_true.npy"), spec_true)
    np.save(os.path.join(OUT_DIR, "spec_refined.npy"), spec_refined)

    # Save metrics text
    metrics_path = os.path.join(OUT_DIR, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Diffusion-SR metrics\n")
        f.write(f"Number of samples: {len(all_true)}\n\n")

        f.write(f"MSE refined: {metrics['mse_refined']:.8e}\n")
        f.write(f"MAE refined: {metrics['mae_refined']:.8e}\n")
        f.write(f"Global NMSE refined: {metrics['global_nmse_refined']:.8e}\n")
        f.write(
            f"Mean NMSE refined: {metrics['mean_nmse_refined']:.8e} "
            f"+- {metrics['std_nmse_refined']:.8e}\n"
        )
        f.write(f"Field Error refined: {metrics['field_err_refined']:.8e}\n")
        f.write(f"Spec Error refined: {metrics['spec_err_refined']:.8e}\n")
        f.write(f"Condition MSE: {metrics['condition_mse']:.8e}\n")
        f.write(f"Condition MAE: {metrics['condition_mae']:.8e}\n")
        f.write(f"Condition NMSE: {metrics['condition_nmse']:.8e}\n")
        f.write(
            f"Mean condition MSE: {metrics['mean_condition_mse']:.8e} "
            f"+- {metrics['std_condition_mse']:.8e}\n"
        )

    # Save figures
    save_spectrum_figure(
        spec_true,
        spec_refined,
        os.path.join(OUT_DIR, "spectrum_comparison.png"),
    )

    save_pdf_figure(
        all_true,
        all_refined,
        os.path.join(OUT_DIR, "vorticity_pdf.png"),
    )

    save_nmse_hist(
        sample_nmse,
        os.path.join(OUT_DIR, "sample_nmse_hist.png"),
    )

    save_condition_hist(
        sample_condition_mse,
        os.path.join(OUT_DIR, "condition_mse_hist.png"),
    )

    # Save sample figures
    num_figs = min(10, len(all_true))
    for i in range(num_figs):
        save_sample_figure(
            lr_cond=all_lr[i, 0],
            lr_from_sr=all_lr_from_sr[i, 0],
            refined=all_refined[i],
            true=all_true[i],
            save_path=os.path.join(OUT_DIR, f"sample_{i:04d}.png"),
            title_prefix=f"Sample {i}",
        )

    # Save all merged arrays for later analysis
    np.save(os.path.join(OUT_DIR, "all_true.npy"), all_true)
    np.save(os.path.join(OUT_DIR, "all_refined.npy"), all_refined)
    np.save(os.path.join(OUT_DIR, "all_lr.npy"), all_lr)
    np.save(os.path.join(OUT_DIR, "all_lr_from_sr.npy"), all_lr_from_sr)

    print(f"[Done] Metrics saved to: {metrics_path}")
    print(f"[Done] Figures and arrays saved to: {OUT_DIR}")

    return metrics


if __name__ == "__main__":
    evaluate_npz_folder(EVAL_DIR)
