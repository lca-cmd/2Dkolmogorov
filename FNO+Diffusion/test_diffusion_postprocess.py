import os
os.environ["DNNL_MAX_CPU_ISA"] = "AVX2"
os.environ["MKL_DEBUG_CPU_TYPE"] = "5"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from utils.architecture import Unet
from utils.diffusion import ElucidatedDiffusion

# ==================== 配置 ====================
EXPORT_DIR = "/home/masters/PycharmProjects/PythonProject1/exports/fno_vorticity_postprocess"
DM_DIR = "/home/masters/PycharmProjects/PythonProject1/outputs/diffusion_vorticity_postprocess"
OUT_DIR = "/home/masters/PycharmProjects/PythonProject1/results/diffusion_vorticity_postprocess_test"
os.makedirs(OUT_DIR, exist_ok=True)

PAR_PATH = os.path.join(DM_DIR, "Par.pkl")
BEST_MODEL_PATH = os.path.join(DM_DIR, "best_dm.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = True
BATCH_SIZE = 16
NUM_SAMPLE_STEPS = 32
FLASH_ATTN = False

# ==================== 工具 ====================
class MyDataset(Dataset):
    def __init__(self, x, y, lr4):
        self.x = x
        self.y = y
        self.lr4 = lr4

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.lr4[idx]

def ensure_4d(x):
    if x.ndim == 3:
        x = x[:, None, :, :]
    return x

def field_get_err(true, pred):
    return np.mean(np.mean((true - pred) ** 2, axis=(2, 3)) / (np.mean(true ** 2, axis=(2, 3)) + 1e-12))

def radial_power_spectrum(batch):
    """
    batch: (N, 1, H, W) or (N, H, W)
    return: (N, K)
    """
    if batch.ndim == 4:
        batch = batch[:, 0]

    n, h, w = batch.shape
    assert h == w
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
        amp2 = np.abs(fft_img) ** 2
        amp2 = amp2.reshape(-1)

        Abins = np.zeros(nbins, dtype=np.float64)
        for b in range(1, len(kbins)):
            mask = (bin_ids == b)
            if np.any(mask):
                Abins[b - 1] = amp2[mask].mean()

        Abins *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
        out[i] = np.log(Abins + 1e-12)

    return out

def spec_get_err(true_spec, pred_spec):
    return np.mean(np.mean((true_spec - pred_spec) ** 2, axis=1) / (np.mean(true_spec ** 2, axis=1) + 1e-12))

def magnitude_like_single_channel(field):
    # 对单通道涡量，直接返回自己
    return field[0]

def save_sample_figure(lr4, rough, refined, true, save_path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(magnitude_like_single_channel(lr4), origin="lower", cmap="twilight", interpolation="nearest")
    axes[0].set_title("Input Vorticity (4x4)")
    axes[0].axis("off")

    axes[1].imshow(magnitude_like_single_channel(rough), origin="lower", cmap="twilight")
    axes[1].set_title("FNO Prediction")
    axes[1].axis("off")

    axes[2].imshow(magnitude_like_single_channel(refined), origin="lower", cmap="twilight")
    axes[2].set_title("FNO + Diffusion")
    axes[2].axis("off")

    axes[3].imshow(magnitude_like_single_channel(true), origin="lower", cmap="twilight")
    axes[3].set_title("Ground Truth")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    with open(PAR_PATH, "rb") as f:
        Par = pickle.load(f)

    inp_shift = Par["inp_shift"].detach().cpu().numpy()
    inp_scale = Par["inp_scale"].detach().cpu().numpy()
    out_shift = Par["out_shift"].detach().cpu().numpy()
    out_scale = Par["out_scale"].detach().cpu().numpy()
    sigma_data = Par["sigma_data"]
    image_size = Par["image_size"]

    x_test = ensure_4d(np.load(os.path.join(EXPORT_DIR, "TEST_PRED.npy"))).astype(np.float32)
    y_test = ensure_4d(np.load(os.path.join(EXPORT_DIR, "TEST_TRUE.npy"))).astype(np.float32)
    lr4_test = ensure_4d(np.load(os.path.join(EXPORT_DIR, "TEST_LR4.npy"))).astype(np.float32)

    x_test_norm = (x_test - inp_shift) / inp_scale
    y_test_norm = (y_test - out_shift) / out_scale

    dataset = MyDataset(
        torch.tensor(x_test_norm, dtype=torch.float32),
        torch.tensor(y_test_norm, dtype=torch.float32),
        torch.tensor(lr4_test, dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    net = Unet(
        dim=16,
        dim_mults=(1, 2, 4, 8, 8),
        channels=1,
        self_condition=True,
        flash_attn=FLASH_ATTN
    ).to(DEVICE).to(torch.float32)

    model = ElucidatedDiffusion(
        net,
        channels=1,
        image_size=image_size,
        sigma_data=sigma_data,
        num_sample_steps=NUM_SAMPLE_STEPS
    ).to(DEVICE)

    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_lr4 = []
    all_rough = []
    all_refined = []
    all_true = []

    with torch.no_grad():
        for l_fidel, h_fidel, lr4_raw in tqdm(loader, desc="Testing diffusion postprocess"):
            l_fidel = l_fidel.to(DEVICE)
            h_fidel = h_fidel.to(DEVICE)

            with torch.amp.autocast("cuda", enabled=(USE_AMP and DEVICE.type == "cuda")):
                pred = model.sample(l_fidel, num_sample_steps=NUM_SAMPLE_STEPS)

            all_rough.append(l_fidel.cpu().numpy())
            all_refined.append(pred.cpu().numpy())
            all_true.append(h_fidel.cpu().numpy())
            all_lr4.append(lr4_raw.numpy())

    all_rough = np.concatenate(all_rough, axis=0)
    all_refined = np.concatenate(all_refined, axis=0)
    all_true = np.concatenate(all_true, axis=0)
    all_lr4 = np.concatenate(all_lr4, axis=0)

    # 反归一化
    all_rough = all_rough * out_scale + out_shift
    all_refined = all_refined * out_scale + out_shift
    all_true = all_true * out_scale + out_shift

    np.save(os.path.join(OUT_DIR, "test_lr4.npy"), all_lr4)
    np.save(os.path.join(OUT_DIR, "test_rough.npy"), all_rough)
    np.save(os.path.join(OUT_DIR, "test_refined.npy"), all_refined)
    np.save(os.path.join(OUT_DIR, "test_true.npy"), all_true)

    # ==================== 指标 ====================
    eps = 1e-12

    mse_rough = np.mean((all_rough - all_true) ** 2)
    mse_refined = np.mean((all_refined - all_true) ** 2)

    mae_rough = np.mean(np.abs(all_rough - all_true))
    mae_refined = np.mean(np.abs(all_refined - all_true))

    nmse_rough_global = np.sum((all_rough - all_true) ** 2) / (np.sum(all_true ** 2) + eps)
    nmse_refined_global = np.sum((all_refined - all_true) ** 2) / (np.sum(all_true ** 2) + eps)

    sample_nmse_rough = np.sum((all_rough - all_true) ** 2, axis=(1, 2, 3)) / (
        np.sum(all_true ** 2, axis=(1, 2, 3)) + eps
    )
    sample_nmse_refined = np.sum((all_refined - all_true) ** 2, axis=(1, 2, 3)) / (
        np.sum(all_true ** 2, axis=(1, 2, 3)) + eps
    )

    field_err_rough = field_get_err(all_true, all_rough)
    field_err_refined = field_get_err(all_true, all_refined)

    spec_true = radial_power_spectrum(all_true)
    spec_rough = radial_power_spectrum(all_rough)
    spec_refined = radial_power_spectrum(all_refined)

    spec_err_rough = spec_get_err(spec_true, spec_rough)
    spec_err_refined = spec_get_err(spec_true, spec_refined)

    print("\n========== Metrics ==========")
    print(f"MSE (rough):            {mse_rough:.8e}")
    print(f"MSE (refined):          {mse_refined:.8e}")
    print(f"MAE (rough):            {mae_rough:.8e}")
    print(f"MAE (refined):          {mae_refined:.8e}")
    print(f"Global NMSE (rough):    {nmse_rough_global:.8e}")
    print(f"Global NMSE (refined):  {nmse_refined_global:.8e}")
    print(f"Mean NMSE (rough):      {np.mean(sample_nmse_rough):.8e} ± {np.std(sample_nmse_rough):.8e}")
    print(f"Mean NMSE (refined):    {np.mean(sample_nmse_refined):.8e} ± {np.std(sample_nmse_refined):.8e}")
    print(f"Field Error (rough):    {field_err_rough:.8e}")
    print(f"Field Error (refined):  {field_err_refined:.8e}")
    print(f"Spec Error (rough):     {spec_err_rough:.8e}")
    print(f"Spec Error (refined):   {spec_err_refined:.8e}")
    print("=============================\n")

    np.save(os.path.join(OUT_DIR, "sample_nmse_rough.npy"), sample_nmse_rough)
    np.save(os.path.join(OUT_DIR, "sample_nmse_refined.npy"), sample_nmse_refined)

    # 保存前 10 个样本图
    for i in range(min(10, len(all_true))):
        save_sample_figure(
            all_lr4[i],
            all_rough[i],
            all_refined[i],
            all_true[i],
            os.path.join(OUT_DIR, f"sample_{i:04d}.png")
        )

    print(f"Results saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
