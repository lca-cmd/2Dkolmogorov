import os
os.environ["DNNL_MAX_CPU_ISA"] = "AVX2"
os.environ["MKL_DEBUG_CPU_TYPE"] = "5"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from neuralop.models import TFNO2d

# ==================== 配置 ====================
NPZ_DIR = "/home/masters/PycharmProjects/PythonProject1/generated_kolmogorov_forced"
FNO_CKPT = "/home/masters/PycharmProjects/PythonProject1/outputs/fno_vorticity_4to256/best_fno_vorticity.pt"
EXPORT_DIR = "/home/masters/PycharmProjects/PythonProject1/exports/fno_vorticity_postprocess"
os.makedirs(EXPORT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 64
NUM_WORKERS = 4
PIN_MEMORY = True

IMAGE_SIZE_LR = 4
IMAGE_SIZE_HR = 256
DOMAIN_LENGTH = 2 * np.pi
FRAMES_PER_TRAJ = 200

MODES1 = 16
MODES2 = 16
WIDTH = 64

# 如果训练时使用了 UpSamplingFNO，设为 True；否则为 False（使用原始插值方式）
USE_UPSAMPLE_FNO = False   # 根据你的实际训练模型选择


# ==================== 数据集 ====================
class VorticityDataset(Dataset):
    def __init__(self, lr_tensor, hr_tensor):
        self.lr = lr_tensor.contiguous()
        self.hr = hr_tensor.contiguous()

    def __len__(self):
        return self.lr.shape[0]

    def __getitem__(self, idx):
        return self.lr[idx], self.hr[idx]


# ==================== 速度转涡量 ====================
def velocity_to_vorticity_np(u, v, domain_length=2 * np.pi):
    h = u.shape[-2]
    w = u.shape[-1]
    dx = domain_length / w
    dy = domain_length / h

    dvdx = (np.roll(v, -1, axis=-1) - np.roll(v, 1, axis=-1)) / (2.0 * dx)
    dudy = (np.roll(u, -1, axis=-2) - np.roll(u, 1, axis=-2)) / (2.0 * dy)
    return (dvdx - dudy).astype(np.float32)


def load_all_vorticity_data(npz_dir):
    file_list = sorted(glob.glob(os.path.join(npz_dir, "pair_batch_*.npz")))
    if not file_list:
        raise FileNotFoundError(f"No npz files found in {npz_dir}")

    lr_list = []
    hr_list = []

    for f in tqdm(file_list, desc="Loading velocity and converting to vorticity"):
        with np.load(f) as data:
            u_lr = data["u_lr"]
            v_lr = data["v_lr"]
            u_hr = data["u_hr"]
            v_hr = data["v_hr"]

        omega_lr = velocity_to_vorticity_np(u_lr, v_lr, DOMAIN_LENGTH)[:, None, :, :]
        omega_hr = velocity_to_vorticity_np(u_hr, v_hr, DOMAIN_LENGTH)[:, None, :, :]

        lr_list.append(torch.from_numpy(omega_lr))
        hr_list.append(torch.from_numpy(omega_hr))

    lr_all = torch.cat(lr_list, dim=0)
    hr_all = torch.cat(hr_list, dim=0)

    print(f"lr_all shape: {tuple(lr_all.shape)}")
    print(f"hr_all shape: {tuple(hr_all.shape)}")
    return lr_all, hr_all


def traj_ids_to_frame_indices(traj_ids):
    """返回轨迹内所有帧的索引（用于训练/验证集）"""
    indices = []
    for tid in traj_ids:
        start = tid * FRAMES_PER_TRAJ
        end = start + FRAMES_PER_TRAJ
        indices.extend(range(start, end))
    return indices


def traj_ids_to_frame_indices_last_frame_only(traj_ids):
    """仅返回每条轨迹的最后一帧索引（用于测试集）"""
    indices = []
    for tid in traj_ids:
        last_frame_idx = tid * FRAMES_PER_TRAJ + (FRAMES_PER_TRAJ - 1)
        indices.append(last_frame_idx)
    return indices


def preprocess_lr(lr_raw, inp_shift, inp_scale, use_upsample_fno):
    """将粗输入移到设备并归一化，若需要则上采样到256x256"""
    lr_raw = lr_raw.to(DEVICE, non_blocking=True)
    lr_norm = (lr_raw - inp_shift) / inp_scale
    if not use_upsample_fno:
        # 原始方式：手动双线性插值到256x256
        lr_norm = F.interpolate(lr_norm, size=(IMAGE_SIZE_HR, IMAGE_SIZE_HR),
                                mode="bilinear", align_corners=False)
    # 如果使用 UpSamplingFNO，模型内部会处理上采样，此处保持4x4
    return lr_norm


# ==================== 模型定义（与训练时保持一致）====================
class UpSamplingFNO(nn.Module):
    """与训练脚本中完全相同的模型定义"""
    def __init__(self, modes1, modes2, width, in_channels=1, out_channels=1,
                 lr_size=4, hr_size=256):
        super().__init__()
        self.lr_size = lr_size
        self.hr_size = hr_size
        scale_factor = hr_size // lr_size

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, width, kernel_size=8, stride=8, padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(width, in_channels, kernel_size=8, stride=8, padding=0),
        )
        self.fno = TFNO2d(
            n_modes=(modes1, modes2),
            hidden_channels=width,
            in_channels=in_channels,
            out_channels=out_channels,
        )

    def forward(self, x):
        x_up = self.upsample(x)
        return self.fno(x_up)


def build_model(use_upsample_fno):
    if use_upsample_fno:
        model = UpSamplingFNO(
            modes1=MODES1, modes2=MODES2, width=WIDTH,
            in_channels=1, out_channels=1,
            lr_size=IMAGE_SIZE_LR, hr_size=IMAGE_SIZE_HR
        )
    else:
        model = TFNO2d(
            MODES1, 
            MODES2,
            WIDTH,
            in_channels=1,
            out_channels=1
        ).to(DEVICE)


@torch.no_grad()
def run_export(model, loader, inp_shift, inp_scale, out_shift, out_scale, use_upsample_fno):
    model.eval()
    pred_list = []
    true_list = []
    lr4_list = []

    for lr_raw, hr_raw in tqdm(loader, desc="Exporting predictions"):
        inp = preprocess_lr(lr_raw, inp_shift, inp_scale, use_upsample_fno)
        pred = model(inp)

        pred = pred * out_scale + out_shift
        true = hr_raw

        pred_list.append(pred.cpu().numpy())
        true_list.append(true.numpy())
        lr4_list.append(lr_raw.numpy())

    pred_all = np.concatenate(pred_list, axis=0)
    true_all = np.concatenate(true_list, axis=0)
    lr4_all = np.concatenate(lr4_list, axis=0)

    return pred_all, true_all, lr4_all


def main():
    ckpt = torch.load(FNO_CKPT, map_location=DEVICE, weights_only=False)

    inp_shift = ckpt["inp_shift"]
    inp_scale = ckpt["inp_scale"]
    out_shift = ckpt["out_shift"]
    out_scale = ckpt["out_scale"]

    train_traj_ids = ckpt["train_traj_ids"].tolist()
    val_traj_ids = ckpt["val_traj_ids"].tolist()
    test_traj_ids = ckpt["test_traj_ids"].tolist()

    lr_all, hr_all = load_all_vorticity_data(NPZ_DIR)

    # 训练集、验证集：使用全部帧（可根据需要修改）
    train_indices = traj_ids_to_frame_indices(train_traj_ids)
    val_indices = traj_ids_to_frame_indices(val_traj_ids)
    # 测试集：仅使用每条轨迹的最后一帧
    test_indices = traj_ids_to_frame_indices_last_frame_only(test_traj_ids)

    print(f"Train frames: {len(train_indices)}")
    print(f"Val frames:   {len(val_indices)}")
    print(f"Test frames (last frame only): {len(test_indices)}")

    train_dataset = VorticityDataset(lr_all[train_indices], hr_all[train_indices])
    val_dataset = VorticityDataset(lr_all[val_indices], hr_all[val_indices])
    test_dataset = VorticityDataset(lr_all[test_indices], hr_all[test_indices])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              persistent_workers=(NUM_WORKERS > 0))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                            persistent_workers=(NUM_WORKERS > 0))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                             persistent_workers=(NUM_WORKERS > 0))

    # 构建与训练时一致的模型
    model = build_model(USE_UPSAMPLE_FNO)
    model.load_state_dict(ckpt["model"])

    print("Export TRAIN split ...")
    train_pred, train_true, train_lr4 = run_export(
        model, train_loader, inp_shift, inp_scale, out_shift, out_scale, USE_UPSAMPLE_FNO
    )

    print("Export VAL split ...")
    val_pred, val_true, val_lr4 = run_export(
        model, val_loader, inp_shift, inp_scale, out_shift, out_scale, USE_UPSAMPLE_FNO
    )

    print("Export TEST split (last frame only) ...")
    test_pred, test_true, test_lr4 = run_export(
        model, test_loader, inp_shift, inp_scale, out_shift, out_scale, USE_UPSAMPLE_FNO
    )

    # 保存结果
    np.save(os.path.join(EXPORT_DIR, "TRAIN_PRED.npy"), train_pred)
    np.save(os.path.join(EXPORT_DIR, "TRAIN_TRUE.npy"), train_true)
    np.save(os.path.join(EXPORT_DIR, "TRAIN_LR4.npy"), train_lr4)

    np.save(os.path.join(EXPORT_DIR, "VAL_PRED.npy"), val_pred)
    np.save(os.path.join(EXPORT_DIR, "VAL_TRUE.npy"), val_true)
    np.save(os.path.join(EXPORT_DIR, "VAL_LR4.npy"), val_lr4)

    np.save(os.path.join(EXPORT_DIR, "TEST_PRED.npy"), test_pred)
    np.save(os.path.join(EXPORT_DIR, "TEST_TRUE.npy"), test_true)
    np.save(os.path.join(EXPORT_DIR, "TEST_LR4.npy"), test_lr4)

    print(f"\nDone. Exports saved to: {EXPORT_DIR}")
    print(f"TRAIN_PRED shape: {train_pred.shape}")
    print(f"VAL_PRED shape:   {val_pred.shape}")
    print(f"TEST_PRED shape:  {test_pred.shape} (should be (N_test_trajs, 1, 256, 256))")


if __name__ == "__main__":
    main()