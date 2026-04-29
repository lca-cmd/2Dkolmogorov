import os

os.environ["DNNL_MAX_CPU_ISA"] = "AVX2"
os.environ["MKL_DEBUG_CPU_TYPE"] = "5"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import glob
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchinfo import summary

# 【修改点 1】：导入 neuralop 库中的 TFNO2d
from neuralop.models import TFNO2d

# ==================== 配置 ====================
NPZ_DIR = "/home/masters/PycharmProjects/PythonProject1/generated_kolmogorov_forced"
OUT_DIR = "/home/masters/PycharmProjects/PythonProject1/outputs/fno_vorticity_4to256"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 16
NUM_EPOCHS = 100
LR = 1e-4
NUM_WORKERS = 4
PIN_MEMORY = True

IMAGE_SIZE_LR = 4
IMAGE_SIZE_HR = 256
DOMAIN_LENGTH = 2 * np.pi

TOTAL_TRAJS = 250
FRAMES_PER_TRAJ = 200

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

SEED = 42
SHUFFLE_TRAJ_IDS = True

MODES1 = 16
MODES2 = 16
WIDTH = 64


# ==================== 随机种子 ====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ==================== 数据集 ====================
class VorticityDataset(Dataset):
    def __init__(self, lr_tensor, hr_tensor):
        self.lr = lr_tensor.contiguous()  # (N, 1, 4, 4)
        self.hr = hr_tensor.contiguous()  # (N, 1, 256, 256)

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

    omega = dvdx - dudy
    return omega.astype(np.float32)


# ==================== 加载全部涡量数据 ====================
def load_all_vorticity_data(npz_dir):
    file_list = sorted(glob.glob(os.path.join(npz_dir, "pair_batch_*.npz")))
    if not file_list:
        raise FileNotFoundError(f"No npz files found in {npz_dir}")

    lr_list = []
    hr_list = []
    total_frames = 0

    for f in tqdm(file_list, desc="Loading velocity and converting to vorticity"):
        with np.load(f) as data:
            u_lr = data["u_lr"]  # (n, 4, 4)
            v_lr = data["v_lr"]  # (n, 4, 4)
            u_hr = data["u_hr"]  # (n, 256, 256)
            v_hr = data["v_hr"]  # (n, 256, 256)

        omega_lr = velocity_to_vorticity_np(u_lr, v_lr, DOMAIN_LENGTH)[:, None, :, :]
        omega_hr = velocity_to_vorticity_np(u_hr, v_hr, DOMAIN_LENGTH)[:, None, :, :]

        lr_list.append(torch.from_numpy(omega_lr))
        hr_list.append(torch.from_numpy(omega_hr))
        total_frames += omega_hr.shape[0]

    lr_all = torch.cat(lr_list, dim=0)
    hr_all = torch.cat(hr_list, dim=0)

    print(f"Total frames: {total_frames}")
    print(f"lr_all shape: {tuple(lr_all.shape)}")
    print(f"hr_all shape: {tuple(hr_all.shape)}")

    return lr_all, hr_all


# ==================== 轨迹划分 ====================
def make_trajectory_split():
    traj_ids = list(range(TOTAL_TRAJS))
    if SHUFFLE_TRAJ_IDS:
        rng = np.random.default_rng(SEED)
        rng.shuffle(traj_ids)

    train_n = int(TRAIN_RATIO * TOTAL_TRAJS)
    val_n = int(VAL_RATIO * TOTAL_TRAJS)
    test_n = TOTAL_TRAJS - train_n - val_n

    train_traj_ids = traj_ids[:train_n]
    val_traj_ids = traj_ids[train_n:train_n + val_n]
    test_traj_ids = traj_ids[train_n + val_n:]

    return traj_ids, train_traj_ids, val_traj_ids, test_traj_ids


def traj_ids_to_frame_indices(traj_ids):
    indices = []
    for tid in traj_ids:
        start = tid * FRAMES_PER_TRAJ
        end = start + FRAMES_PER_TRAJ
        indices.extend(range(start, end))
    return indices


# ==================== 归一化统计 ====================
def compute_minmax(x):
    x_min = float(x.min().item())
    x_max = float(x.max().item())
    x_scale = x_max - x_min
    if x_scale <= 0:
        raise ValueError("Scale must be positive.")
    return x_min, x_scale


# ==================== 预处理 ====================
def preprocess_batch_on_gpu(lr_raw, hr_raw, inp_shift, inp_scale, out_shift, out_scale):
    lr_raw = lr_raw.to(DEVICE, non_blocking=True)
    hr_raw = hr_raw.to(DEVICE, non_blocking=True)

    lr = (lr_raw - inp_shift) / inp_scale
    hr = (hr_raw - out_shift) / out_scale

    # 这里的插值是核心：强行把 4x4 拉伸到 256x256 作为 TFNO 的输入底稿
    lr_up = F.interpolate(
        lr, size=(IMAGE_SIZE_HR, IMAGE_SIZE_HR),
        mode="bilinear", align_corners=False
    )
    return lr_up, hr


# ==================== 验证 ====================
@torch.no_grad()
def evaluate(model, loader, inp_shift, inp_scale, out_shift, out_scale):
    model.eval()
    total_loss = 0.0
    total_steps = 0

    for lr_raw, hr_raw in loader:
        inp, target = preprocess_batch_on_gpu(
            lr_raw, hr_raw, inp_shift, inp_scale, out_shift, out_scale
        )
        pred = model(inp)
        loss = F.mse_loss(pred, target)
        total_loss += loss.item()
        total_steps += 1

    return total_loss / max(total_steps, 1)


# ==================== 主程序 ====================
def main():
    print("Loading all vorticity data...")
    lr_all, hr_all = load_all_vorticity_data(NPZ_DIR)

    traj_order, train_traj_ids, val_traj_ids, test_traj_ids = make_trajectory_split()

    train_indices = traj_ids_to_frame_indices(train_traj_ids)
    val_indices = traj_ids_to_frame_indices(val_traj_ids)
    test_indices = traj_ids_to_frame_indices(test_traj_ids)

    print(f"Train trajectories: {len(train_traj_ids)}, frames: {len(train_indices)}")
    print(f"Val trajectories:   {len(val_traj_ids)}, frames: {len(val_indices)}")
    print(f"Test trajectories:  {len(test_traj_ids)}, frames: {len(test_indices)}")

    train_lr = lr_all[train_indices]
    train_hr = hr_all[train_indices]
    val_lr = lr_all[val_indices]
    val_hr = hr_all[val_indices]
    test_lr = lr_all[test_indices]
    test_hr = hr_all[test_indices]

    inp_shift, inp_scale = compute_minmax(train_lr)
    out_shift, out_scale = compute_minmax(train_hr)

    print("\n===== FNO normalization stats =====")
    print(f"inp_shift = {inp_shift:.6f}")
    print(f"inp_scale = {inp_scale:.6f}")
    print(f"out_shift = {out_shift:.6f}")
    print(f"out_scale = {out_scale:.6f}")
    print("===================================\n")

    train_dataset = VorticityDataset(train_lr, train_hr)
    val_dataset = VorticityDataset(val_lr, val_hr)
    test_dataset = VorticityDataset(test_lr, test_hr)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0)
    )

    # 【修改点 2】：用 neuralop 里的 TFNO2d 替换原来的 FNO2d
    model = TFNO2d(
        MODES1,
        MODES2,
        WIDTH,
        in_channels=1,
        out_channels=1
    ).to(DEVICE)

    # 打印网络结构 (可选，确保模型对接正常)
    try:
        dummy_input = torch.randn(BATCH_SIZE, 1, IMAGE_SIZE_HR, IMAGE_SIZE_HR).to(DEVICE)
        summary(model, input_size=dummy_input.shape)
    except Exception as e:
        pass

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    best_path = os.path.join(OUT_DIR, "best_fno_vorticity.pt")

    print(f"Using device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
        for lr_raw, hr_raw in pbar:
            inp, target = preprocess_batch_on_gpu(
                lr_raw, hr_raw, inp_shift, inp_scale, out_shift, out_scale
            )

            optimizer.zero_grad(set_to_none=True)
            pred = model(inp)
            loss = F.mse_loss(pred, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_loss / max(len(train_loader), 1)
        avg_val_loss = evaluate(model, val_loader, inp_shift, inp_scale, out_shift, out_scale)

        print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "inp_shift": inp_shift,
                    "inp_scale": inp_scale,
                    "out_shift": out_shift,
                    "out_scale": out_scale,
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "traj_order": np.array(traj_order, dtype=np.int64),
                    "train_traj_ids": np.array(train_traj_ids, dtype=np.int64),
                    "val_traj_ids": np.array(val_traj_ids, dtype=np.int64),
                    "test_traj_ids": np.array(test_traj_ids, dtype=np.int64),
                },
                best_path,
            )
            print(f"Best FNO model saved to {best_path}")

    # 用当前模型做一次测试；真正导出时会重新读取 best checkpoint
    test_loss = evaluate(model, test_loader, inp_shift, inp_scale, out_shift, out_scale)
    print(f"Final Test Loss (current model): {test_loss:.6f}")


if __name__ == "__main__":
    main()
