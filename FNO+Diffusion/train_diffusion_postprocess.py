import os
os.environ["DNNL_MAX_CPU_ISA"] = "AVX2"
os.environ["MKL_DEBUG_CPU_TYPE"] = "5"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils.architecture import Unet
from utils.diffusion import ElucidatedDiffusion

# ==================== 配置 ====================
EXPORT_DIR = "/home/masters/PycharmProjects/PythonProject1/exports/fno_vorticity_postprocess"
OUT_DIR = "/home/masters/PycharmProjects/PythonProject1/outputs/diffusion_vorticity_postprocess"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DTYPE = torch.float32
USE_AMP = True

TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 32
NUM_EPOCHS = 500
VAL_EVERY = 10
LR = 1e-4

IMAGE_SIZE = 256
NUM_SAMPLE_STEPS = 32

# 为了稳一些，默认关掉 flash_attn
FLASH_ATTN = False

# ==================== 工具 ====================
def error_metric(pred, true):
    return torch.norm(true - pred, p=2) / torch.norm(true, p=2)

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def ensure_4d(x):
    # 支持 (N,1,H,W) 或 (N,H,W)
    if x.ndim == 3:
        x = x[:, None, :, :]
    return x

def main():
    print("Loading exported FNO predictions...")
    x_train = ensure_4d(np.load(os.path.join(EXPORT_DIR, "TRAIN_PRED.npy"))).astype(np.float32)
    y_train = ensure_4d(np.load(os.path.join(EXPORT_DIR, "TRAIN_TRUE.npy"))).astype(np.float32)

    x_val = ensure_4d(np.load(os.path.join(EXPORT_DIR, "VAL_PRED.npy"))).astype(np.float32)
    y_val = ensure_4d(np.load(os.path.join(EXPORT_DIR, "VAL_TRUE.npy"))).astype(np.float32)

    x_test = ensure_4d(np.load(os.path.join(EXPORT_DIR, "TEST_PRED.npy"))).astype(np.float32)
    y_test = ensure_4d(np.load(os.path.join(EXPORT_DIR, "TEST_TRUE.npy"))).astype(np.float32)

    print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"x_val:   {x_val.shape}, y_val:   {y_val.shape}")
    print(f"x_test:  {x_test.shape}, y_test:  {y_test.shape}")

    # 论文风格：输入和输出分别按训练集 min/max 归一化
    inp_min = np.min(x_train, axis=(0, 2, 3), keepdims=True)
    inp_max = np.max(x_train, axis=(0, 2, 3), keepdims=True)
    out_min = np.min(y_train, axis=(0, 2, 3), keepdims=True)
    out_max = np.max(y_train, axis=(0, 2, 3), keepdims=True)

    inp_scale = inp_max - inp_min
    out_scale = out_max - out_min

    inp_scale[inp_scale == 0] = 1.0
    out_scale[out_scale == 0] = 1.0

    x_train_norm = (x_train - inp_min) / inp_scale
    x_val_norm = (x_val - inp_min) / inp_scale
    x_test_norm = (x_test - inp_min) / inp_scale

    y_train_norm = (y_train - out_min) / out_scale
    y_val_norm = (y_val - out_min) / out_scale
    y_test_norm = (y_test - out_min) / out_scale

    sigma_data = float(np.std(y_train_norm))
    print(f"sigma_data = {sigma_data:.6f}")

    Par = {
        "inp_shift": torch.tensor(inp_min, dtype=DTYPE, device=DEVICE),
        "inp_scale": torch.tensor(inp_scale, dtype=DTYPE, device=DEVICE),
        "out_shift": torch.tensor(out_min, dtype=DTYPE, device=DEVICE),
        "out_scale": torch.tensor(out_scale, dtype=DTYPE, device=DEVICE),
        "sigma_data": sigma_data,
        "channels": 1,
        "image_size": IMAGE_SIZE,
        "self_condition": True,
        "num_sample_steps": NUM_SAMPLE_STEPS,
    }

    with open(os.path.join(OUT_DIR, "Par.pkl"), "wb") as f:
        pickle.dump(Par, f)

    train_dataset = MyDataset(torch.tensor(x_train_norm, dtype=torch.float32),
                              torch.tensor(y_train_norm, dtype=torch.float32))
    val_dataset = MyDataset(torch.tensor(x_val_norm, dtype=torch.float32),
                            torch.tensor(y_val_norm, dtype=torch.float32))
    test_dataset = MyDataset(torch.tensor(x_test_norm, dtype=torch.float32),
                             torch.tensor(y_test_norm, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False)

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
        image_size=IMAGE_SIZE,
        sigma_data=sigma_data,
        num_sample_steps=NUM_SAMPLE_STEPS
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and DEVICE.type == "cuda"))

    best_val_loss = float("inf")
    best_model_id = 0
    best_path = os.path.join(OUT_DIR, "best_dm.pt")

    print("Using device:", DEVICE)
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    t0 = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        for l_fidel, h_fidel in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
            l_fidel = l_fidel.to(DEVICE)
            h_fidel = h_fidel.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(USE_AMP and DEVICE.type == "cuda")):
                # 论文风格：images = truth, self_cond = prediction
                loss = model(h_fidel, l_fidel)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        if (epoch + 1) % VAL_EVERY == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for l_fidel, h_fidel in val_loader:
                    l_fidel = l_fidel.to(DEVICE)
                    h_fidel = h_fidel.to(DEVICE)

                    with torch.amp.autocast("cuda", enabled=(USE_AMP and DEVICE.type == "cuda")):
                        pred = model.sample(l_fidel, num_sample_steps=NUM_SAMPLE_STEPS)
                        loss = error_metric(pred, h_fidel)

                    val_loss += loss.item()

            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_id = epoch + 1
                torch.save(model.state_dict(), best_path)

            print(
                f"[Epoch {epoch + 1}/{NUM_EPOCHS}] "
                f"Train Loss: {train_loss:.6e}, "
                f"Val RelL2: {val_loss:.6e}, "
                f"Best Epoch: {best_model_id}"
            )
        else:
            print(f"[Epoch {epoch + 1}/{NUM_EPOCHS}] Train Loss: {train_loss:.6e}")

    print("Training finished.")
    print(f"Training Time: {time.time() - t0:.1f}s")

    # 最终测试（当前模型，不一定是最佳）
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for l_fidel, h_fidel in test_loader:
            l_fidel = l_fidel.to(DEVICE)
            h_fidel = h_fidel.to(DEVICE)
            with torch.amp.autocast("cuda", enabled=(USE_AMP and DEVICE.type == "cuda")):
                pred = model.sample(l_fidel, num_sample_steps=NUM_SAMPLE_STEPS)
                loss = error_metric(pred, h_fidel)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f"Current model test RelL2: {test_loss:.6e}")
    print(f"Best diffusion checkpoint saved to: {best_path}")

if __name__ == "__main__":
    main()
