# dataset.py

import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

import config


def load_all_npz(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "pair_batch_*.npz")))
    if len(files) == 0:
        raise FileNotFoundError(f"No pair_batch_*.npz files found in {data_dir}")

    lr_list = []
    hr_list = []

    for f in files:
        data = np.load(f)

        if config.LR_KEY not in data or config.HR_KEY not in data:
            raise KeyError(
                f"{f} does not contain {config.LR_KEY} and {config.HR_KEY}. "
                f"Available keys: {data.files}"
            )

        omega_lr = data[config.LR_KEY].astype(np.float32)
        omega_hr = data[config.HR_KEY].astype(np.float32)

        # (T, H, W) -> (T, 1, H, W)
        if omega_lr.ndim == 3:
            omega_lr = omega_lr[:, None, :, :]
        if omega_hr.ndim == 3:
            omega_hr = omega_hr[:, None, :, :]

        if omega_lr.ndim != 4 or omega_hr.ndim != 4:
            raise ValueError(
                f"Unexpected shape in {f}: lr={omega_lr.shape}, hr={omega_hr.shape}"
            )

        if omega_lr.shape[1:] != (1, 4, 4):
            raise ValueError(f"Unexpected lr shape in {f}: {omega_lr.shape}")

        if omega_hr.shape[1:] != (1, 256, 256):
            raise ValueError(f"Unexpected hr shape in {f}: {omega_hr.shape}")

        lr_list.append(omega_lr)
        hr_list.append(omega_hr)

    lr_all = np.concatenate(lr_list, axis=0)
    hr_all = np.concatenate(hr_list, axis=0)

    print(f"[Data] Loaded {len(files)} files")
    print(f"[Data] lr_all shape: {lr_all.shape}, dtype: {lr_all.dtype}")
    print(f"[Data] hr_all shape: {hr_all.shape}, dtype: {hr_all.dtype}")

    return lr_all, hr_all


class KolmogorovVorticityDataset(Dataset):
    def __init__(self, lr_all, hr_all, mean=None, std=None, normalize=True):
        self.lr = torch.from_numpy(lr_all).float()
        self.hr = torch.from_numpy(hr_all).float()

        self.normalize = normalize

        if normalize:
            if mean is None or std is None:
                mean = self.hr.mean()
                std = self.hr.std()
                if std < 1e-8:
                    std = torch.tensor(1.0)

            self.mean = torch.as_tensor(mean).float()
            self.std = torch.as_tensor(std).float()

            self.lr = (self.lr - self.mean) / self.std
            self.hr = (self.hr - self.mean) / self.std
        else:
            self.mean = torch.tensor(0.0)
            self.std = torch.tensor(1.0)

    def __len__(self):
        return self.hr.shape[0]

    def __getitem__(self, idx):
        return {
            "lr": self.lr[idx],
            "hr": self.hr[idx],
            "idx": idx,
        }


def make_dataloaders():
    lr_all, hr_all = load_all_npz(config.DATA_DIR)

    full_raw = KolmogorovVorticityDataset(
        lr_all,
        hr_all,
        normalize=False,
    )

    total_len = len(full_raw)
    train_len = int(total_len * config.TRAIN_RATIO)
    val_len = int(total_len * config.VAL_RATIO)
    test_len = total_len - train_len - val_len

    generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    train_raw, val_raw, test_raw = random_split(
        full_raw,
        [train_len, val_len, test_len],
        generator=generator,
    )

    # 用 train set 的 HR 计算 mean/std
    train_indices = train_raw.indices
    train_hr = full_raw.hr[train_indices]
    mean = train_hr.mean()
    std = train_hr.std()
    if std < 1e-8:
        std = torch.tensor(1.0)

    print(f"[Norm] mean={mean.item():.6e}, std={std.item():.6e}")
    torch.save({"mean": mean, "std": std}, config.NORM_STATS)

    full_norm = KolmogorovVorticityDataset(
        lr_all,
        hr_all,
        mean=mean,
        std=std,
        normalize=config.NORMALIZE,
    )

    train_set = torch.utils.data.Subset(full_norm, train_indices)
    val_set = torch.utils.data.Subset(full_norm, val_raw.indices)
    test_set = torch.utils.data.Subset(full_norm, test_raw.indices)

    train_loader = DataLoader(
        train_set,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    print(f"[Split] train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    return train_loader, val_loader, test_loader, mean, std
