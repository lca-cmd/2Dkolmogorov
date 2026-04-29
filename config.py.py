# config.py

import os
import torch

# =========================
# Data
# =========================
DATA_DIR = "/home/masters/PycharmProjects/PythonProject8/generated_kolmogorov_center_stride"

LR_KEY = "vort_lr"
HR_KEY = "vort_hr"

IMAGE_SIZE = 256
LR_SIZE = 4
IN_CHANNELS = 1

# 你的中心采样点
CENTER_INDICES = [32, 96, 160, 224]

# =========================
# Train / validation split
# =========================
TRAIN_RATIO = 0.90
VAL_RATIO = 0.05
TEST_RATIO = 0.05

RANDOM_SEED = 42

# =========================
# Normalization
# =========================
# 建议先用 train set 的 HR 均值方差归一化。
# LR 条件也用同一套 HR mean/std，因为 LR 是 HR 的中心采样值。
NORMALIZE = True

# =========================
# Diffusion
# =========================
TIMESTEPS = 1000
BETA_SCHEDULE = "cosine"  # "linear" or "cosine"

# =========================
# Model
# =========================
BASE_CHANNELS = 64
CHANNEL_MULTS = (1, 2, 4, 8)
TIME_EMB_DIM = 256
DROPOUT = 0.0

# =========================
# Training
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8
NUM_WORKERS = 4
LR = 2e-4
WEIGHT_DECAY = 1e-6
EPOCHS = 200

GRAD_CLIP = 1.0
USE_AMP = True

SAVE_EVERY = 5
SAMPLE_EVERY = 5

CHECKPOINT_DIR = "checkpoints"
SAMPLE_DIR = "samples"
LOG_DIR = "logs"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# =========================
# Conditional sampling
# =========================
# DDIM 采样步数，越大质量越好但越慢
DDIM_STEPS = 100
DDIM_ETA = 0.0

# 条件梯度强度。
# 对 4x4 极低分辨率，不建议太大。
COND_SCALE = 1.0

# 最后若干步增强条件约束
FINAL_COND_SCALE = 5.0
FINAL_COND_STEPS = 10

# 最终 refine 次数
REFINE_STEPS = 10
REFINE_LR = 0.05

# =========================
# Paths
# =========================
BEST_CKPT = os.path.join(CHECKPOINT_DIR, "best.pt")
LAST_CKPT = os.path.join(CHECKPOINT_DIR, "last.pt")
NORM_STATS = os.path.join(CHECKPOINT_DIR, "norm_stats.pt")
