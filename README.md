 推荐运行顺序
cd kolmogorov_diffusion_sr

# 1. 检查数据
python visualize_dataset.py

# 2. 训练无条件高分辨率扩散模型
python train.py

# 3. 条件采样，4x4 -> 256x256
python sample_sr.py

# 4. 评价
python evaluate.py

重要调参建议
问题 1：生成图像好看，但中心点不匹配 LR
增大：
COND_SCALE = 2.0
FINAL_COND_SCALE = 10.0
REFINE_STEPS = 20
REFINE_LR = 0.1

问题 2：中心点匹配了，但整体流场不自然
减小：
COND_SCALE = 0.5
FINAL_COND_SCALE = 2.0
REFINE_LR = 0.02

问题 3：训练 loss 降不下去
可以先改小模型或训练更久：
BATCH_SIZE = 4
LR = 1e-4
EPOCHS = 300

问题 4：显存不足
改：
BATCH_SIZE = 2
BASE_CHANNELS = 48
CHANNEL_MULTS = (1, 2, 4, 4)
USE_AMP = True


研究表述
这版实现的是：
首先用 25000 帧 256×256 Kolmogorov vorticity field 训练无条件 DDPM，使模型学习高分辨率涡量场分布 p(ωHR)p(\omega_{HR})p(ωHR​)。推理时，将 4×4 中心采样观测 ωLR\omega_{LR}ωLR​ 作为可微条件，通过中心采样算子 D(ωHR)=ωHR[32:224:64,32:224:64]D(\omega_{HR})=\omega_{HR}[32:224:64,32:224:64]D(ωHR​)=ωHR​[32:224:64,32:224:64] 构造条件误差，并在 DDIM 反向采样过程中用该误差对当前隐变量的梯度进行引导，从而采样 p(ωHR∣ωLR)p(\omega_{HR}|\omega_{LR})p(ωHR​∣ωLR​)。
也就是说，这不是普通的 supervised U-Net SR，而是：
unconditional high-resolution prior + differentiable observation constraint





