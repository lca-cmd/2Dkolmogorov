# diffusion.py

import math
import torch
import torch.nn.functional as F

import config


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-5, 0.999)


def linear_beta_schedule(timesteps):
    beta_start = 1e-4
    beta_end = 2e-2
    return torch.linspace(beta_start, beta_end, timesteps)


def extract(a, t, x_shape):
    b = t.shape[0]
    out = a.gather(-1, t.cpu()).to(t.device)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def center_stride_downsample(x):
    """
    x: (B, C, 256, 256)
    return: (B, C, 4, 4)

    采样位置：
    32, 96, 160, 224
    """
    idx = torch.tensor(config.CENTER_INDICES, device=x.device, dtype=torch.long)
    return x.index_select(-2, idx).index_select(-1, idx)


class GaussianDiffusion:
    def __init__(self, timesteps=1000, beta_schedule="cosine", device="cuda"):
        self.timesteps = timesteps
        self.device = device

        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        self.betas = betas.to(device)
        self.alphas = (1.0 - self.betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1],
            (1, 0),
            value=1.0,
        ).to(device)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(device)

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod).to(device)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1).to(device)

        self.posterior_variance = (
            self.betas
            * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        ).to(device)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_x0_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def p_losses(self, model, x_start):
        b = x_start.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x_start.device).long()
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        pred_noise = model(x_noisy, t)

        loss = F.mse_loss(pred_noise, noise)
        return loss

    @torch.no_grad()
    def ddim_sample_unconditional(
        self,
        model,
        shape,
        ddim_steps=100,
        eta=0.0,
    ):
        model.eval()

        device = self.device
        b = shape[0]
        x = torch.randn(shape, device=device)

        times = torch.linspace(
            self.timesteps - 1,
            0,
            steps=ddim_steps,
            device=device,
        ).long()

        time_pairs = list(zip(times[:-1], times[1:]))

        for time, time_next in time_pairs:
            t = torch.full((b,), time.item(), device=device, dtype=torch.long)
            t_next = torch.full((b,), time_next.item(), device=device, dtype=torch.long)

            pred_noise = model(x, t)
            x0 = self.predict_x0_from_noise(x, t, pred_noise)
            x0 = x0.clamp(-5, 5)

            alpha = extract(self.alphas_cumprod, t, x.shape)
            alpha_next = extract(self.alphas_cumprod, t_next, x.shape)

            sigma = (
                eta
                * torch.sqrt((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha))
            )
            c = torch.sqrt(1 - alpha_next - sigma ** 2)

            noise = torch.randn_like(x) if eta > 0 else 0.0
            x = torch.sqrt(alpha_next) * x0 + c * pred_noise + sigma * noise

        # final step at t=0
        t = torch.zeros((b,), device=device, dtype=torch.long)
        pred_noise = model(x, t)
        x0 = self.predict_x0_from_noise(x, t, pred_noise)

        return x0

    def ddim_sample_conditional_sr(
        self,
        model,
        lr_cond,
        shape=None,
        ddim_steps=100,
        eta=0.0,
        cond_scale=1.0,
        final_cond_scale=5.0,
        final_cond_steps=10,
        refine_steps=10,
        refine_lr=0.05,
    ):
        """
        lr_cond: (B, 1, 4, 4), 已经归一化
        shape:   (B, 1, 256, 256)

        返回:
        x0:      (B, 1, 256, 256), 归一化空间
        """

        model.eval()
        device = self.device
        lr_cond = lr_cond.to(device)

        b = lr_cond.shape[0]
        if shape is None:
            shape = (b, 1, config.IMAGE_SIZE, config.IMAGE_SIZE)

        x = torch.randn(shape, device=device)

        times = torch.linspace(
            self.timesteps - 1,
            0,
            steps=ddim_steps,
            device=device,
        ).long()

        time_pairs = list(zip(times[:-1], times[1:]))

        for step_id, (time, time_next) in enumerate(time_pairs):
            is_final_stage = step_id >= len(time_pairs) - final_cond_steps
            current_cond_scale = final_cond_scale if is_final_stage else cond_scale

            t = torch.full((b,), time.item(), device=device, dtype=torch.long)
            t_next = torch.full((b,), time_next.item(), device=device, dtype=torch.long)

            with torch.enable_grad():
                x = x.detach().requires_grad_(True)

                pred_noise = model(x, t)
                x0 = self.predict_x0_from_noise(x, t, pred_noise)

                lr_from_x0 = center_stride_downsample(x0)
                cond_loss = F.mse_loss(lr_from_x0, lr_cond)

                grad = torch.autograd.grad(cond_loss, x)[0]

                # 梯度归一化，避免不稳定
                grad_norm = grad.flatten(1).norm(dim=1).view(b, 1, 1, 1).clamp(min=1e-8)
                grad = grad / grad_norm

                x = x.detach() - current_cond_scale * grad.detach()

            with torch.no_grad():
                pred_noise = model(x, t)
                x0 = self.predict_x0_from_noise(x, t, pred_noise)
                x0 = x0.clamp(-5, 5)

                alpha = extract(self.alphas_cumprod, t, x.shape)
                alpha_next = extract(self.alphas_cumprod, t_next, x.shape)

                sigma = (
                    eta
                    * torch.sqrt((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha))
                )
                c = torch.sqrt(1 - alpha_next - sigma ** 2)

                noise = torch.randn_like(x) if eta > 0 else 0.0
                x = torch.sqrt(alpha_next) * x0 + c * pred_noise + sigma * noise

        # 最后得到 x0
        t = torch.zeros((b,), device=device, dtype=torch.long)
        with torch.no_grad():
            pred_noise = model(x, t)
            x0 = self.predict_x0_from_noise(x, t, pred_noise)

        # 最终 refine：直接在 x0 上让中心点更匹配 LR，同时不要破坏太多整体分布
        # 这里是轻微 refine，不参与模型反传
        x0 = x0.detach()

        for _ in range(refine_steps):
            x0 = x0.detach().requires_grad_(True)
            lr_from_x0 = center_stride_downsample(x0)
            cond_loss = F.mse_loss(lr_from_x0, lr_cond)
            grad = torch.autograd.grad(cond_loss, x0)[0]

            grad_norm = grad.flatten(1).norm(dim=1).view(b, 1, 1, 1).clamp(min=1e-8)
            grad = grad / grad_norm

            x0 = (x0 - refine_lr * grad).detach()

        return x0
