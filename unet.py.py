# unet.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, dropout=0.0):
        super().__init__()

        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch),
        )

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        residual = x

        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        head_dim = c // self.num_heads

        q = q.reshape(b, self.num_heads, head_dim, h * w)
        k = k.reshape(b, self.num_heads, head_dim, h * w)
        v = v.reshape(b, self.num_heads, head_dim, h * w)

        scale = head_dim ** -0.5
        attn = torch.einsum("bhcn,bhcm->bhnm", q * scale, k)
        attn = attn.softmax(dim=-1)

        out = torch.einsum("bhnm,bhcm->bhcn", attn, v)
        out = out.reshape(b, c, h, w)

        return residual + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        time_emb_dim=256,
        dropout=0.0,
    ):
        super().__init__()

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        channels = [base_channels * m for m in channel_mults]

        # Down path
        self.downs = nn.ModuleList()
        in_ch = base_channels
        self.skip_channels = []

        for level, out_ch in enumerate(channels):
            self.downs.append(ResBlock(in_ch, out_ch, time_emb_dim, dropout))
            self.downs.append(ResBlock(out_ch, out_ch, time_emb_dim, dropout))

            # 在 32x32 和 16x16 附近加 attention
            if level >= 2:
                self.downs.append(AttentionBlock(out_ch))

            self.skip_channels.append(out_ch)

            if level != len(channels) - 1:
                self.downs.append(Downsample(out_ch))

            in_ch = out_ch

        # Middle
        mid_ch = channels[-1]
        self.mid1 = ResBlock(mid_ch, mid_ch, time_emb_dim, dropout)
        self.mid_attn = AttentionBlock(mid_ch)
        self.mid2 = ResBlock(mid_ch, mid_ch, time_emb_dim, dropout)

        # Up path
        self.ups = nn.ModuleList()
        rev_channels = list(reversed(channels))
        in_ch = mid_ch

        for level, out_ch in enumerate(rev_channels):
            skip_ch = out_ch

            self.ups.append(ResBlock(in_ch + skip_ch, out_ch, time_emb_dim, dropout))
            self.ups.append(ResBlock(out_ch, out_ch, time_emb_dim, dropout))

            if level <= 1:
                self.ups.append(AttentionBlock(out_ch))

            if level != len(rev_channels) - 1:
                self.ups.append(Upsample(out_ch))

            in_ch = out_ch

        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_conv = nn.Conv2d(base_channels, in_channels, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embedding(t)

        x = self.init_conv(x)
        skips = []

        for module in self.downs:
            if isinstance(module, ResBlock):
                x = module(x, t_emb)
            else:
                x = module(x)

            if isinstance(module, ResBlock):
                # 每个 level 第二个 ResBlock 后会重复存，不过可以工作；
                # 为了稳定，只在 spatial size 变化前保存最后一个特征更好。
                pass

        # 上面简单写法不方便保存 skip，下面改用显式 forward 会更清楚。
        raise RuntimeError("Use UNetV2 below instead of UNet.")


class UNetV2(nn.Module):
    def __init__(
        self,
        in_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        time_emb_dim=256,
        dropout=0.0,
    ):
        super().__init__()

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        chs = [base_channels * m for m in channel_mults]

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        in_ch = base_channels
        for i, out_ch in enumerate(chs):
            use_attn = i >= 2
            block = nn.ModuleDict({
                "res1": ResBlock(in_ch, out_ch, time_emb_dim, dropout),
                "res2": ResBlock(out_ch, out_ch, time_emb_dim, dropout),
                "attn": AttentionBlock(out_ch) if use_attn else nn.Identity(),
            })
            self.down_blocks.append(block)

            if i != len(chs) - 1:
                self.downsamples.append(Downsample(out_ch))
            else:
                self.downsamples.append(nn.Identity())

            in_ch = out_ch

        mid_ch = chs[-1]
        self.mid1 = ResBlock(mid_ch, mid_ch, time_emb_dim, dropout)
        self.mid_attn = AttentionBlock(mid_ch)
        self.mid2 = ResBlock(mid_ch, mid_ch, time_emb_dim, dropout)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        rev_chs = list(reversed(chs))
        in_ch = mid_ch

        for i, out_ch in enumerate(rev_chs):
            use_attn = i <= 1
            block = nn.ModuleDict({
                "res1": ResBlock(in_ch + out_ch, out_ch, time_emb_dim, dropout),
                "res2": ResBlock(out_ch, out_ch, time_emb_dim, dropout),
                "attn": AttentionBlock(out_ch) if use_attn else nn.Identity(),
            })
            self.up_blocks.append(block)

            if i != len(rev_chs) - 1:
                self.upsamples.append(Upsample(out_ch))
            else:
                self.upsamples.append(nn.Identity())

            in_ch = out_ch

        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_conv = nn.Conv2d(base_channels, in_channels, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embedding(t)

        x = self.init_conv(x)
        skips = []

        for block, downsample in zip(self.down_blocks, self.downsamples):
            x = block["res1"](x, t_emb)
            x = block["res2"](x, t_emb)
            x = block["attn"](x)
            skips.append(x)
            x = downsample(x)

        x = self.mid1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid2(x, t_emb)

        for block, upsample in zip(self.up_blocks, self.upsamples):
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = block["res1"](x, t_emb)
            x = block["res2"](x, t_emb)
            x = block["attn"](x)
            x = upsample(x)

        x = self.out_conv(F.silu(self.out_norm(x)))
        return x
