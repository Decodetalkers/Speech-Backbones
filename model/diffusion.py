# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

from emodataset import max_2_div
from utils import EMO_FEATURES
from typing import Optional, Tuple, List
import math
import torch
from einops import rearrange

from model.base import BaseModule
from .unet import Unet


class Mish(BaseModule):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(torch.nn.functional.softplus(x))


class Upsample(BaseModule):
    def __init__(self, dim: int):
        super(Upsample, self).__init__()
        self.conv = torch.nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Downsample(BaseModule):
    def __init__(self, dim: int):
        super(Downsample, self).__init__()
        self.conv = torch.nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Rezero(BaseModule):
    def __init__(self, fn: torch.nn.Module):
        super(Rezero, self).__init__()
        self.fn = fn
        self.g = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x) * self.g


class Block(BaseModule):
    def __init__(self, dim: int, dim_out: int, groups: int = 8):
        super(Block, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(dim, dim_out, 3, padding=1),
            torch.nn.GroupNorm(groups, dim_out),
            Mish(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        output = self.block(x * mask)
        return output * mask


class ResnetBlock(BaseModule):
    def __init__(self, dim: int, dim_out: int, time_emb_dim: int, groups: int = 8):
        super(ResnetBlock, self).__init__()
        self.mlp = torch.nn.Sequential(Mish(), torch.nn.Linear(time_emb_dim, dim_out))

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        if dim != dim_out:
            self.res_conv = torch.nn.Conv2d(dim, dim_out, 1)
        else:
            self.res_conv = torch.nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor, time_emb: torch.Tensor):
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output


class LinearAttention(BaseModule):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super(LinearAttention, self).__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = torch.nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(
            out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w
        )
        return self.to_out(out)


class Residual(BaseModule):
    def __init__(self, fn: torch.nn.Module):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        output = self.fn(x, *args, **kwargs) + x
        return output


class SinusoidalPosEmb(BaseModule):
    def __init__(self, dim: int):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, scale: int = 1000) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# this module will train a random shape data to 4 grade
# flat the n * m data and add embedding
class EmoClassify(torch.nn.Module):
    convs: List[torch.nn.Module]

    def __init__(
        self,
        n_mels: int,
        out_features: int,
        hidden_dim: int = 1024,
        num_layers: int = 1,
        time_embedding_dim: int = 256,
        dropout: float = 0.1,
        tau: float = 0.01,
    ):
        super(EmoClassify, self).__init__()

        self.tau = tau
        self.unet = Unet(time_embedding_dim, in_channels=1, out_channels=1)

        self.lstm = torch.nn.LSTM(
            n_mels,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.linear = torch.nn.Linear(hidden_dim, out_features)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(
        self, x_0: torch.Tensor, timestamp: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x_0 = x_0.unsqueeze(dim=1)
        x_0 = self.unet(x_0, timestamp)
        x_0 = x_0.squeeze()
        x_0 = x_0.transpose(-1, -2)
        x_0, (_hidden, _cell) = self.lstm(x_0)
        x_0 = self.linear(x_0)
        x_0 = x_0.mean(dim=-2)
        x_0 = self.softmax(x_0)

        return x_0

    def train_label(
        self, x_0: torch.Tensor, timestamp: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        this part make the biggest label stronger, in order to predict the right label
        Only used in training process
        """
        x_0 = self.forward(x_0, timestamp)
        x_0 = self.softmax(x_0 / self.tau)
        return x_0


# In fact, it is unet(Like)
class GradLogPEstimator2d(BaseModule):
    def __init__(
        self,
        dim: int,
        dim_mults: List[int] = [1, 2, 4],
        groups: int = 8,
        n_spks: Optional[int] = None,
        spk_emb_dim: int = 64,
        n_feats: int = 80,
        pe_scale: int = 1000,
    ):
        super(GradLogPEstimator2d, self).__init__()
        self.dim = dim
        self.dim_mults = dim_mults
        self.groups = groups
        self.n_spks = n_spks if not isinstance(n_spks, type(None)) else 1
        self.spk_emb_dim = spk_emb_dim
        self.pe_scale = pe_scale

        if n_spks is not None and n_spks > 1:
            self.spk_mlp = torch.nn.Sequential(
                torch.nn.Linear(spk_emb_dim, spk_emb_dim * 4),
                Mish(),
                torch.nn.Linear(spk_emb_dim * 4, n_feats),
            )
        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4), Mish(), torch.nn.Linear(dim * 4, dim)
        )

        dims = [2 + (1 if n_spks > 1 else 0), *map(lambda m: dim * m, dim_mults)]  # ty:ignore[unsupported-operator]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                torch.nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, time_emb_dim=dim),
                        ResnetBlock(dim_out, dim_out, time_emb_dim=dim),
                        Residual(Rezero(LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else torch.nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(
                torch.nn.ModuleList(
                    [
                        ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim),
                        ResnetBlock(dim_in, dim_in, time_emb_dim=dim),
                        Residual(Rezero(LinearAttention(dim_in))),
                        Upsample(dim_in),
                    ]
                )
            )
        self.final_block = Block(dim, dim)
        self.final_conv = torch.nn.Conv2d(dim, 1, 1)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spk: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not isinstance(spk, type(None)):
            s = self.spk_mlp(spk)

        t = self.time_pos_emb(t, scale=self.pe_scale)
        t = self.mlp(t)

        if self.n_spks < 2:
            x = torch.stack([mu, x], 1)
        else:
            s = s.unsqueeze(-1).repeat(1, 1, x.shape[-1])
            x = torch.stack([mu, x, s], 1)
        mask = mask.unsqueeze(1)

        hiddens = []
        masks = [mask]
        for resnet1, resnet2, attn, downsample in self.downs:  # ty:ignore[not-iterable]
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)
            x = resnet2(x, mask_down, t)
            x = attn(x)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]
        x = self.mid_block1(x, mask_mid, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, mask_mid, t)

        for resnet1, resnet2, attn, upsample in self.ups:  # ty:ignore[not-iterable]
            mask_up = masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up, t)
            x = resnet2(x, mask_up, t)
            x = attn(x)
            x = upsample(x * mask_up)

        x = self.final_block(x, mask)
        output = self.final_conv(x * mask)

        return (output * mask).squeeze(1)


# t: timestamp ??
def get_noise(
    t: torch.Tensor, beta_init: float, beta_term: float, cumulative: bool = False
) -> torch.Tensor:
    if cumulative:
        noise = beta_init * t + 0.5 * (beta_term - beta_init) * (t**2)
    else:
        noise = beta_init + (beta_term - beta_init) * t
    return noise


class Diffusion(BaseModule):
    def __init__(
        self,
        n_feats: int,
        dim: int,
        n_spks: int = 1,
        spk_emb_dim: Optional[int] = 64,
        beta_min: float = 0.05,
        beta_max: int | float = 20,
        pe_scale: int = 1000,
        time_embedding_dim: int = 128,
    ):
        super(Diffusion, self).__init__()
        self.n_feats = n_feats
        self.dim = dim
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim if spk_emb_dim is not None else 64
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale

        self.estimator = GradLogPEstimator2d(
            dim, n_spks=n_spks, spk_emb_dim=self.spk_emb_dim, pe_scale=pe_scale
        )
        self.emo_estimtor = EmoClassify(
            n_feats, EMO_FEATURES, time_embedding_dim=time_embedding_dim
        )
        self.mel_div = max_2_div(n_feats)
        self.emo_loss = torch.nn.CrossEntropyLoss()

    # NOTE: I should know how the time is like
    # And maybe I should get the time in every stage?
    def forward_diffusion(
        self, x0: torch.Tensor, mask: torch.Tensor, mu: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        mean = x0 * torch.exp(-0.5 * cum_noise) + mu * (
            1.0 - torch.exp(-0.5 * cum_noise)
        )
        variance = 1.0 - torch.exp(-cum_noise)
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, requires_grad=False)
        xt = mean + z * torch.sqrt(variance)
        return xt * mask, z * mask

    # FIXME: I need it do not grade, but I need the grad
    @torch.no_grad
    def reverse_diffusion(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        n_timesteps: int,
        stoc: bool = False,
        spk: Optional[torch.Tensor] = None,
        emo: Optional[int] = None,
        emo_hydrid: Optional[float] = None,
    ) -> torch.Tensor:
        h = 1.0 / n_timesteps
        xt = z * mask
        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5) * h) * torch.ones(
                z.shape[0], dtype=z.dtype, device=z.device
            )
            emo_noise = torch.zeros(z.shape, device=z.device).detach().clone().requires_grad_(True)
            if emo is not None and emo_hydrid is not None:

                with torch.enable_grad():
                    self.emo_estimtor.train()
                    mel_max_len = xt.shape[-1]
                    for _ in range(self.mel_div):
                        if mel_max_len % self.mel_div == 0:
                            break
                        mel_max_len += 1
                    xt_copy = torch.zeros(
                        (xt.shape[0], xt.shape[1], mel_max_len),
                        dtype=torch.float32,
                        device=xt.device,
                    )
                    xt_copy[:, :, : xt.shape[-1]] = xt
                    xt_copy = xt_copy.detach().clone().requires_grad_(True)
                    target = self.emo_estimtor.forward(xt_copy, t)
                    emo_now = torch.zeros(5, device=z.device)
                    emo_now[emo] = 1
                    loss = self.emo_loss(target, emo_now)
                    grads = torch.autograd.grad(outputs=loss, inputs=xt_copy)[0]

                emo_noise = grads * emo_hydrid
                emo_noise = emo_noise[:,:,:xt.shape[-1]]
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = get_noise(time, self.beta_min, self.beta_max, cumulative=False)
            if stoc:  # adds stochastic term
                dxt_det = (
                    0.5 * (mu - xt) - self.estimator(xt, mask, mu, t, spk) - emo_noise
                )
                dxt_det = dxt_det * noise_t * h
                dxt_stoc = torch.randn(
                    z.shape, dtype=z.dtype, device=z.device, requires_grad=False
                )
                dxt_stoc = dxt_stoc * torch.sqrt(noise_t * h)
                dxt = dxt_det + dxt_stoc
            else:
                dxt = 0.5 * (mu - xt - self.estimator(xt, mask, mu, t, spk) - emo_noise)
                dxt = dxt * noise_t * h
            xt = (xt - dxt) * mask
        return xt

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        n_timesteps: int,
        stoc: bool = False,
        spk: Optional[torch.Tensor] = None,
        emo: Optional[int] = None,
        emo_hydrid: Optional[float] = None,
    ) -> torch.Tensor:
        return self.reverse_diffusion(
            z, mask, mu, n_timesteps, stoc, spk, emo, emo_hydrid
        )

    def loss_t(
        self,
        x0: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        emo_label: Optional[torch.Tensor] = None,
        spk: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xt, z = self.forward_diffusion(x0, mask, mu, t)
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        noise_estimation = self.estimator(xt, mask, mu, t, spk)
        noise_estimation *= torch.sqrt(1.0 - torch.exp(-cum_noise))
        loss = torch.sum((noise_estimation + z) ** 2) / (torch.sum(mask) * self.n_feats)
        # TODO: it is not a good way to let loss together
        if emo_label is not None:
            mel_max_len = xt.shape[-1]
            for _ in range(self.mel_div):
                if mel_max_len % self.mel_div == 0:
                    break
                mel_max_len += 1
                xt_copy = torch.zeros(
                    (xt.shape[0], xt.shape[1], mel_max_len),
                    dtype=torch.float32,
                    device=xt.device,
                )
                xt_copy[:, :, : xt.shape[-1]] = xt
            label_predict = self.emo_estimtor.train_label(xt_copy, t)
            loss += self.emo_loss(label_predict, emo_label.float())
        return loss, xt

    # NOTE: modify it
    def compute_loss(
        self,
        x0: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        emo_label: Optional[torch.Tensor] = None,
        spk: Optional[torch.Tensor] = None,
        offset: float = 1e-5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.rand(
            x0.shape[0], dtype=x0.dtype, device=x0.device, requires_grad=False
        )
        t = torch.clamp(t, offset, 1.0 - offset)
        return self.loss_t(x0, mask, mu, t, emo_label, spk)
