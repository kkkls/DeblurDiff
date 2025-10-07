from typing import overload, Tuple, Optional

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from PIL import Image
from einops import rearrange

from model.cldm import ControlLDM
from model.gaussian_diffusion import Diffusion
from utils.sampler import SpacedSampler
from utils.cond_fn import Guidance
from utils.common import wavelet_decomposition, wavelet_reconstruction, count_vram_usage
from torch import Tensor

def calc_mean_std(feat: Tensor, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.
    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    feat_var = feat.reshape(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().reshape(b, c, 1, 1)
    feat_mean = feat.reshape(b, c, -1).mean(dim=2).reshape(b, c, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat:Tensor, style_feat:Tensor):
    """Adaptive instance normalization.
    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.
    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def bicubic_resize(img: np.ndarray, scale: float) -> np.ndarray:
    pil = Image.fromarray(img)
    res = pil.resize(tuple(int(x * scale) for x in pil.size), Image.BICUBIC)
    return np.array(res)


def resize_short_edge_to(imgs: torch.Tensor, size: int) -> torch.Tensor:
    _, _, h, w = imgs.size()
    if h == w:
        new_h, new_w = size, size
    elif h < w:
        new_h, new_w = size, int(w * (size / h))
    else:
        new_h, new_w = int(h * (size / w)), size
    return F.interpolate(imgs, size=(new_h, new_w), mode="bicubic", antialias=True)


def pad_to_multiples_of(imgs: torch.Tensor, multiple: int) -> torch.Tensor:
    _, _, h, w = imgs.size()
    if h % multiple == 0 and w % multiple == 0:
        return imgs.clone()
    # get_pad = lambda x: (x // multiple + 1) * multiple - x
    get_pad = lambda x: (x // multiple + int(x % multiple != 0)) * multiple - x
    ph, pw = get_pad(h), get_pad(w)
    return F.pad(imgs, pad=(0, pw, 0, ph), mode="constant", value=0)


class Pipeline:

    def __init__(self, cldm: ControlLDM, diffusion: Diffusion, cond_fn: Optional[Guidance], device: str) -> None:
#        self.stage1_model = stage1_model
        self.cldm = cldm
        self.diffusion = diffusion
        self.cond_fn = cond_fn
        self.device = device
        self.final_size: Tuple[int] = None

    def set_final_size(self, lq: torch.Tensor) -> None:
        h, w = lq.shape[2:]
        self.final_size = (h, w)


    @count_vram_usage
    def run_diff(
        self,
        clean: torch.Tensor,
        steps: int,
        strength: float,
        tiled: bool,
        tile_size: int,
        tile_stride: int,
        pos_prompt: str,
        neg_prompt: str,
        cfg_scale: float,
        better_start: float
    ) -> torch.Tensor:
        ### preprocess
        bs, _, ori_h, ori_w = clean.shape
        # pad: ensure that height & width are multiples of 64
        pad_clean = pad_to_multiples_of(clean, multiple=64)
        h, w = pad_clean.shape[2:]
        # prepare conditon
        if not tiled:
            cond = self.cldm.prepare_condition(pad_clean, [pos_prompt] * bs)
            uncond = self.cldm.prepare_condition(pad_clean, [neg_prompt] * bs)
        else:
            cond = self.cldm.prepare_condition_tiled(pad_clean, [pos_prompt] * bs, tile_size, tile_stride)
            uncond = self.cldm.prepare_condition_tiled(pad_clean, [neg_prompt] * bs, tile_size, tile_stride)
        if self.cond_fn:
            self.cond_fn.load_target(pad_clean * 2 - 1)
        old_control_scales = self.cldm.control_scales
        self.cldm.control_scales = [strength] * 13
        if better_start:
            # using noised low frequency part of condition as a better start point of
            # reverse sampling, which can prevent our model from generating noise in
            # image background.
            _, low_freq = wavelet_decomposition(pad_clean)
            if not tiled:
                x_0 = self.cldm.vae_encode(low_freq)
            else:
                x_0 = self.cldm.vae_encode_tiled(low_freq, tile_size, tile_stride)
            x_T = self.diffusion.q_sample(
                x_0,
                torch.full((bs, ), self.diffusion.num_timesteps - 1, dtype=torch.long, device=self.device),
                torch.randn(x_0.shape, dtype=torch.float32, device=self.device)
            )
            # print(f"diffusion sqrt_alphas_cumprod: {self.diffusion.sqrt_alphas_cumprod[-1]}")
        else:
            x_T = torch.randn((bs, 4, h // 8, w // 8), dtype=torch.float32, device=self.device)
        ### run sampler
        sampler = SpacedSampler(self.diffusion.betas)
        z = sampler.sample(
            model=self.cldm, device=self.device, steps=steps, batch_size=bs, x_size=(4, h // 8, w // 8),
            cond=cond, uncond=uncond, cfg_scale=cfg_scale, x_T=x_T, progress=True,
            progress_leave=True, cond_fn=self.cond_fn, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
        )
        if not tiled:
            x = self.cldm.vae_decode(z)
        else:
            x = self.cldm.vae_decode_tiled(z, tile_size // 8, tile_stride // 8)
        ### postprocess
        self.cldm.control_scales = old_control_scales
        sample = x[:, :, :ori_h, :ori_w]
        return sample

    @torch.no_grad()
    def run(
        self,
        lq: np.ndarray,
        steps: int,
        strength: float,
        tiled: bool,
        tile_size: int,
        tile_stride: int,
        pos_prompt: str,
        neg_prompt: str,
        cfg_scale: float,
        better_start: bool
    ) -> np.ndarray:
        # image to tensor
        lq = torch.tensor((lq / 255.).clip(0, 1), dtype=torch.float32, device=self.device)
        lq = rearrange(lq, "n h w c -> n c h w").contiguous()
        # set pipeline output size
        self.set_final_size(lq)
        clean = lq
        sample = self.run_diff(
            clean, steps, strength, tiled, tile_size, tile_stride,
            pos_prompt, neg_prompt, cfg_scale, better_start
        )
        # colorfix (borrowed from StableSR, thanks for their work)
        sample = (sample + 1) / 2
        sample = adaptive_instance_normalization(sample, clean)

        sample = rearrange(sample * 255., "n c h w -> n h w c")
        sample = sample.contiguous().clamp(0, 255).to(torch.uint8).cpu().numpy()
        return sample


