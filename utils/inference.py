import os
from typing import overload, Generator, Dict
from argparse import Namespace

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

from model.cldm import ControlLDM
from model.gaussian_diffusion import Diffusion

from utils.common import instantiate_from_config, load_file_from_url, count_vram_usage
from utils.pipeline import (
    Pipeline,
    bicubic_resize
)
from utils.cond_fn import MSEGuidance, WeightedMSEGuidance
import torch

class InferenceLoop:

    def __init__(self, args: Namespace) -> "InferenceLoop":
        self.args = args
        self.loop_ctx = {}
        self.pipeline: Pipeline = None
        self.init_model()
        self.init_cond_fn()
        self.init_pipeline()


    @count_vram_usage
    def init_model(self) -> None:

        self.cldm: ControlLDM = instantiate_from_config(OmegaConf.load("configs/inference/cldm.yaml"))
        self.cldm.load_state_dict(torch.load(self.args.model))
        self.cldm.eval().to(self.args.device)
        ### load diffusion
        self.diffusion: Diffusion = instantiate_from_config(OmegaConf.load("configs/inference/diffusion.yaml"))
        self.diffusion.to(self.args.device)

    def init_cond_fn(self) -> None:
        if not self.args.guidance:
            self.cond_fn = None
            return
        if self.args.g_loss == "mse":
            cond_fn_cls = MSEGuidance
        elif self.args.g_loss == "w_mse":
            cond_fn_cls = WeightedMSEGuidance
        else:
            raise ValueError(self.args.g_loss)
        self.cond_fn = cond_fn_cls(
            scale=self.args.g_scale, t_start=self.args.g_start, t_stop=self.args.g_stop,
            space=self.args.g_space, repeat=self.args.g_repeat
        )


    def init_pipeline(self) -> None:
        self.pipeline = Pipeline(self.cldm, self.diffusion, self.cond_fn, self.args.device)


    def setup(self) -> None:
        self.output_dir = self.args.output
        os.makedirs(self.output_dir, exist_ok=True)

    def lq_loader(self) -> Generator[np.ndarray, None, None]:
        img_exts = [".png", ".jpg", ".jpeg",".PNG"]
        if os.path.isdir(self.args.input):
            file_names = sorted([
                file_name for file_name in os.listdir(self.args.input) if os.path.splitext(file_name)[-1] in img_exts
            ])
            file_paths = [os.path.join(self.args.input, file_name) for file_name in file_names]
        else:
            assert os.path.splitext(self.args.input)[-1] in img_exts
            file_paths = [self.args.input]

        def _loader() -> Generator[np.ndarray, None, None]:
            for file_path in file_paths:
                ### load lq
                lq = np.array(Image.open(file_path).convert("RGB"))
                print(f"load lq: {file_path}")
                ### set context for saving results
                self.loop_ctx["file_stem"] = os.path.splitext(os.path.basename(file_path))[0]
                for i in range(self.args.n_samples):
                    self.loop_ctx["repeat_idx"] = i
                    yield lq

        return _loader

    def after_load_lq(self, lq: np.ndarray) -> np.ndarray:
        return lq

    @torch.no_grad()
    def run(self) -> None:
        self.setup()
        loader = self.lq_loader()
        for lq in loader():
            lq = self.after_load_lq(lq)
            sample = self.pipeline.run(
                lq[None], self.args.steps, 1.0, self.args.tiled,
                self.args.tile_size, self.args.tile_stride,
                self.args.pos_prompt, self.args.neg_prompt, self.args.cfg_scale,
                self.args.better_start
            )[0]
            self.save(sample)

    def save(self, sample: np.ndarray) -> None:
        file_stem, repeat_idx = self.loop_ctx["file_stem"], self.loop_ctx["repeat_idx"]
        file_name = f"{file_stem}_{repeat_idx}.png" if self.args.n_samples > 1 else f"{file_stem}.png"
        save_path = os.path.join(self.args.output, file_name)
        Image.fromarray(sample).save(save_path)
        print(f"save result to {save_path}")



