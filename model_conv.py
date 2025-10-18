import math
import shutil
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Dict, Optional, List, Any

import kornia
import numpy as np
import torch
import tqdm
import pandas as pd
from torch import nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader

from data_utils import array_to_nifti
from dataset import CardiacUKBBValidationFullImage, CardiacUKBBFullImage
from networks import Encoder
from utils import params_to_mat, make_coordinate_tensor, to_1hot, create_meshplot_visualization, \
    process_segmentation_with_marching_cubes
from model_autoreg import INR_AutoReg


class INR_Conv(INR_AutoReg):
    def __init__(self, *args, **kwargs):
        super(INR_Conv, self).__init__(*args, **kwargs)
        channels = kwargs.get('conv_channels', (32,64,64,128,128,128))
        self.encoder = Encoder(channels, self.latent_size, **kwargs)

    def get_sample_elements_from_batch(self, batch): return batch[:-1]  # Leave out stack of full images

    def get_latents(self, batch: Tuple[Any, ...], aff_def_params: torch.Tensor, *args) -> torch.Tensor:
        aff_params, num_subj_slices, imgs = batch[5], batch[-2], batch[-1]
        aff_params = aff_params + aff_def_params
        subj_latents = self.encoder(imgs, aff_params, num_subj_slices, self.global_step)
        return subj_latents

    def get_inf_dset(self, subj_idx: int, dset: CardiacUKBBFullImage):
        return CardiacUKBBValidationFullImage([dset.data_paths[subj_idx]],
                                              dset.max_slices, dset.max_slice_shape,
                                              dset.num_coords, to_gpu=True)

    def initialize_inference_params(self):
        latent_params = torch.randn((1, self.latent_size), dtype=torch.float32, device="cuda") * 1e-2
        aff_def_params = torch.zeros((1, self.max_slices, 6), dtype=torch.float32, device="cuda")
        intens_scale_params = torch.zeros((1, self.max_slices, 1), dtype=torch.float32, device="cuda")
        return latent_params, aff_def_params, intens_scale_params
