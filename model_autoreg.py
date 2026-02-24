import copy
import math
import shutil
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Dict, Optional, List, Union, Any, Iterable

import kornia
import numpy as np
import torch
import tqdm
import pandas as pd
from torch import nn
from torchvision.utils import save_image
import torch.nn.functional as F
import lightning.pytorch as pl
import wandb
from monai.losses import DiceLoss

from data_utils import array_to_nifti
from dataset import CardiacUKBB, CardiacUKBBValidation
from networks import MLP
from pos_encoding import PosEncodingNeRFAnnealed, PosEncodinFourier
from utils import params_to_mat, make_coordinate_tensor, to_1hot, create_meshplot_visualization, \
    process_segmentation_with_marching_cubes, data_frame_to_line_plot, video_array_to_file, \
    fast_trilinear_interpolation, sdf_hierarchy_to_seg, keep_last_true, burn_contours

RED = (1.0, 0.0, 0.0)
GREEN = (0.0, 1.0, 0.0)
BLUE = (1.0, 1.0, 0.0)
COLORS = {1: RED, 2: GREEN, 3: BLUE}

class INR_AutoReg(pl.LightningModule):
    def __init__(self, coord_size: int, num_subjects: int, max_slices: int, log_path: Optional[Path] = None, **kwargs):
        super(INR_AutoReg, self).__init__()
        self.automatic_optimization = False  # Lightning param
        self.logging_disabled = kwargs['logging_disabled']
        self.logging_wandb_disabled = kwargs['logging_wandb_disabled']
        self.logging_rate = kwargs['logging_rate']
        self.logging_start_rate = kwargs['logging_start_rate']
        self.addit_log_epochs = kwargs['addit_log_epochs']
        self.inference_metrics = {}
        self.log_path = log_path

        self.coord_size = coord_size
        self.intensity_size = 1
        self.num_classes = 3
        self.num_subjects = num_subjects
        self.max_slices = max_slices
        self.norm_min, self.norm_max = 0.0, 1.0
        self.point_spread_start_epoch = kwargs['point_spread_start_epoch']
        self.num_coords_during_point_spread = kwargs['num_coords_during_point_spread']
        self.point_spread_size_after = kwargs['point_spread_size_after']
        self.point_spread_size_before = kwargs['point_spread_size_before']
        self.point_spread_std_before = torch.tensor(kwargs['point_spread_std_before'], dtype=torch.float32, device="cuda"
                                             ).reshape(1, 1, 1, self.coord_size)
        self.point_spread_std_after = torch.tensor(kwargs['point_spread_std_after'], dtype=torch.float32, device="cuda"
                                             ).reshape(1, 1, 1, self.coord_size)
        self.latent_size = kwargs["latent_size"]
        self.spatial_functa_res = kwargs["spatial_functa_resolution"]
        self.use_spatial_functa = self.spatial_functa_res > 1
        assert isinstance(self.spatial_functa_res, int) and self.spatial_functa_res > 0
        latent_vector_size = self.latent_size * self.spatial_functa_res**3
        self.subj_latents = nn.Parameter(torch.randn((self.num_subjects, latent_vector_size),
                                                     dtype=torch.float32, device="cuda") * 1e-2, requires_grad=True)

        self.aff_deform_params = nn.Parameter(torch.zeros((self.num_subjects, self.max_slices, 6),
                                                          dtype=torch.float32, device="cuda"), requires_grad=True)
        self.int_scale_range = kwargs["int_scale_range"]
        self.intensity_scale_params = nn.Parameter(torch.randn((self.num_subjects, self.max_slices, 1),
                                                               dtype=torch.float32, device="cuda") * 1e-3, requires_grad=True)
        self.pos_enc = PosEncodingNeRFAnnealed(in_dim=self.coord_size+1, # 3D + Cyclical time
                                               num_frequencies=kwargs['pe_num_frequencies'],
                                               anneal_max_iter=kwargs['pe_anneal_max_iter'],
                                               anneal_start_prop=kwargs['pe_anneal_start_prop'])
        self.canonical_inr = MLP(self.pos_enc.out_dim, self.latent_size,
                                 num_hidden_layers=kwargs['num_hidden_layers'],
                                 hidden_size=kwargs['hidden_size'],
                                 out_size=self.intensity_size + self.num_classes)

        self.class_weight = torch.tensor([i / sum(kwargs['weight_seg_class']) for i in kwargs['weight_seg_class']])
        self.seg_loss = DiceLoss(softmax=False, reduction="none")
        self.psnr_loss = kornia.losses.PSNRLoss(max_val=1.0)
        self.mse_loss = torch.nn.MSELoss(reduction='none')

        self.weight_reg_inr = kwargs["weight_reg_inr"]
        self.weight_reg_aff = kwargs["weight_reg_aff"]
        self.weight_reg_lat = kwargs["weight_reg_lat"]
        self.weight_reg_intens_scale = kwargs["weight_reg_int_scale"]
        self.weight_loss_deriv = kwargs["weight_loss_deriv"]
        self.weight_loss_seg = kwargs["weight_loss_seg"]
        self.supervise_seg = self.weight_loss_seg != 0
        self.lr_inr = kwargs["learning_rate_inr"]
        self.lr_lat = kwargs["learning_rate_lat"]
        self.lr_aff = kwargs["learning_rate_aff"]
        self.lr_intens_scale = kwargs["learning_rate_int_scale"]
        self.lr_anneal_tmax = kwargs["lr_anneal_tmax"]
        self.lr_anneal_eta_min = kwargs["learning_rate_anneal_eta_min"]
        # Inference hyperparams
        self.inf_weight_loss_seg = kwargs["inf_weight_loss_seg"]
        self.inf_max_epochs = kwargs["inf_max_epochs"]
        self.inf_num_coords = kwargs["inf_num_coords"]
        self.inf_lr_inr = kwargs["inf_learning_rate_inr"]
        self.inf_lr_latent = kwargs["inf_learning_rate_latent"]
        self.inf_lr_aff = kwargs["inf_learning_rate_aff"]
        self.inf_lr_def = kwargs["inf_learning_rate_def"]
        self.inf_point_spread_start_epoch = kwargs['inf_point_spread_start_epoch']

    def configure_optimizers(self):
        opt_inr = torch.optim.AdamW([*self.canonical_inr.parameters()], lr=self.lr_inr)
        opt_latent = torch.optim.AdamW([ self.subj_latents], lr=self.lr_lat)
        opt_aff = torch.optim.AdamW([ self.aff_deform_params], lr=self.lr_aff)
        opt_intens_scale = torch.optim.AdamW([self.intensity_scale_params], lr=self.lr_intens_scale)

        sched_inr = torch.optim.lr_scheduler.CosineAnnealingLR(opt_inr, T_max=self.lr_anneal_tmax, eta_min=self.lr_anneal_eta_min)
        sched_latent = torch.optim.lr_scheduler.CosineAnnealingLR(opt_latent, T_max=self.lr_anneal_tmax, eta_min=self.lr_anneal_eta_min)
        sched_aff = torch.optim.lr_scheduler.CosineAnnealingLR(opt_aff, T_max=self.lr_anneal_tmax, eta_min=self.lr_anneal_eta_min)
        sched_intens_scale = torch.optim.lr_scheduler.CosineAnnealingLR(opt_intens_scale, T_max=self.lr_anneal_tmax, eta_min=self.lr_anneal_eta_min)

        return (
            [opt_inr, opt_latent, opt_aff, opt_intens_scale],
            [
                {"scheduler": sched_inr, "interval": "epoch"},
                {"scheduler": sched_latent, "interval": "epoch"},
                {"scheduler": sched_aff, "interval": "epoch"},
                {"scheduler": sched_intens_scale, "interval": "epoch"},
            ]
        )

    def reset_schedulers(self, T_max: int, eta_min: float):
        """Call this before the second .fit() to restart schedulers fresh."""
        opt_inr, opt_latent, opt_aff, opt_intens_scale = self.optimizers()
        for opt, new_lr in zip([opt_inr, opt_latent, opt_aff, opt_intens_scale],
                               [self.lr_inr, self.lr_lat, self.lr_aff, self.lr_intens_scale]):
            for pg in opt.param_groups:
                pg['lr'] = new_lr
                pg['initial_lr'] = new_lr
        new_scheds = [
            torch.optim.lr_scheduler.CosineAnnealingLR(opt_inr, T_max=T_max, eta_min=eta_min),
            torch.optim.lr_scheduler.CosineAnnealingLR(opt_latent, T_max=T_max, eta_min=eta_min),
            torch.optim.lr_scheduler.CosineAnnealingLR(opt_aff, T_max=T_max, eta_min=eta_min),
            torch.optim.lr_scheduler.CosineAnnealingLR(opt_intens_scale, T_max=T_max, eta_min=eta_min),
        ]
        # Swap out the internal scheduler objects Lightning holds
        for i, new_sched in enumerate(new_scheds):
            self.trainer.lr_scheduler_configs[i].scheduler = new_sched

    @staticmethod
    def loss_reg_inr_params(params, weight: float, dict_name='loss_reg_inr'):
        loss_reg_inr = sum((p * p).sum() for p in params) * weight if weight else 0.0
        return loss_reg_inr, {dict_name: loss_reg_inr}

    @staticmethod
    def loss_reg_aff_params(params: torch.Tensor, weight: float, num_subj_slices: Optional[torch.Tensor] = None, dict_name='loss_reg_aff'):
        if num_subj_slices is not None:
            non_pad_slices = torch.arange(0, params.shape[1], device=params.device).tile((params.shape[0],1)) < num_subj_slices[:, None]
            params_ = params[non_pad_slices]
        else:
            params_ = params
        loss_reg_aff = F.mse_loss(params_, torch.zeros_like(params_)) * weight if weight else 0.0
        return loss_reg_aff, {dict_name: loss_reg_aff}

    @staticmethod
    def loss_reg_latent_params(params: torch.Tensor, weight: float, dict_name='loss_reg_lat'):
        loss_reg_lat = F.mse_loss(params, torch.zeros_like(params)) * weight if weight else 0.0
        return loss_reg_lat, {dict_name: loss_reg_lat}

    @staticmethod
    def loss_reg_int_scale_params(params: torch.Tensor, weight: float, num_subj_slices: Optional[torch.Tensor] = None, dict_name='loss_reg_int_scale'):
        if num_subj_slices is not None:
            non_pad_slices = torch.arange(0, params.shape[1], device=params.device).tile((params.shape[0],1)) < num_subj_slices[:, None]
            params_ = params[non_pad_slices]
        else:
            params_ = params
        loss_reg_int = F.mse_loss(params_, torch.zeros_like(params_)) * weight if weight else 0.0
        return loss_reg_int, {dict_name: loss_reg_int}

    def regularization_criterion(self, subj_idx: torch.Tensor) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        reg_inr_loss, reg_inr_dict = self.loss_reg_inr_params(self.canonical_inr.parameters(), self.weight_reg_inr)
        reg_aff_loss, reg_aff_dict = self.loss_reg_aff_params(self.aff_deform_params[subj_idx], self.weight_reg_aff)
        reg_lat_loss, reg_lat_dict = self.loss_reg_latent_params(self.subj_latents[subj_idx], self.weight_reg_lat)
        reg_int_scale_loss, reg_int_scale_dict = self.loss_reg_int_scale_params(self.intensity_scale_params[subj_idx], self.weight_reg_intens_scale)
        reg_loss = reg_inr_loss + reg_aff_loss + reg_lat_loss + reg_int_scale_loss
        reg_dict = {f"loss_reg": reg_loss,
                    **reg_inr_dict, **reg_lat_dict,
                    **reg_aff_dict, **reg_int_scale_dict,
                    }
        return reg_loss, reg_dict

    def forward(self,
                coords_voxel: torch.Tensor,
                aff_params: torch.Tensor,
                spacings: torch.Tensor,
                needs_flip: torch.BoolTensor,
                slice_idx: torch.LongTensor,
                min_coords: torch.Tensor,
                max_coords: torch.Tensor,
                latent_params: torch.Tensor,
                aff_def_params: Optional[torch.Tensor] = None,
                return_deriv: bool = False,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        world_coords = self.forward_coord_model(coords_voxel, aff_params, spacings, needs_flip,
                                                slice_idx, min_coords, max_coords, aff_def_params)
        values_pred, sdf_pred = self.forward_inr(world_coords, latent_params, **kwargs)
        sdf_pred_d = None
        if return_deriv:
            (B, N, C), D = sdf_pred.shape, world_coords.shape[-1]
            sdf_pred_d = torch.zeros((B, N, C, D), dtype=sdf_pred.dtype, device=sdf_pred.device)
            for i in range(C):
                sdf_pred_d[:, :, i] = torch.autograd.grad(sdf_pred, world_coords, grad_outputs=torch.ones_like(sdf_pred),
                                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
        return values_pred, sdf_pred, sdf_pred_d, world_coords

    def forward_coord_model(self,
                            coords: torch.Tensor,
                            aff_params: torch.Tensor,
                            spacings: torch.Tensor,
                            needs_flip: torch.BoolTensor,
                            slice_idx: torch.LongTensor,
                            min_coords: torch.Tensor,
                            max_coords: torch.Tensor,
                            aff_def_params: Optional[torch.Tensor] = None,
                            normalize: bool = True,
                            inverse: bool = False
                            ) -> torch.Tensor:
        assert coords.shape[-1] == 4
        # normalize coordinates
        if normalize and inverse:
            coords = self.min_max_unnormalize(coords, min_coords[:, None], max_coords[:, None], self.norm_min, self.norm_max)

        # Create indexing tensor to keep track of which batch is each coordinate coming from
        b, _ = torch.meshgrid(torch.arange(0, slice_idx.shape[0]), torch.arange(0, slice_idx.shape[1]))
        coords_flat_ = coords.reshape((-1, coords.shape[-1])).to(torch.float32)

        if aff_def_params is not None:
            aff_params = aff_params + aff_def_params
        aff_params_ = aff_params.reshape((-1, aff_params.shape[-1]))
        spacings_ = spacings.reshape((-1, spacings.shape[-1]))
        needs_flip_ = needs_flip.reshape((-1,))
        affines_ = params_to_mat(aff_params_, spacings_, needs_flip_)
        affines = affines_.reshape((aff_params.shape[0], aff_params.shape[1], 4, 4))

        # In order to move coords from voxel space to world space, we need to have them in (x, y, z, 1)
        time_coord_ = coords_flat_[:, -1:]  # We take out time coordinates. We will replace them back in later.
        spatial_coord_ = torch.cat((coords_flat_[:, :-1], torch.ones_like(coords_flat_[:, :1])), dim=1)  # Moving coords to world space requires (x, y, z, 1)

        # Get affines corresponding to each coordinate
        affines_ = affines[b.reshape(-1), slice_idx.reshape(-1)]
        if inverse:
            affines_ = torch.linalg.inv(affines_)
        # Move coordinates to world space
        coords_transformed_ = torch.bmm(affines_, spatial_coord_[..., None])
        coords_transformed_ = coords_transformed_.squeeze(-1)
        # Add time coordinate back
        coords_transformed_[:, -1:] = time_coord_
        coords_transformed = coords_transformed_.reshape(coords.shape)

        # normalize coordinates
        if normalize and not inverse:
            coords_transformed = self.min_max_normalize(coords_transformed, min_coords[:, None], max_coords[:, None], self.norm_min, self.norm_max)
        return coords_transformed

    def forward_inr(self,
                    coords: torch.Tensor,
                    subject_latent: torch.Tensor,
                    **kwargs) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        coords_enc, subject_latent_tile = self.process_input(coords, subject_latent, **kwargs)
        values_pred, seg_pred = self.forward_view_inr(coords_enc, subject_latent_tile)
        return values_pred, seg_pred

    def process_input(self,
                      coords: torch.Tensor,
                      subject_latent: torch.Tensor,
                      inference=False,
                      **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        # Make time dim cyclical
        coords = torch.cat((coords[..., :3],
                            torch.cos(coords[...,-1:] * torch.pi),
                            torch.sin(coords[...,-1:] * torch.pi)), dim=-1)
        coords_enc = self.pos_enc(coords, None if inference else self.global_step)
        if self.use_spatial_functa:
            r = self.spatial_functa_res
            spatial_latent_params = subject_latent.reshape(-1, r, r, r, self.latent_size)
            latent_coords = coords / 2 + 0.5 * r
            subject_latent = fast_trilinear_interpolation(spatial_latent_params, latent_coords[..., 0],
                                                          latent_coords[..., 1], latent_coords[..., 2])
        else:
            subject_latent = subject_latent[:, None].tile((1, coords.shape[1], 1))
        # x = torch.cat((coords_enc, subject_latent), dim=-1)
        return coords_enc, subject_latent

    def forward_view_inr(self,
                         coords: Optional[torch.Tensor],
                         subject_latent: Optional[torch.Tensor],
                         **kwargs) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        # Forward INR to obtain predicted volume values
        B, N, F = coords.shape
        coords_ = coords.reshape((B*N, -1))
        subject_latent_ = subject_latent.reshape((B*N, -1))
        values_pred_ = self.canonical_inr(coords_, subject_latent_)
        values_pred = values_pred_[:, 0].reshape((B, N))
        values_pred = torch.sigmoid(values_pred)
        seg_pred = values_pred_[:, 1:].reshape((B, N, self.num_classes))
        return values_pred, seg_pred

    def forward_intensity_params(self,
                                 coords_voxel: torch.Tensor,
                                 slice_idx: Optional[torch.Tensor],
                                 inten_scale_params: torch.Tensor) -> torch.Tensor:
        B, N = slice_idx.shape[:2]
        b_idx_tile = torch.arange(B, dtype=torch.long, device=slice_idx.device)[:,None].tile(1, N)
        inten_scale_params_ = inten_scale_params[b_idx_tile.flatten(), slice_idx.flatten()]
        int_deform_ = torch.tanh(inten_scale_params_) * (self.int_scale_range / 2)
        return int_deform_.reshape(*coords_voxel.shape[:2])

    def apply_intensity_scaling(self,
                                intensities: torch.Tensor,
                                coords_voxel: torch.Tensor,
                                slice_idx: torch.Tensor,
                                intens_scale_params: torch.Tensor,
                                inverse: bool = False):
        intens_scale = self.forward_intensity_params(coords_voxel, slice_idx, intens_scale_params)
        if not inverse:
            return intensities * (1 + intens_scale)
        else:
            return intensities / (1 + intens_scale)

    def forward_with_point_spread(self,
                                  coords_voxel: torch.Tensor,
                                  aff_params: torch.Tensor,
                                  spacings: torch.Tensor,
                                  needs_flip: torch.BoolTensor,
                                  slice_idx: torch.LongTensor,
                                  min_coords: torch.Tensor,
                                  max_coords: torch.Tensor,
                                  latent_params: torch.Tensor,
                                  aff_def_params: torch.Tensor,
                                  point_spread_size: int,
                                  point_spread_std: torch.Tensor,
                                  return_deriv: bool = False,
                                  reduction: bool = True,
                                  **kwargs,
                                  ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        B, N, C = coords_voxel.shape
        coords_voxel_tile = coords_voxel[..., None, :].tile(1, 1, point_spread_size, 1)
        coords_tile_delta = torch.randn(coords_voxel_tile.shape, device=coords_voxel.device) * point_spread_std.to(
            coords_voxel.device)
        coords_voxel_tile = coords_voxel_tile + coords_tile_delta
        coords_voxel_tile_ = coords_voxel_tile.reshape((B, N*point_spread_size, C))
        slice_idx_tile = slice_idx[..., None, :].tile(1, 1, point_spread_size, 1)
        slice_idx_tile_ = slice_idx_tile.reshape((B, N*point_spread_size, slice_idx_tile.shape[-1]))
        values_pred_, seg_pred_, seg_pred_d_, world_coords \
            = self.forward(coords_voxel_tile_, aff_params,
                           spacings, needs_flip,
                           slice_idx_tile_,
                           min_coords, max_coords,
                           latent_params=latent_params,
                           aff_def_params=aff_def_params,
                           return_deriv=return_deriv,
                           **kwargs)
        values_pred = values_pred_.reshape(B, N, point_spread_size)
        seg_pred = seg_pred_.reshape(B, N, point_spread_size, seg_pred_.shape[-1])
        seg_pred_d = None
        if seg_pred_d_ is not None:
            seg_pred_d = seg_pred_d_.reshape(B, N, point_spread_size, *seg_pred_d_.shape[-2:])
        if reduction:
            values_pred = values_pred.mean(2)
            seg_pred = seg_pred.mean(2)
            if seg_pred_d is not None:
                seg_pred_d = seg_pred_d.mean(2)
        return values_pred, seg_pred, seg_pred_d, world_coords

    def get_train_set_learnable_params(self, subj_idx):
        aff_def_params = self.aff_deform_params[subj_idx]
        latent_params = self.subj_latents[subj_idx]
        intens_scale_params = self.intensity_scale_params[subj_idx]
        return latent_params, aff_def_params, intens_scale_params

    def training_step(self, batch):
        opt_inr, opt_latent, opt_aff, opt_intens_scale = self.optimizers()
        opt_inr.zero_grad()
        opt_latent.zero_grad()
        opt_aff.zero_grad()
        opt_intens_scale.zero_grad()

        # Get batch elements
        (coords_voxel, intens_values, seg_values, gt_avail, aff_params, spacings, needs_flip,
         subject_idx, slice_idx, min_coords, max_coords, num_subj_slices,
         coords_surface, coords_surface_slice_idx, coords_surface_class) = batch
        latent_params, aff_def_params, intens_scale_params = self.get_train_set_learnable_params(subject_idx)
        # Forward INR with coordinates
        loss_recon =  0.0
        warm_up = self.current_epoch < self.point_spread_start_epoch
        values_pred, _, sdf_pred_d, _ = self.forward_with_point_spread(coords_voxel, aff_params, spacings, needs_flip,
            slice_idx, min_coords, max_coords, latent_params, aff_def_params,
            self.point_spread_size_before if warm_up else self.point_spread_size_after,
            self.point_spread_std_before if warm_up else self.point_spread_std_after,
            return_deriv=False)
        # Apply learnt intensity scaling to each slice
        values_deform = self.apply_intensity_scaling(intens_values, coords_voxel, slice_idx, intens_scale_params)
        # Recon loss
        loss_recon = self.psnr_loss(values_pred, values_deform)
        # Seg metrics and loss
        loss_sdf, loss_euk, loss_sdf_per_class = 0.0, 0.0, torch.tensor((0.,0.,0.,0.))
        sdf_pred_d_mag = torch.tensor((0.,))
        loss_sdf_sign = 0.0
        loss_thickness = 0.0
        if self.supervise_seg:
            # Eikonal loss
            if sdf_pred_d is None:
                # When using PSF, the deriv is very expensive.
                # After the warmup when PSF starts, we compute it here only for the voxel centers
                _, sdf_pred_voxel, sdf_pred_d, _ = self.forward(coords_voxel, aff_params, spacings, needs_flip, slice_idx,
                                                    min_coords, max_coords, latent_params, aff_def_params,
                                                    return_deriv=True)
            sdf_pred_d_mag = torch.linalg.norm(sdf_pred_d[..., :-1], dim=-1)  # Get magnitude of spatial dimensions
            loss_euk = (sdf_pred_d_mag - 1)
            loss_euk = (loss_euk * loss_euk).mean()
            loss_euk = self.weight_loss_deriv * loss_euk
            # Boundary loss
            _, sdf_pred, _, _ = self.forward(coords_surface, aff_params, spacings, needs_flip, coords_surface_slice_idx,
                                             min_coords, max_coords, latent_params, aff_def_params,
                                             return_deriv=False)
            sdf_pred_ = sdf_pred.reshape(-1, sdf_pred.shape[-1])
            p_idx = torch.arange(sdf_pred_.shape[0], device=sdf_pred.device)
            sdf_pred_ = sdf_pred_[p_idx, coords_surface_class.reshape(-1)]
            loss_sdf_per_class = (sdf_pred_ * sdf_pred_).reshape(sdf_pred.shape[0], 3, coords_surface_class.shape[1] // 3)
            loss_sdf_per_class = loss_sdf_per_class.mean(-1).mean(0)
            loss_sdf = (loss_sdf_per_class * self.class_weight.to(loss_sdf_per_class.device)).mean()
            loss_sdf = self.weight_loss_seg * loss_sdf

            # # The distance from the endo myo contour to the epi contour should be at least 3 voxels
            # thickness = sdf_pred[...,0] - sdf_pred[...,1]
            # thickness_ = thickness[coords_surface_slice_idx.squeeze(-1) > 6]  # This only applies to mid-ventr and apical slices
            # thickness_ = thickness_.clamp(max=2/80)
            # loss_thickness = -thickness_.mean()
            # thickness2 = sdf_pred[...,1] - sdf_pred[...,2]
            # thickness2_ = thickness2[coords_surface_slice_idx.squeeze(-1) > 6]  # This only applies to mid-ventr and apical slices
            # thickness2_ = thickness2_.clamp(max=0)
            # loss_thickness += -thickness2_.mean()

            seg_hierarchy = seg_values[..., 1:].clone()
            seg_hierarchy[..., 1] = seg_hierarchy[..., 0] + seg_hierarchy[..., 1]
            seg_hierarchy[..., 2] = seg_hierarchy[..., 0] + seg_hierarchy[..., 1] + seg_hierarchy[...,2]
            is_foreground = seg_hierarchy * gt_avail[..., None]
            foreground_sdf = sdf_pred_voxel * is_foreground
            foreground_sdf = foreground_sdf.clamp(min=0.0)
            loss_sdf_sign = foreground_sdf.mean()
            background_sdf = sdf_pred_voxel * seg_values[...,:1] * gt_avail[..., None]
            background_sdf = background_sdf.clamp(max=0.0)
            loss_sdf_sign += -background_sdf.mean()

        # Regularization losses
        loss_regul, loss_reg_dict = self.regularization_criterion(subject_idx)

        # Backprop losses and update params
        loss = loss_recon + loss_regul + loss_sdf + loss_euk + loss_thickness + loss_sdf_sign
        self.manual_backward(loss)
        opt_inr.step()
        opt_latent.step()
        opt_aff.step()
        opt_intens_scale.step()
        # Logging
        log_name = "train_metrics"
        self.log_dict({f"{log_name}/{k}": v for k, v in
                       {"loss": loss, "loss_recon": 0.0, "loss_seg": loss_sdf, "loss_seg_euk": loss_euk,
                        # "loss_thickness": loss_thickness,
                        'loss_sdf_sign': loss_sdf_sign,
                        }.items()}, prog_bar=True)
        self.log_dict({f"{log_name}/{k}": v for k, v in
                       {"loss_LV": loss_sdf_per_class[-3], "loss_MYO": loss_sdf_per_class[-2],
                        "loss_RV": loss_sdf_per_class[-1],
                        "seg_deriv_std": sdf_pred_d_mag.std().item(),
                        **loss_reg_dict}.items()}, prog_bar=False)
        self.log_dict({f"{log_name}/{k}": v for k, v in
                       {"lr_inr": opt_inr.param_groups[0]['lr'],
                        "lr_latent": opt_latent.param_groups[0]['lr'],
                        "lr_aff": opt_aff.param_groups[0]['lr'],
                        "lr_intens_scale": opt_intens_scale.param_groups[0]['lr'],
                        }.items()}, prog_bar=False)

    @staticmethod
    def min_max_normalize(X, x_min, x_max, s_min=0., s_max=1.):
        return (X - x_min) / (x_max - x_min) * (s_max - s_min) + s_min

    @staticmethod
    def min_max_unnormalize(X, x_min, x_max, s_min=0., s_max=1.):
        return (X - s_min) / (s_max - s_min) * (x_max - x_min) + x_min

    def on_train_epoch_start(self):
        if (self.current_epoch > 0 and self.current_epoch >= self.logging_start_rate and self.current_epoch % self.logging_rate == 0
                or self.current_epoch in self.addit_log_epochs):
            self.do_logging()

    def on_train_epoch_end(self) -> None:
        for sched in self.lr_schedulers():
            sched.step()


    def do_logging(self, dset=None):
        dset_str = 'train'
        if dset is None:
            dset = eval(f"self.trainer.datamodule.{dset_str}_dset")
        for i in range(0, 2):
            batch = tuple(b[None].cuda() for b in dset[i])
            latent_params, aff_def_params, intens_scale_params = self.get_train_set_learnable_params(batch[5+2])
            self.log_images(i, dset, mode=dset_str,
                            latent_params=latent_params,
                            aff_def_params=aff_def_params,
                            intens_scale_params=intens_scale_params)
            self.log_volume(i, dset, mode=dset_str,
                            latent_params=latent_params,
                            aff_def_params=aff_def_params)
        # dset_str = 'val'
        # dset = eval(f"self.trainer.datamodule.{dset_str}_dset")
        # for i in range(0, 1):
        #     latent_params, aff_def_params, intens_scale_params = self.initialize_inference_params()
        #     opt_latent = torch.optim.Adam([latent_params], lr=self.lr)
        #     opt_affine_def = torch.optim.Adam([aff_def_params], lr=self.lr_aff)
        #     opt_intensity_def = torch.optim.Adam([intens_scale_params], lr=self.lr_def)
        #     optimized_latent, optimized_affine_def, optimized_intensity_def, best_step_num, best_score, *_ \
        #         = self.inference(i, dset,
        #                          latent_params=latent_params,
        #                          aff_def_params=aff_def_params,
        #                          intens_scale_params=intens_scale_params,
        #                          opt_latent=opt_latent,
        #                          opt_affine_def=opt_affine_def,
        #                          opt_intensity_def=opt_intensity_def,
        #                          dset_str = dset_str,
        #                          max_epochs=self.inf_max_epochs,
        #                          point_spread_size_before = self.point_spread_size_before,
        #                          point_spread_size_after = self.point_spread_size_after,
        #                          point_spread_std_before = self.point_spread_std_before.squeeze().tolist(),
        #                          point_spread_std_after = self.point_spread_std_after.squeeze().tolist(),
        #                          point_spread_start_epoch = self.inf_point_spread_start_epoch,
        #                          weight_loss_seg = self.inf_weight_loss_seg,
        #                          )
        #     if not self.logging_wandb_disabled:
        #         wandb.log({f'{dset_str}/inf_best_step_num': best_step_num})
        #         wandb.log({f'{dset_str}/inf_best_score': best_score})
        #     self.log_images(i, dset, mode=dset_str,
        #                     latent_params=optimized_latent,
        #                     aff_def_params=optimized_affine_def,
        #                     intens_scale_params=optimized_intensity_def)
        #     self.log_volume(i, dset, mode=dset_str,
        #                     latent_params=optimized_latent,
        #                     aff_def_params=optimized_affine_def)

    def do_testing(self, dset=None, dset_str = 'test'):
        if dset is None:
            dset = eval(f"self.trainer.datamodule.{dset_str}_dset")
        for i in range(0, len(dset)):
            latent_params, aff_def_params, intens_scale_params = self.initialize_inference_params()
            opt_latent = torch.optim.Adam([latent_params], lr=self.inf_lr_latent)
            opt_affine_def = torch.optim.Adam([aff_def_params], lr=self.inf_lr_aff)
            opt_intensity_def = torch.optim.Adam([intens_scale_params], lr=self.inf_lr_def)
            opt_inr = torch.optim.Adam([*self.canonical_inr.parameters()], lr=self.inf_lr_inr)
            (optimized_latent, optimized_affine_def, optimized_intensity_def, best_step_num, best_score,
             opt_latent, opt_affine_def, opt_intensity_def, opt_inr) \
                = self.inference(i, dset,
                                 latent_params=latent_params,
                                 aff_def_params=aff_def_params,
                                 intens_scale_params=intens_scale_params,
                                 opt_latent=opt_latent,
                                 opt_affine_def=opt_affine_def,
                                 opt_intensity_def=opt_intensity_def,
                                 opt_inr=opt_inr,
                                 dset_str=dset_str,
                                 max_epochs=self.inf_max_epochs,
                                 point_spread_size_before=self.point_spread_size_before,
                                 point_spread_size_after=self.point_spread_size_after,
                                 point_spread_std_before=self.point_spread_std_before.squeeze().tolist(),
                                 point_spread_std_after=self.point_spread_std_after.squeeze().tolist(),
                                 point_spread_start_epoch=self.inf_point_spread_start_epoch,
                                 weight_loss_seg=self.inf_weight_loss_seg,
                                 weight_loss_deriv=self.weight_loss_deriv,
                                 )
            starting_inr_weights = self.canonical_inr.state_dict()
            curr_step = self.inf_max_epochs
            for s in [0,50, 100, 250, 500]:
                dset_str_ft = dset_str + f"_opt{self.inf_max_epochs:04d}_ft{s:04d}"
                if s > 0:
                    (optimized_latent, optimized_affine_def, optimized_intensity_def, best_step_num, best_score,
                     opt_latent, opt_affine_def, opt_intensity_def, opt_inr) \
                        = self.inference(i, dset,
                                         latent_params=optimized_latent,
                                         aff_def_params=optimized_affine_def,
                                         intens_scale_params=optimized_intensity_def,
                                         opt_latent=opt_latent,
                                         opt_affine_def=opt_affine_def,
                                         opt_intensity_def=opt_intensity_def,
                                         opt_inr=opt_inr,
                                         dset_str=dset_str_ft,
                                         max_epochs=self.inf_max_epochs+s,
                                         epochs_elapsed=curr_step,
                                         point_spread_size_before=self.point_spread_size_before,
                                         point_spread_size_after=self.point_spread_size_after,
                                         point_spread_std_before=self.point_spread_std_before.squeeze().tolist(),
                                         point_spread_std_after=self.point_spread_std_after.squeeze().tolist(),
                                         point_spread_start_epoch=self.inf_point_spread_start_epoch,
                                         weight_loss_seg=self.inf_weight_loss_seg,
                                         weight_loss_deriv=self.weight_loss_deriv,
                                         optimize_latent_params=True,
                                         optimize_aff_def_params=True,
                                         optimize_intens_scale_params=True,
                                         optimize_decoder=True,
                    )
                    self.log_volume(i, dset, mode=dset_str_ft,
                                             latent_params=optimized_latent,
                                             aff_def_params=optimized_affine_def)
                curr_step = self.inf_max_epochs + s
                self.log_images(i, dset, mode=dset_str_ft,
                                         latent_params=optimized_latent,
                                         aff_def_params=optimized_affine_def,
                                         intens_scale_params=optimized_intensity_def,)
            self.canonical_inr.load_state_dict(starting_inr_weights)

    def initialize_inference_params(self):
        latent_params = nn.Parameter(torch.randn((1, self.subj_latents.shape[1]), dtype=torch.float32, device="cuda")*1e-3, requires_grad=True)
        aff_def_params = nn.Parameter(torch.zeros((1, self.max_slices, 6), dtype=torch.float32, device="cuda"), requires_grad=True)
        intens_scale_params = nn.Parameter(torch.zeros((1, self.max_slices, 1), dtype=torch.float32, device="cuda"), requires_grad=True)
        return latent_params, aff_def_params, intens_scale_params

    def inference(self, subj_idx: int,
                  dset: CardiacUKBB,
                  latent_params: torch.Tensor,
                  aff_def_params: torch.Tensor,
                  intens_scale_params: torch.Tensor,
                  opt_latent: Optional[torch.optim.Optimizer] = None,
                  opt_affine_def: Optional[torch.optim.Optimizer] = None,
                  opt_intensity_def: Optional[torch.optim.Optimizer] = None,
                  opt_inr: Optional[torch.optim.Optimizer] = None,
                  dset_str: str = 'val',
                  optimize_latent_params: bool = True,
                  optimize_aff_def_params: bool = True,
                  optimize_intens_scale_params: bool = True,
                  optimize_decoder: bool = False,  # By default, we don't want to optimise network
                  max_epochs: int = 2000,
                  epochs_elapsed: int = 0,  # If you want to continue inference
                  point_spread_size_before: int = 1,
                  point_spread_size_after: int = 16,
                  point_spread_std_before: Iterable[float] = (0.01, 0.01, 0.01, 0.01),
                  point_spread_std_after: Iterable[float] = (0.4, 0.4, 0.4, 0.4),
                  point_spread_start_epoch: int = 0,
                  weight_loss_seg: float = 0.0,  # By default, we assume we don't have segmentation at inf time
                  score_window_size=20,
                  log=True,
                  ):
        # Optimizers (won't be used if not supervised)
        opt_latent = torch.optim.Adam([latent_params], lr=self.lr) if opt_latent is None else opt_latent
        opt_affine_def = torch.optim.Adam([aff_def_params], lr=self.lr_aff) if opt_affine_def is None else opt_affine_def
        opt_intensity_def = torch.optim.Adam([intens_scale_params], lr=self.lr_def) if opt_intensity_def is None else opt_intensity_def
        opt_inr = torch.optim.Adam([*self.canonical_inr.parameters()], lr=self.lr) if opt_inr is None else opt_inr

        point_spread_std_before = torch.tensor(point_spread_std_before, dtype=torch.float32, device="cuda"
                                               ).reshape(1, 1, 1, self.coord_size)
        point_spread_std_after = torch.tensor(point_spread_std_after, dtype=torch.float32, device="cuda"
                                              ).reshape(1, 1, 1, self.coord_size)
        supervise_seg = weight_loss_seg > 0
        instance_dset = CardiacUKBBValidation([dset.data_paths[subj_idx]],
                                              dset.max_slices, dset.max_slice_shape,
                                              dset.num_coords, cache_data=True, cache_to_gpu=True)
        subj_id = Path(instance_dset.data_paths[0]).parent.name
        # instance_dloader = DataLoader(instance_dset, shuffle=False, num_workers=0)
        # Save to disk
        if not self.logging_disabled:
            save_path = self.log_path / 'inf_sanity_check' / f"{self.current_epoch:06d}_{subj_idx}_inf.png"
            save_path.parent.parent.mkdir(exist_ok=True)
            save_path.parent.mkdir(exist_ok=True)
            save_image(instance_dset.image_pad[0, 0, :6].float().cpu().reshape(-1, instance_dset.image_pad.shape[-3]), str(save_path))

        best_score = None
        best_inf_step_num = 0
        metrics = defaultdict(list)
        for i in tqdm.tqdm(range(epochs_elapsed, max_epochs), desc=f"Performing inference {'with' if supervise_seg else 'WITHOUT'} "
                                                   f"seg supervision for subject idx {subj_idx} (id {subj_id}) until epoch {max_epochs}"):
            # Get batch elements
            batch = (b[None].cuda() for b in instance_dset.__getitem__(0))
            (coords_voxel, img_values, seg_gt, gt_avail,
             aff_params_padded, spacings_padded, needs_flip_padded,
             subject_idx, slice_idx, min_coords, max_coords, num_subj_slices, *_) = batch

            # Reset gradients (if optimizers exist for those params)
            if optimize_latent_params: opt_latent.zero_grad()
            if optimize_aff_def_params: opt_affine_def.zero_grad()
            if optimize_intens_scale_params: opt_intensity_def.zero_grad()
            if optimize_decoder: opt_inr.zero_grad()
            # Make predictions for this batch
            pred_values, pred_sdf, pred_sdf_d, _ = self.forward_with_point_spread(
                   coords_voxel, aff_params_padded,
                   spacings_padded, needs_flip_padded,
                   slice_idx,
                   min_coords, max_coords,
                   latent_params=latent_params[subject_idx],
                   aff_def_params=aff_def_params[subject_idx],
                   point_spread_size=point_spread_size_before if i < point_spread_start_epoch else point_spread_size_after,
                   point_spread_std=point_spread_std_before if i < point_spread_start_epoch else point_spread_std_after,
                   return_deriv=False,
                   inference=True,
                   reduction=True)
            values_deform = self.apply_intensity_scaling(img_values, coords_voxel, slice_idx, intens_scale_params)
            # Recon loss
            loss_recon = self.psnr_loss(pred_values, values_deform)
            # Segmentation metrics

            pred_seg = sdf_hierarchy_to_seg(pred_sdf).float()
            # pred_seg_1hot = keep_last_true(pred_seg)
            # pred_seg_argmax = pred_seg_1hot.long().argmax(-1)
            pred_seg, segs = pred_seg * gt_avail[..., None], seg_gt * gt_avail[..., None]
            pred_seg[~gt_avail] = torch.tensor((1., 0., 0., 0.), device=pred_seg.device)
            seg_gt[~gt_avail] = torch.tensor((1., 0., 0., 0.), device=pred_seg.device)
            loss_seg_per_class = self.seg_loss(pred_seg.moveaxis(-1, 1), seg_gt.moveaxis(-1, 1)).mean(-1).mean(0)
            dice_per_class = 1 - loss_seg_per_class # TODO: doesn't properly handle gt_not_available
            # Regularization losses
            reg_aff_loss, reg_aff_dict = self.loss_reg_aff_params(latent_params[subject_idx], self.weight_reg_aff, num_subj_slices=num_subj_slices)
            reg_lat_loss, reg_lat_dict = self.loss_reg_latent_params(aff_def_params[subject_idx], self.weight_reg_lat)
            reg_int_scale_loss, reg_int_scale_dict = self.loss_reg_int_scale_params(intens_scale_params[subject_idx], self.weight_reg_intens_scale, num_subj_slices=num_subj_slices)
            loss_reg = reg_aff_loss + reg_lat_loss + reg_int_scale_loss
            reg_dict = {f"loss_reg": loss_reg,
                        **reg_aff_dict, **reg_lat_dict,
                        **reg_int_scale_dict,}
            loss = loss_recon + loss_reg
            # Backprop only image-based losses and regularization losses (we assume we don't have seg GT)
            loss.backward()
            # Update parameters (if optimizers exist for those params)
            if optimize_latent_params: opt_latent.step()
            if optimize_aff_def_params: opt_affine_def.step()
            if optimize_intens_scale_params: opt_intensity_def.step()
            if optimize_decoder: opt_inr.step()
            # Metrics for logging
            metrics['step'].append(i)
            metrics['loss_recon'].append(loss_recon.item())
            metrics['PSNR'].append(-1. * loss_recon.item())
            [metrics[k].append(v.item()) for k, v in reg_dict.items()]
            metrics['dice_BG'].append(dice_per_class[0].item())
            metrics['dice_FG'].append(dice_per_class[1:].mean().item())
            metrics['dice_LV'].append(dice_per_class[1].item())
            metrics['dice_MYO'].append(dice_per_class[2].item())
            metrics['dice_RV'].append(dice_per_class[3].item())
            curr_score = np.mean(metrics['PSNR'][-score_window_size:])
            metrics['best_step'].append(curr_score)
            if best_score is None or curr_score > best_score:
                best_score = curr_score
                best_inf_step_num = i
        # Logging  -------------------------------------------------------------------------
        print('Best inf step num:', best_inf_step_num)
        if not self.logging_disabled and log:
            log_name = f"{dset_str}_inf_metrics"
            log_dir = self.log_path / log_name / subj_id
            log_dir.mkdir(exist_ok=True, parents=True)
            # if 'step' not in self.inference_metrics:
            #     self.inference_metrics['step'] = {}
            # steps = [sum(metrics['step'][i:i+score_window_size]) / len(metrics['step'][i:i+score_window_size]) for i in range(0, len(metrics['step']), score_window_size)]
            # if subj_id not in self.inference_metrics['step']:
            #     self.inference_metrics['step'][subj_id] = pd.DataFrame(steps, columns=['step'])
            # else:
            #     self.inference_metrics['step'][subj_id]['step']._append(pd.Series(steps))
            for k, v in metrics.items():
                v = [sum(v[i:i+score_window_size]) / len(v[i:i+score_window_size]) for i in range(0, len(v), score_window_size)]
                v = pd.DataFrame(v, columns=[f"Epoch {self.current_epoch:06d}"])
                if k not in self.inference_metrics:
                    self.inference_metrics[k] = {}
                if subj_id not in self.inference_metrics[k]:
                    # If no previous dataframe, create dataframe with column for this epoch
                    self.inference_metrics[k][subj_id] = v
                else:
                    if f"Epoch {self.current_epoch:06d}" not in self.inference_metrics[k][subj_id]:
                        # Add column for this epoch
                        self.inference_metrics[k][subj_id][f"Epoch {self.current_epoch:06d}"] = v
                    else:
                        self.inference_metrics[k][subj_id] = pd.concat((self.inference_metrics[k][subj_id], v))

                self.inference_metrics[k][subj_id].to_csv(str(log_dir / f'{k}.csv'))
                save_path = str(log_dir / f"inf_lines_{k}.png")
                data_frame_to_line_plot(self.inference_metrics[k][subj_id], self.inference_metrics['step'][subj_id],
                                        k, str(subj_idx), save_path=save_path)
                # if not self.logging_wandb_disabled:
                #     wandb.log({f'{log_name}/subj_{str(subj_id)}_inf_metric_{k}_plot': wandb.Image(save_path)})
                #     plot = wandb.plot.line_series(xs=list(self.inference_metrics['step'][subj_id]),
                #                                   ys=[list(self.inference_metrics[k][subj_id][i]) for i in list(self.inference_metrics[k][subj_id].columns)],
                #                                   keys=list(self.inference_metrics[k][subj_id].columns),
                #                                   title=f"{k} metric over inference optimization",
                #                                   xname="Optimization steps")
                #     wandb.log({f'{log_name}/subj_{str(subj_id)}_inf_metric_{k}': plot})
        return (latent_params, aff_def_params, intens_scale_params,
                best_inf_step_num, best_score, opt_latent, opt_affine_def, opt_intensity_def, opt_inr)

    @torch.no_grad()
    def log_images(self,
                   subj_idx: int,
                   dataset: CardiacUKBB,
                   latent_params: torch.Tensor,
                   aff_def_params: torch.Tensor,
                   intens_scale_params: torch.Tensor,
                   video_duration: float = 4,
                   with_psf: bool = False,
                   mode="train"):
        subj_path = dataset.data_paths[subj_idx]
        subj_id = Path(subj_path).parent.name
        gt_ims = [[] for _ in range(20)]
        gt_segs = [[] for _ in range(20)]
        videos = [[] for _ in range(20)]
        preds = [[] for _ in range(20)]
        segs = [[] for _ in range(20)]
        psnrs = [[] for _ in range(20)]
        ssims = [[] for _ in range(20)]
        dices = [[] for _ in range(20)]
        for t in tqdm.tqdm(range(0, 50), desc=f"Logging {mode} slices for subject {subj_idx} (UKBB id: {subj_id})"):
            images, seg_argmax, _, full_indices, coord_min, coord_max, \
                aff_params_padded, spacings_padded, flippings_padded, num_subj_slices, contours \
                = dataset.load_subject_data(subj_idx, t)
            images, num_subj_slices = images.cuda()[None], num_subj_slices.cuda()[None]
            B, S, H, W = images.shape
            seg_argmax = seg_argmax.cuda()[None]
            full_indices, coord_max, coord_min = full_indices.cuda()[None], coord_max.cuda()[None], coord_min.cuda()[None]
            aff_params_padded, spacings_padded, flippings_padded = aff_params_padded.cuda()[None], spacings_padded.cuda()[None], flippings_padded.cuda()[None]
            full_indices = make_coordinate_tensor(images.shape[1:], device=images.device)
            slice_idx = full_indices[..., :1]
            voxel_indices = torch.cat((full_indices[..., 1:3], torch.zeros_like(full_indices[..., :1]),
                                       torch.full_like(full_indices[..., :1], t)), dim=-1).float()
            for s in range(num_subj_slices.item()):
                gt_ims[s].append((images[0,s] * 255).cpu().numpy().astype(np.uint8))
                gt_segs[s].append((seg_argmax[0,s]).cpu().numpy().astype(np.uint8))
                voxel_indices_ = voxel_indices[s, ..., :].reshape(1, -1, 4)
                slice_idx_ = slice_idx[s, ..., :].reshape(1, -1, 1)
                with torch.enable_grad():
                    pred_vals_, pred_sdf_, pred_sdf_d_, _ = self.forward_with_point_spread(
                        voxel_indices_, aff_params_padded,
                        spacings_padded, flippings_padded,
                        slice_idx_,
                        coord_min, coord_max,
                        latent_params=latent_params,
                        aff_def_params=aff_def_params,
                        point_spread_size=self.point_spread_size_after if with_psf else self.point_spread_size_before,
                        point_spread_std=self.point_spread_std_after if with_psf else self.point_spread_std_before,
                        return_deriv=True, inference=True, reduction=True)
                pred_sdf_, pred_vals_, pred_sdf_d_ \
                    = pred_sdf_.detach(), pred_vals_.detach(), pred_sdf_d_.detach()

                image_scaled = self.apply_intensity_scaling(images[0, s].reshape(1, -1), images[0, s].reshape(1, -1),
                                                      slice_idx_, intens_scale_params).reshape(H, W)

                pred_img = pred_vals_.reshape(H, W)
                psnr_metric = kornia.metrics.psnr(pred_img, image_scaled, max_val=1.0)
                pred_img_diff = (pred_img - images[0, s]).abs()
                psnrs[s].append(psnr_metric.mean().detach().cpu().item())
                ssim_metric = kornia.metrics.ssim(pred_img[None, None], image_scaled[None,None], window_size=11, max_val=1.0)
                ssims[s].append(ssim_metric.mean().detach().cpu().item())
                pred = pred_img.clip(0.0, 1.0)
                pred = (pred * 255).cpu().numpy().astype(np.uint8)
                preds[s].append(pred)
                pred_sdf = pred_sdf_.reshape(H, W, pred_sdf_.shape[-1])
                pred_seg = sdf_hierarchy_to_seg(pred_sdf)
                pred_seg_1hot = keep_last_true(pred_seg)
                pred_seg_argmax = pred_seg_1hot.long().argmax(-1)
                pred_seg_diff = (pred_seg_argmax != seg_argmax[0,s]).to(torch.float32)
                segs[s].append(pred_seg_argmax.cpu().numpy().astype(np.uint8))
                seg_gt = to_1hot(seg_argmax[0,s].reshape(-1), pred_seg_1hot.shape[-1]).reshape(pred_seg_1hot.shape)
                dice = 1 - self.seg_loss(pred_seg_1hot[None].moveaxis(-1,1), seg_gt[None].moveaxis(-1,1)).mean(0).squeeze()
                if s >= 3 or (t in {0,16,32} and seg_gt[...,1].any().item()):
                    dices[s].append(dice.detach().cpu())

                # Image
                img = torch.stack([torch.cat([images[0,s,...], pred_img], 0)]*3, 0)
                # Segmentation
                contour_im = torch.stack([images[0,s,...]]*3, -1)
                for contour_class, c_per_slice in reversed(contours.items()):
                    if c_per_slice[s]:
                        contour_im = burn_contours(contour_im, c_per_slice[s], COLORS[contour_class])
                contour_im = contour_im.moveaxis(-1, 0)
                seg_frames = torch.stack([images[0,s,...]]*3, 0)
                red = torch.tensor(RED, device=seg_frames.device).reshape((3, 1, 1)).tile((1, *seg_frames.shape[-2:]))
                green = torch.tensor(GREEN, device=seg_frames.device).reshape((3, 1, 1)).tile((1, *seg_frames.shape[-2:]))
                blue = torch.tensor(BLUE, device=seg_frames.device).reshape((3, 1, 1)).tile((1, *seg_frames.shape[-2:]))
                seg_mask = torch.stack([pred_seg_argmax == 1] * 3, dim=0)
                seg_frames = torch.where(seg_mask, red, seg_frames)
                seg_mask = torch.stack([pred_seg_argmax == 2] * 3, dim=0)
                seg_frames = torch.where(seg_mask, green, seg_frames)
                seg_mask = torch.stack([pred_seg_argmax == 3] * 3, dim=0)
                seg_frames = torch.where(seg_mask, blue, seg_frames)
                seg_frames = torch.cat([contour_im, seg_frames], 1)
                # Derivatives
                pred_img_dt = pred_sdf[...,-1].float()*4 + 0.5
                pred_img_dt = torch.stack([pred_img_dt]*3, dim=0)
                # pred_img_dt = torch.tanh(pred_img_dt)/2+0.5
                pred_img2_dt = (pred_sdf<=0).float().moveaxis(-1, 0)
                # pred_img2_dt = torch.tanh(pred_img2_dt)/2+0.5
                # pred_img_dt = torch.stack([pred_img_dt.abs()]*3, dim=0)
                # pred_img2_dt = torch.stack([pred_img2_dt.abs()]*3, dim=0)
                # Learnable intensity scale
                # pred_img_int_scale_undo_ = self.apply_intensity_scaling(pred_vals_, voxel_indices_, slice_idx_,
                #                                                 intens_scale_params=intens_scale_params, inverse=True)
                # pred_img_int_scale_undo = pred_img_int_scale_undo_.reshape(pred_img.shape)
                # pred_img_int_scale_undo = torch.stack([pred_img_int_scale_undo.abs()] * 3, dim=0)
                # # Learnable affine params
                # with torch.enable_grad():
                #     pred_vals_no_def_, *_ = self.forward(
                #         voxel_indices_, aff_params_padded,
                #         spacings_padded, flippings_padded,
                #         slice_idx_,
                #         coord_min, coord_max,
                #         latent_params=latent_params,
                #         aff_def_params=None,
                #         return_deriv=False,
                #         return_def=True,
                #         inference=True)
                # pred_img_no_def = pred_vals_no_def_.detach().reshape(H, W)
                # pred_img_no_def_diff = (pred_img_no_def - pred_img).abs()
                # pred_img_no_def_diff = torch.stack([pred_img_no_def_diff.abs()] * 3, dim=0)


                pred_img_diff = torch.stack([pred_img_diff]*3, dim=0).to(torch.float32)
                pred_seg_diff = torch.stack([pred_seg_diff]*3, dim=0).to(torch.float32)

                col_img = torch.cat([img, seg_frames], dim=1)
                col_d = torch.cat([pred_img_dt, pred_img2_dt, pred_img_diff, pred_seg_diff], dim=1)
                frame = torch.cat([col_img, col_d], dim=2)
                frame = frame.clip(0.0, 1.0)
                frame = (frame * 255).cpu().numpy().astype(np.uint8)
                videos[s].append(frame)
        videos = [np.stack(v, 0) for v in videos if v]

        if self.logging_disabled:  # Logging  -------------------------------------------------------------------------
            return
        save_dir = self.log_path / f"{mode}_slices" / str(subj_id) / f"epoch_{self.current_epoch:06d}"
        save_dir.mkdir(exist_ok=True, parents=True)
        # Save metrics to file
        psnrs = [np.mean(i) for i in psnrs if i]
        ssims = [np.mean(i) for i in ssims if i]
        dices = [torch.stack(d, 0).mean(0) if d else torch.tensor([-1.]*4) for i, d in enumerate(dices) if i <= 2 or d]
        metrics = {"PSNR": psnrs, "SSIM": ssims,
                   "Dice_BG": [i[0].item() for i in dices],
                   "Dice_LV": [i[1].item() for i in dices],
                   "Dice_MYO": [i[2].item() for i in dices],
                   "Dice_RV": [i[3].item() for i in dices],
                   }
        pd.DataFrame(metrics).to_csv(str(save_dir / f"inplane_metrics.csv"), index=False)
        # Save to WANDB
        if not self.logging_wandb_disabled:
            psnr_strings = [f"PSNR:{i:.1f}" for i in psnrs]
            dices_strings = [f"Dice:" + f"{d[1].item():.2f}, " + f"{d[2].item():.2f}, " + f"{d[3].item():.2f}"
                             if i >= 3 or d[1] >= 0. else "Dice: -, -, -" for i, d in enumerate(dices)]
            wandb_videos = [wandb.Video(v, fps=max(1, int(50 / video_duration)), format='gif',
                                        caption=f"Slice:{i}, Epoch:{self.current_epoch}, {psnr_strings[i]}  {dices_strings[i]}") for i, v in enumerate(videos)]
            wandb.log({f"{mode}_videos/subj_{subj_id}": wandb_videos})
        # Save series to file as mp4
        save_dir_vid = save_dir / "videos"
        save_dir_vid.parent.mkdir(exist_ok=True)
        save_dir_vid.mkdir(exist_ok=True)
        for i, v in enumerate(videos):
            video_array_to_file(v, save_dir_vid / f"slice_{i:02d}.mp4")

        # Save series to file as nifti
        images, _, _, full_indices, coord_max, coord_min, \
            aff_params_padded, spacings_padded, flippings_padded, _, _ = dataset.load_subject_data(subj_idx, 0)
        save_dir_nif = save_dir / "niftis"
        save_dir_nif.mkdir(exist_ok=True, parents=True)
        save_dir_nif_og = save_dir_nif / 'original'
        save_dir_nif_og.mkdir(exist_ok=True)
        preds = [np.stack(v, 0) for v in preds if v]
        segs = [np.stack(v, 0) for v in segs if v]
        gt_ims = [np.stack(v, 0) for v in gt_ims if v]
        gt_segs = [np.stack(v, 0) for v in gt_segs if v]
        for i, (gt_im, gt_seg, v, s) in enumerate(zip(gt_ims, gt_segs, preds, segs)):
            aff_params_def = aff_params_padded[i][None] + aff_def_params[0, i].cpu()
            aff_def = params_to_mat(aff_params_def, spacings_padded[i][None], flippings_padded[i][None])
            aff_def = aff_def[0].cpu().numpy()
            v = v.astype(np.int32)
            v = np.moveaxis(v[..., None], 0, -1)
            array_to_nifti(str(save_dir_nif / f"slice_{i:02d}.nii.gz"), v, aff_def)
            s = s.astype(np.int32)
            s = np.moveaxis(s[..., None], 0, -1)
            array_to_nifti(str(save_dir_nif / f"seg_slice_{i:02d}.nii.gz"), s, aff_def)
            aff_params = aff_params_padded[i][None]
            aff = params_to_mat(aff_params, spacings_padded[i][None], flippings_padded[i][None])
            aff = aff[0].cpu().numpy()
            gt_im = gt_im.astype(np.int32)
            gt_im = np.moveaxis(gt_im[..., None], 0, -1)
            array_to_nifti(str(save_dir_nif_og / f"slice_{i:02d}.nii.gz"), gt_im, aff)
            gt_seg = gt_seg.astype(np.int32)
            gt_seg = np.moveaxis(gt_seg[..., None], 0, -1)
            array_to_nifti(str(save_dir_nif_og / f"seg_slice_{i:02d}.nii.gz"), gt_seg, aff)

    @torch.no_grad()
    def log_volume(self,
                   subj_idx: int,
                   dataset: CardiacUKBB,
                   latent_params: torch.Tensor,
                   aff_def_params: torch.Tensor,
                   video_duration: float = 4,
                   tds: int = 2,
                   mode="train",
                   res=(200, 200, 200),
                   margin=(0.0, 0.0, 0.0)):
        res_tensor = torch.tensor(res, dtype=torch.float32)
        subj_path = dataset.data_paths[subj_idx]
        subj_id = Path(subj_path).parent.name
        coords = torch.stack(torch.meshgrid(*[torch.linspace(self.norm_min+margin[i], self.norm_max-margin[i], r) for i, r in enumerate(res)]), dim=-1)
        ims = []
        segs = []
        for t in tqdm.tqdm(range(0, 50, tds), desc=f"Logging {mode} volume for subject {subj_idx} (UKBB id: {subj_id})"):
            t_norm = t / 50
            c = torch.concatenate((coords, torch.full((*res, 1), t_norm)), dim=-1).cuda()
            z_im_slices = []
            z_seg_slices = []
            for z in range(res[-1]):
                c_z = c[:,:,z:z+1]
                pred_vals_, pred_sdf_ = self.forward_inr(c_z.reshape(1, -1, 4), latent_params, inference=True)
                pred_sdf = pred_sdf_.reshape((*c_z.shape[:-1], pred_sdf_.shape[-1]))
                pred_seg = sdf_hierarchy_to_seg(pred_sdf)
                pred_seg = keep_last_true(pred_seg).long().argmax(-1)
                pred_vals_ = pred_vals_.clip(0.0, 1.0)
                pred_val = pred_vals_.reshape(c_z.shape[:-1])
                pred_seg = pred_seg.cpu().numpy()
                pred_seg = pred_seg.astype(np.uint8)
                pred_val = pred_val.cpu().numpy()
                pred_val = (pred_val*255).astype(np.uint8)
                z_im_slices.append(pred_val)
                z_seg_slices.append(pred_seg)
            pred_im_t = np.concatenate(z_im_slices, -1)
            pred_seg_t = np.concatenate(z_seg_slices, -1)
            ims.append(pred_im_t)
            segs.append(pred_seg_t)
        ims = np.stack(ims, -1)
        segs = np.stack(segs, -1)

        if self.logging_disabled:  # Logging  -------------------------------------------------------------------------
            return
        gt_images, _, gt_segs, _, full_indices, coord_max, coord_min, \
            aff_params_padded, spacings_padded, flippings_padded, _ = dataset.load_subject_data(subj_idx, 0)
        save_dir = self.log_path / f"{mode}_volumes" / f"epoch_{self.current_epoch}" / str(subj_id)
        save_dir.parent.parent.mkdir(exist_ok=True)
        save_dir.parent.mkdir(exist_ok=True)
        save_dir.mkdir(exist_ok=True)
        # Save volumes to disk as nifti
        # Voxel (0,0,0) -> (domain_min, domain_min, domain_min)
        spacing = (self.norm_max - self.norm_min) / (res_tensor - 1)  # voxel spacing along each axis
        aff = np.array([
            [spacing[0], 0, 0, self.norm_min],
            [0, spacing[1], 0, self.norm_min],
            [0, 0, spacing[2], self.norm_min],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        save_dir_pred = save_dir / 'pred'
        save_dir_pred.mkdir(exist_ok=True)
        array_to_nifti(str(save_dir_pred / f"full.nii.gz"), ims.astype(np.int32), aff)
        array_to_nifti(str(save_dir_pred / f"full_seg.nii.gz"), segs.astype(np.int32), aff)

        save_dir_gt = save_dir / "gt"
        save_dir_gt.mkdir(exist_ok=True)
        for i in range(gt_images.shape[0]):
            gt_im_frames, gt_seg_frames = [], []
            for t in range(0, 50, tds):
                gt_images, gt_segs, _, full_indices, coord_max, coord_min, \
                    aff_params_padded, spacings_padded, flippings_padded, _, _ \
                    = dataset.load_subject_data(subj_idx, t)
                gt_images = (gt_images * 255).to(torch.uint8)
                gt_segs = gt_segs.to(torch.uint8)
                gt_im_frames.append(gt_images)
                gt_seg_frames.append(gt_segs)
            gt_im_frames = torch.stack(gt_im_frames, -1).unsqueeze(-2)
            gt_seg_frames = torch.stack(gt_seg_frames, -1).unsqueeze(-2)
            param = aff_params_padded[i] + aff_def_params[0, i].cpu()
            aff = params_to_mat(param[None], torch.ones_like(spacings_padded[i][None]), flippings_padded[i][None])
            aff = aff[0].cpu().numpy()
            array_to_nifti(str(save_dir_gt / f"slice{i}.nii.gz"), gt_im_frames[i].numpy().astype(np.int32), aff)
            array_to_nifti(str(save_dir_gt / f"slice{i}_seg.nii.gz"), gt_seg_frames[i].numpy().astype(np.int32), aff)

        slice_indices = range(20, res[2] - 20, res[2] // 10)
        ims_rgb = np.stack([ims]*3, axis=0)
        segs_rgb = (np.stack([segs]*3, axis=0) / 4 * 255).astype(np.uint8)
        content = [np.concatenate((ims_rgb[..., i, :], segs_rgb[..., i, :]), 2) for i in slice_indices]
        content = [np.moveaxis(c, -1, 0) for c in content]
        # Save series to file as mp4
        save_dir_gt = save_dir / "mp4"
        save_dir_gt.mkdir(exist_ok=True)
        for v, i in zip(content, slice_indices):
            video_array_to_file(v, save_dir / f"slice_{i:03d}-{res[2]}.mp4", video_duration=video_duration)
        # Log to WANDB
        if not self.logging_wandb_disabled:
            # Log video slices
            videos = [wandb.Video(c, fps=max(1, int(50 / video_duration)), format='gif',
                                  caption=f"Epoch:{self.current_epoch}") for c in content]
            wandb.log({f"{mode}_volumes/subj_{subj_id}": videos})
            if not self.supervise_seg:
                return
            # Log meshes
            meshes = process_segmentation_with_marching_cubes(segs[..., 0], level=0.5, step_size=1)
            if not meshes:
                # Dummy mesh
                a = np.zeros_like(segs[..., 0])
                a[5:15,5:15,5:15] = 1
                a[20:25,20:25,20:25] = 2
                meshes = process_segmentation_with_marching_cubes(a, level=0.5, step_size=1)
            import tempfile
            for i in range(5):
                save_dir = self.log_path / f"temp_mesh_files/{subj_id}"
                save_dir.parent.mkdir(exist_ok=True)
                save_dir.mkdir(exist_ok=True)
                try:
                    with tempfile.NamedTemporaryFile(dir=str(save_dir.absolute()), suffix='.html', delete=False) as f:
                        plot = create_meshplot_visualization(meshes, f.name)
                        temp_file_path = f.name
                        with open(f.name, 'r') as html_file:
                            wandb.log({f"{mode}_mesh/subj_{subj_id}": wandb.Html(html_file.read())})
                    break
                except Exception as e:
                    print("Error while logging mesh:")
                    traceback.print_exc()
                shutil.rmtree(save_dir.parent, ignore_errors=True)
