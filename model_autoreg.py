import copy
import math
import shutil
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Dict, Optional, List, Union, Any

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
from torch.utils.data import DataLoader

from data_utils import array_to_nifti
from dataset import CardiacUKBBValidation, CardiacUKBB
from networks import MLP
from pos_encoding import PosEncodingNeRFAnnealed, PosEncodinFourier
from utils import params_to_mat, make_coordinate_tensor, to_1hot, create_meshplot_visualization, \
    process_segmentation_with_marching_cubes, data_frame_to_line_plot, video_array_to_file, draw_3d_vectors_on_image


class INR_AutoReg(pl.LightningModule):
    def __init__(self, coord_size: int, num_subjects: int, max_slices: int, log_path: Optional[Path] = None, **kwargs):
        super(INR_AutoReg, self).__init__()
        self.automatic_optimization = False
        self.logging_disabled = kwargs['logging_disabled']
        self.logging_wandb_disabled = kwargs['logging_wandb_disabled']
        self.logging_rate = kwargs['logging_rate']
        self.addit_log_epochs = kwargs['addit_log_epochs']
        self.inference_metrics = {}
        self.log_path = log_path

        self.coord_size = coord_size
        self.intensity_size = 1
        self.num_subjects = num_subjects
        self.max_slices = max_slices
        self.norm_min, self.norm_max = 0.0, 1.0
        self.point_spread_size = kwargs['point_spread_size']
        self.point_spread_std = torch.tensor(kwargs['point_spread_std'], dtype=torch.float32).reshape(1, 1, 1, -1)

        self.latent_size = kwargs["latent_size"]
        self.subj_latents = nn.Parameter(torch.randn((self.num_subjects, self.latent_size),
                                                     dtype=torch.float32, device="cuda") * 1e-2, requires_grad=True)

        self.aff_deform_params = nn.Parameter(torch.zeros((self.num_subjects, self.max_slices, 6),
                                                          dtype=torch.float32, device="cuda"), requires_grad=True)
        self.int_scale_range = kwargs["int_scale_range"]
        self.intensity_scale_params = nn.Parameter(torch.randn((self.num_subjects, self.max_slices, 1),
                                                               dtype=torch.float32, device="cuda") * 1e-3, requires_grad=True)
        self.pos_enc = PosEncodingNeRFAnnealed(in_dim=5, # 3D + Cyclical time
                                               num_frequencies=kwargs['pe_num_frequencies'],
                                               anneal_max_iter=kwargs['pe_anneal_max_iter'],
                                               anneal_start_prop=kwargs['pe_anneal_start_prop'])
        self.canonical_inr = MLP(self.pos_enc.out_dim + self.latent_size,
                                 num_hidden_layers=kwargs['num_hidden_layers'],
                                 hidden_size=kwargs['hidden_size'],
                                 out_size=1+4+3)
        self.target_network = copy.deepcopy(self.canonical_inr)

        self.class_weight = torch.tensor([i / sum(kwargs['weight_seg_class']) for i in kwargs['weight_seg_class']])
        self.seg_loss = DiceLoss(softmax=False, reduction="none")
        self.psnr_loss = kornia.losses.PSNRLoss(max_val=1.0)

        self.weight_reg_inr = kwargs["weight_reg_inr"]
        self.weight_reg_aff = kwargs["weight_reg_aff"]
        self.weight_reg_lat = kwargs["weight_reg_lat"]
        self.weight_intensity_scale = kwargs["weight_reg_int_scale"]
        self.weight_loss_deriv = kwargs["weight_loss_deriv"]
        self.supervise_deriv = self.weight_loss_deriv != 0
        self.weight_loss_seg = kwargs["weight_loss_seg"]
        self.weight_loss_regist_recon = kwargs['weight_loss_regist_recon']
        self.weight_loss_regist_seg = kwargs['weight_loss_regist_seg']
        self.weight_loss_regist_reg = kwargs['weight_loss_regist_reg']
        self.regist_task_start_epoch = kwargs['regist_task_start_epoch']
        self.supervise_regist = self.weight_loss_regist_seg != 0 or self.weight_loss_regist_recon != 0
        self.lr = kwargs["learning_rate"]
        self.lr_aff = kwargs["learning_rate_aff"]
        self.lr_def = kwargs["learning_rate_def"]
        self.inf_lr = kwargs["inf_learning_rate"]
        self.inf_lr_aff = kwargs["inf_learning_rate_aff"]
        self.inf_lr_def = kwargs["inf_learning_rate_def"]
        self.inf_max_epochs = kwargs["inf_max_epochs"]

    def configure_optimizers(self):
        opt_inr = torch.optim.Adam([*self.canonical_inr.parameters(), self.subj_latents], lr=self.lr)
        opt_deform = torch.optim.Adam([self.aff_deform_params], lr=self.lr_aff)
        opt_intensity = torch.optim.Adam([self.intensity_scale_params], lr=self.lr_def)
        return opt_inr, opt_deform, opt_intensity

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
        loss_reg_aff = nn.functional.mse_loss(params_, torch.zeros_like(params_)) * weight if weight else 0.0
        return loss_reg_aff, {dict_name: loss_reg_aff}

    @staticmethod
    def loss_reg_latent_params(params: torch.Tensor, weight: float, dict_name='loss_reg_lat'):
        loss_reg_lat = nn.functional.mse_loss(params, torch.zeros_like(params)) * weight if weight else 0.0
        return loss_reg_lat, {dict_name: loss_reg_lat}

    @staticmethod
    def loss_reg_int_scale_params(params: torch.Tensor, weight: float, num_subj_slices: Optional[torch.Tensor] = None, dict_name='loss_reg_int_scale'):
        if num_subj_slices is not None:
            non_pad_slices = torch.arange(0, params.shape[1], device=params.device).tile((params.shape[0],1)) < num_subj_slices[:, None]
            params_ = params[non_pad_slices]
        else:
            params_ = params
        loss_reg_int = nn.functional.mse_loss(params_, torch.zeros_like(params_)) * weight if weight else 0.0
        return loss_reg_int, {dict_name: loss_reg_int}

    def regularization_criterion(self, subj_idx: torch.Tensor) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        reg_inr_loss, reg_inr_dict = self.loss_reg_inr_params(self.canonical_inr.parameters(), self.weight_reg_inr)
        reg_aff_loss, reg_aff_dict = self.loss_reg_aff_params(self.aff_deform_params[subj_idx], self.weight_reg_aff)
        reg_lat_loss, reg_lat_dict = self.loss_reg_latent_params(self.subj_latents[subj_idx], self.weight_reg_lat)
        reg_int_scale_loss, reg_int_scale_dict = self.loss_reg_int_scale_params(self.intensity_scale_params[subj_idx], self.weight_intensity_scale)
        reg_loss = reg_inr_loss + reg_aff_loss + reg_lat_loss + reg_int_scale_loss
        reg_dict = {f"loss_reg": reg_loss,
                    **reg_inr_dict, **reg_lat_dict,
                    **reg_aff_dict, **reg_int_scale_dict,
                    }
        return reg_loss, reg_dict

    def forward_registration(self,
                             latent_params: torch.Tensor,
                             deform_pred: torch.Tensor,
                             world_coords: torch.Tensor,
                             target_t: float = 0.0
                             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_coords = world_coords.clone()
        target_coords[..., -1] = target_t
        target_coords[..., :3] += deform_pred
        target_values_pred, target_seg_pred, _ = self.forward_inr(target_coords, latent_params, use_target_net=True)
        return target_values_pred, target_seg_pred, target_coords

    def compute_registration_losses(self,
                                    latent_params: torch.Tensor,
                                    source_inten_pred: torch.Tensor,
                                    source_seg_pred: torch.Tensor,
                                    source_deform_pred: torch.Tensor,
                                    source_world_coords: torch.Tensor,
                                    target_t: float = 0.0):
        # source_world_coords = source_world_coords.detach()
        source_inten_pred = source_inten_pred.detach()
        source_seg_pred = source_seg_pred.detach()
        target_inten_pred, target_seg_pred, target_coords \
             = self.forward_registration(latent_params, source_deform_pred, source_world_coords, target_t)
        target_inten_pred = target_inten_pred.detach()
        target_seg_pred = target_seg_pred.detach()
        loss_recon = self.psnr_loss(target_inten_pred, source_inten_pred) * self.weight_loss_regist_recon
        target_seg_pred, source_seg_pred = torch.softmax(target_seg_pred, -1), torch.softmax(source_seg_pred, -1)
        loss_seg = self.seg_loss(target_seg_pred.moveaxis(-1,1), source_seg_pred.moveaxis(-1,1)).mean() * self.weight_loss_regist_seg
        jac_det_loss = self.compute_jacobian_determinant_loss(source_world_coords, source_deform_pred) * self.weight_loss_regist_reg
        return loss_recon, loss_seg, jac_det_loss

    def compute_jacobian_determinant_loss(self, coords, deforms, add_identity=True):
        # Compute determinants and take norm
        dx = torch.autograd.grad(deforms[..., 0:1], coords, grad_outputs=torch.ones_like(deforms[..., :1]),
                            create_graph=True, retain_graph=True)[0][...,:3]
        dy = torch.autograd.grad(deforms[..., 1:2], coords, grad_outputs=torch.ones_like(deforms[..., :1]),
                            create_graph=True, retain_graph=True)[0][...,:3]
        dz = torch.autograd.grad(deforms[..., 2:3], coords, grad_outputs=torch.ones_like(deforms[..., :1]),
                            create_graph=True, retain_graph=True)[0][...,:3]
        jac_mat = torch.stack((dx, dy, dz), dim=-2)
        if add_identity:
            jac_mat[..., 0, 0] = 1.
            jac_mat[..., 1, 1] = 1.
            jac_mat[..., 2, 2] = 1.
        loss = torch.det(jac_mat) - 1
        loss_ = loss.reshape(-1)
        # We want to enforce det to be 1.0. Taking the mean wouldn't work since det-1 could be <0.
        # Hence we take the norm to minimize the distance from 0 (|det-1| -> 0)
        loss = torch.linalg.norm(loss_, ord=1) / loss_.shape[0]
        return loss

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
                return_jac_det: bool = True,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        # if return_deriv:
        #     latent_params.requires_grad = True
        #     aff_def_params.requires_grad = True
        world_coords = self.forward_coord_model(coords_voxel, aff_params, spacings, needs_flip,
                                                slice_idx, min_coords, max_coords, aff_def_params)
        values_pred, seg_pred, regist_pred = self.forward_inr(world_coords, latent_params)
        values_pred_d = None
        if return_deriv:
            values_pred_d = torch.autograd.grad(values_pred, world_coords, grad_outputs=torch.ones_like(values_pred),
                                                create_graph=True, retain_graph=True)[0]
        return values_pred, seg_pred, regist_pred, values_pred_d, world_coords

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
                    use_target_net: bool = False) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Make time dim cyclical
        coords = torch.cat((coords[..., :3],
                            torch.cos(coords[...,-1:] * torch.pi),
                            torch.sin(coords[...,-1:] * torch.pi)), dim=-1)
        coords_enc = self.pos_enc(coords, self.global_step)
        subject_latent = subject_latent[:, None].tile((1, coords.shape[1], 1))
        x = torch.cat((coords_enc, subject_latent), dim=-1)
        # Forward INR to obtain predicted volume values
        x_ = x.reshape((-1, x.shape[-1]))
        if use_target_net:
            values_pred_ = self.target_network(x_)
        else:
            values_pred_ = self.canonical_inr(x_)
        values_pred = values_pred_[:, 0].reshape((coords.shape[0], coords.shape[1]))
        values_pred = torch.sigmoid(values_pred)
        seg_pred = values_pred_[:, 1:5].reshape((coords.shape[0], coords.shape[1], 4))
        regist_pred = values_pred_[:, 5:].reshape((coords.shape[0], coords.shape[1], 3))
        return values_pred, seg_pred, regist_pred

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
                                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        B, N, C = coords_voxel.shape
        coords_voxel_tile = coords_voxel[..., None, :].tile(1, 1, point_spread_size, 1)
        coords_tile_delta = torch.randn(coords_voxel_tile.shape, device=coords_voxel.device) * point_spread_std.to(
            coords_voxel.device)
        coords_voxel_tile = coords_voxel_tile + coords_tile_delta
        coords_voxel_tile_ = coords_voxel_tile.reshape((B, N*point_spread_size, C))
        slice_idx_tile = slice_idx[..., None, :].tile(1, 1, point_spread_size, 1)
        slice_idx_tile_ = slice_idx_tile.reshape((B, N*point_spread_size, slice_idx_tile.shape[-1]))
        values_pred_, seg_pred_, regist_pred_, values_pred_d_, world_coords \
            = self.forward(coords_voxel_tile_, aff_params,
                           spacings, needs_flip,
                           slice_idx_tile_,
                           min_coords, max_coords,
                           latent_params=latent_params,
                           aff_def_params=aff_def_params,
                           return_deriv=return_deriv)
        values_pred = values_pred_.reshape(B, N, point_spread_size).mean(2)
        seg_pred = seg_pred_.reshape(B, N, point_spread_size, -1).mean(2)
        regist_pred = regist_pred_.reshape(B, N, point_spread_size, -1).mean(2)
        values_pred_d = None
        if values_pred_d_ is not None:
            values_pred_d = values_pred_d_.reshape(B, N, point_spread_size, -1).mean(2)
        return values_pred, seg_pred, regist_pred, values_pred_d, world_coords

    def get_sample_elements_from_batch(self, batch): return batch

    def get_latents(self, batch: Tuple[Any, ...], *args) -> torch.Tensor:
        """ This function is meant to be overwritten by sub-classes that may for ex. predict the latent """
        return self.subj_latents[batch[-5]]  # batch[-5] is subject index

    def get_train_set_learnable_params(self, batch):
        aff_def_params = self.aff_deform_params[batch[8]]
        latent_params = self.get_latents(batch, aff_def_params)
        intens_scale_params = self.intensity_scale_params[batch[8]]
        return latent_params, aff_def_params, intens_scale_params

    def training_step(self, batch):
        opt_inr, opt_deform, opt_inten = self.optimizers()
        # Get conditioning/deformation params
        (coords_voxel, values, values_dt, segs, gt_avail,
         aff_params, spacings, needs_flip, subject_idx, slice_idx,
         min_coords, max_coords, num_subj_slices) = self.get_sample_elements_from_batch(batch)
        latent_params, aff_def_params, intens_scale_params = self.get_train_set_learnable_params(batch)
        # Forward INR with coordinates
        values_pred, seg_pred, deform_pred, values_pred_d, world_coords = self.forward_with_point_spread(
            coords_voxel, aff_params,
            spacings, needs_flip,
            slice_idx,
            min_coords, max_coords,
            latent_params, aff_def_params,
            self.point_spread_size,
            self.point_spread_std,
            return_deriv=self.supervise_deriv)
        # Apply learnt intensity scaling to each slice
        values_deform = self.apply_intensity_scaling(values, coords_voxel, slice_idx, intens_scale_params)
        # Recon loss
        loss_recon = self.psnr_loss(values_pred, values_deform)
        # Seg metrics and loss
        seg_pred, segs = torch.softmax(seg_pred, -1)*gt_avail[...,None], segs*gt_avail[...,None]
        loss_seg_per_class = self.seg_loss(seg_pred.moveaxis(-1,1), segs.moveaxis(-1,1)).mean(-1).mean(0)
        dice_per_class = 1 - loss_seg_per_class
        loss_seg = (loss_seg_per_class * self.class_weight.to(loss_seg_per_class.device)).mean() * self.weight_loss_seg
        # Registration metrics_and loss
        loss_regist, loss_regist_recon, loss_regist_seg, loss_regist_reg = 0.0, 0.0, 0.0, 0.0
        if self.supervise_regist and self.current_epoch >= self.regist_task_start_epoch:
            loss_regist_recon, loss_regist_seg, loss_regist_reg = self.compute_registration_losses(
                latent_params, values_pred, seg_pred, deform_pred, world_coords)
            loss_regist = loss_regist_recon + loss_regist_seg + loss_regist_reg
        # Recon derivative loss (if user decided to supervise it)
        loss_dt = 0.0
        if self.supervise_deriv:
            loss_dt = self.psnr_loss(values_pred_d[...,-1], values_dt*50) * self.weight_loss_deriv
        # Regularization losses
        loss_regul, loss_reg_dict = self.regularization_criterion(subject_idx)
        # Backprop losses and update params
        loss = loss_recon + loss_regul + loss_seg + loss_regist + loss_dt
        opt_inr.zero_grad()
        opt_deform.zero_grad()
        opt_inten.zero_grad()
        self.manual_backward(loss)
        opt_inr.step()
        opt_deform.step()
        opt_inten.step()
        # Copy weights to target net
        self.target_network.load_state_dict(self.canonical_inr.state_dict())
        # Logging
        log_name = "train_metrics"
        self.log_dict({f"{log_name}/{k}": v for k, v in
                       {"loss": loss, "loss_recon": loss_recon, "loss_seg": loss_seg, "loss_regist": loss_regist
                        }.items()}, prog_bar=True)
        self.log_dict({f"{log_name}/{k}": v for k, v in
                       {"loss_dt": loss_dt,
                        "dice_BG": dice_per_class[0], "dice_FG": dice_per_class[1:].mean(),
                        "dice_LV": dice_per_class[1], "dice_MYO": dice_per_class[2],
                        "dice_RV": dice_per_class[3],
                        "registration_recon": loss_regist_recon,
                        "registration_seg": loss_regist_seg,
                        "registration_reg": loss_regist_reg,
                        **loss_reg_dict}.items()}, prog_bar=False)

    @staticmethod
    def min_max_normalize(X, x_min, x_max, s_min=0., s_max=1.):
        return (X - x_min) / (x_max - x_min) * (s_max - s_min) + s_min

    @staticmethod
    def min_max_unnormalize(X, x_min, x_max, s_min=0., s_max=1.):
        return (X - s_min) / (s_max - s_min) * (x_max - x_min) + x_min

    def on_train_epoch_start(self):
        if (self.current_epoch % self.logging_rate == 0 and self.current_epoch > 0) or self.current_epoch in self.addit_log_epochs:
            dset_str = 'train'
            dset = eval(f"self.trainer.datamodule.{dset_str}_dset")
            # for i in range(0, min(len(dset), 8)):
            for i in range(0, 8):
                batch = tuple(b[None].cuda() for b in dset[i])
                latent_params, aff_def_params, intens_scale_params = self.get_train_set_learnable_params(batch)
                self.log_images(i, dset, mode=dset_str,
                                latent_params=latent_params,
                                aff_def_params=aff_def_params,
                                intens_scale_params=intens_scale_params)
                self.log_volume(i, dset, mode=dset_str,
                                latent_params=latent_params,
                                aff_def_params=aff_def_params)
        if (self.current_epoch % self.logging_rate == 0 and self.current_epoch > 0) or self.current_epoch in self.addit_log_epochs:
            dset_str = 'val'
            dset = eval(f"self.trainer.datamodule.{dset_str}_dset")
            for i in range(0, len(dset)):
                latent_params, aff_def_params, intens_scale_params = self.initialize_inference_params()
                optimized_latent, optimized_affine_def, optimized_intensity_def, best_step_num, best_score \
                    = self.inference(i, dset,
                                     latent_params=latent_params,
                                     aff_def_params=aff_def_params,
                                     intens_scale_params=intens_scale_params)
                if not self.logging_wandb_disabled:
                    wandb.log({f'{dset_str}/inf_best_step_num': best_step_num})
                    wandb.log({f'{dset_str}/inf_best_score': best_score})
                self.log_images(i, dset, mode=dset_str,
                                latent_params=optimized_latent,
                                aff_def_params=optimized_affine_def,
                                intens_scale_params=optimized_intensity_def)
                self.log_volume(i, dset, mode=dset_str,
                                latent_params=optimized_latent,
                                aff_def_params=optimized_affine_def)

    def get_inf_dset(self, subj_idx: int, dset: CardiacUKBB):
        return CardiacUKBBValidation([dset.data_paths[subj_idx]],
                                     dset.max_slices, dset.max_slice_shape,
                                     dset.num_coords, to_gpu=True)

    def initialize_inference_params(self):
        latent_params = torch.randn((1, self.latent_size), dtype=torch.float32, device="cuda") * 1e-2
        aff_def_params = torch.zeros((1, self.max_slices, 6), dtype=torch.float32, device="cuda")
        intens_scale_params = torch.zeros((1, self.max_slices, 1), dtype=torch.float32, device="cuda")
        return latent_params, aff_def_params, intens_scale_params

    def inference(self, subj_idx: int,
                  dset: CardiacUKBB,
                  latent_params: torch.Tensor,
                  aff_def_params: torch.Tensor,
                  intens_scale_params: torch.Tensor,
                  supervize_seg: bool = False,  # By default, we assume we don't have segmentation at inf time
                  optimize_latent_params: bool = True,
                  optimize_aff_def_params: bool = True,
                  optimize_intens_scale_params: bool = True,
                  dset_str: str = 'val'):
        instance_dset = CardiacUKBBValidation([dset.data_paths[subj_idx]],
                                              dset.max_slices, dset.max_slice_shape,
                                              dset.num_coords, to_gpu=True)
        subj_id = Path(instance_dset.data_paths[0]).parent.name
        # instance_dloader = DataLoader(instance_dset, shuffle=False, num_workers=0)
        # Save to disk
        if not self.logging_disabled:
            save_path = self.log_path / 'inf_sanity_check' / f"{self.current_epoch:06d}_{subj_idx}_inf.png"
            save_path.parent.parent.mkdir(exist_ok=True)
            save_path.parent.mkdir(exist_ok=True)
            save_image(instance_dset.image_pad[0, :6, ..., 0].reshape(-1, instance_dset.image_pad.shape[-3]), str(save_path))
        # Parameters to optimize
        inf_subj_latents = nn.Parameter(latent_params, requires_grad=optimize_latent_params)
        inf_aff_def_params = nn.Parameter(aff_def_params, requires_grad=optimize_aff_def_params)
        inf_intens_scale_params = nn.Parameter(intens_scale_params, requires_grad=optimize_intens_scale_params)
        best_inf_subj_latents = torch.clone(inf_subj_latents)
        best_inf_aff_def_params = torch.clone(inf_aff_def_params)
        best_inf_intens_scale_params = torch.clone(inf_intens_scale_params)
        best_score = None
        best_inf_step_num = 0
        best_score_window_size = 20

        # Optimizers (None if we do not want to optimize)
        opt_latent = None if not optimize_latent_params else torch.optim.Adam([inf_subj_latents], lr=self.inf_lr)
        opt_affine_def = None if not optimize_aff_def_params else torch.optim.Adam([inf_aff_def_params], lr=self.inf_lr_aff)
        opt_intensity_def = None if not optimize_intens_scale_params else torch.optim.Adam([inf_intens_scale_params], lr=self.inf_lr_def)
        metrics = defaultdict(list)
        for i in tqdm.tqdm(range(self.inf_max_epochs), desc=f"Performing inference for subject idx {subj_idx} (id {subj_id})"):
            # Get batch elements
            batch = (b[None].cuda() for b in instance_dset.__getitem__(0))
            (coords_voxel, img_values, img_dt_values, seg_gt, gt_avail,
             aff_params_padded, spacings_padded, needs_flip_padded,
             subject_idx, slice_idx, min_coords, max_coords, num_subj_slices) = batch

            # Reset gradients (if optimizers exist for those params)
            if optimize_latent_params: opt_latent.zero_grad()
            if optimize_aff_def_params: opt_affine_def.zero_grad()
            if optimize_intens_scale_params: opt_intensity_def.zero_grad()
            # Make predictions for this batch
            values_pred, seg_pred, _, values_pred_d = self.forward_with_point_spread(
                                                                   coords_voxel, aff_params_padded,
                                                                   spacings_padded, needs_flip_padded,
                                                                   slice_idx,
                                                                   min_coords, max_coords,
                                                                   latent_params=inf_subj_latents[subject_idx],
                                                                   aff_def_params=inf_aff_def_params[subject_idx],
                                                                   point_spread_size=self.point_spread_size,
                                                                   point_spread_std=self.point_spread_std,
                                                                   return_deriv=self.supervise_deriv)
            values_deform = self.apply_intensity_scaling(img_values, coords_voxel, slice_idx, intens_scale_params)
            # Recon loss
            loss_recon = self.psnr_loss(values_pred, values_deform)
            # Segmentation metrics
            seg_pred, segs = torch.softmax(seg_pred, -1) * gt_avail[..., None], seg_gt * gt_avail[..., None]
            loss_seg_per_class = self.seg_loss(seg_pred.moveaxis(-1, 1), segs.moveaxis(-1, 1)).mean(-1).mean(0)
            dice_per_class = 1 - loss_seg_per_class
            # Seg loss is only supervized if user explicitly wants to
            loss_seg = torch.tensor((0.0,), device=loss_recon.device)
            if supervize_seg:
                loss_seg = (loss_seg_per_class * self.class_weight.to(loss_seg_per_class.device)).mean() * self.weight_loss_seg
            # Regularization losses
            reg_aff_loss, reg_aff_dict = self.loss_reg_aff_params(inf_aff_def_params[subject_idx], self.weight_reg_aff, num_subj_slices=num_subj_slices)
            reg_lat_loss, reg_lat_dict = self.loss_reg_latent_params(inf_subj_latents[subject_idx], self.weight_reg_lat)
            reg_int_scale_loss, reg_int_scale_dict = self.loss_reg_int_scale_params(inf_intens_scale_params[subject_idx], self.weight_intensity_scale, num_subj_slices=num_subj_slices)
            loss_reg = reg_aff_loss + reg_lat_loss + reg_int_scale_loss
            reg_dict = {f"loss_reg": loss_reg,
                        **reg_aff_dict, **reg_lat_dict,
                        **reg_int_scale_dict,}
            # Recon derivative loss
            loss_dt = torch.tensor((0.0,), device=loss_recon.device)
            if self.supervise_deriv:
                loss_dt = self.psnr_loss(values_pred_d[...,-1:], img_dt_values*50) * self.weight_loss_deriv
            loss = loss_recon + loss_seg + loss_reg + loss_dt
            # Backprop only image-based losses and regularization losses (we assume we don't have seg GT)
            loss.backward()
            # Update parameters (if optimizers exist for those params)
            if optimize_latent_params: opt_latent.step()
            if optimize_aff_def_params: opt_affine_def.step()
            if optimize_intens_scale_params: opt_intensity_def.step()
            # Metrics for logging
            metrics['step'].append(i)
            metrics['loss_recon'].append(loss_recon.item())
            metrics['PSRN'].append(-1. * loss_recon.item())
            metrics['loss_dt'].append(loss_dt.item())
            [metrics[k].append(v.item()) for k, v in reg_dict.items()]
            metrics['loss_seg'].append(loss_seg.item())
            metrics['dice_BG'].append(dice_per_class[0].item())
            metrics['dice_FG'].append(dice_per_class[1:].mean().item())
            metrics['dice_LV'].append(dice_per_class[1].item())
            metrics['dice_MYO'].append(dice_per_class[2].item())
            metrics['dice_RV'].append(dice_per_class[3].item())
            curr_score = np.mean(metrics['dice_FG'][-best_score_window_size:])
            metrics['best_step'].append(curr_score)
            if best_score is None or curr_score > best_score:
                best_score = curr_score
                best_inf_subj_latents = torch.clone(inf_subj_latents)
                best_inf_aff_def_params = torch.clone(inf_aff_def_params)
                best_inf_intens_scale_params = torch.clone(inf_intens_scale_params)
                best_inf_step_num = i
        # Logging  -------------------------------------------------------------------------
        if self.logging_disabled:
            return inf_subj_latents, inf_aff_def_params, inf_intens_scale_params
        log_name = f"{dset_str}_inf_metrics"
        log_dir = self.log_path / log_name
        log_dir.mkdir(exist_ok=True)
        window_size = 20
        if 'step' not in self.inference_metrics:
            steps = [sum(metrics['step'][i:i+window_size]) / len(metrics['step'][i:i+window_size])
                     for i in range(0, len(metrics['step']), window_size)]
            self.inference_metrics['step'] = pd.DataFrame(steps, columns=['step'])
        for k, v in metrics.items():
            if k == 'step':
                continue
            v = [sum(v[i:i+window_size]) / len(v[i:i+window_size]) for i in range(0, len(v), window_size)]
            if k not in self.inference_metrics:
                # If no previous dataframe, create dataframe with column for this epoch
                self.inference_metrics[k] = pd.DataFrame(v, columns=[f"{k}_{self.current_epoch:06d}"])
            else:
                # Add column for this epoch
                self.inference_metrics[k][f"{k}_{self.current_epoch:06d}"] = v

            self.inference_metrics[k].to_csv(str(self.log_path / f'{k}_{str(subj_idx)}.csv'))
            save_path = str(log_dir / f"inf_lines_{str(subj_idx)}_{k}.png")
            data_frame_to_line_plot(self.inference_metrics[k], self.inference_metrics['step'], k, str(subj_idx),
                                    save_path=save_path)
            if not self.logging_wandb_disabled:
                wandb.log({f'{log_name}/subj_{str(subj_id)}_inf_metric_{k}_plot': wandb.Image(save_path)})
                plot = wandb.plot.line_series(xs=list(self.inference_metrics['step']),
                                              ys=[list(self.inference_metrics[k][i]) for i in list(self.inference_metrics[k].columns)],
                                              keys=list(self.inference_metrics[k].columns),
                                              title=f"{k} metric over inference optimization",
                                              xname="Optimization steps")
                wandb.log({f'{log_name}/subj_{str(subj_id)}_inf_metric_{k}': plot})

        return best_inf_subj_latents, best_inf_aff_def_params, best_inf_intens_scale_params, best_inf_step_num, best_score

    @torch.no_grad()
    def log_images(self,
                   subj_idx: int,
                   dataset: CardiacUKBB,
                   latent_params: torch.Tensor,
                   aff_def_params: torch.Tensor,
                   intens_scale_params: torch.Tensor,
                   video_duration: float = 4,
                   mode="train"):
        subj_path = dataset.data_paths[subj_idx]
        subj_id = Path(subj_path).parent.name
        gt_ims = [[] for _ in range(20)]
        videos = [[] for _ in range(20)]
        preds = [[] for _ in range(20)]
        segs = [[] for _ in range(20)]
        psnrs = [[] for _ in range(20)]
        ssims = [[] for _ in range(20)]
        dices = [[] for _ in range(20)]
        for t in tqdm.tqdm(range(0, 50), desc=f"Logging {mode} slices for subject {subj_idx} (UKBB id: {subj_id})"):
            images, images_dt, seg_argmax, _, full_indices, coord_min, coord_max, \
                aff_params_padded, spacings_padded, flippings_padded, num_subj_slices \
                = self.get_sample_elements_from_batch(dataset.load_subject_data(subj_idx, t))
            images, num_subj_slices = images.cuda()[None], num_subj_slices.cuda()[None]
            B, S, H, W = images.shape
            images_dt, seg_argmax = images_dt[...,-1].cuda()[None], seg_argmax.cuda()[None]
            full_indices, coord_max, coord_min = full_indices.cuda()[None], coord_max.cuda()[None], coord_min.cuda()[None]
            aff_params_padded, spacings_padded, flippings_padded = aff_params_padded.cuda()[None], spacings_padded.cuda()[None], flippings_padded.cuda()[None]
            full_indices = make_coordinate_tensor(images.shape[1:], device=images.device)
            slice_idx = full_indices[..., :1]
            voxel_indices = torch.cat((full_indices[..., 1:3], torch.zeros_like(full_indices[..., :1]),
                                       torch.full_like(full_indices[..., :1], t)), dim=-1).float()
            for s in range(num_subj_slices.item()):
                gt_ims[s].append((images[0,s] * 255).cpu().numpy().astype(np.uint8))
                voxel_indices_ = voxel_indices[s, ..., :].reshape(1, -1, 4)
                slice_idx_ = slice_idx[s, ..., :].reshape(1, -1, 1)
                with torch.enable_grad():
                    pred_vals_, pred_seg_, pred_deform_, pred_vals_d_, world_coords_ = self.forward(
                        voxel_indices_, aff_params_padded,
                        spacings_padded, flippings_padded,
                        slice_idx_,
                        coord_min, coord_max,
                        latent_params=latent_params,
                        aff_def_params=aff_def_params,
                        return_deriv=True)
                pred_seg_, pred_vals_, pred_vals_d_ = pred_seg_.detach(), pred_vals_.detach(), pred_vals_d_.detach()
                pred_img = pred_vals_.reshape(H, W)

                psnr_metric = kornia.metrics.psnr(pred_img, images[0,s,...], max_val=1.0)
                psnrs[s].append(psnr_metric.mean().detach().cpu().item())
                ssim_metric = kornia.metrics.ssim(pred_img[None, None], images[:,s,None,...], window_size=11, max_val=1.0)
                ssims[s].append(ssim_metric.mean().detach().cpu().item())
                pred = pred_img.clip(0.0, 1.0)
                pred = (pred * 255).cpu().numpy().astype(np.uint8)
                preds[s].append(pred)
                pred_seg = pred_seg_.reshape(H, W, pred_seg_.shape[-1])
                pred_seg_argmax = pred_seg.argmax(-1)
                segs[s].append(pred_seg_argmax.cpu().numpy().astype(np.uint8))
                seg_gt = to_1hot(seg_argmax[0,s].reshape(-1), pred_seg.shape[-1]).reshape(pred_seg.shape)
                pred_seg_1hot = to_1hot(pred_seg_argmax.reshape(-1), pred_seg.shape[-1]).reshape(pred_seg.shape)
                dice = 1 - self.seg_loss(pred_seg_1hot[None].moveaxis(-1,1), seg_gt[None].moveaxis(-1,1)).mean(0).squeeze()
                dices[s].append(dice.detach().cpu())

                # Image
                img = torch.stack([torch.cat([images[0,s,...], pred_img], 0)]*3, 0)
                # Segmentation
                segs_argmax = torch.cat([seg_argmax[0,s], pred_seg_argmax], 0)
                seg_frames = torch.stack([torch.cat([images[0,s,...], images[0,s,...]], 0)]*3, 0)
                red = torch.tensor((1.0, 0.0, 0.0), device=seg_frames.device).reshape((3, 1, 1)).tile((1, *seg_frames.shape[-2:]))
                green = torch.tensor((0.0, 1.0, 0.0), device=seg_frames.device).reshape((3, 1, 1)).tile((1, *seg_frames.shape[-2:]))
                blue = torch.tensor((0.0, 1.0, 1.0), device=seg_frames.device).reshape((3, 1, 1)).tile((1, *seg_frames.shape[-2:]))
                seg_mask = torch.stack([segs_argmax == 1] * 3, dim=0)
                seg_frames = torch.where(seg_mask, red, seg_frames)
                seg_mask = torch.stack([segs_argmax == 2] * 3, dim=0)
                seg_frames = torch.where(seg_mask, green, seg_frames)
                seg_mask = torch.stack([segs_argmax == 3] * 3, dim=0)
                seg_frames = torch.where(seg_mask, blue, seg_frames)
                # Derivatives
                pred_img_dt = pred_vals_d_.reshape(H, W, pred_vals_d_.shape[-1])[...,-1]
                images_dt_norm = images_dt * (coord_max[:, None, -1:] - coord_min[:, None, -1:])
                images_dt_norm_abs = torch.stack([images_dt_norm[0, s].abs()] * 3, dim=0)
                pred_img_dt = torch.stack([pred_img_dt.abs()] * 3, dim=0)
                # Learnable intensity scale
                pred_img_int_scale_undo_ = self.apply_intensity_scaling(pred_vals_, voxel_indices_, slice_idx_,
                                                                intens_scale_params=intens_scale_params, inverse=True)
                pred_img_int_scale_undo = pred_img_int_scale_undo_.reshape(pred_img.shape)
                pred_img_int_scale_undo = torch.stack([pred_img_int_scale_undo.abs()] * 3, dim=0)
                # Learnable affine params
                with torch.enable_grad():
                    pred_vals_no_def_, *_ = self.forward(
                        voxel_indices_, aff_params_padded,
                        spacings_padded, flippings_padded,
                        slice_idx_,
                        coord_min, coord_max,
                        latent_params=latent_params,
                        aff_def_params=None,
                        return_deriv=False)
                pred_img_no_def = pred_vals_no_def_.detach().reshape(H, W)
                pred_img_no_def_diff = (pred_img_no_def - pred_img).abs()
                pred_img_no_def_diff = torch.stack([pred_img_no_def_diff.abs()] * 3, dim=0)

                # Registration
                target_values_pred_, target_seg_pred_, target_coords_ = self.forward_registration(
                    latent_params, pred_deform_, world_coords_)
                # deform = voxel_indices_.clone().reshape(*pred_img.shape, -1)
                # deform[pred_img.shape[0]//2:, pred_img.shape[1]//2:, 2] += 5
                # # deform += 1
                # deform_ = deform.reshape(world_coords_.shape)
                # deform_w_ = self.forward_coord_model(deform_, aff_params_padded,
                #                                      spacings_padded, flippings_padded,
                #                                      slice_idx_,
                #                                      coord_min, coord_max,
                #                                      aff_def_params=aff_def_params,
                #                                      inverse=False)
                # from sa_la_interp import interpolate_sa_segs_to_la
                # a = interpolate_sa_segs_to_la(
                #     [seg_argmax[0, s, ..., None, None].cpu().numpy() for s in range(num_subj_slices.item())],
                #     [images[0, s, ..., None, None].cpu().numpy() for s in range(num_subj_slices.item())],
                #     [params_to_mat(aff_params_padded[:, s], spacings_padded[:, s],
                #                    flippings_padded[:, s]).cpu().numpy()[0] for s in range(num_subj_slices.item())],
                #     None, None,
                #     self.min_max_unnormalize(world_coords_, coord_min[:, None], coord_max[:, None], self.norm_min, self.norm_max
                #                              ).reshape(*pred_img.shape, 4)[..., None, :3].cpu().numpy(),
                #     )
                # pred_deform_w_ = deform_w_ - world_coords_
                pred_deform_w_ = torch.zeros_like(world_coords_)
                pred_deform_w_[...,1] = 0.3

                # pred_deform[pred_img.shape[0]//2:, pred_img.shape[1]//2:, -2] = 0.1
                # pred_deform_ = pred_deform.reshape(pred_deform_.shape)
                target_coords_ = world_coords_.clone()
                target_coords_[...,:3] += pred_deform_w_[...,:3]
                deformed_coords_image_ = self.forward_coord_model(target_coords_, aff_params_padded,
                                                                  spacings_padded, flippings_padded,
                                                                  slice_idx_,
                                                                  coord_min, coord_max,
                                                                  aff_def_params=aff_def_params,
                                                                  inverse=True)
                deform_pred_image_ = deformed_coords_image_ - voxel_indices_
                deform_pred_image = deform_pred_image_.reshape(*pred_img.shape, 4)
                quiver_image = draw_3d_vectors_on_image(images[0, s].cpu().numpy(),
                                                        deform_pred_image[..., :3].cpu().numpy())
                quiver_image = torch.tensor(quiver_image, dtype=torch.float32, device=pred_img.device) / 255.
                target_values_pred = target_values_pred_.reshape(pred_img.shape)
                regist_values_diff = (target_values_pred - pred_img).abs()
                regist_values_diff = torch.stack([regist_values_diff]*3, dim=0)
                target_seg_pred = target_seg_pred_.reshape(*pred_img.shape, -1)
                regist_seg_diff = (target_seg_pred.argmax(-1) != pred_seg.argmax(-1))
                regist_seg_diff = torch.stack([regist_seg_diff]*3, dim=0)
                target_values_pred = torch.stack([target_values_pred]*3, dim=0).to(torch.float32)

                col_img = torch.cat([img, seg_frames], dim=1)
                col_d = torch.cat([images_dt_norm_abs, pred_img_dt, pred_img_int_scale_undo, pred_img_no_def_diff], dim=1)
                col_regist = torch.cat([target_values_pred, quiver_image, regist_values_diff, regist_seg_diff], dim=1)
                frame = torch.cat([col_img, col_d, col_regist], dim=2)
                frame = frame.clip(0.0, 1.0)
                frame = (frame * 255).cpu().numpy().astype(np.uint8)
                a = np.moveaxis(frame, 0, -1)
                videos[s].append(frame)
        videos = [np.stack(v, 0) for v in videos if v]

        if self.logging_disabled:  # Logging  -------------------------------------------------------------------------
            return
        save_dir = self.log_path / f"{mode}_slices" / str(subj_id) / f"epoch_{self.current_epoch:06d}"
        save_dir.mkdir(exist_ok=True, parents=True)
        # Save metrics to file
        psnrs = [np.mean(i) for i in psnrs if i]
        ssims = [np.mean(i) for i in ssims if i]
        dices = [torch.stack(d, 0).mean(0) for i, d in enumerate(dices) if d]
        metrics = {"PSNR": psnrs, "SSIM": ssims,
                   "Dice_BG": [i[0].item() for i in dices],
                   "Dice_LV": [i[1].item() for i in dices],
                   "Dice_MYO": [i[2].item() for i in dices],
                   "Dice_RV": [i[3].item() for i in dices],
                   }
        pd.DataFrame(metrics).to_csv(str(self.log_path / f"epoch_{self.current_epoch:06d}_subj_{subj_idx}.csv"), index=False)
        # Save to WANDB
        if not self.logging_wandb_disabled:
            psnr_strings = [f"PSNR:{i:.1f}" for i in psnrs]
            dices_strings = [f"Dice:" + f"{d[1].item():.2f}, " + f"{d[2].item():.2f}, " + f"{d[3].item():.2f}"
                             if i >= 3 else "Dice: -, -, -" for i, d in enumerate(dices)]
            wandb_videos = [wandb.Video(v, fps=max(1, int(50 / video_duration)), format='gif',
                                        caption=f"Slice:{i}, {psnr_strings[i]}  {dices_strings[i]}") for i, v in enumerate(videos)]
            wandb.log({f"{mode}_videos/subj_{subj_id}": wandb_videos})
        # Save series to file as mp4
        save_dir_vid = save_dir / "videos"
        save_dir_vid.parent.mkdir(exist_ok=True)
        save_dir_vid.mkdir(exist_ok=True)
        for i, v in enumerate(videos):
            video_array_to_file(v, save_dir_vid / f"slice_{i:02d}.mp4")

        # Save series to file as nifti
        images, _, _, _, full_indices, coord_max, coord_min, \
            aff_params_padded, spacings_padded, flippings_padded, _ \
            = self.get_sample_elements_from_batch(dataset.load_subject_data(subj_idx, 0))
        save_dir_nif = save_dir / "niftis"
        save_dir_nif.mkdir(exist_ok=True, parents=True)
        save_dir_nif_og = save_dir_nif / 'original'
        save_dir_nif_og.mkdir(exist_ok=True)
        preds = [np.stack(v, 0) for v in preds if v]
        segs = [np.stack(v, 0) for v in segs if v]
        gt_ims = [np.stack(v, 0) for v in gt_ims if v]
        for i, (gt, v, s) in enumerate(zip(gt_ims, preds, segs)):
            aff_params = aff_params_padded[i][None] + aff_def_params[0, i].cpu()
            aff = params_to_mat(aff_params, spacings_padded[i][None], flippings_padded[i][None])
            aff = aff[0].cpu().numpy()
            v = v.astype(np.uint8)
            v = np.moveaxis(v[..., None], 0, -1)
            array_to_nifti(str(save_dir_nif / f"slice_{i:02d}.nii.gz"), v, aff)
            s = s.astype(np.uint8)
            s = np.moveaxis(s[..., None], 0, -1)
            array_to_nifti(str(save_dir_nif / f"seg_slice_{i:02d}.nii.gz"), s, aff)
            gt = gt.astype(np.uint8)
            gt = np.moveaxis(gt[..., None], 0, -1)
            array_to_nifti(str(save_dir_nif_og / f"slice_{i:02d}.nii.gz"), gt, aff)

    @torch.no_grad()
    def log_volume(self,
                   subj_idx: int,
                   dataset: CardiacUKBB,
                   latent_params: torch.Tensor,
                   aff_def_params: torch.Tensor,
                   video_duration: float = 4,
                   mode="train",
                   res=(200, 200, 200)):
        subj_path = dataset.data_paths[subj_idx]
        subj_id = Path(subj_path).parent.name
        coords = torch.stack(torch.meshgrid(*[torch.linspace(self.norm_min, self.norm_max, i) for i in res]), dim=-1)
        ims = []
        segs = []
        defs = []
        for t in tqdm.tqdm(range(0, 50, 5), desc=f"Logging {mode} volume for subject {subj_idx} (UKBB id: {subj_id})"):
            t_norm = t / 50
            c = torch.concatenate((coords, torch.full((*res, 1), t_norm)), dim=-1).cuda()
            z_im_slices = []
            z_seg_slices = []
            z_def_slices = []
            for z in range(res[-1]):
                c_z = c[:,:,z:z+1]
                pred_vals_, pred_seg_, pred_def_ = self.forward_inr(c_z.reshape(1, -1, 4), latent_params)
                pred_seg = pred_seg_.argmax(-1).reshape(c_z.shape[:-1])
                pred_vals_ = pred_vals_.clip(0.0, 1.0)
                pred_val = pred_vals_.reshape(c_z.shape[:-1])
                pred_def = pred_def_.reshape(*c_z.shape[:-1], 3)
                pred_def = pred_def * torch.tensor(res, device=pred_def.device)
                quiver_image = draw_3d_vectors_on_image(pred_val.cpu().numpy(),
                                                        pred_def[...,0, :3].cpu().numpy())
                quiver_image = quiver_image[...,None]
                pred_seg = pred_seg.cpu().numpy()
                pred_seg = pred_seg.astype(np.uint8)
                pred_val = pred_val.cpu().numpy()
                pred_val = (pred_val*255).astype(np.uint8)
                z_im_slices.append(pred_val)
                z_seg_slices.append(pred_seg)
                z_def_slices.append(quiver_image)
            pred_im_t = np.concatenate(z_im_slices, 2)
            pred_seg_t = np.concatenate(z_seg_slices, 2)
            pred_def_t = np.concatenate(z_def_slices, 3)
            ims.append(pred_im_t)
            segs.append(pred_seg_t)
            defs.append(pred_def_t)
        ims = np.stack(ims, -1)
        segs = np.stack(segs, -1)
        defs = np.stack(defs, -1)

        if self.logging_disabled:  # Logging  -------------------------------------------------------------------------
            return
        gt_images, _, gt_segs, _, full_indices, coord_max, coord_min, \
            aff_params_padded, spacings_padded, flippings_padded, _ \
                = self.get_sample_elements_from_batch(dataset.load_subject_data(subj_idx, 0))
        save_dir = self.log_path / f"{mode}_volumes" / str(subj_id)
        save_dir.parent.parent.mkdir(exist_ok=True)
        save_dir.parent.mkdir(exist_ok=True)
        save_dir.mkdir(exist_ok=True)
        # Save volumes to disk as nifti
        # Use the coordinate system of the top-most SA slice (ie. 3)
        aff_params = aff_params_padded[3][None] + aff_def_params[0, 3].cpu()
        aff = params_to_mat(aff_params, torch.ones_like(spacings_padded[3][None]), flippings_padded[3][None])
        aff = aff[0].cpu().numpy()
        save_dir_pred = save_dir / 'pred'
        save_dir_pred.mkdir(exist_ok=True)
        array_to_nifti(str(save_dir_pred / f"full.nii.gz"), ims, aff)
        array_to_nifti(str(save_dir_pred / f"full_seg.nii.gz"), segs, aff)
        save_dir_gt = save_dir.parent / "gt"
        save_dir_gt.mkdir(exist_ok=True)
        for i in range(gt_images.shape[0]):
            aff = params_to_mat(aff_params, torch.ones_like(spacings_padded[i][None]), flippings_padded[i][None])
            aff = aff[0].cpu().numpy()
            array_to_nifti(str(save_dir_gt / f"slice{i}.nii.gz"), gt_images[i, ..., None, None].numpy(), aff)
            array_to_nifti(str(save_dir_gt / f"slice{i}_seg.nii.gz"), gt_segs[i, ..., None, None].numpy(), aff)

        slice_indices = range(20, res[2] - 20, res[2] // 10)
        ims_rgb = np.stack([ims]*3, axis=0)
        segs_rgb = (np.stack([segs]*3, axis=0) / 4 * 255).astype(np.uint8)
        content = [np.concatenate((ims_rgb[..., i, :], segs_rgb[..., i, :], defs[...,i,:]), 2) for i in slice_indices]
        content = [np.moveaxis(c, -1, 0) for c in content]
        # Save series to file as mp4
        for v, i in zip(content, slice_indices):
            video_array_to_file(v, save_dir / f"epoch_{self.current_epoch:06d}_subj_{subj_idx}_slice_{i:02d}-{res[2]}.mp4",
                                video_duration=2.0)
        # Log to WANDB
        if not self.logging_wandb_disabled:
            # Log video slices
            videos = [wandb.Video(c, fps=max(1, int(50 / video_duration)), format='gif') for c in content]
            wandb.log({f"{mode}_volumes/subj_{subj_id}": videos})
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
