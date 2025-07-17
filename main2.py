import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Optional, List

import kornia
import numpy as np
import torch
import tqdm
from kornia.filters import spatial_gradient
from torch import nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
import wandb
from monai.losses import DiceLoss

from data_utils import array_to_nifti
from dataloader import CMRDataModule
from pos_encoding import PosEncodingNeRFAnnealed, PosEncodinFourier
from layers import Relu, ConvBlock
from utils import params_to_mat, make_coordinate_tensor, to_1hot, create_meshplot_visualization, \
    process_segmentation_with_marching_cubes
from lightning.pytorch.loggers import WandbLogger


class Encoder(nn.Module):
    """ Simple decoder that uses a single averaged latent vector in its input to
    condition the way coordinates are processed. """

    def __init__(self, filters: Tuple[int, ...], **kwargs):
        super(Encoder, self).__init__()
        self.poolings = [(2,2,2), (2,2,1), (2,2,5), (2,2,5), (2,2,1), (2,2,1)]
        a = []
        for i in range(len(filters) - 1):
            a.append(ConvBlock(filters[i], filters[i+1]))
        a.append(ConvBlock(filters[-1], filters[-1]))
        self.layers = nn.Sequential(*a)
        self.out = nn.AdaptiveAvgPool3d(1)
        self.out_dim = filters[-1]
        self.aff_emb = PosEncodingNeRFAnnealed(in_dim=6, # 3D + Cyclical time
                                               num_frequencies=(7, 7, 7, 7, 7, 7),
                                               anneal_max_iter=kwargs['pe_anneal_max_iter'],
                                               anneal_start_prop=kwargs['pe_anneal_start_prop'])
        self.aff_enc = nn.Linear(self.aff_emb.out_dim, self.out_dim)

    def forward(self, x: torch.Tensor, aff_params: torch.Tensor) -> torch.Tensor:
        for l in self.layers[:-1]:
            x = l(x)
            x = F.avg_pool3d(x, 2)
        x = x + self.layers[-1](x)
        x = self.out(x).squeeze()
        aff = self.aff_enc(self.aff_emb(aff_params))
        return x + aff


class MLP(nn.Module):
    """ Simple decoder that uses a single averaged latent vector in its input to
    condition the way coordinates are processed. """

    def __init__(self, coord_size: int, num_hidden_layers: int, hidden_size: int, out_size: int, **kwargs):
        super(MLP, self).__init__()
        self.start = Relu(coord_size, hidden_size)
        # a = []
        # for i in range(num_hidden_layers - 2):
        #     a.append(Relu(hidden_size, hidden_size))
        # a.append(nn.Linear(hidden_size, out_size))
        self.mlp1 = nn.Sequential(*[Relu(hidden_size, hidden_size) for _ in range(num_hidden_layers//4)])
        self.mlp2 = nn.Sequential(*[Relu(hidden_size, hidden_size) for _ in range(num_hidden_layers//4)])
        self.mlp3 = nn.Sequential(*[Relu(hidden_size, hidden_size) for _ in range(num_hidden_layers//4)])
        self.mlp4 = nn.Sequential(*[Relu(hidden_size, hidden_size) for _ in range(num_hidden_layers//4)])
        self.out = nn.Linear(hidden_size, out_size)
        self.out_size = out_size

    def forward(self, coord: torch.Tensor) -> torch.Tensor:
        x_s = self.start(coord)
        x = x_s + self.mlp1(x_s)
        x = x_s + self.mlp2(x)
        x = x_s + self.mlp3(x)
        x = x_s + self.mlp4(x)
        return self.out(x)


class MLPDeform(nn.Module):
    """ Simple decoder that uses a single averaged latent vector in its input to
    condition the way coordinates are processed. """

    def __init__(self, coord_size: int, num_hidden_layers: int, hidden_size: int, out_size: int, out_init_std=0.01, **kwargs):
        super(MLPDeform, self).__init__()
        self.start = Relu(coord_size, hidden_size)
        a = []
        for i in range(num_hidden_layers - 1):
            a.append(Relu(hidden_size, hidden_size))
        self.middle = nn.Sequential(*a)
        self.out = nn.Linear(hidden_size, out_size)
        nn.init.xavier_uniform_(self.out.weight, out_init_std)
        self.out_size = out_size

    def forward(self, coord: torch.Tensor) -> torch.Tensor:
        x_s = self.start(coord)
        x = x_s + self.middle(x_s)
        return self.out(x)


class INR(pl.LightningModule):
    def __init__(self, coord_size: int, num_subjects: int, max_slices: int, **kwargs):
        super(INR, self).__init__()
        self.automatic_optimization = False
        self.logging_rate = kwargs['logging_rate']

        self.coord_size = coord_size
        self.intensity_size = 1
        self.num_subjects = num_subjects
        self.max_slices = max_slices
        self.norm_min, self.norm_max = 0.0, 1.0

        self.latent_size = kwargs["latent_size"]
        self.subj_latents = nn.Parameter(torch.randn((self.num_subjects, self.latent_size),
                                                     dtype=torch.float32, device="cuda") * 1e-2, requires_grad=True)

        self.aff_deform_params = nn.Parameter(torch.zeros((self.num_subjects, self.max_slices, 6),
                                                          dtype=torch.float32, device="cuda"), requires_grad=True)
        self.deform_latent_size = kwargs["deform_latent_size"]
        self.deform_latents = nn.Parameter(torch.randn((self.num_subjects, self.max_slices, self.deform_latent_size),
                                                     dtype=torch.float32, device="cuda") * 1e-2, requires_grad=True)
        self.coord_deform_inr = MLPDeform(2 + self.deform_latent_size,
                                          num_hidden_layers=kwargs['deform_num_hidden_layers'],
                                          hidden_size=kwargs['deform_hidden_size'],
                                          out_size=3)
        self.encoder = Encoder([16, 32, 64, 128, 256, 512], **kwargs)
        # self.intensity_deform_latent_size = kwargs["deform_latent_size"]
        # self.intensity_deform_latents = nn.Parameter(torch.randn((self.num_subjects, self.max_slices, self.intensity_deform_latent_size),
        #                                              dtype=torch.float32, device="cuda") * 1e-2, requires_grad=True)
        # self.intensity_deform_inr = MLPDeform(2 + self.deform_latent_size,
        #                                   num_hidden_layers=kwargs['deform_num_hidden_layers'],
        #                                   hidden_size=kwargs['deform_hidden_size'],
        #                                   out_size=1)
        self.intensity_deform_scale = nn.Parameter(torch.randn((self.num_subjects, self.max_slices, 1),
                                                     dtype=torch.float32, device="cuda") * 1e-3, requires_grad=True)
        # self.pos_enc = PosEncodinFourier(in_dim=5, # 3D + Cyclical time
        #                                  num_frequencies=kwargs['pe_num_frequencies'],
        #                                  coords_freq_scale=kwargs['pe_freq_scale'],
        #                                  )
        self.pos_enc = PosEncodingNeRFAnnealed(in_dim=5, # 3D + Cyclical time
                                               num_frequencies=kwargs['pe_num_frequencies'],
                                               anneal_max_iter=kwargs['pe_anneal_max_iter'],
                                               anneal_start_prop=kwargs['pe_anneal_start_prop'])
        self.canonical_inr = MLP(self.pos_enc.out_dim + self.latent_size,
                                 num_hidden_layers=kwargs['num_hidden_layers'],
                                 hidden_size=kwargs['hidden_size'],
                                 out_size=5)

        self.recon_loss = torch.nn.MSELoss()
        self.class_weight = torch.tensor([i / sum(kwargs['weight_seg_class']) for i in kwargs['weight_seg_class']])
        self.seg_loss = DiceLoss(softmax=False, reduction="none")
        self.psnr_loss = kornia.losses.PSNRLoss(max_val=1.0)

        self.weight_reg_inr = kwargs["weight_reg_inr"]
        self.weight_reg_aff = kwargs["weight_reg_aff"]
        self.weight_reg_lat = kwargs["weight_reg_lat"]
        self.weight_intensity_scale = kwargs["weight_reg_int_scale"]
        self.weight_reg_deform = kwargs["weight_reg_deform"]
        self.weight_reg_deform_lat = kwargs["weight_reg_deform_lat"]
        self.weight_loss_deriv = kwargs["weight_loss_deriv"]
        self.weight_loss_hess = kwargs["weight_loss_hess"]
        self.weight_loss_seg = kwargs["weight_loss_seg"]
        self.lr = kwargs["learning_rate"]
        self.lr_aff = kwargs["learning_rate_aff"]
        self.lr_def = kwargs["learning_rate_def"]


    def configure_optimizers(self):
        opt_inr = torch.optim.Adam([*self.canonical_inr.parameters(), self.subj_latents], lr=self.lr)
        opt_deform = torch.optim.Adam([self.aff_deform_params], lr=self.lr_def)#, *self.intensity_deform_inr.parameters(), self.intensity_deform_latents], lr=self.lr_def)
        opt_intensity = torch.optim.Adam([self.intensity_deform_scale], lr=self.lr_def)
        return opt_inr, opt_deform, opt_intensity

    def regularization_criterion(self, subj_idx: torch.Tensor) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        loss_dict = dict()
        loss_reg = torch.zeros((1,), dtype=torch.float32, device=self.device)
        if self.weight_reg_inr:
            loss_reg_inr = sum((p * p).sum() for p in self.canonical_inr.parameters())
            loss_reg_inr = loss_reg_inr * self.weight_reg_inr
            loss_reg += loss_reg_inr
            loss_dict["loss_reg_inr"] = loss_reg_inr
        if self.weight_reg_aff:
            loss_reg_aff = nn.functional.mse_loss(self.aff_deform_params[subj_idx],
                                                  torch.zeros_like(self.aff_deform_params[subj_idx]))
            loss_reg_aff = loss_reg_aff * self.weight_reg_aff
            loss_reg += loss_reg_aff
            loss_dict["loss_reg_aff"] = loss_reg_aff
        if self.weight_reg_lat:
            loss_reg_lat = nn.functional.mse_loss(self.subj_latents[subj_idx],
                                                  torch.zeros_like(self.subj_latents[subj_idx]))
            loss_reg_lat = loss_reg_lat * self.weight_reg_lat
            loss_reg += loss_reg_lat
            loss_dict["loss_reg_lat"] = loss_reg_lat
        if self.weight_intensity_scale:
            loss_reg_int_scale = nn.functional.mse_loss(self.intensity_deform_scale[subj_idx],
                                                        torch.zeros_like(self.intensity_deform_scale[subj_idx]))
            loss_reg_int_scale = loss_reg_int_scale * self.weight_intensity_scale
            loss_reg += loss_reg_int_scale
            loss_dict["loss_reg_int_scale"] = loss_reg_int_scale
        if self.weight_reg_deform:
            loss_reg_c_deform = sum((p * p).sum() for p in self.coord_deform_inr.parameters())
            loss_reg_c_deform = loss_reg_c_deform * self.weight_reg_deform
            loss_reg += loss_reg_c_deform
            loss_dict["loss_reg_deform"] = loss_reg_c_deform
        if self.weight_reg_deform_lat:
            loss_reg_lat = nn.functional.mse_loss(self.deform_latents[subj_idx],
                                                  torch.zeros_like(self.deform_latents[subj_idx]))
            loss_reg_lat = loss_reg_lat * self.weight_reg_deform_lat
            loss_reg += loss_reg_lat
            loss_dict["loss_reg_deform_lat"] = loss_reg_lat
        loss_dict["loss_reg"] = loss_reg
        return loss_reg, loss_dict

    def forward(self,
                imgs,
                coords_voxel: torch.Tensor,
                aff_params: torch.Tensor,
                spacings: torch.Tensor,
                needs_flip: torch.BoolTensor,
                subject_idx: torch.LongTensor,
                slice_idx: torch.LongTensor,
                min_coords: torch.Tensor,
                max_coords: torch.Tensor,
                return_deriv: bool = False,
                return_hessian: bool = False) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        world_coords = self.forward_coord_model(coords_voxel, aff_params, spacings, needs_flip,
                                                subject_idx, slice_idx, min_coords, max_coords)
        seg_pred, values_pred  = self.forward_inr(world_coords, subject_idx)
        values_pred_d = None
        values_pred_dd = None
        if return_deriv:
            values_pred_d = torch.autograd.grad(values_pred, world_coords, grad_outputs=torch.ones_like(values_pred),
                                                 create_graph=True, retain_graph=True)[0]
            if return_hessian:
                # values_pred_ddx = torch.autograd.grad(values_pred_d[..., 0], world_coords,
                #                                       grad_outputs=torch.ones_like(values_pred_d[..., 0]),
                #                                       retain_graph=True)[0]
                # values_pred_ddy = torch.autograd.grad(values_pred_d[..., 1], world_coords,
                #                                       grad_outputs=torch.ones_like(values_pred_d[..., 1]),
                #                                       retain_graph=True)[0]
                # values_pred_ddz = torch.autograd.grad(values_pred_d[..., 2], world_coords,
                #                                       grad_outputs=torch.ones_like(values_pred_d[..., 2]),
                #                                       retain_graph=True)[0]
                values_pred_ddt = torch.autograd.grad(values_pred_d[..., 3], world_coords,
                                                      grad_outputs=torch.ones_like(values_pred_d[..., 3]),
                                                      retain_graph=True)[0]
                # values_pred_dd = torch.stack((values_pred_ddx, values_pred_ddy, values_pred_ddz, values_pred_ddt), -2)
                values_pred_dd = values_pred_ddt
        return seg_pred, values_pred, values_pred_d, values_pred_dd

    def forward_coord_model(self,
                            coords_voxel: torch.Tensor,
                            aff_params: torch.Tensor,
                            spacings: torch.Tensor,
                            needs_flip: torch.BoolTensor,
                            subject_idx: torch.LongTensor,
                            slice_idx: torch.LongTensor,
                            min_coords: torch.Tensor,
                            max_coords: torch.Tensor,
                            normalize=True):
        # Create indexing tensor to keep track of which batch is each coordinate coming from
        b, _ = torch.meshgrid(torch.arange(0, slice_idx.shape[0]), torch.arange(0, slice_idx.shape[1]))
        coords_flat_ = coords_voxel.reshape((-1, coords_voxel.shape[-1])).to(torch.float32)
        # deform_in = torch.cat((coords_flat_[...,:2], self.deform_latents.cuda()[subject_idx][b.flatten(), slice_idx.flatten()]), -1)
        # coord_deform = self.coord_deform_inr(deform_in)
        # coords_flat_[..., :3] += coord_deform

        aff_params_deform = aff_params + self.aff_deform_params[subject_idx]
        aff_params_deform_ = aff_params_deform.reshape((-1, aff_params_deform.shape[-1]))
        spacings_ = spacings.reshape((-1, spacings.shape[-1]))
        needs_flip_ = needs_flip.reshape((-1,))
        affines_ = params_to_mat(aff_params_deform_, spacings_, needs_flip_)
        affines = affines_.reshape((aff_params.shape[0], aff_params.shape[1], 4, 4))

        # In order to move coords from voxel space to world space, we need to have them in (x, y, z, 1)
        time_coord_ = coords_flat_[:, -1:]  # We take out time coordinates. We will replace them back in later.
        spatial_coord_ = torch.cat((coords_flat_[:, :-1], torch.ones_like(coords_flat_[:, :1])), dim=1)  # Moving coords to world space requires (x, y, z, 1)

        # Get affines corresponding to each coordinate
        affines_ = affines[b.reshape(-1), slice_idx.reshape(-1)]
        # Move coordinates to world space
        coords_world_ = torch.bmm(affines_, spatial_coord_[..., None])
        coords_world_ = coords_world_.squeeze(-1)
        # Add time coordinate back
        coords_world_[:, -1:] = time_coord_
        coords_world = coords_world_.reshape(coords_voxel.shape)

        # normalize coordinates
        if normalize:
            return self.min_max_normalize(coords_world, min_coords[:, None], max_coords[:, None], self.norm_min, self.norm_max)
        return coords_world

    def plane_deriv_to_world(self, derivs, *args, min_coords: torch.Tensor, max_coords: torch.Tensor) -> torch.Tensor:
        derivs_world = self.forward_coord_model(derivs, *args, None, None, normalize=False)
        derivs_world_norm = self.min_max_unnormalize(derivs_world, min_coords[:, None], max_coords[:, None], self.norm_min, self.norm_max)
        return derivs_world_norm

    def forward_inr(self, coords: torch.Tensor, subj_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        coords = torch.cat((coords[..., :3],
                            torch.cos(coords[...,-1:] * torch.pi),
                            torch.sin(coords[...,-1:] * torch.pi)), dim=-1)
        coords_enc = self.pos_enc(coords, self.global_step)
        # Get a subject latent for each coordinate
        # subject_latent = self.subj_latents[subj_idx]
        subject_latent = self.encoder()
        subject_latent = subject_latent[:, None].tile((1, coords.shape[1], 1))
        x = torch.cat((coords_enc, subject_latent), dim=-1)
        # Forward INR to obtain predicted volume values
        x_ = x.reshape((-1, x.shape[-1]))
        values_pred_ = self.canonical_inr(x_)
        seg_pred = values_pred_[:, 1:].reshape((coords.shape[0], coords.shape[1], 4))
        values_pred = values_pred_[:, 0].reshape((coords.shape[0], coords.shape[1]))
        return seg_pred, values_pred

    def forward_intensity_deform(self, coords_voxel, subject_idx, slice_idx):
        subject_idx = subject_idx[:,None].tile((1, slice_idx.shape[1]))
        int_deform = self.intensity_deform_scale[subject_idx.flatten(), slice_idx.flatten()]
        return int_deform.reshape(*coords_voxel.shape[:2])
        # latents_ = self.intensity_deform_latents[subject_idx.flatten(), slice_idx.flatten()]
        # coords_voxel_ = coords_voxel.reshape(latents_.shape[0], coords_voxel.shape[-1])[..., :2]
        # coords_voxel_norm_ = coords_voxel_ / 250
        # inr_int_ = torch.cat((coords_voxel_norm_, latents_), -1)
        # int_deform = self.intensity_deform_inr(inr_int_)
        # int_deform = torch.tanh(int_deform)
        # return int_deform.reshape(*coords_voxel.shape[:2])

    def deform_intensities(self, intensities, coords_voxel, subject_idx, slice_idx, inverse=False):
        if not inverse:
            return intensities * (1 + self.forward_intensity_deform(coords_voxel, subject_idx, slice_idx))
        else:
            return intensities / (1 + self.forward_intensity_deform(coords_voxel, subject_idx, slice_idx))


    def training_step(self, batch):
        opt_inr, opt_deform, opt_inten = self.optimizers()

        (imgs, coords_voxel, values, values_dt, values_ddt, segs, gt_avail,
         aff_params, spacings, needs_flip, subject_idx, slice_idx, min_coords, max_coords) = batch

        coords_voxel = coords_voxel + torch.randn(coords_voxel.shape, device=coords_voxel.device) * 5e-2  # TODO
        seg_pred, values_pred, values_pred_d, values_pred_dd = self.forward(imgs, coords_voxel, aff_params,
                                                                             spacings, needs_flip,
                                                                             subject_idx, slice_idx,
                                                                             min_coords, max_coords,
                                                                             return_deriv=True,
                                                                             return_hessian=True,)
        values_deform = self.deform_intensities(values, coords_voxel, subject_idx, slice_idx)
        loss_recon = self.psnr_loss(values_pred, values_deform)
        seg_pred, segs = torch.softmax(seg_pred, -1)*gt_avail[...,None], segs*gt_avail[...,None]
        loss_seg_per_class = self.seg_loss(seg_pred.moveaxis(-1,1), segs.moveaxis(-1,1)).mean(-1).mean(0)
        dice_per_class = 1 - loss_seg_per_class
        loss_seg = (loss_seg_per_class * self.class_weight.to(loss_seg_per_class.device)).mean() * self.weight_loss_seg
        # values_dt_world = self.plane_deriv_to_world(values_dt, aff_params, spacings, needs_flip, subject_idx, slice_idx,
        #                                             min_coords=min_coords, max_coords=max_coords)
        loss_dt = 0.0 #self.psnr_loss(values_pred_d[...,-1:], values_dt*50) * self.weight_loss_deriv
        loss_ddt = 0.0 #self.psnr_loss(values_pred_dd[..., -1:], values_ddt*50) * self.weight_loss_hess

        loss_reg, loss_reg_dict = self.regularization_criterion(subject_idx)
        loss = loss_recon + loss_reg + loss_seg + loss_dt + loss_ddt

        opt_inr.zero_grad()
        opt_deform.zero_grad()
        opt_inten.zero_grad()
        self.manual_backward(loss)
        opt_inr.step()
        opt_deform.step()
        opt_inten.step()
        self.log_dict({"loss": loss, "loss_recon": loss_recon, **loss_reg_dict,
                       "loss_dt": loss_dt, "loss_ddt": loss_ddt, "loss_seg": loss_seg}, prog_bar=True)
        self.log_dict({"dice_BG": dice_per_class[0], "dice_LV": dice_per_class[1], "dice_MYO": dice_per_class[2],
                       "dice_RV": dice_per_class[3]}, prog_bar=False)

    def on_train_epoch_end(self):
        if self.current_epoch > 0 and (self.current_epoch in {100, 500} or self.current_epoch % self.logging_rate == 0):
            for i in range(8):
                self.log_images(i)
            for i in range(5):
                self.log_volume(i)

    @staticmethod
    def min_max_normalize(X, x_min, x_max, s_min=0., s_max=1.):
        return (X - x_min) / (x_max - x_min) * (s_max - s_min) + s_min

    @staticmethod
    def min_max_unnormalize(X, x_min, x_max, s_min=0., s_max=1.):
        return (X - s_min) / (s_max - s_min) * (x_max - x_min) + x_min

    @torch.no_grad()
    def log_images(self, subj_idx, video_duration: float = 4, mode="train"):
        dataset = eval(f"self.trainer.datamodule.{mode}_dset")
        subj_path = dataset.data_paths[subj_idx]
        subj_id = Path(subj_path).parent.name
        videos = [[] for _ in range(20)]
        preds = [[] for _ in range(20)]
        segs = [[] for _ in range(20)]
        psnrs = [[] for _ in range(20)]
        dices = [[] for _ in range(20)]
        for t in tqdm.tqdm(range(0, 50, 5), desc=f"Logging slices subj {subj_id}"):
            images, images_dt, images_ddt, seg_argmax, _, full_indices, coord_min, coord_max, \
                aff_params_padded, spacings_padded, flippings_padded = dataset.load_subject_data(subj_idx, t)
            images, images_dt, images_ddt, seg_argmax = images.cuda()[None], images_dt[...,-1].cuda()[None], images_ddt[...,-1].cuda()[None], seg_argmax.cuda()[None]
            full_indices, coord_max, coord_min = full_indices.cuda()[None], coord_max.cuda()[None], coord_min.cuda()[None]
            aff_params_padded, spacings_padded, flippings_padded = aff_params_padded.cuda()[None], spacings_padded.cuda()[None], flippings_padded.cuda()[None]
            full_indices = make_coordinate_tensor(images.shape[1:]).cuda()
            slice_idx = full_indices[..., :1]
            voxel_indices = torch.cat((full_indices[..., 1:3], torch.zeros_like(full_indices[..., :1]),
                                       torch.full_like(full_indices[..., :1], t)), dim=-1).float()
            for s in range(images.shape[1]):
                voxel_indices_ = voxel_indices[s].reshape(1, -1, 4)
                slice_idx_ = slice_idx[s].reshape(1, -1, 1)
                with torch.enable_grad():
                    pred_seg_, pred_vals_, pred_vals_d_, pred_vals_dd_ = self.forward(voxel_indices_, aff_params_padded,
                                                                          spacings_padded, flippings_padded,
                                                                          torch.tensor((subj_idx,)), slice_idx_,
                                                                          coord_min, coord_max,
                                                                          return_deriv=True,
                                                                          return_hessian=True)
                pred_seg_, pred_vals_, pred_vals_d_, pred_vals_dd_ = pred_seg_.detach(), pred_vals_.detach(), pred_vals_d_.detach(), pred_vals_dd_.detach()
                pred_img = pred_vals_.reshape(images.shape[2:])
                value_deform_ = self.forward_intensity_deform(voxel_indices_, torch.tensor((subj_idx,)), slice_idx_)
                value_deform = value_deform_.reshape(pred_img.shape)
                pred_img_dt = pred_vals_d_.reshape(*images.shape[2:], pred_vals_d_.shape[-1])[...,-1]
                # pred_img_ddt = pred_vals_dd_.reshape(*images.shape[2:], *pred_vals_dd_.shape[2:])[...,-1]
                pred_img_deform_ = self.deform_intensities(pred_vals_, voxel_indices_, torch.tensor((subj_idx,)), slice_idx_, inverse=True)
                pred_img_deform = pred_img_deform_.reshape(pred_img.shape)
                psnr_metric = kornia.metrics.psnr(pred_img, images[0,s], max_val=1.0)
                psnrs[s].append(psnr_metric.mean().item())
                pred = pred_img.clip(0.0, 1.0)
                pred = (pred * 255).cpu().numpy()
                preds[s].append(pred)
                pred_seg = pred_seg_.reshape(*images.shape[2:], pred_seg_.shape[-1])
                pred_seg_argmax = pred_seg.argmax(-1)
                segs[s].append(pred_seg_argmax)
                seg_gt = to_1hot(seg_argmax[0,s].reshape(-1), pred_seg.shape[-1]).reshape(pred_seg.shape)
                dice = 1 - self.seg_loss(pred_seg[None].moveaxis(-1,1), seg_gt[None].moveaxis(-1,1)).mean(-1).mean(0)
                dices[s].append(dice)

                # Image
                img = torch.stack([torch.cat([images[0,s], pred_img], 0)]*3, 0)
                # Segmentation
                segs_argmax = torch.cat([seg_argmax[0,s], pred_seg_argmax], 0)
                seg_frames = torch.stack([torch.cat([images[0,s], images[0,s]], 0)]*3, 0)
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
                images_dt_norm = images_dt * (coord_max[:, None, -1:] - coord_min[:, None, -1:])
                img_dt = torch.stack([torch.cat([images_dt_norm[0,s].abs(), pred_img_dt.abs()], 0)]*3, 0)
                # images_ddt_norm = images_ddt * (coord_max[:, None, -1:] - coord_min[:, None, -1:])
                # img_ddt = torch.stack([torch.cat([images_ddt_norm[0,s].abs(), pred_img_ddt.abs()], 0)]*3, 0)
                img_ddt = torch.stack([torch.cat([pred_img_deform.abs(), value_deform], 0)]*3, 0)

                frame_img = torch.cat([img, seg_frames], 1)
                frame_d = torch.cat([img_dt, img_ddt], 1)
                frame = torch.cat([frame_img, frame_d], 2)
                frame = frame.clip(0.0, 1.0)
                frame = (frame * 255).cpu().numpy().astype(np.uint8)
                videos[s].append(frame)
        videos = [np.stack(v, 0) for v in videos if v]
        psnrs = [np.mean(i) for i in psnrs if i]
        psnr_strings = [f"PSNR:{i:.1f}" for i in psnrs]
        dices = [torch.stack(d, 0).mean(-1).mean(0) for i, d in enumerate(dices) if d]
        dices_strings = [f"Dice:" + f"{d[1]:.2f}, " + f"{d[2]:.2f}, " + f"{d[3]:.2f}" if i >= 3 else "Dice: -, -, -" for i, d in enumerate(dices)]
        wandb_videos = [wandb.Video(v, fps=max(1, int(50 / video_duration)),
                              caption=f"Slice:{i}, {psnr_strings[i]}  {dices_strings[i]}") for i, v in enumerate(videos)]
        wandb.log({f"{mode}_videos/subj_{subj_id}": wandb_videos}, step=self.current_epoch)

        if self.current_epoch > 0 and self.current_epoch % (self.logging_rate * 10) == 0:
            images, _, _, _, _, full_indices, coord_max, coord_min, \
                aff_params_padded, spacings_padded, flippings_padded = dataset.load_subject_data(subj_idx, 0)
            save_dir = Path(f"niftis/{subj_id}")
            save_dir.parent.mkdir(exist_ok=True)
            save_dir.mkdir(exist_ok=True)
            preds = [np.stack(v, 0) for v in preds if v]
            for i, v in enumerate(preds):
                aff_params = aff_params_padded[i][None] + self.aff_deform_params[subj_idx][i].cpu()
                aff = params_to_mat(aff_params, spacings_padded[i][None], flippings_padded[i][None])
                aff = aff[0].cpu().numpy()
                v = (v > 100).astype(np.uint8)
                v = np.moveaxis(v[..., None], 0, -1)
                array_to_nifti(str(save_dir / f"{self.current_epoch}_{i}.nii.gz"), v[:,:,None], aff)

    @torch.no_grad()
    def log_volume(self, subj_idx, video_duration: float = 4, mode="train", res=(200, 200, 200)):
        dataset = eval(f"self.trainer.datamodule.{mode}_dset")
        subj_path = dataset.data_paths[subj_idx]
        subj_id = Path(subj_path).parent.name
        coords = torch.stack(torch.meshgrid(*[torch.linspace(self.norm_min+.25, self.norm_max-.25, i) for i in res]), dim=-1)
        ims = []
        segs = []
        for t in tqdm.tqdm(range(0, 50, 5), desc=f"Logging volume subj {subj_id}"):
            t_norm = t / 50
            c = torch.concatenate((coords, torch.full((*res, 1), t_norm)), dim=-1).cuda()
            z_im_slices = []
            z_seg_slices = []
            for z in range(res[-1]):
                c_z = c[:,:,z:z+1]
                pred_seg_, pred_vals_ = self.forward_inr(c_z.reshape(1, -1, 4), torch.tensor((subj_idx,)))
                pred_seg = pred_seg_.argmax(-1).reshape(c_z.shape[:-1])
                pred_seg = pred_seg.cpu().numpy()
                pred_seg = pred_seg.astype(np.uint8)
                pred_vals_ = pred_vals_.clip(0.0, 1.0)
                pred_val = pred_vals_.reshape(c_z.shape[:-1])
                pred_val = pred_val.cpu().numpy()
                pred_val = (pred_val*255).astype(np.uint8)
                z_im_slices.append(pred_val)
                z_seg_slices.append(pred_seg)
            pred_im_t = np.concatenate(z_im_slices, 2)
            pred_seg_t = np.concatenate(z_seg_slices, 2)
            ims.append(pred_im_t)
            segs.append(pred_seg_t)
        ims = np.stack(ims, -1)
        segs = np.stack(segs, -1)

        # Save niftis
        if self.current_epoch > 0 and self.current_epoch % (self.logging_rate * 5) == 0:
            gt_images, _, _, gt_segs, _, full_indices, coord_max, coord_min, \
                aff_params_padded, spacings_padded, flippings_padded = dataset.load_subject_data(subj_idx, 0)
            aff_params = aff_params_padded[3][None] + self.aff_deform_params[subj_idx][3].cpu()
            aff = params_to_mat(aff_params, torch.ones_like(spacings_padded[3][None]), flippings_padded[3][None])
            aff = aff[0].cpu().numpy()
            save_dir = Path(f"niftis_vol/{subj_id}/{self.current_epoch}")
            save_dir.parent.mkdir(exist_ok=True)
            save_dir.mkdir(exist_ok=True)
            array_to_nifti(str(save_dir / f"full.nii.gz"), ims, aff)
            array_to_nifti(str(save_dir / f"full_seg.nii.gz"), segs, aff)
            save_dir = save_dir / "gt"
            save_dir.mkdir(exist_ok=True)
            for i in range(gt_images.shape[0]):
                aff = params_to_mat(aff_params, torch.ones_like(spacings_padded[i][None]), flippings_padded[i][None])
                aff = aff[0].cpu().numpy()
                array_to_nifti(str(save_dir / f"slice{i}.nii.gz"), gt_images[i, ..., None, None].numpy(), aff)
                array_to_nifti(str(save_dir / f"slice{i}_seg.nii.gz"), gt_segs[i, ..., None, None].numpy(), aff)

            # Log videos
            content = [np.stack([np.concatenate((ims[..., i, :], (segs[..., i, :] / 4 * 255).astype(np.uint8)), 1)] * 3,
                                axis=0)
                       for i in range(20, res[2] - 20, res[2] // 10)]
            videos = [wandb.Video(np.moveaxis(c, -1, 0), fps=max(1, int(50 / video_duration))) for c in content]
            wandb.log({f"{mode}_volumes/subj_{subj_id}": videos}, step=self.current_epoch)

        # Log meshes
        meshes = process_segmentation_with_marching_cubes(segs[..., 0], level=0.5, step_size=1)
        if not meshes:
            # Dummy mesh
            a = np.zeros_like(segs[..., 0])
            a[5:15,5:15,5:15] = 1
            a[20:25,20:25,20:25] = 2
            meshes = process_segmentation_with_marching_cubes(a, level=0.5, step_size=1)
        import tempfile
        save_dir = Path(f"temp_mesh_files/{subj_id}")
        save_dir.parent.mkdir(exist_ok=True)
        save_dir.mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=str(save_dir.absolute()), suffix='.html', delete=False) as f:
            plot = create_meshplot_visualization(meshes, f.name)
            temp_file_path = f.name
            with open(f.name, 'r') as html_file:
                wandb.log({f"{mode}_mesh/subj_{subj_id}": wandb.Html(html_file.read())})
        shutil.rmtree(save_dir.parent)


@dataclass
class Params:
    # Epochs -------------------------------------------------------------------
    max_epochs: int = 1_000_000
    num_coords: int = 20_000
    batch_size: int = 4
    check_val_every_n_epoch: int = 1000000000
    logging_rate: int = 2000
    # Model -------------------------------------------------------------------
    num_hidden_layers: int = 16
    enc_num_hidden_layers: int = 4
    hidden_size: int = 512
    latent_size: int = 512
    deform_latent_size: int = 16
    deform_num_hidden_layers: int = 1
    deform_hidden_size: int = 128
    # Regularization -------------------------------------------------------------------
    weight_reg_inr: float = 1e-5
    weight_reg_aff: float = 1e-4
    weight_reg_lat: float = 1e-4
    weight_reg_int_scale: float = 1e-3
    weight_reg_deform: float = 0e-2
    weight_reg_deform_lat: float = 1e-2
    weight_loss_deriv: float = 0e0
    weight_loss_hess: float = 0e0
    weight_loss_seg: float = 1e0
    weight_seg_class: Tuple[float] = (1,5,15,10)  # Will be normalized
    # Learning rates -------------------------------------------------------------------
    learning_rate: float = 1e-4
    learning_rate_aff: float = 1e-4
    learning_rate_def: float = 1e-4
    # Positional encoder -------------------------------------------------------------------
    pe_num_frequencies: List[int] = (8,8,8,5,5)
    pe_anneal_max_iter: int = 2000
    pe_anneal_start_prop: float = 0.2
    pe_freq_scale: float = 1.0


def main(data_dir, wandb_disabled="true"):
    os.environ['WANDB_DISABLED'] = wandb_disabled

    # configure accelerator and devices
    accelerator = "gpu"
    devices = 1  # one GPU only
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    params = Params()
    data_module = CMRDataModule(load_la_dir=data_dir, load_sa_dir=data_dir,
                                preprocessed_store_path=r"/home/nil/data/ukbb/cardiac/unaligned_h5",
                                batch_size=params.batch_size, num_coords=params.num_coords, num_workers=0)
    data_module.prepare_data()

    logger = WandbLogger(project="CMR-Align")
    logger.log_hyperparams(params.__dict__)

    model = INR(coord_size=data_module.get_coord_size(), num_subjects=data_module.num_train, max_slices=data_module.get_max_slices(), **params.__dict__)

    os.makedirs(r'./checkpoints', exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/", every_n_epochs=1, save_last=True, verbose=True
    )

    trainer = Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator=accelerator,
        devices=devices,
        max_epochs=params.max_epochs,
        check_val_every_n_epoch=params.check_val_every_n_epoch,
        fast_dev_run=False,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        num_sanity_val_steps=1,
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    main(r"/home/nil/data/ukbb/cardiac/unaligned_subjects")
