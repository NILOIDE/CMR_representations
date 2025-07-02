import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Optional, List

import numpy as np
import torch
from kornia.filters import spatial_gradient
from torch import nn
import lightning.pytorch as pl
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
import wandb
from data_utils import array_to_nifti

from dataloader import CMRDataModule
from pos_encoding import PosEncodingNeRFAnnealed, PosEncodinFourier
from layers import Relu
from utils import params_to_mat, make_coordinate_tensor
from lightning.pytorch.loggers import WandbLogger
from losses import SegmentationCriterion, ReconstructionCriterion, SegmentationMetrics, ReconstructionMetrics


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

    def __init__(self, coord_size: int, num_hidden_layers: int, hidden_size: int, out_size: int, **kwargs):
        super(MLPDeform, self).__init__()
        self.start = Relu(coord_size, hidden_size)
        a = []
        for i in range(num_hidden_layers - 1):
            a.append(Relu(hidden_size, hidden_size))
        self.middle = nn.Sequential(*a)
        self.out = nn.Linear(hidden_size, out_size)
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
        self.internsity_size = 1
        self.num_subjects = num_subjects
        self.max_slices = max_slices

        self.latent_size = kwargs["latent_size"]
        self.subj_latents = nn.Parameter(torch.randn((self.num_subjects, self.latent_size),
                                                     dtype=torch.float32, device="cuda") * 1e-2, requires_grad=True)

        self.aff_deform_params = nn.Parameter(torch.zeros((self.num_subjects, self.max_slices, 6),
                                                          dtype=torch.float32, device="cuda"), requires_grad=True)
        self.coord_deform_inrs = {subj_idx: {i: MLP(self.coord_size,
                                                    num_hidden_layers=kwargs['deform_num_hidden_layers'],
                                                    hidden_size=kwargs['hidden_size'],
                                                    out_size=3)
                                             for i in range(self.max_slices)}
                                  for subj_idx in range(self.num_subjects)}
        self.deform_latent_size = kwargs["deform_latent_size"]
        self.deform_latents = nn.Parameter(torch.randn((self.num_subjects, self.max_slices, self.deform_latent_size),
                                                     dtype=torch.float32, device="cuda") * 1e-2, requires_grad=True)
        self.coord_deform_inr = MLPDeform(2 + self.deform_latent_size,
                                          num_hidden_layers=kwargs['deform_num_hidden_layers'],
                                          hidden_size=kwargs['deform_hidden_size'],
                                          out_size=3)
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
                                 out_size=1)

        self.recon_loss = torch.nn.MSELoss()

        self.weight_reg_inr = kwargs["weight_reg_inr"]
        self.weight_reg_aff = kwargs["weight_reg_aff"]
        self.weight_reg_lat = kwargs["weight_reg_lat"]
        self.weight_reg_deform = kwargs["weight_reg_deform"]
        self.weight_reg_deform_lat = kwargs["weight_reg_deform_lat"]
        self.lr = kwargs["learning_rate"]
        self.lr_aff = kwargs["learning_rate_aff"]
        self.lr_def = kwargs["learning_rate_def"]

        self.segmentation_metrics = SegmentationMetrics(**kwargs)
        self.reconstruction_metrics = ReconstructionMetrics(**kwargs)


    def configure_optimizers(self):
        opt_inr = torch.optim.Adam([*self.canonical_inr.parameters(), self.subj_latents], lr=self.lr)
        opt_aff = torch.optim.Adam([self.aff_deform_params], lr=self.lr_aff)
        opt_deform = torch.optim.Adam([*self.coord_deform_inr.parameters(), self.deform_latents], lr=self.lr_def)
        # opt_deform = torch.optim.Adam([w for inr in self.coord_deform_inrs.values() for w in inr.weights] +
        #                               [b for inr in self.coord_deform_inrs.values() for b in inr.biases], lr=1e-3)
        return opt_inr, opt_aff, opt_deform

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

    @torch.enable_grad
    def forward_with_dt(self, coords_voxel, aff_params, spacings, needs_flip,
                subject_idx, slice_idx, min_coords, max_coords):
        world_coords = self.forward_coord_model(coords_voxel, aff_params, spacings, needs_flip,
                                                subject_idx, slice_idx, min_coords, max_coords)
        values_pred = self.forward_inr(world_coords, subject_idx)
        values_pred_dt = torch.autograd.grad(values_pred.sum(), world_coords, retain_graph=True)[0][..., -1]
        return values_pred, values_pred_dt

    def forward(self, coords_voxel, aff_params, spacings, needs_flip,
                subject_idx, slice_idx, min_coords, max_coords):
        world_coords = self.forward_coord_model(coords_voxel, aff_params, spacings, needs_flip,
                                                subject_idx, slice_idx, min_coords, max_coords)
        values_pred = self.forward_inr(world_coords, subject_idx)
        return values_pred

    def forward_coord_model(self, coords_voxel, aff_params, spacings, needs_flip,
                subject_idx, slice_idx, min_coords, max_coords):
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

        # normalize coordinates [-1, 1]
        norm_coords_ = self.min_max_scale(coords_world_, min_coords[b.reshape(-1)], max_coords[b.reshape(-1)],
                                          s_min=-1, s_max=1)
        norm_coords = norm_coords_.reshape(coords_voxel.shape)
        return norm_coords

    def forward_inr(self, coords, subj_idx):
        coords = torch.cat((coords[..., :3],
                            torch.cos(coords[...,-1:] * torch.pi),
                            torch.sin(coords[...,-1:] * torch.pi)), dim=-1)
        coords_enc = self.pos_enc(coords, self.global_step)
        # Get a subject latent for each coordinate
        subject_latent = self.subj_latents[subj_idx]
        subject_latent = subject_latent[:, None].tile((1, coords.shape[1], 1))
        x = torch.cat((coords_enc, subject_latent), dim=-1)
        # Forward INR to obtain predicted volume values
        x_ = x.reshape((-1, x.shape[-1]))
        values_pred_ = self.canonical_inr(x_)
        values_pred = values_pred_.reshape((coords.shape[0], coords.shape[1]))
        return values_pred

    def forward_value_deform(self, coords_voxel, values, subject_idx, slice_idx):
        # int_deform_inr = self.int_deform_inrs[subject_idx]
        # int_deform_inr = int_deform_inr.cuda()

        # values = values + int_deform_inr(coords_voxel, slice_idx)
        return values

    def training_step(self, batch):
        opt_inr, opt_aff, opt_deform = self.optimizers()

        (coords_voxel, values, values_dt, aff_params, spacings, needs_flip,
         subject_idx, slice_idx, min_coords, max_coords) = batch

        coords_voxel = coords_voxel + torch.randn(coords_voxel.shape, device=coords_voxel.device) * 5e-2  # TODO
        values_pred, values_pred_dt = self.forward_with_dt(coords_voxel, aff_params, spacings, needs_flip,
                                                           subject_idx, slice_idx, min_coords, max_coords)
        values_deform = self.forward_value_deform(coords_voxel, values, subject_idx, slice_idx)
        loss_recon = self.recon_loss(values_pred, values_deform)
        dt_diff = (values_pred_dt * 1/50) - values_dt
        loss_dt = (dt_diff * dt_diff).mean()

        loss_reg, loss_reg_dict = self.regularization_criterion(subject_idx)
        loss = loss_recon + loss_reg + loss_dt

        opt_inr.zero_grad()
        opt_aff.zero_grad()
        # opt_deform.zero_grad()
        self.manual_backward(loss)
        opt_inr.step()
        opt_aff.step()
        # opt_deform.step()
        self.log_dict({"loss": loss, "loss_recon": loss_recon, **loss_reg_dict, "loss_dt": loss_dt}, prog_bar=True)

    def on_train_epoch_end(self):
        if self.current_epoch >= 0 and self.current_epoch % self.logging_rate == 0:
            self.log_images(0)
            self.log_images(1)
            self.log_images(2)

            self.log_volume(0)
            self.log_volume(1)
            self.log_volume(2)

    @staticmethod
    def min_max_scale(X, x_min, x_max, s_min=-1., s_max=1.):
        return (X - x_min) / (x_max - x_min) * (s_max - s_min) + s_min

    @torch.no_grad()
    def log_images(self, subj_idx, video_duration: float = 4, mode="train"):
        dataset = eval(f"self.trainer.datamodule.{mode}_dset")
        videos = [[] for _ in range(20)]
        preds = [[] for _ in range(20)]
        psnrs = [[] for _ in range(20)]
        for t in range(0, 50, 5):
            images, images_dt, full_indices, coord_min, coord_max, \
                aff_params_padded, spacings_padded, flippings_padded = dataset.load_subject_data(subj_idx, t)
            images, images_dt = images.cuda()[None], images_dt.cuda()[None]
            full_indices, coord_max, coord_min = full_indices.cuda()[None], coord_max.cuda()[None], coord_min.cuda()[None]
            aff_params_padded, spacings_padded, flippings_padded = aff_params_padded.cuda()[None], spacings_padded.cuda()[None], flippings_padded.cuda()[None]
            full_indices = make_coordinate_tensor(images.shape[1:]).cuda()
            slice_idx = full_indices[..., :1]
            voxel_indices = torch.cat((full_indices[..., 1:3], torch.zeros_like(full_indices[..., :1]),
                                       torch.full_like(full_indices[..., :1], t)), dim=-1).float()
            for s in range(images.shape[1]):
                voxel_indices_ = voxel_indices[s].reshape(1, -1, 4)
                slice_idx_ = slice_idx[s].reshape(1, -1, 1)
                pred_vals_, pred_vals_dt_ = self.forward_with_dt(voxel_indices_, aff_params_padded,
                                                                 spacings_padded, flippings_padded,
                                                                 torch.tensor((subj_idx,)), slice_idx_,
                                                                 coord_min, coord_max)
                pred_img = pred_vals_.reshape(images.shape[2:])
                pred_img_dt = pred_vals_dt_.reshape(images.shape[2:])
                psnr_metric = self.reconstruction_metrics.psnr(pred_img, images[0,s])

                psnrs[s].append(psnr_metric.mean().item())
                pred = pred_img.clip(0.0, 1.0)
                pred = (pred * 255).cpu().numpy()
                preds[s].append(pred)

                frame = torch.cat([images[0,s], pred_img, images_dt[0,s], pred_img_dt], 0)
                frame = frame.clip(0.0, 1.0)
                frame = (frame * 255).cpu().numpy().astype(np.uint8)
                videos[s].append(frame)
        videos = [np.stack(v, 0) for v in videos if v]
        psnrs = [np.mean(i) for i in psnrs]
        wandb_videos = [wandb.Video(np.stack([v]*3, 1), fps=max(1, int(50 / video_duration)),
                              caption=f"Slice:{i}, PSNR:{psnrs[i]:.2f}") for i, v in enumerate(videos)]
        subj_path = dataset.data_paths[subj_idx]
        subj_id = Path(subj_path).parent.name
        wandb.log({f"{mode}_videos/subj_{subj_id}": wandb_videos}, step=self.current_epoch)

        images, _, full_indices, coord_max, coord_min, \
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
    def log_volume(self, subj_idx, video_duration: float = 4, mode="train", res=(100, 100, 100)):
        dataset = eval(f"self.trainer.datamodule.{mode}_dset")
        subj_path = dataset.data_paths[subj_idx]
        subj_id = Path(subj_path).parent.name
        coords = torch.stack(torch.meshgrid(*[torch.arange(-.5, .5, 1/i) for i in res]), dim=-1)
        ims = []
        for t in range(0, 50, 5):
            t_norm = t / 50 * 2 - 1
            c = torch.concatenate((coords, torch.full((*res, 1), t_norm)), dim=-1).cuda()

            pred_vals_ = self.forward_inr(c.reshape(1, -1, 4), torch.tensor((subj_idx,)))
            pred_vals_ = pred_vals_.clip(0.0, 1.0)
            pred_val = pred_vals_.reshape(c.shape[:-1])
            pred_val = pred_val.cpu().numpy()
            pred_val = (pred_val*255).astype(np.uint8)
            ims.append(pred_val)
        wandb.log({f"{mode}_volumes/subj_{subj_id}": [wandb.Image(ims[0][...,i]) for i in range(0,100,10)]}, step=self.current_epoch)

        images, _, full_indices, coord_max, coord_min, \
            aff_params_padded, spacings_padded, flippings_padded = dataset.load_subject_data(subj_idx, 0)
        aff_params = aff_params_padded[3][None] + self.aff_deform_params[subj_idx][3].cpu()
        aff = params_to_mat(aff_params, torch.ones_like(spacings_padded[3][None]), flippings_padded[3][None])
        aff = aff[0].cpu().numpy()
        v = np.stack(ims, -1)
        save_dir = Path(f"niftis_vol/{subj_id}")
        save_dir.parent.mkdir(exist_ok=True)
        save_dir.mkdir(exist_ok=True)
        array_to_nifti(str(save_dir / f"{self.current_epoch}_full.nii.gz"), v, aff)


@dataclass
class Params:
    # Epochs -------------------------------------------------------------------
    max_epochs: int = 1_000_000
    num_coords: int = 50_000
    batch_size: int = 4
    check_val_every_n_epoch: int = 1000000000
    logging_rate: int = 200
    # Model -------------------------------------------------------------------
    num_hidden_layers: int = 16
    enc_num_hidden_layers: int = 4
    hidden_size: int = 512
    latent_size: int = 512
    deform_latent_size: int = 16
    deform_num_hidden_layers: int = 1
    deform_hidden_size: int = 32
    # Regularization -------------------------------------------------------------------
    weight_reg_inr: float = 1e-5
    weight_reg_aff: float = 1e-4
    weight_reg_lat: float = 1e-4
    weight_reg_deform: float = 0e-2
    weight_reg_deform_lat: float = 1e-2
    # Learning rates -------------------------------------------------------------------
    learning_rate: float = 1e-4
    learning_rate_aff: float = 1e-5
    learning_rate_def: float = 1e-5
    # Positional encoder -------------------------------------------------------------------
    pe_num_frequencies: List[int] = (10,10,10, 6,6)
    pe_anneal_max_iter: int = 5000
    pe_anneal_start_prop: float = 0.2
    pe_freq_scale: float = 1.0


def main(data_dir, wandb_disabled="false"):
    os.environ['WANDB_DISABLED'] = wandb_disabled
    logger = WandbLogger(project="CMR-Align")

    # configure accelerator and devices
    accelerator = "gpu"
    devices = 1  # one GPU only
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    params = Params()
    logger.log_hyperparams(params.__dict__)
    data_module = CMRDataModule(load_la_dir=data_dir, load_sa_dir=data_dir,
                                preprocessed_store_path=r"/home/nil/data/ukbb/cardiac/aligned_h5",
                                batch_size=params.batch_size, num_coords=params.num_coords, num_workers=0)
    data_module.setup(stage="fit")

    coord_size = data_module.get_coord_size()

    model = INR(coord_size=coord_size, num_subjects=data_module.num_train, max_slices=data_module.get_max_slices(), **params.__dict__)

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
    main(r"/home/nil/data/ukbb/cardiac/aligned_subjects")
