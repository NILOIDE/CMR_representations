import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import torch
import lightning.pytorch as pl
import wandb
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import nn
# import pytorch_lightning as pl


from dataloader import CMRDataModule
from encoders import PerceiverEncoder
from layers import Relu
from utils import params_to_mat
from lightning.pytorch.loggers import WandbLogger


class ReconstructionHead(nn.Module):
    def __init__(self, input_size, output_size, **kwargs):
        super(ReconstructionHead, self).__init__()
        self.out_layer = nn.Linear(input_size, output_size)
        self.out_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.out_layer(x)
        out = torch.sigmoid(out)
        return out


class MLP(nn.Module):
    """ Simple decoder that uses a single averaged latent vector in its input to
    condition the way coordinates are processed. """
    LUT_NAME = "mlp"

    def __init__(self, coord_size: int, num_hidden_layers: int, hidden_size: int, out_size: int, **kwargs):
        super(MLP, self).__init__()
        a = [Relu(coord_size, hidden_size)]
        for i in range(num_hidden_layers - 2):
            a.append(Sine(hidden_size, hidden_size))
        a.append(nn.Linear(hidden_size, out_size))
        self.mlp = nn.Sequential(*a)
        self.out_size = out_size

    def forward(self, coord: torch.Tensor) -> torch.Tensor:
        return self.mlp(coord)


class MLPBackbone(nn.Module):
    """ Simple decoder that uses a single averaged latent vector in its input to
    condition the way coordinates are processed. """
    LUT_NAME = "mlp"

    def __init__(self, coord_size: int, num_hidden_layers: int = 4, hidden_size: int = 128, **kwargs):
        super(MLPBackbone, self).__init__()
        a = [Sine(coord_size, hidden_size, kwargs["siren_factor"])]
        for i in range(num_hidden_layers - 1):
            a.append(Sine(hidden_size, hidden_size, kwargs["siren_factor"]))
        self.mlp = nn.Sequential(*a)
        self.out_size = hidden_size

    def forward(self, coord: torch.Tensor) -> torch.Tensor:
        return self.mlp(coord)


class MultiSliceMLP(nn.Module):
    def __init__(self, coord_size: int, num_slices: int, num_hidden_layers: int = 1, hidden_size: int = 128, siren_factor=50.):
        super(MultiSliceMLP, self).__init__()
        self.siren_factor = siren_factor

        in_sizes = [coord_size - 1] + [hidden_size] * num_hidden_layers
        out_sizes = [hidden_size] * num_hidden_layers + [coord_size]
        self.weights = []
        self.biases = []
        for i, o in zip(in_sizes, out_sizes):
            w_range = math.sqrt(6 / i) / self.siren_factor
            self.weights.append(nn.Parameter((w_range + w_range) * torch.rand(size=(num_slices, i, o), device="cpu") - w_range, requires_grad=True))
            self.biases.append(nn.Parameter((w_range + w_range) * torch.rand(size=(num_slices, i, o), device="cpu") - w_range, requires_grad=True))

    def forward(self, coords, slice_indices):
        x = coords[:, None]
        for w, b in zip(self.weights, self.biases):
            x = torch.bmm(x, w[slice_indices]) + b[slice_indices]
        return x


class INR(pl.LightningModule):
    def __init__(self, coord_size: int, num_subjects: int, max_slices: int, **kwargs):
        super(INR, self).__init__()
        self.automatic_optimization = False

        self.coord_size = coord_size
        self.internsity_size = 1
        self.num_subjects = num_subjects
        self.max_slices = max_slices

        self.latent_size = kwargs.get("latent_size", 128)

        self.encoder = PerceiverEncoder(self.coord_size, self.internsity_size, **kwargs)
        self.model = MLPBackbone(self.coord_size + self.encoder.out_size, **kwargs)
        self.recon_layer = ReconstructionHead(self.model.out_size, self.internsity_size)
        self.aff_deform_params = nn.Parameter(torch.zeros((self.num_subjects, self.max_slices, 6),
                                                          dtype=torch.float32, device="cuda:0"))
        self.coord_deform_inrs = {subj_idx: MultiSliceMLP(self.coord_size, self.max_slices,
                                                          num_hidden_layers=kwargs.get("deform_num_hidden_layers", 1))
                                  for subj_idx in range(self.num_subjects)}

        self.recon_loss = torch.nn.MSELoss()
        self.weight_reg_inr = 1e-5
        self.weight_reg_aff = 1e-3
        self.weight_reg_deform = 0.0 #1e-3
        self.weight_reg_enc = 1e-4

    def configure_optimizers(self):
        opt_inr = torch.optim.Adam([*self.model.parameters(),
                                    *self.recon_layer.parameters(), *self.encoder.parameters()], lr=1e-3)
        opt_aff = torch.optim.Adam([self.aff_deform_params], lr=1e-3)
        # opt_deform = torch.optim.Adam([w for inr in self.coord_deform_inrs.values() for w in inr.weights] +
        #                               [b for inr in self.coord_deform_inrs.values() for b in inr.biases], lr=1e-3)
        return opt_inr, opt_aff

    def regularization_criterion(self, subj_idx: torch.Tensor) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        loss_dict = dict()
        loss_reg = torch.zeros((1,), dtype=torch.float32, device=self.device)
        if self.weight_reg_inr:
            loss_reg_inr = sum((p * p).sum() for p in self.model.parameters())
            loss_reg_inr = loss_reg_inr * self.weight_reg_inr
            loss_reg += loss_reg_inr
            loss_dict["loss_reg_inr"] = loss_reg_inr
        if self.weight_reg_enc:
            loss_reg_enc = sum((p * p).sum() for p in self.encoder.parameters())
            loss_reg_enc = loss_reg_enc * self.weight_reg_enc
            loss_reg += loss_reg_enc
            loss_dict["loss_reg_enc"] = loss_reg_enc
        if self.weight_reg_aff:
            loss_reg_aff = nn.functional.mse_loss(self.aff_deform_params[subj_idx],
                                                  torch.zeros_like(self.aff_deform_params[subj_idx]))
            loss_reg_aff = loss_reg_aff * self.weight_reg_aff
            loss_reg += loss_reg_aff
            loss_dict["loss_reg_aff"] = loss_reg_aff
        # if self.weight_reg_deform:  TODO: Needs fixing
        #     loss_reg_c_deform_w = sum((p * p).sum() for p in self.coord_deform_inrs[subj_idx].weights)
        #     loss_reg_c_deform_b = sum((p * p).sum() for p in self.coord_deform_inrs[subj_idx].biases)
        #     loss_reg_c_deform = (loss_reg_c_deform_w + loss_reg_c_deform_b) * self.weight_reg_deform
        #     loss_reg += loss_reg_c_deform
        #     loss_dict["loss_reg_c_deform"] = loss_reg_c_deform
        loss_dict["loss_reg"] = loss_reg
        return loss_reg, loss_dict

    def forward(self, coords_voxel, values, aff_params, spacings, needs_flip,
                subject_idx, slice_idx, min_coords, max_coords):
        world_coords = self.forward_coord_model(coords_voxel, aff_params, spacings, needs_flip,
                                                subject_idx, slice_idx, min_coords, max_coords)
        values_pred = self.forward_inr(world_coords, values)
        return values_pred

    def forward_coord_model(self, coords_voxel, aff_params, spacings, needs_flip,
                subject_idx, slice_idx, min_coords, max_coords):
        # coord_deform_inr = self.coord_deform_inrs[subject_idx]
        # coord_deform_inr = coord_deform_inr.cuda()

        # coords_voxel = coords_voxel + coord_deform_inr(coords_voxel, slice_idx)
        aff_params_deform = aff_params + self.aff_deform_params[subject_idx]
        aff_params_deform_ = aff_params_deform.reshape((-1, aff_params_deform.shape[-1]))
        spacings_ = spacings.reshape((-1, spacings.shape[-1]))
        needs_flip_ = needs_flip.reshape((-1,))
        affines_ = params_to_mat(aff_params_deform_, spacings_, needs_flip_)
        affines = affines_.reshape((aff_params.shape[0], aff_params.shape[1], 4, 4))

        # In order to move coords from voxel space to world space, we need to have them in (x, y, z, 1)
        coords_flat_ = coords_voxel.reshape((-1, coords_voxel.shape[-1])).to(torch.float32)
        time_coord_ = coords_flat_[:, -1:]  # We take out time coordinates. We will replace them back in later.
        spatial_coord_ = torch.cat((coords_flat_[:, :-1], torch.ones_like(time_coord_)), dim=1)  # Moving coords to world space requires (x, y, z, 1)

        # Create indixing tensor to keep track of which batch is each coordinate coming from
        b, _ = torch.meshgrid(torch.arange(0, slice_idx.shape[0]), torch.arange(0, slice_idx.shape[1]))
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

    def forward_inr(self, coords, values):
        # Get a subject latent for each coordinate
        subject_latent = self.encoder(coords, values)
        subject_latent = subject_latent[:, None].tile((1, coords.shape[1], 1))
        x = torch.cat((coords, subject_latent), dim=-1)
        # Forward INR to obtain predicted volume values
        x_ = x.reshape((-1, x.shape[-1]))
        features_ = self.model(x_)
        values_pred_ = self.recon_layer(features_)
        values_pred = values_pred_.reshape((coords.shape[0], coords.shape[1]))
        return values_pred

    def forward_value_deform(self, coords_voxel, values, subject_idx, slice_idx):
        # int_deform_inr = self.int_deform_inrs[subject_idx]
        # int_deform_inr = int_deform_inr.cuda()

        # values = values + int_deform_inr(coords_voxel, slice_idx)
        return values

    def training_step(self, batch):
        opt_inr, opt_aff = self.optimizers()

        coords_voxel, values, aff_params, spacings, needs_flip, subject_idx, slice_idx, min_coords, max_coords = batch

        coords_world = self.forward_coord_model(coords_voxel, aff_params, spacings, needs_flip,
                                                subject_idx, slice_idx, min_coords, max_coords)
        values_pred = self.forward_inr(coords_world, values, subject_idx)
        values_deform = self.forward_value_deform(coords_voxel, values, subject_idx, slice_idx)
        loss_recon = self.recon_loss(values_pred, values_deform)

        loss_reg, loss_reg_dict = self.regularization_criterion(subject_idx)
        loss = loss_recon + loss_reg

        opt_inr.zero_grad()
        opt_aff.zero_grad()
        # opt_deform.zero_grad()
        self.manual_backward(loss)
        opt_inr.step()
        opt_aff.step()
        # opt_deform.step()
        self.log_dict({"loss": loss, "loss_recon": loss_recon, **loss_reg_dict}, prog_bar=True)

    def validation_step(self, batch, batch_idx):

        self.trainer.val_dataloader.dataset.generate_item(batch_idx, )

    @staticmethod
    def min_max_scale(X, x_min, x_max,  s_min=-1, s_max=1):
        return (X - x_min) / (x_max - x_min) * (s_max - s_min) + s_min


@dataclass
class Params:
    check_val_every_n_epoch: int = 10
    num_hidden_layers: int = 4
    enc_num_hidden_layers: int = 4
    deform_num_hidden_layers: int = 4
    hidden_size: int = 64
    max_epochs: int = 1000
    siren_factor: float = 50.0


def main(work_dir, wandb_disabled="true"):
    # os.environ['WANDB_DISABLED'] = wandb_disabled
    logger = WandbLogger(save_dir=work_dir, project="CMR-Align")

    # configure accelerator and devices
    accelerator = "gpu"
    devices = 1  # one GPU only
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    data_module = CMRDataModule(load_la_dir=r"D:\UKBB_subjects", load_sa_dir=r"D:\UKBB_subjects_unaligned",
                                batch_size=4, num_coords=4000, num_workers=0)
    data_module.setup(stage="fit")

    coord_size = data_module.get_coord_size()
    params = Params()

    model = INR(coord_size=coord_size, num_subjects=data_module.num_train, max_slices=data_module.get_max_slices(), **params.__dict__)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{work_dir}/checkpoints/", every_n_epochs=1, save_last=True, verbose=True
    )

    trainer = Trainer(
        default_root_dir=work_dir,
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

    main(work_dir=r"D:\logs_inr", wandb_disabled="true")

    a = MultiSliceMLP(4, 1, 64, 10)
    x = torch.normal(0.0, 1.0, (20, 3))
    i = torch.randint(0, 10, size=(20,))
    a(x, i)
