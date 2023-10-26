import math
import os
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

import torch
import lightning.pytorch as pl
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import nn
from tqdm import tqdm

# import pytorch_lightning as pl


from dataloader import CMRDataModule
from decoders import MLPBackbone, ReconstructionHead, CADecoder
from encoders import PerceiverEncoder
from layers import Sine, Relu
from pos_encoding import PosEncodingNeRFOptimized, PosEncodingGaussian
from utils import params_to_mat, compute_neighbourhood_matrix
from lightning.pytorch.loggers import WandbLogger
import wandb


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
        self.intensity_size = 1
        self.num_subjects = num_subjects
        self.max_slices = max_slices

        self.enc_pos_encoding = PosEncodingGaussian(self.coord_size, self.intensity_size,
                                                    nerf_num_frequencies=[8]*(self.coord_size + self.intensity_size),
                                                    gauss_num_frequencies=(64,),
                                                    freq_scale=[5.0])
        print(self.enc_pos_encoding)
        self.dec_pos_encoding = PosEncodingGaussian(self.coord_size,
                                                    nerf_num_frequencies=[8]*self.coord_size,
                                                    gauss_num_frequencies=(64,),
                                                    freq_scale=[5.0])
        print(self.dec_pos_encoding)

        self.encoder = PerceiverEncoder(self.enc_pos_encoding.out_dim, **kwargs)
        self.decoder = CADecoder(self.dec_pos_encoding.out_dim, self.encoder.out_size, **kwargs)
        self.recon_layer = ReconstructionHead(self.decoder.out_size, self.intensity_size)
        self.aff_deform_params = nn.Parameter(torch.zeros((self.num_subjects, self.max_slices, 6),
                                                          dtype=torch.float32, device="cuda:0"))
        # self.coord_deform_inrs = {subj_idx: MultiSliceMLP(self.coord_size, self.max_slices,
        #                                                   num_hidden_layers=kwargs.get("deform_num_hidden_layers", 1))
        #                           for subj_idx in range(self.num_subjects)}

        self.recon_loss = torch.nn.MSELoss()
        self.weight_reg_inr = 1e-4
        self.weight_reg_enc = 1e-4
        self.weight_reg_aff = 1e-1
        # self.weight_reg_deform = 1e-3
        # self.inference_opt_steps = 100

    def configure_optimizers(self):
        opt_inr = torch.optim.Adam([*self.decoder.parameters(), *self.recon_layer.parameters()], lr=1e-4)
        opt_enc = torch.optim.Adam(self.encoder.parameters(), lr=1e-4)
        opt_aff = torch.optim.Adam([self.aff_deform_params], lr=1e-3)
        # opt_deform = torch.optim.Adam([w for inr in self.coord_deform_inrs.values() for w in inr.weights] +
        #                               [b for inr in self.coord_deform_inrs.values() for b in inr.biases], lr=1e-3)
        return opt_inr, opt_enc, opt_aff

    def regularization_criterion(self, subj_idx: torch.Tensor) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        loss_dict = dict()
        loss_reg = torch.zeros((1,), dtype=torch.float32, device=self.device)
        if self.weight_reg_inr:
            loss_reg_inr = sum((p * p).sum() for p in self.decoder.parameters())
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
                subject_idx, slice_idx, min_coords, max_coords, aff_params_deform=None):
        if aff_params_deform is None:
            assert subject_idx is not None
            aff_params_deform = self.aff_deform_params[subject_idx, slice_idx]
        world_coords = self.forward_coord_model(coords_voxel, aff_params, aff_params_deform, spacings, needs_flip,
                                                slice_idx, min_coords, max_coords)
        values_pred = self.forward_inr(world_coords, values)
        return values_pred

    def forward_coord_model(self, coords_voxel, aff_params, aff_params_deform, spacings, needs_flip,
                            slice_idx, min_coords, max_coords):
        # coords_voxel[:, :, 2] = slice_idx
        # max_values = torch.stack([coords_voxel.amax(dim=1)[:, 0], coords_voxel.amax(dim=1)[:, 1],
        #                           spacings.any(dim=2).sum(dim=1),
        #                           torch.full((coords_voxel.shape[0],), 50, device="cuda")], dim=1)
        # world_coords = coords_voxel / max_values
        # return world_coords

        # TODO: Change to einsum instead of flattening
        # If we want to account for non-rigid deformations, we want to uncomment this
        # coord_deform_inr = self.coord_deform_inrs[subject_idx]
        # coord_deform_inr = coord_deform_inr.cuda()
        # coords_voxel = coords_voxel + coord_deform_inr(coords_voxel, slice_idx)

        # Create affine matrices for each slice from affine params
        # TODO: change from affine params to affine matrix multiplication
        aff_params_deform = aff_params + aff_params_deform
        aff_params_deform_ = aff_params_deform.reshape((-1, aff_params_deform.shape[-1]))
        spacings_ = spacings.reshape((-1, spacings.shape[-1]))
        needs_flip_ = needs_flip.reshape((-1,))
        affines_ = params_to_mat(aff_params_deform_, spacings_, needs_flip_)
        affines = affines_.reshape((aff_params.shape[0], aff_params.shape[1], 4, 4))

        # In order to move coords from voxel space to world space, we need to have them in (x, y, z, 1)
        coords_flat_ = coords_voxel.reshape((-1, coords_voxel.shape[-1])).to(torch.float32)
        time_coord_ = coords_flat_[:, -1:]  # We take out time coordinates. We will replace them back in later.
        spatial_coord_ = torch.cat((coords_flat_[:, :-1], torch.ones_like(time_coord_)), dim=1)  # Moving coords to world space requires (x, y, z, 1)

        # Create indexing tensor to keep track of which batch is each coordinate coming from
        b, _ = torch.meshgrid(torch.arange(0, slice_idx.shape[0]), torch.arange(0, slice_idx.shape[1]))
        # Get affines corresponding to each coordinate
        affines_ = affines[b.reshape(-1), slice_idx.reshape(-1)]
        # Move coordinates to world space
        coords_world_ = torch.bmm(affines_, spatial_coord_[..., None])
        coords_world_ = coords_world_.squeeze(-1)
        # Add time coordinate back
        coords_world_[:, -1:] = time_coord_

        # normalize world coordinates to [-1, 1]
        norm_coords_ = self.min_max_scale(coords_world_, min_coords[b.reshape(-1)], max_coords[b.reshape(-1)],
                                          s_min=-1, s_max=1)
        norm_coords = norm_coords_.reshape(coords_voxel.shape)

        return norm_coords

    def forward_inr(self, coords, values):
        enc_x = self.enc_pos_encoding(coords, values)
        dec_x = self.dec_pos_encoding(coords)
        # Get a subject latent for each coordinate
        subject_latent = self.encoder(enc_x)
        # Forward INR to obtain predicted volume values
        features = self.decoder(dec_x, subject_latent)
        features_ = features.reshape((-1, features.shape[-1]))
        values_pred_ = self.recon_layer(features_)
        values_pred = values_pred_.reshape((coords.shape[0], coords.shape[1]))
        return values_pred

    def forward_value_deform(self, coords_voxel, values, subject_idx, slice_idx):
        # int_deform_inr = self.int_deform_inrs[subject_idx]
        # int_deform_inr = int_deform_inr.cuda()

        # values = values + int_deform_inr(coords_voxel, slice_idx)
        return values

    def training_step(self, batch):
        opt_inr, opt_enc, opt_aff = self.optimizers()

        coords_voxel, values, aff_params, spacings, needs_flip, subject_idx, slice_idx, min_coords, max_coords, _ = batch

        aff_params_deform = self.aff_deform_params[subject_idx]
        coords_world = self.forward_coord_model(coords_voxel, aff_params, aff_params_deform, spacings, needs_flip,
                                                slice_idx, min_coords, max_coords)
        values_pred = self.forward_inr(coords_world, values)
        values_deform = self.forward_value_deform(coords_voxel, values, subject_idx, slice_idx)
        loss_recon = self.recon_loss(values_pred, values_deform[..., 0])

        loss_reg, loss_reg_dict = self.regularization_criterion(subject_idx)
        loss = loss_recon + loss_reg

        opt_inr.zero_grad()
        opt_enc.zero_grad()
        opt_aff.zero_grad()
        # opt_deform.zero_grad()
        self.manual_backward(loss)
        opt_inr.step()
        opt_enc.step()
        opt_aff.step()
        # opt_deform.step()
        self.log_dict({"loss": loss, "loss_recon": loss_recon, **loss_reg_dict}, prog_bar=True)

    # @torch.enable_grad()
    def validation_step(self, batch, batch_idx):
        coords_voxel, values, aff_params, spacings, needs_flip, subject_idx, slice_idx, min_coords, max_coords, _ = batch
        # We will only be optimizing the affine params. We ignore the INR and encoder optimizer.
        # self.train()
        # _, opt_aff = self.optimizers()
        aff_params_deform = nn.Parameter(torch.zeros((1, self.max_slices, 6), dtype=torch.float32, device="cuda:0"))
        # start_loss = self.validation_loss(batch_idx, aff_params_deform)
        # self.log("validation/start_loss", start_loss)
        # start_recon = self.val_image(batch_idx, aff_params_deform)
        # self.logger.log_image("validation/start_recon", start_recon)
        #
        # for i in range(self.inference_opt_steps):
        #     batch = self.trainer.datamodule.val_dset.generate_item(batch_idx, 1.0)
        #     coords_voxel, values, aff_params, spacings, needs_flip, subject_idx, slice_idx, min_coords, max_coords = batch
        #     coords_voxel = coords_voxel.cuda()[None]
        #     values = values.cuda()[None]
        #     aff_params = aff_params.cuda()[None]
        #     spacings = spacings.cuda()[None]
        #     needs_flip = needs_flip.cuda()[None]
        #     subject_idx = subject_idx.cuda()[None]
        #     slice_idx = slice_idx.cuda()[None]
        #     min_coords = min_coords.cuda()[None]
        #     max_coords = max_coords.cuda()[None]
        #
        #     values_pred = self.forward(coords_voxel, values, aff_params, spacings, needs_flip, None,
        #                                 slice_idx, min_coords, max_coords, aff_params_deform=aff_params_deform)
        #     values_deform = self.forward_value_deform(coords_voxel, values, subject_idx, slice_idx)
        #     loss_recon = self.recon_loss(values_pred, values_deform)
        #     opt_aff.zero_grad()
        #     self.manual_backward(loss_recon)
        #     opt_aff.step()

        # end_loss = self.validation_loss(batch_idx, self.aff_deform_params[batch_idx])
        # self.log("validation/end_loss", end_loss)

        with torch.no_grad():
            # Log point cloud
            self.log_val_point_cloud(int(subject_idx[0]), draw_seg=True)
            # Log individual 2D slices
            self.log_val_slices(int(subject_idx[0]), aff_params_deform)

    # def validation_loss(self, batch_idx, aff_params_deform):
    #     batch = self.trainer.datamodule.val_dset.generate_item(batch_idx, 1.0)
    #     coords_voxel, values, aff_params, spacings, needs_flip, subject_idx, slice_idx, min_coords, max_coords = batch
    #     coords_voxel = coords_voxel.cuda()[None]
    #     values = values.cuda()[None]
    #     aff_params = aff_params.cuda()[None]
    #     spacings = spacings.cuda()[None]
    #     needs_flip = needs_flip.cuda()[None]
    #     subject_idx = subject_idx.cuda()[None]
    #     slice_idx = slice_idx.cuda()[None]
    #     min_coords = min_coords.cuda()[None]
    #     max_coords = max_coords.cuda()[None]
    #
    #     # values_pred = self.forward(coords_voxel, values, aff_params, spacings, needs_flip, None,
    #     #                            slice_idx, min_coords, max_coords, aff_params_deform=aff_params_deform)
    #     # values_deform = self.forward_value_deform(coords_voxel, values, subject_idx, slice_idx)
    #     # loss_recon = self.recon_loss(values_pred, values_deform)
    #     return 0.0

    def log_val_slices(self, subj_idx, aff_params_deform):
        # Visualize slices
        frame_idx = 0
        batch = self.trainer.datamodule.val_dset.load_subject_data(subj_idx, frame_idx=frame_idx)
        batch = [i.cuda() for i in batch]
        img_values, padding_mask, indices, min_coords, max_coords, \
            aff_params_padded, spacings_padded, needs_flip_padded, _ = batch
        slices = []
        for i in range(0, img_values.shape[0]):
            x, y = torch.where(padding_mask[i])
            img = img_values[i, :x.max().item(), :y.max().item()]
            indices = torch.meshgrid(torch.arange(0, img.shape[0]), torch.arange(0, img.shape[1]), torch.tensor([0]), torch.tensor([frame_idx]))
            indices_ = torch.stack([i.reshape(-1) for i in indices], dim=1).cuda()
            slice_pred_ = self.forward(indices_[None], img.reshape((-1, 1))[None], aff_params_padded[None],
                                       spacings_padded[None], needs_flip_padded[None], None,
                                       torch.full((1, indices_.shape[0],), i, device="cuda"), min_coords[None], max_coords[None],
                                       aff_params_deform=aff_params_deform)
            slice_pred = slice_pred_.reshape(img.shape)
            diff = (img-slice_pred).abs()
            vis = torch.cat((img, slice_pred, diff), dim=1)
            vis = (vis.clamp(min=0.0, max=1.0) * 255).to(torch.uint8)
            vis = vis.detach().cpu().numpy()
            slices.append(vis)
        self.logger.log_image(f"validation/end_recon_{subj_idx}", slices)


    def log_val_point_cloud(self, subj_idx: int, draw_seg: bool = True,
                            seg_dist_thresh: float = 0.15, coord_split: int = 500):
        # Log cloud point
        batch = self.trainer.datamodule.val_dset.generate_item(subj_idx, 1.0, frame=0)
        coords_voxel, values, aff_params, spacings, needs_flip, _, slice_idx, min_coords, max_coords, segs = batch
        coords_voxel = coords_voxel.cuda()[None]
        values = values.cuda()
        aff_params = aff_params.cuda()[None]
        spacings = spacings.cuda()[None]
        needs_flip = needs_flip.cuda()[None]
        slice_idx = slice_idx.cuda()[None]
        min_coords = min_coords.cuda()[None]
        max_coords = max_coords.cuda()[None]
        segs = segs.cuda()
        world_coords = self.forward_coord_model(coords_voxel, aff_params, torch.zeros_like(aff_params),
                                                spacings, needs_flip, slice_idx, min_coords, max_coords)
        del min_coords, max_coords, slice_idx, needs_flip, spacings, aff_params
        world_coords = world_coords[0, ..., :-1]
        values = (values.clamp(min=0.0, max=1.0) * 255)

        selection = []
        for i in tqdm(range(0, coord_split)):
            idx1 = int(world_coords.shape[0] * (i / coord_split))
            idx2 = int(world_coords.shape[0] * (i + 1) / coord_split)
            s = compute_neighbourhood_matrix(world_coords[idx1:idx2], world_coords[segs.any(dim=-1)],
                                             dist_thresh=seg_dist_thresh)
            selection.append(s)
        del s
        selection = torch.cat(selection)
        # selection = torch.logical_and(selection, values[..., 0] > 70)
        values = torch.cat([values[selection]] * 3, dim=-1)
        if draw_seg:
            segs = torch.cat([segs[selection]] * 3, dim=-1)
            values = torch.where(segs == 1, torch.tensor((255, 0, 0)).cuda(), values)
            values = torch.where(segs == 2, torch.tensor((0, 255, 0)).cuda(), values)
            values = torch.where(segs == 3, torch.tensor((0, 255, 255)).cuda(), values)

        cloud_point = torch.cat([world_coords[selection], values], dim=-1).detach().cpu().numpy()
        wandb.log({f"validation/img_cloud_{subj_idx}": wandb.Object3D(cloud_point)})
        del selection, cloud_point

    @staticmethod
    def min_max_scale(X, x_min, x_max,  s_min=-1, s_max=1):
        return (X - x_min) / (x_max - x_min) * (s_max - s_min) + s_min


@dataclass
class Params:
    check_val_every_n_epoch: int = 10
    num_hidden_layers: int = 4
    enc_num_hidden_layers: int = 4
    deform_num_hidden_layers: int = 4
    hidden_size: int = 128
    max_epochs: int = 1_000_000
    siren_factor: float = 30.0


def main(work_dir, wandb_disabled="true"):
    # os.environ['WANDB_DISABLED'] = wandb_disabled
    logger = WandbLogger(save_dir=work_dir, project="CMR-Align")

    # configure accelerator and devices
    accelerator = "gpu"
    devices = 1  # one GPU only
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    data_module = CMRDataModule(load_la_dir=r"D:\UKBB_subjects", load_sa_dir=r"D:\UKBB_subjects_unaligned",
                                batch_size=1, num_coords=20000, num_workers=0)
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
        log_every_n_steps=params.check_val_every_n_epoch,
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':

    main(work_dir=r"D:\logs_inr", wandb_disabled="true")

    a = MultiSliceMLP(4, 1, 64, 10)
    x = torch.normal(0.0, 1.0, (20, 3))
    i = torch.randint(0, 10, size=(20,))
    a(x, i)
