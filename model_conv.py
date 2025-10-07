import math
import shutil
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Dict, Optional, List

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

    def forward(self,
                imgs: torch.Tensor,
                num_subj_slices: torch.Tensor,
                coords_voxel: torch.Tensor,
                aff_params: torch.Tensor,
                spacings: torch.Tensor,
                needs_flip: torch.BoolTensor,
                slice_idx: torch.LongTensor,
                min_coords: torch.Tensor,
                max_coords: torch.Tensor,
                subject_idx: Optional[torch.LongTensor] = None,
                latent_params: Optional[torch.Tensor] = None,
                aff_def_params: Optional[torch.Tensor] = None,
                return_deriv: bool = False,
                return_hessian: bool = False,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        latent_params, aff_def_params = self.get_params(imgs, num_subj_slices, subject_idx, latent_params=latent_params, aff_def_params=aff_def_params)
        world_coords = self.forward_coord_model(coords_voxel, aff_params, spacings, needs_flip,
                                                slice_idx, min_coords, max_coords, aff_def_params)
        seg_pred, values_pred  = self.forward_inr(world_coords, latent_params)
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

    def get_params(self,
                   imgs: torch.Tensor,
                   num_subj_slices: torch.Tensor,
                   subj_idx: Optional[torch.Tensor]=None,
                   latent_params: Optional[torch.Tensor]=None,
                   aff_def_params: Optional[torch.Tensor]=None) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        # Get a subject latent for each coordinate
        assert subj_idx is not None or latent_params is not None
        assert subj_idx is not None or aff_def_params is not None
        aff_def_params = self.aff_deform_params[subj_idx] if aff_def_params is None else aff_def_params
        if latent_params is None:
            latent_params = self.encoder(imgs, aff_def_params, num_subj_slices, self.global_step)
        return latent_params, aff_def_params

    def forward_inr(self,
                    coords: torch.Tensor,
                    subject_latent: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        # Make time dim cyclical
        coords = torch.cat((coords[..., :3],
                            torch.cos(coords[...,-1:] * torch.pi),
                            torch.sin(coords[...,-1:] * torch.pi)), dim=-1)
        coords_enc = self.pos_enc(coords, self.global_step)
        subject_latent = subject_latent[:, None].tile((1, coords.shape[1], 1))
        x = torch.cat((coords_enc, subject_latent), dim=-1)
        # Forward INR to obtain predicted volume values
        x_ = x.reshape((-1, x.shape[-1]))
        values_pred_ = self.canonical_inr(x_)
        seg_pred = values_pred_[:, 1:].reshape((coords.shape[0], coords.shape[1], 4))
        values_pred = values_pred_[:, 0].reshape((coords.shape[0], coords.shape[1]))
        return seg_pred, values_pred

    def forward_intensity_params(self,
                                 subject_idx: Optional[torch.Tensor],
                                 slice_idx: Optional[torch.Tensor],
                                 inten_scale_params: Optional[torch.Tensor] = None) -> torch.Tensor:
        inten_scale_params = self.intensity_scale_params if inten_scale_params is None else inten_scale_params
        subject_idx_tile = subject_idx[:,None].tile((1, slice_idx.shape[1]))
        inten_scale_params_ = inten_scale_params[subject_idx_tile.flatten(), slice_idx.flatten()]
        int_deform_ = torch.tanh(inten_scale_params_) * (self.int_scale_range / 2)
        return int_deform_.reshape(*slice_idx.shape[:2])

    def apply_intensity_scaling(self,
                                intensities: torch.Tensor,
                                subject_idx: torch.Tensor,
                                slice_idx: torch.Tensor,
                                intens_scale_params: Optional[torch.Tensor] = None,
                                inverse: bool = False):
        intens_scale = self.forward_intensity_params(subject_idx, slice_idx, intens_scale_params)
        if not inverse:
            return intensities * (1 + intens_scale)
        else:
            return intensities / (1 + intens_scale)

    def training_step(self, batch):
        opt_inr, opt_deform, opt_inten = self.optimizers()

        (imgs, num_subj_slices, coords_voxel, values, values_dt, segs, gt_avail,
         aff_params, spacings, needs_flip, subject_idx, slice_idx, min_coords, max_coords) = batch

        coords_voxel = coords_voxel + torch.randn(coords_voxel.shape, device=coords_voxel.device) * 5e-2  # TODO
        seg_pred, values_pred, values_pred_d, values_pred_dd = self.forward(imgs, num_subj_slices,
                                                                            coords_voxel, aff_params,
                                                                            spacings, needs_flip,
                                                                            slice_idx,
                                                                            min_coords, max_coords,
                                                                            subject_idx=subject_idx,
                                                                            return_deriv=self.supervise_deriv,
                                                                            return_hessian=self.supervise_hess,)
        values_deform = self.apply_intensity_scaling(values, subject_idx, slice_idx)
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
        log_name = "train_metrics"
        self.log_dict({f"{log_name}/{k}": v for k, v in
                       {"loss": loss, "loss_recon": loss_recon, "loss_seg": loss_seg
                        }.items()}, prog_bar=True)
        self.log_dict({f"{log_name}/{k}": v for k, v in
                       {"loss_dt": loss_dt, "loss_ddt": loss_ddt,
                        "dice_BG": dice_per_class[0], "dice_FG": dice_per_class[1:].mean(),
                        "dice_LV": dice_per_class[1], "dice_MYO": dice_per_class[2],
                        "dice_RV": dice_per_class[3], **loss_reg_dict,
                        }.items()}, prog_bar=False)

    def inference(self, subj_idxs, dset: CardiacUKBBFullImage, dset_str: str = 'val'):
        instance_dset = CardiacUKBBValidationFullImage([dset.data_paths[i] for i in subj_idxs], dset.max_slices, dset.max_slice_shape, dset.num_coords)
        instance_dloader = DataLoader(instance_dset, batch_size=len(subj_idxs), shuffle=False,
                                      num_workers=len(subj_idxs), pin_memory=True,
                                      persistent_workers=len(subj_idxs)>0)
        inf_subj_latents = None
        inf_aff_def_params = nn.Parameter(torch.zeros((len(subj_idxs), self.max_slices, 6),
                                                         dtype=torch.float32, device="cuda"), requires_grad=True)
        inf_intensity_scale_params = nn.Parameter(torch.randn((len(subj_idxs), self.max_slices, 1),
                                                              dtype=torch.float32, device="cuda") * 1e-3, requires_grad=True)
        opt_affine_def = torch.optim.Adam([inf_aff_def_params], lr=1e-4)
        opt_intensity_def = torch.optim.Adam([inf_intensity_scale_params], lr=1e-4)
        metrics = defaultdict(list)
        for i in tqdm.tqdm(range(2000), desc=f"Performing inference for subjects {subj_idxs}"):
            for batch in instance_dloader:
                batch = (b.cuda() for b in batch)
                (imgs, num_subj_slices, coords_voxel, values, values_dt, segs, gt_avail,
                 aff_params, spacings, needs_flip, subject_idx, slice_idx, min_coords, max_coords) = batch
                opt_intensity_def.zero_grad()
                B, S, H, W, T = imgs.shape
                imgs_ = imgs.reshape((B, -1))
                slice_idx_tile_ = slice_idx.tile(T).reshape(B, -1, 1)
                imgs_deform_ = self.apply_intensity_scaling(imgs_, subject_idx, slice_idx_tile_,
                                                             intens_scale_params=inf_intensity_scale_params)
                imgs_deform = imgs_deform_.reshape(imgs.shape)
                # Make predictions for this batch
                inf_subj_latents = self.encoder(imgs, inf_aff_def_params, num_subj_slices, self.global_step)
                coords_voxel_ = coords_voxel.reshape(B, -1, coords_voxel.shape[-1])
                slice_idx_ = slice_idx.reshape(B, -1, 1)
                seg_pred, values_pred, values_pred_d, values_pred_dd = self.forward(imgs_deform, num_subj_slices,
                                                                                    coords_voxel_, aff_params,
                                                                                    spacings, needs_flip,
                                                                                    slice_idx_,
                                                                                    min_coords, max_coords,
                                                                                    subject_idx=subject_idx,
                                                                                    latent_params=inf_subj_latents,
                                                                                    aff_def_params=inf_aff_def_params,
                                                                                    return_deriv=self.supervise_deriv,
                                                                                    return_hessian=self.supervise_hess,)
                seg_pred = seg_pred.reshape(B, S, H, W, -1)
                values_pred = values_pred.reshape(B, S, H, W)
                if values_pred_d is not None:
                    values_pred_d = values_pred_d.reshape(B, S, H, W, -1)
                if values_pred_dd is not None:
                    values_pred_dd = values_pred_dd.reshape(B, S, H, W, -1)
                loss_recon = self.psnr_loss(values_pred, values)
                reg_inr_loss, reg_inr_dict = self.loss_reg_inr_params(self.canonical_inr.parameters(), self.weight_reg_inr)
                reg_aff_loss, reg_aff_dict = self.loss_reg_aff_params(inf_aff_def_params[subject_idx], self.weight_reg_aff, num_subj_slices=num_subj_slices)
                reg_lat_loss, reg_lat_dict = self.loss_reg_latent_params(inf_subj_latents, self.weight_reg_lat)
                reg_int_scale_loss, reg_int_scale_dict = self.loss_reg_int_scale_params(inf_intensity_scale_params[subject_idx], self.weight_intensity_scale, num_subj_slices=num_subj_slices)
                # reg_c_deform_loss, reg_c_deform_dict = self.loss_reg_deform_inr_params(coord_deform_inr.parameters(), self.weight_reg_deform)
                # reg_c_deform_lat_loss, reg_c_deform_lat_dict = self.loss_reg_deform_lat_params(deform_latents[subj_idx], self.weight_reg_deform_lat)
                reg_loss = reg_inr_loss + reg_aff_loss + reg_lat_loss + reg_int_scale_loss #+ reg_c_deform_loss + reg_c_deform_lat_loss
                reg_dict = {f"loss_reg": reg_loss,
                            **reg_inr_dict, **reg_aff_dict, **reg_lat_dict,
                            **reg_int_scale_dict,}
                loss_dt = torch.tensor((0.0,), device=loss_recon.device)  # self.psnr_loss(values_pred_d[...,-1:], values_dt*50) * self.weight_loss_deriv
                loss_ddt = torch.tensor((0.0,), device=loss_recon.device)  # self.psnr_loss(values_pred_dd[..., -1:], values_ddt*50) * self.weight_loss_hess
                loss = loss_recon + reg_loss + loss_dt + loss_dt
                # Backprop only image-based losses and regularization losses (we assume we don't have seg GT)
                loss.backward()
                opt_affine_def.step()
                opt_intensity_def.step()

                seg_pred, segs = torch.softmax(seg_pred, -1) * gt_avail[..., None], segs * gt_avail[..., None]
                loss_seg_per_class = self.seg_loss(seg_pred.moveaxis(-1, 1), segs.moveaxis(-1, 1)).mean(-1).mean(0)
                dice_per_class = 1 - loss_seg_per_class
                loss_seg = (loss_seg_per_class * self.class_weight.to(loss_seg_per_class.device)).mean() * self.weight_loss_seg

                metrics['step'].append(i)
                metrics['loss_recon'].append(loss_recon.item())
                metrics['loss_dt'].append(loss_dt.item())
                metrics['loss_ddt'].append(loss_ddt.item())
                [metrics[k].append(v.item()) for k, v in reg_dict.items()]
                metrics['loss_seg'].append(loss_seg.item())
                metrics['dice_BG'].append(dice_per_class[0].item())
                metrics['dice_FG'].append(dice_per_class[1:].mean().item())
                metrics['dice_LV'].append(dice_per_class[1].item())
                metrics['dice_MYO'].append(dice_per_class[2].item())
                metrics['dice_RV'].append(dice_per_class[3].item())

        log_name = f"{dset_str}_inf_metrics"
        for k, v in metrics.items():
            if k == 'step':
                continue
            if k not in self.inference_metrics:
                # If no previous dataframe, create dataframe with column for this epoch
                self.inference_metrics[k] = pd.DataFrame(v, columns=[f"{k}_{self.current_epoch}"])
            else:
                # Add column for this epoch
                self.inference_metrics[k][f"{k}_{self.current_epoch}"] = v
            # wandb.log({f'{log_name}/{str(subj_idxs)}_inf_table_{k}': wandb.Table(dataframe=self.inference_metrics[k])})
            plot = wandb.plot.line_series(xs=metrics['step'],
                                          ys=[list(self.inference_metrics[k][i]) for i in list(self.inference_metrics[k].columns)],
                                          keys=list(self.inference_metrics[k].columns),
                                          title=f"{k} metric over inference optimization",
                                          xname="Optimization steps")
            wandb.log({f'{log_name}/{str(subj_idxs)}_inf_metric_{k}': plot})
        return inf_subj_latents, inf_aff_def_params, inf_intensity_scale_params

    @torch.no_grad()
    def log_images(self,
                   subj_idx: int,
                   latent_params: Optional[torch.Tensor] = None,
                   aff_def_params: Optional[torch.Tensor] = None,
                   intens_scale_params: Optional[torch.Tensor] = None,
                   video_duration: float = 4,
                   mode="train"):
        dataset = eval(f"self.trainer.datamodule.{mode}_dset")
        subj_path = dataset.data_paths[subj_idx]
        subj_id = Path(subj_path).parent.name
        videos = [[] for _ in range(20)]
        preds = [[] for _ in range(20)]
        segs = [[] for _ in range(20)]
        psnrs = [[] for _ in range(20)]
        dices = [[] for _ in range(20)]
        for t in tqdm.tqdm(range(0, 50, 5), desc=f"Logging slices subj  {subj_idx} (id:{subj_id})"):
            images, num_subj_slices, images_t, images_dt, seg_argmax, _, full_indices, coord_min, coord_max, \
                aff_params_padded, spacings_padded, flippings_padded = dataset.load_subject_data(subj_idx, t)
            images, num_subj_slices = images.cuda()[None], num_subj_slices.cuda()[None]
            B, S, H, W, T = images.shape
            images_t, images_dt, seg_argmax = images_t.cuda()[None], images_dt[...,-1].cuda()[None], seg_argmax.cuda()[None]
            full_indices, coord_max, coord_min = full_indices.cuda()[None], coord_max.cuda()[None], coord_min.cuda()[None]
            aff_params_padded, spacings_padded, flippings_padded = aff_params_padded.cuda()[None], spacings_padded.cuda()[None], flippings_padded.cuda()[None]
            full_indices = make_coordinate_tensor(images.shape[1:]).cuda()
            slice_idx = full_indices[..., :1]
            voxel_indices = torch.cat((full_indices[..., 1:3], torch.zeros_like(full_indices[..., :1]),
                                       torch.full_like(full_indices[..., :1], t)), dim=-1).float()
            for s in range(num_subj_slices.item()):
                voxel_indices_ = voxel_indices[s, ..., t, :].reshape(1, -1, 4)
                slice_idx_ = slice_idx[s, ..., t, :].reshape(1, -1, 1)
                with torch.enable_grad():
                    pred_seg_, pred_vals_, pred_vals_d_, pred_vals_dd_ = self.forward(
                        images, num_subj_slices,
                        voxel_indices_, aff_params_padded,
                        spacings_padded, flippings_padded,
                        slice_idx_,
                        coord_min, coord_max,
                        subject_idx=torch.LongTensor((subj_idx,)),
                        latent_params=latent_params,
                        aff_def_params=aff_def_params,
                        return_deriv=True,
                        return_hessian=True)
                pred_seg_, pred_vals_, pred_vals_d_, pred_vals_dd_ = pred_seg_.detach(), pred_vals_.detach(), pred_vals_d_.detach(), pred_vals_dd_.detach()
                pred_img = pred_vals_.reshape(H, W)
                dummy_subj_idx = torch.LongTensor((0,))
                intens_scale_ = self.forward_intensity_params(dummy_subj_idx, slice_idx_, intens_scale_params)
                intens_scale = intens_scale_.reshape(pred_img.shape)
                pred_img_dt = pred_vals_d_.reshape(H, W, pred_vals_d_.shape[-1])[...,-1]
                # pred_img_ddt = pred_vals_dd_.reshape(B, H, W, *pred_vals_dd_.shape[2:])[...,-1]
                pred_img_deform_ = self.apply_intensity_scaling(pred_vals_, dummy_subj_idx, slice_idx_,
                                                                intens_scale_params=intens_scale_params, inverse=True)
                pred_img_deform = pred_img_deform_.reshape(pred_img.shape)
                psnr_metric = kornia.metrics.psnr(pred_img, images[0,s,...,t], max_val=1.0)
                psnrs[s].append(psnr_metric.mean().detach().cpu().item())
                pred = pred_img.clip(0.0, 1.0)
                pred = (pred * 255).cpu().numpy()
                preds[s].append(pred)
                pred_seg = pred_seg_.reshape(H, W, pred_seg_.shape[-1])
                pred_seg_argmax = pred_seg.argmax(-1)
                segs[s].append(pred_seg_argmax)
                seg_gt = to_1hot(seg_argmax[0,s].reshape(-1), pred_seg.shape[-1]).reshape(pred_seg.shape)
                pred_seg_1hot = to_1hot(pred_seg_argmax.reshape(-1), pred_seg.shape[-1]).reshape(pred_seg.shape)
                dice = 1 - self.seg_loss(pred_seg_1hot[None].moveaxis(-1,1), seg_gt[None].moveaxis(-1,1)).mean(0).squeeze()
                dices[s].append(dice.detach().cpu())

                # Image
                img = torch.stack([torch.cat([images[0,s,...,t], pred_img], 0)]*3, 0)
                # Segmentation
                segs_argmax = torch.cat([seg_argmax[0,s], pred_seg_argmax], 0)
                seg_frames = torch.stack([torch.cat([images[0,s,...,t], images[0,s,...,t]], 0)]*3, 0)
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
                img_ddt = torch.stack([torch.cat([pred_img_deform.abs(), intens_scale], 0)]*3, 0)

                frame_img = torch.cat([img, seg_frames], 1)
                frame_d = torch.cat([img_dt, img_ddt], 1)
                frame = torch.cat([frame_img, frame_d], 2)
                frame = frame.clip(0.0, 1.0)
                frame = (frame * 255).cpu().numpy().astype(np.uint8)
                videos[s].append(frame)
        videos = [np.stack(v, 0) for v in videos if v]
        psnrs = [np.mean(i) for i in psnrs if i]
        psnr_strings = [f"PSNR:{i:.1f}" for i in psnrs]
        dices = [torch.stack(d, 0).mean(0) for i, d in enumerate(dices) if d]
        dices_strings = [f"Dice:" + f"{d[1].item():.2f}, " + f"{d[2].item():.2f}, " + f"{d[3].item():.2f}"
                         if i >= 3 else "Dice: -, -, -" for i, d in enumerate(dices)]
        wandb_videos = [wandb.Video(v, fps=max(1, int(50 / video_duration)),
                                    caption=f"Slice:{i}, {psnr_strings[i]}  {dices_strings[i]}") for i, v in enumerate(videos)]
        wandb.log({f"{mode}_videos/subj_{subj_id}": wandb_videos}, step=self.current_epoch)

        if self.current_epoch > 0 and self.current_epoch % (self.logging_rate * 5) == 0:
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
    def log_volume(self,
                   subj_idx: int,
                   latent_params: Optional[torch.Tensor] = None,
                   aff_def_params: Optional[torch.Tensor] = None,
                   video_duration: float = 4,
                   mode="train",
                   res=(200, 200, 200)):
        dataset = eval(f"self.trainer.datamodule.{mode}_dset")
        subj_path = dataset.data_paths[subj_idx]
        subj_id = Path(subj_path).parent.name
        coords = torch.stack(torch.meshgrid(*[torch.linspace(self.norm_min, self.norm_max, i) for i in res]), dim=-1)
        images, num_subj_slices, _, _, _, _, _, _, _, aff_params_padded, _, _ = dataset.load_subject_data(subj_idx, 0)
        images, num_subj_slices, aff_params_padded = images[None].cuda(), num_subj_slices[None].cuda(), aff_params_padded[None].cuda()
        ims = []
        segs = []
        latent_params, aff_def_params = self.get_params(images, num_subj_slices, torch.LongTensor((subj_idx,)),
                                                        latent_params=latent_params, aff_def_params=aff_def_params)
        for t in tqdm.tqdm(range(0, 50, 5), desc=f"Logging volume subj {subj_idx} (id:{subj_id})"):
            t_norm = t / 50
            c = torch.concatenate((coords, torch.full((*res, 1), t_norm)), dim=-1).cuda()
            z_im_slices = []
            z_seg_slices = []
            for z in range(res[-1]):
                c_z = c[:,:,z:z+1]
                pred_seg_, pred_vals_ = self.forward_inr(c_z.reshape(1, -1, 4), latent_params)
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
            gt_images, _, gt_segs, _, full_indices, coord_max, coord_min, \
                aff_params_padded, spacings_padded, flippings_padded, _ = dataset.load_subject_data(subj_idx, 0)
            # Use the coordinate system of the top-most SA slice (ie. 3)
            aff_params = aff_params_padded[3][None] + aff_def_params[0, 3].cpu()
            aff = params_to_mat(aff_params, torch.ones_like(spacings_padded[3][None]), flippings_padded[3][None])
            aff = aff[0].cpu().numpy()
            save_dir = Path(f"niftis_vol/{subj_id}/{self.current_epoch}")
            save_dir.parent.parent.mkdir(exist_ok=True)
            save_dir.parent.mkdir(exist_ok=True)
            save_dir.mkdir(exist_ok=True)
            array_to_nifti(str(save_dir / f"full.nii.gz"), ims, aff)
            array_to_nifti(str(save_dir / f"full_seg.nii.gz"), segs, aff)
            save_dir_gt = save_dir.parent / "gt"
            save_dir_gt.mkdir(exist_ok=True)
            for i in range(gt_images.shape[0]):
                aff = params_to_mat(aff_params, torch.ones_like(spacings_padded[i][None]), flippings_padded[i][None])
                aff = aff[0].cpu().numpy()
                array_to_nifti(str(save_dir_gt / f"slice{i}.nii.gz"), gt_images[i, ..., None, None].numpy(), aff)
                array_to_nifti(str(save_dir_gt / f"slice{i}_seg.nii.gz"), gt_segs[i, ..., None, None].numpy(), aff)

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
        for i in range(5):
            try:
                save_dir = Path(f"temp_mesh_files/{subj_id}")
                save_dir.parent.mkdir(exist_ok=True)
                save_dir.mkdir(exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=str(save_dir.absolute()), suffix='.html', delete=False) as f:
                    plot = create_meshplot_visualization(meshes, f.name)
                    temp_file_path = f.name
                    with open(f.name, 'r') as html_file:
                        wandb.log({f"{mode}_mesh/subj_{subj_id}": wandb.Html(html_file.read())})
                shutil.rmtree(save_dir.parent)
                break
            except Exception as e:
                print("Error while logging mesh:")
                traceback.print_exc()
