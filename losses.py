from typing import Optional

import monai.metrics
import torch
from torch import nn
from monai.losses import DiceLoss, GeneralizedDiceLoss
from monai.metrics import PSNRMetric, SSIMMetric, DiceMetric, HausdorffDistanceMetric


class SegmentationCriterion(nn.Module):
    def __init__(self, **kwargs):
        super(SegmentationCriterion, self).__init__()
        self.criterion = DiceLoss(reduction="none")
        self.seg_weights = torch.tensor(kwargs.get("seg_weights", (1.0, 1.0, 1.0, 1.0)))

    def forward(self, pred: torch.Tensor, target: torch.Tensor, slice_mask: Optional[torch.Tensor] = None):
        assert len(pred.shape) == 6, f"Should be (b, s, t, h, w, c). Received: {pred.shape}"

        # First three slice are LA, we don't have seg GT for LA
        pred, target, slice_mask = pred[:, 3:], target[:, 3:], slice_mask[:, 3:]
        B, S, T, H, W, C = pred.shape
        slice_mask = slice_mask if slice_mask is not None else torch.ones((B, S), dtype=torch.bool, device=pred.device)
        # Mask out padding slices
        pred__ = pred[slice_mask]  # (B*S, T, H, W, C)
        target__ = target[slice_mask]  # (B*S, T, H, W, C)
        # Remove time dimension (batch of 2D images)
        pred__ = pred__.view((-1, H, W, C))  # (B*S*T, H, W, C)
        target__ = target__.view((-1, H, W, C))  # (B*S*T, H, W, C)
        # Monai wants channels first
        pred__ = pred__.moveaxis(-1, 1)  # (B*S*T, C, H, W)
        target__ = target__.moveaxis(-1, 1)  # (B*S*T, C, H, W)
        # Loss per 2D image
        loss__ = self.criterion(pred__, target__).squeeze()
        if loss__.shape[1] == self.seg_weights.shape[0]:
            loss__ = loss__ * self.seg_weights[None].to(loss__.device)
            loss__ = loss__.mean(1)
        loss = loss__.reshape(-1, T)  # Bring back time dimension

        # The slices of a given subject should be weighted depending on amount of non-padding slices subject has
        subj_weight = (1 / slice_mask.sum(1))[:, None].tile((1, S))
        subj_weight_ = subj_weight[slice_mask][:, None]
        loss = loss * subj_weight_
        loss = loss.mean(1)  # mean across time
        loss = loss.sum(0) / B  # mean across subjects
        return loss
    

class ReconstructionCriterion(nn.Module):
    def __init__(self, **kwargs):
        super(ReconstructionCriterion, self).__init__()
        # self.criterion = nn.MSELoss(reduction="none")
        self.criterion = lambda x, y: torch.abs((x - y) / (x.detach() + 1e-3))
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, slice_mask: Optional[torch.Tensor] = None):
        assert len(pred.shape) == 6, f"Should be (b, s, t, h, w, c). Received: {pred.shape}"

        B, S, T, H, W, C = pred.shape
        # Mask out padding slices
        slice_mask = slice_mask if slice_mask is not None else torch.ones((B, S), dtype=torch.bool, device=pred.device)
        pred_ = pred[slice_mask]  # (B*S, T, H, W, C)
        target_ = target[slice_mask]  # (B*S, T, H, W, C)
        # Remove time dimension (batch of 2D images)
        pred_ = pred_.view((-1, H, W, C))  # (B*S*T, H, W, C)
        target_ = target_.view((-1, H, W, C))  # (B*S*T, H, W, C)
        # Loss per 2D image
        loss_ = self.criterion(pred_, target_).mean((-3, -2, -1))
        loss = loss_.reshape(-1, T)  # Bring back time dimension

        # The slices of a given subject should be weighted depending on amount of non-padding slices subject has
        subj_weight = (1 / slice_mask.sum(1))[:, None].tile((1, S))
        subj_weight_ = subj_weight[slice_mask][:, None]
        loss = loss * subj_weight_
        loss = loss.mean(1)  # mean across time
        loss = loss.sum(0) / B  # mean across subjects

        return loss


class SegmentationMetrics:
    def __init__(self, **kwargs):
        self.dice = DiceMetric(include_background=True,
                               ignore_empty=False,  # Will set 1.0 to channels with no foreground
                               reduction="none")
        self.hd = HausdorffDistanceMetric(include_background=True, reduction="none")

    def __call__(self, pred: torch.Tensor, target: torch.Tensor, slice_mask: Optional[torch.Tensor] = None):
        assert len(pred.shape) == 6, f"Should be (b, s, t, h, w, c). Received: {pred.shape}"

        # First three slice are LA, we don't have seg GT for LA
        pred, target, slice_mask = pred[:, 3:], target[:, 3:], slice_mask[:, 3:]
        B, S, T, H, W, C = pred.shape
        slice_mask = slice_mask if slice_mask is not None else torch.ones((B, S), dtype=torch.bool, device=pred.device)

        # Monai wants channels first
        pred = pred.moveaxis(-1, -3)  # (B, S, T, C, H, W)
        target = target.moveaxis(-1, -3)  # (B, S, T, C, H, W)
        # Remove slice and time dimension (batch of 2D images)
        pred__ = pred.view((-1, C, H, W))  # (B*S*T, C, H, W)
        target__ = target.view((-1, C, H, W))  # (B*S*T, C, H, W)

        # Metric per 2D image
        dice__ = self.dice(pred__, target__)
        dice = dice__.reshape(B, S, T, C)  # Bring back time dimension
        # HD does not natively handle channels with no foreground, we only eval those channels that contain something
        target_not_empty = target.any(-2).any(-1)  # (B, S, T, C)
        pred_not_empty = pred.any(-2).any(-1)  # (B, S, T, C)
        both_not_empty = torch.logical_and(target_not_empty, pred_not_empty)
        hd = torch.zeros((pred.shape[:4]), dtype=pred.dtype, device=pred.device)
        if both_not_empty.any().item():
            hd[both_not_empty] = self.hd(pred[both_not_empty][:, None], target[both_not_empty][:, None]).squeeze(-1)
        only_pred_empty = torch.logical_and(target_not_empty, ~pred_not_empty)
        hd[only_pred_empty] = 1000.0
        only_target_empty = torch.logical_and(~target_not_empty, pred_not_empty)
        hd[only_target_empty] = 1000.0
        only_target_empty = torch.logical_and(~target_not_empty, ~pred_not_empty)
        hd[only_target_empty] = 0.0

        # Compute region-wise metrics.
        num_slices = slice_mask.sum(1)
        dice_base = dice[:, :3].mean((0, 1, 2))
        dice_apex = torch.cat((dice[:, num_slices-1], dice[:, num_slices-2], dice[:, num_slices-3]), dim=1)
        dice_apex = dice_apex.mean((0, 1, 2))
        hd_base = hd[:, :3].mean((0, 1, 2))
        hd_apex = torch.cat((hd[:, num_slices - 1], hd[:, num_slices - 2], hd[:, num_slices - 3]), dim=1)
        hd_apex = hd_apex.mean((0, 1, 2))

        dice_mask__ = dice[slice_mask]
        hd_mask__ = hd[slice_mask]
        # The slices of a given subject should be weighted depending on amount of non-padding slices subject has
        subj_weight = (1 / slice_mask.sum(1))[:, None].tile((1, S))
        subj_weight_ = subj_weight[slice_mask][:, None, None]
        dice_mask__ = dice_mask__ * subj_weight_
        hd_mask__ = hd_mask__ * subj_weight_
        dice_global = dice_mask__.mean(1)  # mean across time
        dice_global = dice_global.sum(0) / B  # mean across subjects
        hd_global = hd_mask__.mean(1)  # mean across time
        hd_global = hd_global.sum(0) / B   # mean across subjects
        return {**{f"Dice_{i}": dice_global[i] for i in range(C)},
                **{f"HD_{i}": hd_global[i] for i in range(C)},
                "Dice_FG": dice_global[1:].mean(),
                "HD_FG": hd_global[1:].mean(),
                **{f"Dice_base_{i}": dice_base[i] for i in range(C)},
                **{f"HD_base_{i}": hd_base[i] for i in range(C)},
                "Dice_base_FG": dice_base[1:].mean(),
                "HD_base_FG": hd_base[1:].mean(),
                **{f"Dice_apex_{i}": dice_apex[i] for i in range(C)},
                **{f"HD_apex_{i}": hd_apex[i] for i in range(C)},
                "Dice_apex_FG": dice_apex[1:].mean(),
                "HD_apex_FG": hd_apex[1:].mean(),
                }


class ReconstructionMetrics:
    def __init__(self, **kwargs):
        self.psnr = PSNRMetric(max_val=1.0, reduction="none")
        self.ssim = SSIMMetric(spatial_dims=2, reduction="none")

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor, slice_mask: Optional[torch.Tensor] = None):
        assert len(pred.shape) == 6, f"Should be (b, s, t, h, w, c). Received: {pred.shape}"

        B, S, T, H, W, C = pred.shape
        # Mask out padding slices
        slice_mask = slice_mask if slice_mask is not None else torch.ones((B, S), dtype=torch.bool, device=pred.device)
        # Monai wants channels first
        pred = pred.moveaxis(-1, -3)  # (B, S, T, C, H, W)
        target = target.moveaxis(-1, -3)  # (B, S, T, C, H, W)
        # Remove slice and time dimension (batch of 2D images)
        pred__ = pred.view((-1, C, H, W))  # (B*S*T, C, H, W)
        target__ = target.view((-1, C, H, W))  # (B*S*T, C, H, W)
        # Metric per 2D image
        psnr__ = self.psnr(pred__, target__).squeeze(-1).clamp(max=1000.0)
        psnr = psnr__.reshape(B, S, T)
        psnr = psnr.mean(2)  # mean across time
        ssim__ = self.ssim(pred__, target__).squeeze(-1)
        ssim = ssim__.reshape(B, S, T)
        ssim = ssim.mean(2)  # mean across time

        # Metrics globally
        # The slices of a given subject should be weighted depending on amount of non-padding slices subject has
        subj_weight = (1 / slice_mask.sum(1))[:, None].tile((1, S))
        subj_weight__ = subj_weight[slice_mask]
        psnr_mask__ = psnr[slice_mask] * subj_weight__
        psnr_global = psnr_mask__.sum(0) / B  # mean across subjects
        ssim_mask__ = ssim[slice_mask] * subj_weight__
        ssim_global = ssim_mask__.sum(0) / B  # mean across subjects

        # Metrics on SA slices
        # The slices of a given subject should be weighted depending on amount of non-padding slices subject has
        psnr_sa = psnr[:, 3:]
        ssim_sa = ssim[:, 3:]
        mask_sa = slice_mask[:, 3:]
        subj_weight = (1 / mask_sa.sum(1))[:, None].tile((1, S-3))
        subj_weight_ = subj_weight[mask_sa]
        psnr_sa = psnr_sa[mask_sa] * subj_weight_
        ssim_sa = ssim_sa[mask_sa] * subj_weight_
        psnr_sa = psnr_sa.sum(0) / B  # mean across subjects
        ssim_sa = ssim_sa.sum(0) / B  # mean across subjects

        # Metrics on LA slices
        psnr_la = psnr[:, :3].mean()
        ssim_la = ssim[:, :3].mean()

        return {"PSNR": psnr_global,
                "SSIM": ssim_global,
                "PSNR_LA": psnr_la,
                "SSIM_LA": ssim_la,
                "PSNR_SA": psnr_sa,
                "SSIM_SA": ssim_sa,
                }
