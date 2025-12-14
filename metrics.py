import torch
from monai.losses import DiceLoss

from utils import to_1hot


class Metric:
    def mask_out_metric(self, metric: torch.Tensor, mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
        mask = self.merge_masks(mask1, mask2)
        # Mask out section of line out of bounds
        metric *= mask
        metric = torch.sum(metric, dim=(1, 2)) / (mask.sum(dim=(1, 2)) + 1e-8)
        metric[~mask.any(2).any(1)] = 9999.0
        return metric

    def merge_masks(self, mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
        # Line section where both images are in-bounds
        mask = torch.ones_like(mask1)
        mask = torch.logical_and(mask, mask1)
        mask = torch.logical_and(mask, mask2)
        return mask


class L2(Metric):
    def __call__(self, sapmle1: torch.Tensor, sample2: torch.Tensor,
                 mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
        diff = sapmle1 - sample2
        metric = diff * diff  # L2
        metric = self.mask_out_metric(metric, mask1, mask2)
        return metric


class L1(Metric):
    def __call__(self, sapmle1: torch.Tensor, sample2: torch.Tensor,
                 mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
        diff = sapmle1 - sample2
        metric = torch.abs(diff)  # L1
        metric = self.mask_out_metric(metric, mask1, mask2)
        return metric


class NCC(Metric):
    def __call__(self, sample1: torch.Tensor, sample2: torch.Tensor,
                 mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
        mask = self.merge_masks(mask1, mask2)
        s1_masked = sample1 * mask
        s2_masked = sample2 * mask
        # Calculate means over valid regions
        mean1 = s1_masked.sum(1) / mask1.sum(1)
        mean2 = s2_masked.sum(1) / mask2.sum(1)

        # Center the signals
        s1_centered = (sample1 - mean1[:,None])
        s2_centered = (sample2 - mean2[:,None])

        # Calculate normalized cross-correlation
        numerator = (s1_centered * s2_centered).sum(1)
        std1 = torch.sqrt((s1_centered * s1_centered).sum(1) + 1e-8)
        std2 = torch.sqrt((s2_centered * s2_centered).sum(1) + 1e-8)

        ncc = numerator / (std1 * std2 + 1e-8)

        # Convert to distance metric (1 - NCC gives range [0, 2])
        # NCC ranges from -1 (anti-correlated) to 1 (perfectly correlated)
        loss = 1.0 - ncc
        return loss


class Dice(Metric):
    def __call__(self, sample1: torch.Tensor, sample2: torch.Tensor,
                 mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice coefficient for segmentation.
        Assumes sample1 and sample2 contain class labels: 0 (background), 1, 2, 3

        Args:
            sample1: (batch, signal, time) with integer class labels
            sample2: (batch, signal, time) with integer class labels
            mask1: (batch, signal, time) validity mask for sample1
            mask2: (batch, signal, time) validity mask for sample2

        Returns:
            metric: (batch,) Dice distance (1 - Dice coefficient)
        """
        # Get merged mask
        mask = self.merge_masks(mask1, mask2)
        # sample1 = sample1 * mask
        # sample2 = sample2 * mask
        B, L, T = sample1.shape

        sample1_1hot = to_1hot(sample1.reshape(-1), 4).reshape(*sample1.shape, 4)
        sample1_1hot = sample1_1hot.moveaxis(-2,1).reshape(-1, L, 4)
        sample2_1hot = to_1hot(sample2.reshape(-1), 4).reshape(*sample2.shape, 4)
        sample2_1hot = sample2_1hot.moveaxis(-2,1).reshape(-1, L, 4)
        loss = DiceLoss(reduction='none')(sample1_1hot.moveaxis(-1, 1), sample2_1hot.moveaxis(-1, 1))
        loss = loss.reshape(B, T, 4)
        return loss
