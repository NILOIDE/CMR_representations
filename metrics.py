import numpy as np
import torch


class L2:
    def __init__(self):
        pass

    def __call__(self, sapmle1, sample2, mask1=None, mask2=None):
        diff = sapmle1 - sample2
        diff = diff * diff
        if mask1 is not None or mask2 is not None:
            mask = torch.ones_like(diff)
            if mask1 is not None:
                mask = torch.logical_and(mask, mask1)
            if mask2 is not None:
                mask = torch.logical_and(mask, mask2)
            diff *= mask
        diff = torch.sum(diff)
        return diff


class L1:
    def __init__(self):
        pass

    def __call__(self, sapmle1, sample2, mask1=None, mask2=None):
        diff = sapmle1 - sample2
        diff = torch.abs(diff)
        if mask1 is not None or mask2 is not None:
            mask = torch.ones_like(diff)
            if mask1 is not None:
                mask = torch.logical_and(mask, mask1)
            if mask2 is not None:
                mask = torch.logical_and(mask, mask2)
            diff *= mask
        diff = torch.sum(diff)
        return diff
