import numpy as np


class L2:
    def __init__(self):
        pass

    def __call__(self, sapmle1, sample2):
        diff = sapmle1 - sample2
        diff = diff * diff
        diff = np.sum(diff)
        return diff


class L1:
    def __init__(self):
        pass

    def __call__(self, sapmle1, sample2):
        diff = sapmle1 - sample2
        diff = np.abs(diff)
        diff = np.sum(diff)
        return diff
