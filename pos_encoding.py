from typing import Any, Optional
import torch
from torch import nn
import math
import numpy as np


class PosEncodingNone(nn.Module):
    LUT_NAME = "none"

    def __init__(self, coord_dim: Optional[int] = None, value_dim: Optional[int] = None, **kwargs):
        super(PosEncodingNone, self).__init__()
        self.coord_dim = coord_dim if coord_dim is not None else 0
        self.value_dim = value_dim if value_dim is not None else 0
        self.in_dim = self.coord_dim + self.value_dim
        assert self.in_dim is not None
        self.out_dim = self.in_dim
        self.dim_repr = (f"{'xyzt'[:self.coord_dim]}" if self.coord_dim else "") + ("v" if self.value_dim else "")

    def forward(self, coords: Optional[torch.Tensor] = None, values: Optional[torch.Tensor] = None):
        if self.coord_dim:
            if self.value_dim:
                # Merge coords and values into a single input
                return torch.cat((coords, values), dim=-1)
            else:
                return coords  # Values were not passed and we return only coords
        else:
            return values  # Coords were not passed and we return only values

    def __repr__(self):
        return f"None ({self.dim_repr})"


class PosEncodingNeRF(PosEncodingNone):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, *args, **kwargs):
        super(PosEncodingNeRF, self).__init__(*args, **kwargs)
        self.num_frequencies = kwargs.get("nerf_num_frequencies")
        assert isinstance(self.num_frequencies, (tuple, list,))
        assert len(self.num_frequencies) == self.in_dim, f"{self.num_frequencies}, {self.in_dim}"

        self.out_dim = self.in_dim + 2 * np.sum(self.num_frequencies)

    def __repr__(self):
        return f"NeRF ({self.dim_repr}   Freqs: {self.num_frequencies}, Scales: {self.freq_scale}, Out-dim: {self.out_dim})"

    def forward(self, coords: Optional[torch.Tensor] = None, values: Optional[torch.Tensor] = None):
        x = super().forward(coords, values)
        x = x.view(x.shape[0], self.in_dim)

        x_pos_enc = x
        for j, dim_freqs in enumerate(self.num_frequencies):
            for i in range(dim_freqs):
                c = x[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

                x_pos_enc = torch.cat((x_pos_enc, sin, cos), axis=-1)

        return x_pos_enc.reshape(x.shape[0], self.out_dim)


class PosEncodingNeRFOptimized(PosEncodingNeRF):
    ''' Vectorized version of the class above. LOOK MA, NO LOOPS! '''
    LUT_NAME = "nerf"

    def __init__(self, *args, **kwargs):
        super(PosEncodingNeRFOptimized, self).__init__(*args, **kwargs)
        device = "cpu"
        self.freq_scale = kwargs.get("freq_scale", [1.0])
        assert isinstance(self.freq_scale, (tuple, list,))
        assert len(self.freq_scale) == 1 or len(self.freq_scale) == len(self.num_frequencies)
        if len(self.freq_scale) == 1:
            self.freq_scale = self.freq_scale * self.in_dim
        self.exp_i_pi = torch.cat([2**torch.arange(f, dtype=torch.float32, device=device, requires_grad=False)[None] * s * np.pi for f, s in zip(self.num_frequencies, self.freq_scale)], dim=1)

    def __repr__(self):
        return f"NeRF Optimized ({self.dim_repr}    Freqs: {self.num_frequencies}, Scales: {self.freq_scale}, Out-dim: {self.out_dim})"

    def forward(self, coords: Optional[torch.Tensor] = None, values: Optional[torch.Tensor] = None):
        x = super().forward(coords, values)
        x_ = torch.cat([torch.tile(x[..., j:j+1], (1, n)) for j, n in enumerate(self.num_frequencies)], dim=-1)
        exp_i_pi = torch.tile(self.exp_i_pi.to(x_.device), (x_.shape[0], x_.shape[1], 1))
        prod = exp_i_pi * x_
        out = torch.cat((x, torch.sin(prod), torch.cos(prod)), dim=-1)
        return out


class PosEncodingGaussian(PosEncodingNone):
    ''' https://github.com/tancik/fourier-feature-networks/blob/master/Demo.ipynb '''
    LUT_NAME = "gaussian"

    def __init__(self, *args, **kwargs):
        super(PosEncodingGaussian, self).__init__(*args, **kwargs)
        self.num_frequencies = kwargs.get("gauss_num_frequencies")
        self.freq_scale = kwargs.get("freq_scale", [1.0])
        assert isinstance(self.num_frequencies, (tuple, list,))
        assert len(self.num_frequencies) == 1
        device = "cpu"
        if not isinstance(self.freq_scale, float):
            self.freq_scale = torch.as_tensor(self.freq_scale)[:, None].to(device)
        self.B_gauss = torch.normal(0.0, 1.0, size=(self.in_dim, self.num_frequencies[0]), requires_grad=False).to(device) * self.freq_scale
        self.B_gauss_pi = 2. * np.pi * self.B_gauss

        # self.out_dim = 2 * total_freqs
        self.out_dim = 2 * self.num_frequencies[0]

    def __repr__(self):
        return f"Gaussian ({self.dim_repr}   Freqs: {self.num_frequencies}, Scales: {self.freq_scale}, Out-dim: {self.out_dim})"

    def get_extra_state(self) -> Any:
        return {"B_gauss_pi": self.B_gauss_pi}  # Required to store gaussian array into network state dict

    def set_extra_state(self, state: Any):
        self.B_gauss_pi = state["B_gauss_pi"]  # Required to store gaussian array into network state dict

    def forward(self, coords: Optional[torch.Tensor] = None, values: Optional[torch.Tensor] = None):
        x = super().forward(coords, values)
        prod = x @ self.B_gauss_pi.to(x.device)
        out = torch.cat((torch.sin(prod), torch.cos(prod)), dim=-1)
        return out


POS_ENCODING_LUT = {PosEncodingNone.LUT_NAME: PosEncodingNone,
                    PosEncodingNeRFOptimized.LUT_NAME: PosEncodingNeRFOptimized,
                    PosEncodingGaussian.LUT_NAME: PosEncodingGaussian,
                    }
