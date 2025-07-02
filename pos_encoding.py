import math
from typing import Any, Optional
import torch
from torch import nn
import numpy as np


class PosEncodingNone(nn.Module):
    LUT_NAME = "none"

    def __init__(self, in_dim: Optional[int] = None, **kwargs):
        super(PosEncodingNone, self).__init__()
        self.in_dim = in_dim
        assert self.in_dim is not None
        self.out_dim = self.in_dim
        self.device = kwargs.get("device", "cuda")

    def forward(self, coords):
        return coords

    def __repr__(self):
        d = "xyzt"
        return f"None ({d[:self.in_dim]})"


class PosEncodingNeRF(PosEncodingNone):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, *args, **kwargs):
        super(PosEncodingNeRF, self).__init__(*args, **kwargs)
        self.num_frequencies = kwargs.get("num_frequencies")
        assert isinstance(self.num_frequencies, (tuple, list,))
        assert len(self.num_frequencies) == self.in_dim, f"{self.num_frequencies}, {self.in_dim}"

        self.out_dim = 2 * np.sum(self.num_frequencies)

    def __repr__(self):
        d = "xyzt"
        return f"NeRF ({d[:self.in_dim]}   Freqs: {self.num_frequencies}, Scales: {self.freq_scale}, Out-dim: {self.out_dim})"

    def forward(self, coords):
        coords = coords.view(coords.shape[0], self.in_dim)

        coords_pos_enc = coords
        for j, dim_freqs in enumerate(self.num_frequencies):
            for i in range(dim_freqs):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc.reshape(coords.shape[0], self.out_dim)


class PosEncodingNeRFOptimized(PosEncodingNeRF):
    ''' Vectorized version of the class above. LOOK MA, NO LOOPS! '''
    LUT_NAME = "nerf"

    def __init__(self, *args, **kwargs):
        super(PosEncodingNeRFOptimized, self).__init__(*args, **kwargs)
        self.freq_scale = kwargs.get("coords_freq_scale", [1.0])
        assert isinstance(self.freq_scale, (tuple, list,))
        assert len(self.freq_scale) == 1 or len(self.freq_scale) == len(self.num_frequencies)
        if len(self.freq_scale) == 1:
            self.freq_scale = self.freq_scale * self.in_dim
        self.exp_i_pi = torch.cat([2**torch.arange(f, dtype=torch.float32, device=self.device, requires_grad=False)[None] * s * np.pi for f, s in zip(self.num_frequencies, self.freq_scale)], dim=1)

    def __repr__(self):
        d = "xyzt"
        return f"NeRF Optimized ({d[:self.in_dim]}   Freqs: {self.num_frequencies}, Scales: {self.freq_scale}, Out-dim: {self.out_dim})"

    def forward(self, coords):
        coords_ = torch.cat([torch.tile(coords[..., j:j+1], (1, n)) for j, n in enumerate(self.num_frequencies)], dim=-1)
        exp_i_pi = torch.tile(self.exp_i_pi.to(coords_.device), (coords_.shape[0], 1))
        prod = exp_i_pi * coords_
        out = torch.cat((coords, torch.sin(prod), torch.cos(prod)), dim=-1)
        return out


class PosEncodingNeRFAnnealed(PosEncodingNeRFOptimized):
    ''' Vectorized version of the class above. LOOK MA, NO LOOPS! '''
    LUT_NAME = "nerf"

    def __init__(self, *args, **kwargs):
        super(PosEncodingNeRFOptimized, self).__init__(*args, **kwargs)
        self.max_iter = kwargs.get("anneal_max_iter", 10_000)
        self.start_frequency_prop = kwargs.get("anneal_start_prop", 0.2)
        self.start_frequencies = [max(int(round(i * self.start_frequency_prop)), 1) for i in self.num_frequencies]
        self.freq_scale = kwargs.get("coords_freq_scale", [1.0])
        assert isinstance(self.freq_scale, (tuple, list,))
        assert len(self.freq_scale) == 1 or len(self.freq_scale) == len(self.num_frequencies)
        if len(self.freq_scale) == 1:
            self.freq_scale = self.freq_scale * self.in_dim
        self.exp_i_pi = torch.cat(
            [2 ** torch.arange(f, dtype=torch.float32, device=self.device, requires_grad=False) * s * np.pi
             for f, s in zip(self.num_frequencies, self.freq_scale)], dim=0)

    def __repr__(self):
        d = "xyzt"
        return f"NeRF Annealed ({d[:self.in_dim]}   Freqs: {self.num_frequencies}, Scales: {self.freq_scale}, Out-dim: {self.out_dim}, Max-iters: {self.max_iter})"

    def get_freq_mask_alpha(self, current_iter):
        # based on https://github.com/Jiawei-Yang/FreeNeRF/blob/main/internal/math.py#L277
        if current_iter is not None and current_iter < self.max_iter:
            mask_per_dim = []
            for freqs, start_freqs in zip(self.num_frequencies, self.start_frequencies):
                freq_mask = np.zeros(freqs)
                ptr = (freqs - start_freqs) * (current_iter / self.max_iter) + start_freqs
                int_ptr = int(ptr)
                freq_mask[: int_ptr + 1] = 1.0  # assign the integer part
                freq_mask[int_ptr: int_ptr + 1] = (ptr - int_ptr)  # assign the fractional part
                freq_mask_alpha = torch.clip(torch.from_numpy(freq_mask), 1e-8,
                                                  1 - 1e-8).float()  # for numerical stability
                # windowed_alpha = ptr
                mask_per_dim.append(freq_mask_alpha)
            freq_mask_alpha = torch.cat(mask_per_dim, dim=0)
            return freq_mask_alpha
        else:
            freq_mask_alpha = torch.ones(sum(self.num_frequencies)).float()
            # windowed_alpha = self.num_frequencies + 1
            return freq_mask_alpha

    def forward(self, coords, curr_iter=None):
        coords_ = torch.cat([torch.tile(coords[..., j:j + 1], (1, n)) for j, n in enumerate(self.num_frequencies)],
                            dim=-1)
        # exp_i_pi = torch.tile(self.exp_i_pi.to(coords_.device), (coords_.shape[0], 1))
        prod = coords_ * self.exp_i_pi
        anneal_mask = self.get_freq_mask_alpha(curr_iter).to(coords_.device)
        out = torch.cat((torch.sin(prod) * anneal_mask,
                         torch.cos(prod) * anneal_mask), dim=-1)
        return out


class PosEncodinFourier(PosEncodingNone):
    ''' https://github.com/tancik/fourier-feature-networks/blob/master/Demo.ipynb '''
    LUT_NAME = "fourier"

    def __init__(self, *args, **kwargs):
        super(PosEncodinFourier, self).__init__(*args, **kwargs)
        self.num_frequencies = kwargs.get("num_frequencies")
        if len(self.num_frequencies) > 1:
            self.num_frequencies = [max(self.num_frequencies)]
        self.freq_scale = kwargs.get("coords_freq_scale", [1.0])
        assert isinstance(self.num_frequencies, (tuple, list,))
        assert len(self.num_frequencies) == 1
        if not isinstance(self.freq_scale, float):
            self.freq_scale = torch.as_tensor(self.freq_scale)[:, None].to(self.device)
        self.B_gauss = torch.normal(0.0, 1.0, size=(self.in_dim, self.num_frequencies[0]), requires_grad=False).to(self.device) * self.freq_scale
        self.B_gauss_pi = np.pi * self.B_gauss

        # self.out_dim = 2 * total_freqs
        self.out_dim = 2 * self.num_frequencies[0]

    def __repr__(self):
        d = "xyzt"
        return f"Gaussian ({d[:self.in_dim]}   Freqs: {self.num_frequencies}, Scales: {self.freq_scale}, Out-dim: {self.out_dim})"

    def get_extra_state(self) -> Any:
        return {"B_gauss_pi": self.B_gauss_pi}  # Required to store gaussian array into network state dict

    def set_extra_state(self, state: Any):
        self.B_gauss_pi = state["B_gauss_pi"]  # Required to store gaussian array into network state dict

    def forward(self, coords, *args, **kwargs):
        prod = coords @ self.B_gauss_pi.to(coords.device)
        out = torch.cat((torch.sin(prod), torch.cos(prod)), dim=-1)
        return out


class PosEncodingCAPE(PosEncodingNone):
    LUT_NAME = "CAPE"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.K = kwargs["cape_K"]
        channels = self.K // 2
        rho = 10 ** torch.linspace(0, 1, channels)
        w_x = rho * torch.cos(torch.arange(channels))
        w_y = rho * torch.sin(torch.arange(channels))
        # w_t = rho * (-torch.cos(torch.arange(channels)))
        self.w = torch.stack([w_x, w_y], dim=0)[:self.in_dim].cuda()

        self.freq = 1.0 * torch.exp(-2.0 * torch.floor(torch.arange(self.K, device="cuda") / 2)
                                      * (math.log(1e4) / self.K))
        _sin2cos_phase_shift = torch.pi / 2.0
        self.cos_shifts = _sin2cos_phase_shift * (torch.arange(self.K, device="cuda") % 2)
        # self.out_dim = self.K//self.in_dim * self.in_dim
        # k = torch.arange(0, self.K//self.in_dim, device=self.device)[None]
        # # This extends the original paper by allowing 4 dimensions (more could be added)
        # w = torch.cat([torch.pow(10, 2 * k / self.K) * torch.cos(k),
        #                torch.pow(10, 2 * k / self.K) * torch.sin(k),
        #                torch.pow(10, 2 * k / self.K) * (-torch.cos(k)),
        #                torch.pow(10, 2 * k / self.K) * (-torch.sin(k)),
        #                ], dim=0)

    def __call__(self, coord: torch.Tensor):
        picw = torch.pi * (coord[:, :2] @ self.w)
        cos = torch.cos(picw)
        sin = torch.sin(picw)
        out = torch.cat([cos, sin], -1)
        if coord.shape[-1] == 3:
            t_emb = torch.sin(coord[:, -1:] * self.freq + self.cos_shifts)
            out = out + t_emb
        return out


POS_ENCODING_LUT = {PosEncodingNone.LUT_NAME: PosEncodingNone,
                    PosEncodingNeRFOptimized.LUT_NAME: PosEncodingNeRFOptimized,
                    PosEncodinFourier.LUT_NAME: PosEncodinFourier,
                    PosEncodingCAPE.LUT_NAME: PosEncodingCAPE,
                    }
