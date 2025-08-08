import abc
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
import math


class Layer(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0, **kwargs):
        super(Layer, self).__init__()
        self.dropout = None
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        self.in_size = in_size
        self.out_size = out_size

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Relu(Layer):
    LUT_NAME = "relu"

    def __init__(self, in_size, out_size, **kwargs):
        super(Relu, self).__init__(in_size, out_size, **kwargs)
        self.linear = nn.Linear(in_size, out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = torch.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class Sine(Layer):
    """ See SIREN paper and github. """
    LUT_NAME = "sine"

    def __init__(self, in_size, out_size, siren_factor=30., **kwargs):
        super(Sine, self).__init__(in_size, out_size, **kwargs)
        self.linear = nn.Linear(in_size, out_size)
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        self.siren_factor = siren_factor
        self.weight_init()

    def forward(self, x):
        x = self.linear(x)
        x = torch.sin(self.siren_factor * x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

    def weight_init(self):
        with torch.no_grad():
            num_input = self.linear.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            self.linear.weight.uniform_(-math.sqrt(6 / num_input) / self.siren_factor,
                                        math.sqrt(6 / num_input) / self.siren_factor)


class WIRE(Layer):
    '''
        Implicit representation with Gabor nonlinearity

        Inputs;
            in_size: Input features
            out_size; Output features
            bias: if True, enable bias for the linear operation
            omega_0: Legacy SIREN parameter
            omega: Frequency of Gabor sinusoid term
            scale: Scaling of Gabor Gaussian term
    '''
    LUT_NAME = "wire"

    def __init__(self, in_size, out_size, bias=True, **kwargs):
        super(WIRE, self).__init__(in_size, out_size, **kwargs)
        self.omega_0 = kwargs.get("wire_omega_0", 10.0)  # Freq
        self.scale_0 = kwargs.get("wire_scale_0", 10.0)
        self.freqs = nn.Linear(in_size, out_size, bias=bias)
        self.scale = nn.Linear(in_size, out_size, bias=bias)

    def forward(self, x):
        omega = self.omega_0 * self.freqs(x)
        scale = self.scale(x) * self.scale_0
        x = torch.cos(omega) * torch.exp(-(scale * scale))
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_size, out_size, do_3d=True, bias=True, dropout=0.0,  **kwargs):
        super(ConvBlock, self).__init__()
        self.do_3d = do_3d
        t_channels = 3 if do_3d else 1
        self.conv1 = nn.Conv3d(in_size, out_size, (3, 3, t_channels), bias=bias)
        self.conv2 = nn.Conv3d(out_size, out_size, (3, 3, t_channels), bias=bias)
        self.dropout = nn.Dropout3d(dropout)

    def pad(self, x):
        if self.do_3d:
            x = F.pad(x, (1,1, 0,0,0,0), mode='circular')
        x = F.pad(x, (0,0, 1,1,1,1), mode='constant')
        return x

    def forward(self, x):
        x = self.dropout(F.relu(self.conv1(self.pad(x))))
        x = x + F.relu(self.conv2(self.pad(x)))
        return x
