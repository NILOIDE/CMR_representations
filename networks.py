from typing import Tuple, List, Optional

import torch
from torch import nn
import torch.nn.functional as F
from layers import Relu, ConvBlock
from pos_encoding import PosEncodingNeRFAnnealed


class Encoder(nn.Module):
    """ Simple decoder that uses a single averaged latent vector in its input to
    condition the way coordinates are processed. """

    def __init__(self, filters: Tuple[int, ...], out_dim: int, **kwargs):
        super(Encoder, self).__init__()
        # self.poolings = [(2,2,2), (2,2,1), (2,2,5), (2,2,1), (2,2,1), (2,2,5), (2,2,1)]  # For full img
        self.poolings = [(2,2,2), (2,2,1), (2,2,5), (2,2,5), (2,2,1), (2,2,1), (2,2,1)]  # For cropped
        a = []
        filters = [1, *filters]
        t_size = 50
        for i in range(len(filters) - 1):
            do_3d = t_size > 1
            a.append(ConvBlock(filters[i], filters[i+1], do_3d))
            t_size //= self.poolings[i][-1]
        self.layers = nn.Sequential(*a)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.out_dim = out_dim
        self.aff_emb = PosEncodingNeRFAnnealed(in_dim=6, # 3D + Cyclical time
                                               num_frequencies=(7, 7, 7, 7, 7, 7),
                                               anneal_max_iter=kwargs['pe_anneal_max_iter'],
                                               anneal_start_prop=kwargs['pe_anneal_start_prop'])
        self.out_enc = nn.Linear(self.aff_emb.out_dim + filters[-1], self.out_dim)

    def forward(self, x: torch.Tensor, aff_params: torch.Tensor, num_subj_slices: torch.Tensor, t: int) -> torch.Tensor:
        # Make sure we have a channel dim
        B, S, *_, H, W, T = x.shape
        x = x.reshape((B, S, 1, H, W, T))
        # We don't want to process padding slices
        pad_slices = torch.arange(0, x.shape[1], device=x.device).tile((x.shape[0],1)) < num_subj_slices[:, None]
        x_ = x[pad_slices]
        # Apply convolutional layers and pooling operations
        for l, pool_size in zip(self.layers[:-1], self.poolings):
            x_ = l(x_)
            if x_.shape[-3] % 2 == 1:
                x_ = F.pad(x_, (0,0,0,0,0,1), mode='replicate')
            if x_.shape[-2] % 2 == 1:
                x_ = F.pad(x_, (0,0,0,1,0,0), mode='replicate')
            x_ = F.avg_pool3d(x_, pool_size)
        x_ = self.layers[-1](x_)
        # Apply global pooling along spatial and temporal dims. We obtain 1 embedding per slice
        x_pooled_ = self.global_pool(x_).squeeze((-3, -2, -1))
        # Encode affine parameters into learned slice embeddings
        aff_ = aff_params[pad_slices]
        aff_enc_ = self.aff_emb(aff_, t)
        out_ = torch.cat((x_pooled_, aff_enc_), -1)
        out_ = self.out_enc(out_)
        # Average across slices. Take into account padding slices into avg operation
        latents = torch.zeros((B, S, out_.shape[-1]), dtype=x_.dtype, device=x_.device)
        latents[pad_slices] = out_
        latents = latents.sum(1) / num_subj_slices[:, None]
        return latents


class MLP(nn.Module):
    """ Simple decoder that uses a single averaged latent vector in its input to
    condition the way coordinates are processed. """

    def __init__(self, coord_size: int, num_hidden_layers: int, hidden_size: int, out_size: int, **kwargs):
        super(MLP, self).__init__()
        self.start = Relu(coord_size, hidden_size)
        # a = []
        # for i in range(num_hidden_layers - 2):
        #     a.append(Relu(hidden_size, hidden_size))
        # a.append(nn.Linear(hidden_size, out_size))
        self.mlp1 = nn.Sequential(*[Relu(hidden_size, hidden_size) for _ in range(num_hidden_layers//4)])
        self.mlp2 = nn.Sequential(*[Relu(hidden_size, hidden_size) for _ in range(num_hidden_layers//4)])
        self.mlp3 = nn.Sequential(*[Relu(hidden_size, hidden_size) for _ in range(num_hidden_layers//4)])
        self.mlp4 = nn.Sequential(*[Relu(hidden_size, hidden_size) for _ in range(num_hidden_layers//4)])
        self.out = nn.Linear(hidden_size, out_size)
        self.out_size = out_size

    def forward(self, coord: torch.Tensor) -> torch.Tensor:
        x_s = self.start(coord)
        x = x_s + self.mlp1(x_s)
        x = x_s + self.mlp2(x)
        x = x_s + self.mlp3(x)
        x = x_s + self.mlp4(x)
        return self.out(x)


class MLPDeform(nn.Module):
    """ Simple decoder that uses a single averaged latent vector in its input to
    condition the way coordinates are processed. """

    def __init__(self, coord_size: int, num_hidden_layers: int, hidden_size: int, out_size: int, out_init_std=0.01, **kwargs):
        super(MLPDeform, self).__init__()
        self.start = Relu(coord_size, hidden_size)
        a = []
        for i in range(num_hidden_layers - 1):
            a.append(Relu(hidden_size, hidden_size))
        self.middle = nn.Sequential(*a)
        self.out = nn.Linear(hidden_size, out_size)
        nn.init.xavier_uniform_(self.out.weight, out_init_std)
        self.out_size = out_size

    def forward(self, coord: torch.Tensor) -> torch.Tensor:
        x_s = self.start(coord)
        x = x_s + self.middle(x_s)
        return self.out(x)

