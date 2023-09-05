from typing import Optional, Union, List

import torch
from torch import nn
import math

from layers import AttentionLayer


class PerceiverEncoder(nn.Module):
    LUT_NAME = "perceiver"

    def __init__(self, coord_size: int, in_features: int, hidden_size: int = 128, num_hidden_layers: int = 2,
                 latent_nodes: int = 128, latent_size: int = 128, **kwargs):
        super(PerceiverEncoder, self).__init__()
        self.att_heads = kwargs.get("enc_att_num_heads", 1)

        self.latent = nn.Parameter(torch.randn(latent_nodes, latent_size))

        a = [AttentionLayer(latent_size,
                            coord_size + in_features,
                            coord_size + in_features,
                            latent_size,
                            latent_size,
                            self.att_heads,
                            dim_feedforward=hidden_size,
                            **kwargs)
             for i in range(num_hidden_layers)]
        self.ca_layers = nn.ModuleList(a)

        a = [AttentionLayer(latent_size,
                            latent_size,
                            latent_size,
                            latent_size,
                            latent_size,
                            self.att_heads,
                            dim_feedforward=hidden_size,
                            **kwargs)
             for i in range(num_hidden_layers)]
        self.sa_layers = nn.ModuleList(a)

        self.out_size = latent_size

    def forward(self, coords: torch.Tensor, intensities: torch.Tensor, **kwargs) -> Union[torch.Tensor, List[torch.Tensor]]:
        cat_input = torch.cat((coords, intensities), dim=-1)
        x = self.latent[None].tile(coords.shape[0], 1, 1)
        outs = []
        for ca_layer, sa_layer in zip(self.ca_layers, self.sa_layers):
            x = x + ca_layer(x, cat_input, cat_input)
            x = x + sa_layer(x, x, x)
            outs.append(x)
        mean_latent = outs[-1].mean(1)
        return mean_latent
