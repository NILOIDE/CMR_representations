from typing import Optional, Union, List

import torch
from torch import nn
import math

from layers import AttentionLayer, CrossAttentionLayer, SelfAttentionLayer


class PerceiverEncoder(nn.Module):
    LUT_NAME = "perceiver"

    def __init__(self, in_size: int, hidden_size: int = 128, enc_num_hidden_layers: int = 2,
                 latent_nodes: int = 64, latent_size: int = 128, **kwargs):
        super(PerceiverEncoder, self).__init__()
        self.att_heads = kwargs.get("enc_att_num_heads", 1)

        self.latent = nn.Parameter(torch.randn(latent_nodes, latent_size))

        a = [AttentionLayer(latent_size,
                            in_size,
                            in_size,
                            latent_size,
                            latent_size,
                            self.att_heads,
                            dim_feedforward=hidden_size,
                            **kwargs)
             for i in range(enc_num_hidden_layers)]
        self.ca_layers = nn.ModuleList(a)

        a = [AttentionLayer(latent_size,
                            latent_size,
                            latent_size,
                            latent_size,
                            latent_size,
                            self.att_heads,
                            dim_feedforward=hidden_size,
                            **kwargs)
             for i in range(enc_num_hidden_layers)]

        self.sa_layers = nn.ModuleList(a)

        self.out_size = latent_size

    def forward(self, x: torch.Tensor, **kwargs) -> Union[torch.Tensor, List[torch.Tensor]]:
        out = self.latent[None].tile(x.shape[0], 1, 1)
        outs = []
        for ca_layer, sa_layer in zip(self.ca_layers, self.sa_layers):
            out = out + ca_layer(out, x)
            out = out + sa_layer(out)
            outs.append(out)
        return outs



