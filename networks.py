from typing import Tuple, List, Optional

import torch
from torch import nn
import torch.nn.functional as F
from layers import Relu

class MLP(nn.Module):
    """ Simple MLP decoder. Skip-connections with concatenated coords every block. """

    def __init__(self, coord_size: int,
                 latent_size: int,
                 num_hidden_layers: int,
                 hidden_size: int,
                 out_size: int,
                 num_blocks: int = 4,
                 **kwargs):
        super(MLP, self).__init__()
        self.start = Relu(latent_size, hidden_size)
        self.layers = nn.Sequential(*[Block(hidden_size+coord_size, num_hidden_layers//num_blocks, hidden_size)
                                      for _ in range(num_blocks)])
        self.out = nn.Linear(hidden_size, out_size)
        self.out_size = out_size

    def forward(self, coord: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        x = self.start(latent)
        for layer in self.layers:
            x = x + layer(torch.cat((x, coord), dim=-1))
        return self.out(x)

class Block(nn.Module):
    def __init__(self, input_size: int, num_hidden_layers: int, hidden_size: int, **kwargs):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            Relu(input_size, hidden_size),
            *[Relu(hidden_size, hidden_size) for _ in range(num_hidden_layers-1)])

    def forward(self, x):
        return self.block(x)


