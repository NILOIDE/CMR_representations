from typing import Tuple, List, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F
from layers import Relu, Sine, WIRE


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
        self.start = nn.Linear(latent_size, hidden_size)
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


class MLP_FiLM(nn.Module):
    """ MLP decoder with FiLM layers. Each layer has their features modulated by a latent vector. """

    def __init__(self, coord_size: int,
                 latent_size: Union[int, Tuple[int, ...]],
                 num_hidden_layers: int,
                 hidden_size: int,
                 out_size: int,
                 num_blocks: int = 4,
                 layer_type: str = "relu",
                 **kwargs):
        super(MLP_FiLM, self).__init__()
        self.start = nn.Linear(coord_size, hidden_size)
        # Projection of latents from features to frequency and phase modulations (hence the *2)
        self.embeds = nn.Sequential(*[nn.Linear(i, hidden_size*2) for i in latent_size])
        self.layers = nn.Sequential(*[Block_FiLM(hidden_size if i == 0 else hidden_size + coord_size,
                                                 num_hidden_layers//num_blocks,
                                                 hidden_size, layer_type=layer_type)
                                      for i in range(num_blocks)])
        self.out = nn.Linear(hidden_size, out_size)
        self.out_size = out_size

    def forward(self,
                coords: torch.Tensor,
                latents: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """"
        coord: coordinate to be decoded (B, N)
        latents: conditioning latents to modulate layer pre-activations (B, F) or List[(B, F1), (B, F2), ...]
        """
        if isinstance(latents, torch.Tensor):
            latents = [latents]*len(self.embeds)
        x =  self.start(coords)
        for i, (block, embed, lat) in enumerate(zip(self.layers, self.embeds, latents)):
            z = embed(lat)
            gamma, beta = torch.chunk(z, 2, dim=-1)
            if i == 0:
                x_in = x
            else:
                x_in = torch.cat((x, coords), dim=-1)
            z = block(x_in, gamma, beta)
            x = x + z
        return self.out(x)


class Block_FiLM(nn.Module):
    def __init__(self, input_size: int, num_hidden_layers: int, hidden_size: int, layer_type: str = "relu", **kwargs):
        super(Block_FiLM, self).__init__()
        assert layer_type in layer_LUT, "Layer type undefined"
        self.block = nn.Sequential(
            layer_LUT[layer_type](input_size, hidden_size),
            *[layer_LUT[layer_type](hidden_size, hidden_size) for _ in range(num_hidden_layers-1)])

    def forward(self,
                x: torch.Tensor,
                gamma: Optional[torch.Tensor] = None,
                beta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """"
        x: layer input (B, F)
        gamma: pre-activation scaling (frequency modulation) (B, F)
        beta: pre-activation shift (phase modulation) (B, F)
        """
        for layer in self.block[:-1]:
            x = layer(x)
        x = self.block[-1](x, gamma, beta)
        return x


layer_LUT = {Relu.LUT_NAME: Relu, Sine.LUT_NAME: Sine, WIRE.LUT_NAME: WIRE}
