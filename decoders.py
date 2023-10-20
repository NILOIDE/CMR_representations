from torch import nn
import torch

from layers import Sine, Relu, AttentionLayer


class ReconstructionHead(nn.Module):
    def __init__(self, input_size, output_size, **kwargs):
        super(ReconstructionHead, self).__init__()
        self.out_layer = nn.Linear(input_size, output_size)
        self.out_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.out_layer(x)
        out = torch.sigmoid(out)
        return out


class MLP(nn.Module):
    """ Simple decoder that uses a single averaged latent vector in its input to
    condition the way coordinates are processed. """
    LUT_NAME = "mlp"

    def __init__(self, coord_size: int, latent_size: int, num_hidden_layers: int, hidden_size: int, out_size: int, **kwargs):
        super(MLP, self).__init__()
        a = [Sine(coord_size, hidden_size)]
        for i in range(num_hidden_layers - 2):
            a.append(Sine(hidden_size, hidden_size))
        a.append(nn.Linear(hidden_size, out_size))
        self.mlp = nn.Sequential(*a)
        self.out_size = out_size

    def forward(self, coord: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        x = torch.cat((coord, latents), dim=1)
        return self.mlp(x)


class MLPBackbone(nn.Module):
    """ Simple decoder that uses a single averaged latent vector in its input to
    condition the way coordinates are processed. """
    LUT_NAME = "mlp"

    def __init__(self, coord_size: int, latent_size: int, num_hidden_layers: int = 4, hidden_size: int = 128, **kwargs):
        super(MLPBackbone, self).__init__()
        a = [Sine(coord_size, hidden_size, siren_factor=kwargs["siren_factor"])]
        for i in range(num_hidden_layers - 1):
            a.append(Sine(hidden_size, hidden_size, siren_factor=kwargs["siren_factor"]))
        self.mlp = nn.Sequential(*a)
        self.out_size = hidden_size

    def forward(self, coord: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        x = torch.cat((coord, latents.mean(1)), dim=1)
        return self.mlp(x)


class CADecoder(nn.Module):
    """ Simple decoder that uses a single averaged latent vector in its input to
    condition the way coordinates are processed. """
    LUT_NAME = "ca"

    def __init__(self, coord_size: int, latent_size: int, num_hidden_layers: int = 4, hidden_size: int = 128, **kwargs):
        super(CADecoder, self).__init__()
        a = [AttentionLayer(coord_size if i == 0 else hidden_size,
                            latent_size,
                            latent_size,
                            8,
                            hidden_size,
                            activation_class=Relu,
                            )
             for i in range(num_hidden_layers)]
        self.mlp = nn.ModuleList(a)
        self.out_size = hidden_size

    def forward(self, coord: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        out = coord
        for layer in self.mlp:
            out = layer(out, latents, latents)
        return out

