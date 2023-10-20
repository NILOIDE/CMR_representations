import abc
from typing import Optional

import torch
from torch import nn
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


class MultiHeadAttention(nn.Module):
    def __init__(self, d_q, d_k, d_v, d_model, d_out=None, num_heads=1):
        """
        :param d_q: Dimension size of Query input
        :param d_k: Dimension size of Keys input
        :param d_v: Dimension size of Values input
        :param d_model: Dimensions of intermediate mapping of all 3 inputs
        :param d_out: Dimensions of output
        """
        super(MultiHeadAttention, self).__init__()
        d_out = d_out if d_out is not None else d_model

        self.q_d = d_q
        self.k_d = d_k
        self.v_d = d_v
        self.d_out = d_model
        self.num_heads = num_heads
        self.d_k = d_k // num_heads

        self.W_q = nn.Linear(d_q, d_model)
        self.W_k = nn.Linear(d_k, d_model)
        self.W_v = nn.Linear(d_v, d_model)
        self.W_o = nn.Linear(d_model, d_out)

    def scaled_dot_product_attention(self, Q, K, V, attn_mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if attn_mask is not None:
            attn_scores[attn_mask[:, None].tile((1, attn_scores.shape[1], 1, 1))] = -1e9
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, -1).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_out)

    def forward(self, Q, K, V, attn_mask=None):
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        attn_output = self.scaled_dot_product_attention(Q, K, V, attn_mask)
        output = self.combine_heads(attn_output)
        output = self.W_o(output)
        return output


class AttentionLayer(nn.Module):
    def __init__(self, d_q: int, d_k: int, d_v: int, d_model: int, d_out: int, nhead: int = 1, dim_feedforward: int = 128,
                 activation_class: Layer = Relu,
                 layer_norm_eps: float = 1e-5, **kwargs) -> None:
        factory_kwargs = {'device': kwargs.get("device", None), 'dtype': kwargs.get("dtype", None)}
        super(AttentionLayer, self).__init__()
        self.attn = MultiHeadAttention(d_q, d_k, d_v, d_model, d_out, nhead)
        self.linear1 = activation_class(d_out, dim_feedforward, **kwargs)
        self.linear2 = nn.Linear(dim_feedforward, d_out, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_out, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_out, eps=layer_norm_eps, **factory_kwargs)
        self.dropout = nn.Dropout(kwargs.get("dropout", 0.0))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.norm1(self.dropout(self.attn(q, k, v, attn_mask=attn_mask)))
        x = self.norm2(x + self.dropout(self.linear2(self.linear1(x))))
        return x
