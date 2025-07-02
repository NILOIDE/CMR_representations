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


class MultiHeadAttention(nn.Module):
    def __init__(self, d_q, d_k, d_v, d_model, d_out=None, num_heads=1, **kwargs):
        """
        :param d_q: Dimension size of Query input
        :param d_k: Dimension size of Keys input
        :param d_v: Dimension size of Values input
        :param d_model: Dimensions of intermediate mapping of all 3 inputs
        :param d_out: Dimensions of output
        """
        super(MultiHeadAttention, self).__init__()
        d_out = d_out if d_out is not None else d_model
        factory_kwargs = {'device': kwargs.get("device", None), 'dtype': kwargs.get("dtype", None)}

        self.q_d = d_q
        self.k_d = d_k
        self.v_d = d_v
        self.d_out = d_model
        self.num_heads = num_heads
        self.d_k = d_k // num_heads

        self.W_q = torch.nn.Parameter(torch.empty((d_q, d_model), **factory_kwargs))
        self.W_k = torch.nn.Parameter(torch.empty((d_k, d_model), **factory_kwargs))
        self.W_v = torch.nn.Parameter(torch.empty((d_v, d_model), **factory_kwargs))
        self.W_o = torch.nn.Parameter(torch.empty((d_model, d_out), **factory_kwargs))
        torch.nn.init.xavier_uniform_(self.W_q)
        torch.nn.init.xavier_uniform_(self.W_k)
        torch.nn.init.xavier_uniform_(self.W_v)
        torch.nn.init.xavier_uniform_(self.W_o)

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
        Q = Q @ self.W_q
        K = K @ self.W_k
        V = V @ self.W_v
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        attn_output = self.scaled_dot_product_attention(Q, K, V, attn_mask)
        output = self.combine_heads(attn_output)
        output = output @ self.W_o
        return output


class AttentionLayer(nn.Module):
    def __init__(self, d_q: int, d_k: int, d_v: int, d_model: int, d_out: int, nhead: int = 1, dim_feedforward: int = 128,
                 activation_class: Layer = Relu,
                 layer_norm_eps: float = 1e-5, **kwargs) -> None:
        factory_kwargs = {'device': kwargs.get("device", None), 'dtype': kwargs.get("dtype", None)}
        super(AttentionLayer, self).__init__()
        self.attn = MultiHeadAttention(d_q, d_k, d_v, d_model, d_out, nhead, **kwargs)
        self.linear1 = activation_class(d_out, dim_feedforward, **kwargs)
        self.linear2 = nn.Linear(dim_feedforward, d_out, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_out, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_out, eps=layer_norm_eps, **factory_kwargs)
        self.dropout = nn.Dropout(kwargs.get("dropout", 0.0))

    def forward(self, q: torch.Tensor, k: torch.Tensor = None, v: torch.Tensor = None,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if k is None:
            v = k = q  # If user only provives q, we're dealing with self-attention and all 3 sets should be the same
        elif v is None:
            v = k  # If user provided q and k, we're dealing with cross-atention and v should be the same as k
        x = self.norm1(self.dropout(self.attn(q, k, v, attn_mask=attn_mask)))
        x = self.norm2(x + self.dropout(self.linear2(self.linear1(x))))
        return x



class CrossAttentionLayer(nn.Module):
    r"""
    -- TAKEN FROM nn.TransformerEncoderLayer AND MODIFIED TO ACCEPT OUR CUSTOM ACTIVATION LAYERS --

    CrossAttentionLayer is made up of attn and feedforward network.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation_class: custom callable layer class (see model/layers.py).
            These typically include a linear layer followed by an activation function.
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, d_out: int, nhead: int, dim_feedforward: int = 128, dropout: float = 0.1,
                 activation_class: Layer = Relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 **kwargs) -> None:
        factory_kwargs = {'device': kwargs.get("device", None), 'dtype': kwargs.get("dtype", None)}
        super(CrossAttentionLayer, self).__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = activation_class(d_model, dim_feedforward, **kwargs)
        self.linear2 = nn.Linear(dim_feedforward, d_out, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_out, eps=layer_norm_eps, **factory_kwargs)
        self.dropout = nn.Dropout(kwargs.get("dropout", 0.0))

    def forward(self, decoding_tokens: torch.Tensor, info_tokens: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Pass the input through the decoder layer.

        Args:
            decoding_tokens: tokens to be processed by the decoder (required).
                These will be the embedded reconstruction coordinates.
            info_tokens: tokens meant to introduce information to the decoding process (required).
                These will be the encoder's output tokens (separate input coordinates passed by the encoder).
            attn_mask: a 2D or 3D mask preventing attention to certain positions.
                Must be of shape (L,S) or (N⋅num_heads,L,S), where N is the batch size,
                L is the target sequence length, and S is the source sequence length.
                A 2D mask will be broadcasted across the batch while a 3D mask allows for a
                different mask for each entry in the batch.
                Binary and float masks are supported.
                For a binary mask, a True value indicates that the corresponding position is not allowed to attend.
                For a float mask, the mask values will be added to the attention weight.
                If both attn_mask and key_padding_mask are supplied, their types should match.
            key_padding_mask: a mask of shape (N,S) indicating which elements within key to ignore for
                the purpose of attention (i.e. treat as “padding”). For unbatched query, shape should be (S).
                Binary and float masks are supported.
                For a binary mask, a True value indicates that the
                corresponding key value will be ignored for the purpose of attention.
                For a float mask, it will be directly added to the corresponding key value.

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = decoding_tokens
        if self.norm_first:
            x = x + self._ca_block(self.norm1(x), self.norm1(info_tokens), attn_mask, key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._ca_block(x, info_tokens, attn_mask, key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # attention block
    def _ca_block(self, decode_tokens: torch.Tensor, info_tokens: torch.Tensor,
                  attn_mask: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.cross_attn(decode_tokens, info_tokens, info_tokens,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            need_weights=False)[0]
        return self.dropout(x)

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.linear2(x)
        return self.dropout(x)


class SelfAttentionLayer(nn.Module):
    r"""
    -- TAKEN FROM nn.TransformerEncoderLayer AND MODIFIED TO ACCEPT OUR CUSTOM ACTIVATION LAYERS --

    TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation_class: custom callable layer class (see model/layers.py).
            These typically include a linear layer followed by an activation function.
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, d_out: int, nhead: int, dim_feedforward: int = 128, dropout: float = 0.1,
                 activation_class: Layer = Relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 **kwargs) -> None:
        factory_kwargs = {'device': kwargs.get("device", None), 'dtype': kwargs.get("dtype", None)}
        super(SelfAttentionLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                               **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = activation_class(d_model, dim_feedforward, **kwargs)
        self.linear2 = nn.Linear(dim_feedforward, d_out, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_out, eps=layer_norm_eps, **factory_kwargs)
        self.dropout = nn.Dropout(kwargs.get("dropout", 0.0))

    def forward(self, encoding_tokens: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            encoding_tokens: the token set to the encoder layer (required).
            attn_mask: a 2D or 3D mask preventing attention to certain positions.
                Must be of shape (L,S) or (N⋅num_heads,L,S), where N is the batch size,
                L is the target sequence length, and S is the source sequence length.
                A 2D mask will be broadcasted across the batch while a 3D mask allows for a
                different mask for each entry in the batch.
                Binary and float masks are supported.
                For a binary mask, a True value indicates that the corresponding position is not allowed to attend.
                For a float mask, the mask values will be added to the attention weight.
                If both attn_mask and key_padding_mask are supplied, their types should match.
            key_padding_mask: a mask of shape (N,S) indicating which elements within key to ignore for
                the purpose of attention (i.e. treat as “padding”). For unbatched query, shape should be (S).
                Binary and float masks are supported.
                For a binary mask, a True value indicates that the
                corresponding key value will be ignored for the purpose of attention.
                For a float mask, it will be directly added to the corresponding key value.

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = encoding_tokens
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), attn_mask, key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, attn_mask, key_padding_mask))
            x = self.norm2(x + self._ff_block(x))  # TODO: Might wanna remove skip connection to allow for variable ouput size

        return x

    # self-attention block
    def _sa_block(self, x: torch.Tensor,
                  attn_mask: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout(x)

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.linear2(x)
        return self.dropout(x)
