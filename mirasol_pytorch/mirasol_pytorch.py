import torch
import torch.nn.functional as F
from torch import Tensor, nn, einsum
from torch.nn import Module, ModuleList

from beartype import beartype
from einops import rearrange, repeat

from x_transformers import (
    Encoder,
    Decoder,
    TransformerWrapper,
    AutoregressiveWrapper
)

from x_transformers.x_transformers import RotaryEmbedding

# helper functions

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

# main class

class Mirasol(Module):

    @beartype
    def __init__(
        self,
        *,
        dim,
        num_text_tokens,
        text_max_seq_len = 2048,
        encoder_depth = 6,
        decoder_depth = 6,
        combiner_depth = 2,
        combiner_output_num_tokens = 3,
        attn_dim_head = 64,
        attn_heads = 8,
        attn_layers_kwargs: dict = dict(),
        autoregressive_wrapper_kwargs: dict = dict(
            pad_value = 0,
            ignore_index = -100
        )
    ):
        super().__init__()

        # combiner, which they found another transformer (followed by splicing the output) is sufficient

        self.combiner = Encoder(
            dim = dim,
            depth = combiner_depth,
            dim_head = attn_dim_head,
            heads = attn_heads,
            *attn_layers_kwargs
        )

        self.combiner_output_num_tokens = combiner_output_num_tokens

        # a/v rotary embedding

        self.audio_video_rotary_pos_emb = RotaryEmbedding(attn_dim_head)

        # a/v encoder

        self.encoder = Encoder(
            dim = dim,
            depth = encoder_depth,
            dim_head = attn_dim_head,
            heads = attn_heads,
            **attn_layers_kwargs
        )

        # text decoder

        self.decoder = TransformerWrapper(
            num_tokens = num_text_tokens,
            max_seq_len = text_max_seq_len,
            attn_layers = Decoder(
                dim = dim,
                depth = decoder_depth,
                dim_head = attn_dim_head,
                heads = attn_heads,
                rotary_pos_emb = True,
                **attn_layers_kwargs
            )
        )

        self.wrapped_decoder = AutoregressiveWrapper(
            self.decoder,
            **autoregressive_wrapper_kwargs
        )

    @beartype
    def forward(
        self,
        *,
        audio: Tensor,
        video: Tensor,
        text: Tensor,
        return_loss = True
    ):

        if not return_loss:
            return self.decoder(text)

        return self.wrapped_decoder(text)
