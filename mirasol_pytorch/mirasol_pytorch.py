import torch
import torch.nn.functional as F
from torch import Tensor, nn, einsum
from torch.nn import Module, ModuleList

from beartype import beartype
from beartype.typing import Optional, Union, Tuple
from einops import rearrange, repeat, pack, unpack

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

def only_one_true(*bools):
    return sum(*[map(int, bools)])

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# main class

class Mirasol(Module):

    @beartype
    def __init__(
        self,
        *,
        dim,
        num_text_tokens,
        audio_timechunk_tokens = 16,
        video_timechunk_tokens = 16,
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

        # number of tokens per chunk for a/v

        self.audio_timechunk_tokens = audio_timechunk_tokens
        self.video_timechunk_tokens = video_timechunk_tokens

        self.encoded_audio_shape = (audio_timechunk_tokens, dim)
        self.encoded_video_shape = (video_timechunk_tokens, dim)

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
                cross_attend = True,
                rotary_pos_emb = True,
                **attn_layers_kwargs
            )
        )

        self.wrapped_decoder = AutoregressiveWrapper(
            self.decoder,
            **autoregressive_wrapper_kwargs
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @beartype
    def forward(
        self,
        *,
        audio: Optional[Tensor] = None,
        video: Optional[Tensor] = None,
        encoded_audio: Optional[Tensor] = None,
        encoded_video: Optional[Tensor] = None,
        text: Tensor,
        return_loss = True
    ):
        assert only_one_true(exists(audio), exists(encoded_audio))
        assert only_one_true(exists(video), exists(encoded_video))

        assert encoded_audio.shape[:2] == encoded_video.shape[:2]

        assert encoded_audio.shape[-2:] == self.encoded_audio_shape
        assert encoded_video.shape[-2:] == self.encoded_video_shape

        # use the transformer combiner strategy for combining audio and video tokens

        audio_and_video_tokens, _ = pack((encoded_audio, encoded_video), 'b n * d')
        audio_and_video_tokens, combine_ps = pack_one(audio_and_video_tokens, '* n d')

        combined_audio_video_tokens = self.combiner(audio_and_video_tokens)

        combined_audio_video_tokens = combined_audio_video_tokens[..., -self.combiner_output_num_tokens:, :]

        combined_audio_video_tokens = unpack_one(combined_audio_video_tokens, combine_ps, '* n d')

        num_time_steps = combined_audio_video_tokens.shape[1]

        av_encoder_input = rearrange(combined_audio_video_tokens, 'b ... d -> b (...) d')

        # custom rotary positions for the autoregressive a/v encoder
        # for example, post combined tokens of 3 would be [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, ...]

        seq_positions = torch.arange(num_time_steps, device = self.device)
        seq_positions = repeat(seq_positions, 'n -> (n c)', c = self.combiner_output_num_tokens)

        rotary_emb = self.audio_video_rotary_pos_emb(seq_positions)

        # encode the audio / video tokens autoregressively

        av_embeddings = self.encoder(av_encoder_input, rotary_pos_emb = rotary_emb)

        if not return_loss:
            return self.decoder(text, context = av_embeddings)

        return self.wrapped_decoder(text, context = av_embeddings)
