import torch
import torch.nn.functional as F
from torch import Tensor, nn, einsum
from torch.nn import Module, ModuleList

from beartype import beartype
from beartype.typing import Optional, Union, Tuple

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

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
    return sum(*[map(int, bools)]) == 1

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def l2norm(t):
    return F.normalize(t, dim = -1)

def cosine_sim_loss(x, y):
    x, y = map(l2norm, (x, y))
    return 1. - einsum('b n d, b n d -> b n', x, y).mean()

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
        combiner_kwargs: dict = dict(),
        autoregressive_wrapper_kwargs: dict = dict(
            pad_value = 0,
            ignore_index = -100
        ),
        av_autoregressive_loss_weight = 1.
    ):
        super().__init__()

        # number of tokens per chunk for a/v

        self.audio_timechunk_tokens = audio_timechunk_tokens
        self.video_timechunk_tokens = video_timechunk_tokens

        self.encoded_audio_shape = (audio_timechunk_tokens, dim)
        self.encoded_video_shape = (video_timechunk_tokens, dim)

        # combiner, which they found another transformer (followed by splicing the output) is sufficient

        default_combiner_kwargs = dict(
            dim = dim,
            depth = combiner_depth,
            dim_head = attn_dim_head,
            heads = attn_heads,
        )

        self.combiner = Encoder(
            **{
                **default_combiner_kwargs,
                **attn_layers_kwargs,
                **combiner_kwargs
            }
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

        # for audio/video autoregressive loss

        self.to_encoder_next_token_pred = nn.Sequential(
            Rearrange('b (n c) d -> b n (c d)', c = combiner_output_num_tokens),
            nn.Linear(combiner_output_num_tokens * dim, dim)
        )

        self.av_autoregressive_loss_weight = av_autoregressive_loss_weight

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
        return_loss = True,
        return_loss_breakdown = False
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

        # chunked causal attn mask

        causal_mask = torch.ones((num_time_steps, num_time_steps), device = self.device, dtype = torch.bool).triu(1)
        causal_mask = repeat(causal_mask, 'i j -> (i c1) (j c2)', c1 = self.combiner_output_num_tokens, c2 = self.combiner_output_num_tokens)

        # custom rotary positions for the autoregressive a/v encoder
        # for example, post combined tokens of 3 would be [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, ...]

        seq_positions = torch.arange(num_time_steps, device = self.device)
        seq_positions = repeat(seq_positions, 'n -> (n c)', c = self.combiner_output_num_tokens)

        rotary_emb = self.audio_video_rotary_pos_emb(seq_positions)

        # encode the audio / video tokens autoregressively

        av_embeddings = self.encoder(
            av_encoder_input,
            attn_mask = ~causal_mask,
            rotary_pos_emb = rotary_emb
        )

        if not return_loss:
            return self.decoder(text, context = av_embeddings)

        # av autoregressive cosine sim loss

        next_token_predictions = self.to_encoder_next_token_pred(av_embeddings)

        past, future = next_token_predictions[:, :-1], next_token_predictions[:, 1:]

        av_autoregressive_loss = cosine_sim_loss(past, future)

        # text autoregressive loss

        text_autoregressive_loss = self.wrapped_decoder(text, context = av_embeddings)

        # total loss

        total_loss = text_autoregressive_loss + \
                     av_autoregressive_loss * self.av_autoregressive_loss_weight

        if not return_loss_breakdown:
            return total_loss

        loss_breakdown = (text_autoregressive_loss, av_autoregressive_loss)

        return total_loss, loss_breakdown
