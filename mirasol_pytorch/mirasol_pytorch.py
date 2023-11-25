import operator
from functools import partial
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn, einsum
from torch.nn import Module, ModuleList

from beartype import beartype
from beartype.typing import Optional, Union, Tuple, Dict, Any

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

from x_transformers import (
    Encoder,
    Decoder,
    TransformerWrapper,
    AutoregressiveWrapper
)

from x_transformers.x_transformers import RotaryEmbedding

from mirasol_pytorch.distributed import all_gather, get_is_distributed

# helper functions

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def divisible_by(num, den):
    return (num % den) == 0

def only_one_true(*bools):
    return sum(*[map(int, bools)]) == 1

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# tensor helpers

def l2norm(t):
    return F.normalize(t, dim = -1)

def cosine_sim_loss(x, y):
    x, y = map(l2norm, (x, y))
    return 1. - einsum('b n d, b n d -> b n', x, y).mean()

def posemb_sincos_nd(
    t: Tensor,
    temperature: int = 10000,
    dtype = torch.float32
):
    b, *dims, feat_dim, device = *t.shape, t.device
    seq_len = torch.tensor(dims).cumprod(dim = -1)[-1].item()

    arange = partial(torch.arange, device = device)

    num_dims = len(dims)
    two_times_num_dims = 2 * num_dims # 2 because sin and cos of same position

    rounded_feat_dim = feat_dim // num_dims * num_dims
    feat_dim_remainder = feat_dim % num_dims

    omega = arange(rounded_feat_dim // two_times_num_dims) / (rounded_feat_dim // two_times_num_dims - 1)
    omega = 1.0 / (temperature ** omega)
    meshed = torch.meshgrid(*[*map(arange, dims)], indexing = 'ij')

    pos = torch.cat(tuple(m.flatten()[..., None] for m in meshed), dim = 0)
    pos = pos * omega[None, :]

    pos = torch.cat((pos.sin(), pos.cos()))

    pos = rearrange(pos, '(n f) d -> n (f d)', n = seq_len)
    pos = pos.type(dtype)

    return F.pad(pos, (0, feat_dim_remainder))

def mask_with_prob(
    shape: Tuple[int, ...],
    prob: float,
    device = None
) -> Tensor:
    length = shape[-1]
    num_mask = int(prob * length)
    randperm = torch.randn(shape, device = device).argsort(dim = -1)
    return randperm >= num_mask

# main class

Losses = namedtuple('Losses', [
    'text_autoregressive',
    'av_autoregressive',
    'av_recon',
    'text_av_sim_reg'
])

class Mirasol(Module):

    @beartype
    def __init__(
        self,
        *,
        dim,
        num_text_tokens,
        video_image_size,
        video_frames_per_timechunk,
        audio_freq_dim,
        audio_time_dim_per_timechunk,
        audio_patch_size: Tuple[int, int],                          # (freq, time)
        video_patch_size: Tuple[int, int],                          # (spatial, time)
        video_recon_patch_size: Optional[Tuple[int, int]] = None,   # (spatial, time) - they use a smaller video for reconstruction loss
        video_recon_interpolate_mode = 'nearest',
        audio_encoder: Union[Module, Dict[str, Any]],
        video_encoder: Union[Module, Dict[str, Any]],
        num_audio_video_register_tokens = 8,                        # https://arxiv.org/abs/2309.16588
        audio_video_mask_prob = 0.15,                         # in the paper, they used masked tokens presumably, but from the Berkeley forgetful-causal-mask paper, a simple key-value mask should suffice
        text_max_seq_len = 2048,
        text_forgetful_causal_mask_prob = 0.1,                      # https://arxiv.org/abs/2210.13432
        encoder_depth = 6,
        decoder_depth = 6,
        combiner_depth = 2,
        combiner_output_num_tokens = 3,
        video_channels = 3,
        attn_dim_head = 64,
        attn_heads = 8,
        flash_attn = True,
        attn_layers_kwargs: dict = dict(),
        combiner: Optional[Module] = None,
        combiner_kwargs: dict = dict(),
        autoregressive_wrapper_kwargs: dict = dict(
            pad_value = 0,
            ignore_index = -100
        ),
        av_autoregressive_loss_weight = 1.,
        av_reconstruction_loss_weight = 1.,
        sim_reg_loss_weight = 0.
    ):
        super().__init__()

        audio_freq_patch_size, audio_time_patch_size = audio_patch_size
        video_spatial_patch_size, video_time_patch_size = video_patch_size

        assert divisible_by(audio_time_dim_per_timechunk, audio_time_patch_size)
        assert divisible_by(video_frames_per_timechunk, video_time_patch_size)

        assert divisible_by(audio_freq_dim, audio_freq_patch_size)
        assert divisible_by(video_image_size, video_spatial_patch_size)

        audio_timechunk_tokens = (audio_freq_dim // audio_freq_patch_size) * (audio_time_dim_per_timechunk // audio_time_patch_size)
        video_timechunk_tokens = ((video_image_size // video_spatial_patch_size) ** 2) * (video_frames_per_timechunk // video_time_patch_size)

        self.audio_freq_dim = audio_freq_dim

        self.video_channels = video_channels
        self.video_image_size = video_image_size

        self.video_frames_per_timechunk = video_frames_per_timechunk
        self.audio_time_dim_per_timechunk = audio_time_dim_per_timechunk

        video_patch_dim = video_channels * (video_spatial_patch_size ** 2) * video_time_patch_size
        audio_patch_dim = audio_freq_patch_size * audio_time_patch_size

        self.to_timechunked_video = Rearrange('b c (f pf) (h ph) (w pw) -> b f h w (c pf ph pw)', pf = video_time_patch_size, ph = video_spatial_patch_size, pw = video_spatial_patch_size)

        self.to_video_tokens = nn.Sequential(
            nn.Linear(video_patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.to_timechunked_audio = Rearrange('b (f pf) (t pt) -> b f t (pf pt)', pf = audio_freq_patch_size, pt = audio_time_patch_size)

        self.to_audio_tokens = nn.Sequential(
            nn.Linear(audio_patch_dim, dim),
            nn.LayerNorm(dim)
        )

        default_vit_kwargs = dict(
            dim = dim,
            flash_attn = flash_attn
        )

        if isinstance(video_encoder, dict):
            video_encoder = Encoder(**{
                **default_vit_kwargs,
                **video_encoder
            })

        if isinstance(audio_encoder, dict):
            audio_encoder = Encoder(**{
                **default_vit_kwargs,
                **audio_encoder
            })

        self.video_encoder = video_encoder
        self.audio_encoder = audio_encoder

        self.video_register_tokens = nn.Parameter(torch.randn(num_audio_video_register_tokens, dim))
        self.audio_register_tokens = nn.Parameter(torch.randn(num_audio_video_register_tokens, dim))

        # number of tokens per chunk for a/v

        self.audio_timechunk_tokens = audio_timechunk_tokens
        self.video_timechunk_tokens = video_timechunk_tokens

        self.encoded_audio_shape = (audio_timechunk_tokens, dim)
        self.encoded_video_shape = (video_timechunk_tokens, dim)

        # combiner, which they found another transformer (followed by splicing the output) is sufficient

        if not exists(combiner):
            default_combiner_kwargs = dict(
                dim = dim,
                depth = combiner_depth,
                dim_head = attn_dim_head,
                heads = attn_heads,
                flash_attn = flash_attn
            )

            combiner = Encoder(
                **{
                    **default_combiner_kwargs,
                    **attn_layers_kwargs,
                    **combiner_kwargs
                }
            )

        self.combiner = combiner
        self.combiner_output_num_tokens = combiner_output_num_tokens

        # a/v rotary embedding

        self.audio_video_rotary_pos_emb = RotaryEmbedding(attn_dim_head)

        # masking of combined a/v tokens

        self.audio_video_mask_prob = audio_video_mask_prob
        self.get_audio_video_self_attn_mask = partial(mask_with_prob, prob = audio_video_mask_prob)

        # a/v encoder

        self.encoder = Encoder(
            dim = dim,
            depth = encoder_depth,
            dim_head = attn_dim_head,
            heads = attn_heads,
            num_mem_kv = 1,
            flash_attn = flash_attn,
            **attn_layers_kwargs
        )

        # for audio/video autoregressive loss

        self.to_flattened_combined_tokens = Rearrange('b (n c) d -> b n (c d)', c = combiner_output_num_tokens)

        flattened_embedding_dim = combiner_output_num_tokens * dim
        self.to_encoder_next_token_pred = nn.Linear(flattened_embedding_dim, dim)

        self.av_autoregressive_loss_weight = av_autoregressive_loss_weight

        # for autoregressive reconstruction loss

        self.should_resize_video_for_recon = exists(video_recon_patch_size)

        if self.should_resize_video_for_recon:
            assert all([*map(operator.le, video_recon_patch_size, video_patch_size)])

            video_recon_spatial_size, video_recon_time_size = video_recon_patch_size

            self.video_recon_shape = (video_recon_spatial_size, video_recon_spatial_size, video_recon_time_size)
            self.video_recon_interpolate_mode = video_recon_interpolate_mode

            cumprod_video_chunk_dims = (video_recon_spatial_size ** 2) * video_recon_time_size
        else:
            cumprod_video_chunk_dims = (video_image_size ** 2) * video_frames_per_timechunk

        self.to_reconstructed_video = nn.Linear(flattened_embedding_dim, cumprod_video_chunk_dims * video_channels)

        self.to_reconstructed_audio = nn.Linear(flattened_embedding_dim, audio_freq_dim * audio_time_dim_per_timechunk)

        self.av_reconstruction_loss_weight = av_reconstruction_loss_weight

        # text decoder

        self.text_max_seq_len = text_max_seq_len

        self.start_token_id = num_text_tokens

        self.decoder = TransformerWrapper(
            num_tokens = num_text_tokens + 1,
            max_seq_len = text_max_seq_len,
            attn_layers = Decoder(
                dim = dim,
                depth = decoder_depth,
                dim_head = attn_dim_head,
                heads = attn_heads,
                num_mem_kv = 1,
                cross_attend = True,
                rotary_pos_emb = True,
                flash_attn = flash_attn,
                **attn_layers_kwargs
            )
        )

        self.wrapped_decoder = AutoregressiveWrapper(
            self.decoder,
            mask_prob = text_forgetful_causal_mask_prob,
            **autoregressive_wrapper_kwargs
        )

        # similarity reg loss

        has_sim_reg = sim_reg_loss_weight > 0.
        self.has_sim_reg = has_sim_reg
        self.sim_reg_loss_weight = sim_reg_loss_weight

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def generate(
        self,
        *,
        seq_len: int,
        prompt: Optional[Tensor] = None,
        **kwargs
    ):
        was_training = self.training
        self.eval()

        assert 'generate' not in kwargs
        assert 'generate_seq_len' not in kwargs

        out = self.forward(
            text = prompt,
            generate = True,
            generate_seq_len = seq_len,
            **kwargs
        )

        self.train(was_training)
        return out

    @beartype
    def forward(
        self,
        *,
        audio: Optional[Tensor] = None,
        video: Optional[Tensor] = None,
        encoded_audio: Optional[Tensor] = None,
        encoded_video: Optional[Tensor] = None,
        text: Optional[Tensor] = None,
        text_mask: Optional[Tensor] = None,
        return_loss = True,
        return_loss_breakdown = False,
        generate = False,
        generate_seq_len = None
    ):
        assert only_one_true(exists(audio), exists(encoded_audio))
        assert only_one_true(exists(video), exists(encoded_video))

        # handle encoding of video

        if not exists(encoded_video):
            _, c, t, h, w = video.shape

            assert c == self.video_channels
            assert (h == self.video_image_size) and (w == self.video_image_size)
            assert divisible_by(t, self.video_frames_per_timechunk)

            video = rearrange(video, 'b c (f fc) h w -> b f c fc h w', fc = self.video_frames_per_timechunk)
            video, video_frame_ps = pack_one(video, '* c fc h w')

            timechunked_video = self.to_timechunked_video(video)

            video_tokens = self.to_video_tokens(timechunked_video)
            video_pos_emb = posemb_sincos_nd(video_tokens)

            video_tokens = rearrange(video_tokens, 'b ... d -> b (...) d')

            video_tokens = video_tokens + video_pos_emb

            video_register_tokens = repeat(self.video_register_tokens, 'n d -> b n d', b = video_tokens.shape[0])
            video_tokens, register_ps = pack([video_register_tokens, video_tokens], 'b * d')

            encoded_video = self.video_encoder(video_tokens)

            _, encoded_video = unpack(video_tokens, register_ps, 'b * d')

            encoded_video = unpack_one(encoded_video, video_frame_ps, '* n d')

        # handle encoding of audio

        if not exists(encoded_audio):
            _, f, t = audio.shape

            assert f == self.audio_freq_dim
            assert divisible_by(t, self.audio_time_dim_per_timechunk)

            audio = rearrange(audio, 'b f (t tc) -> b tc f t', tc = self.audio_time_dim_per_timechunk)
            audio, audio_time_ps = pack_one(audio, '* f t')

            timechunked_audio = self.to_timechunked_audio(audio)

            audio_tokens = self.to_audio_tokens(timechunked_audio)
            audio_pos_emb = posemb_sincos_nd(audio_tokens)

            audio_tokens = rearrange(audio_tokens, 'b ... d -> b (...) d')

            audio_tokens = audio_tokens + audio_pos_emb

            audio_register_tokens = repeat(self.audio_register_tokens, 'n d -> b n d', b = audio_tokens.shape[0])
            audio_tokens, register_ps = pack([audio_register_tokens, audio_tokens], 'b * d')

            encoded_audio = self.audio_encoder(audio_tokens)

            _, encoded_audio = unpack(encoded_audio, register_ps, 'b * d')

            encoded_audio = unpack_one(encoded_audio, audio_time_ps, '* n d')

        # ensure audio and video is time aligned

        batch = encoded_audio.shape[0]

        audio_time_frames = encoded_audio.shape[1]
        video_time_frames = encoded_video.shape[1]

        frames = min(audio_time_frames, video_time_frames)

        encoded_audio = encoded_audio[:, :frames]
        encoded_video = encoded_video[:, :frames]

        # validate encoded audio / video

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

        # determine which tokens are masked
        # and use `self_attn_kv_mask` kwarg on x-transformers

        self_attn_kv_mask = None

        if self.audio_video_mask_prob > 0.:
            self_attn_kv_mask = self.get_audio_video_self_attn_mask((batch, num_time_steps), device = self.device)

            self_attn_kv_mask = repeat(self_attn_kv_mask, 'b n -> b (n c)', c = self.combiner_output_num_tokens)

        # encode the audio / video tokens autoregressively

        av_embeddings = self.encoder(
            av_encoder_input,
            attn_mask = ~causal_mask,
            self_attn_kv_mask = self_attn_kv_mask,
            rotary_pos_emb = rotary_emb
        )

        # handle start token for text

        if exists(text):
            text = F.pad(text, (1, 0), value = self.start_token_id)
        else:
            text = torch.full((batch, 1), self.start_token_id, device = self.device, dtype = torch.long)

        # if generate flag is passed in, generate using `text` as prompt

        if generate:
            generate_seq_len = default(generate_seq_len, self.text_max_seq_len)
            return self.wrapped_decoder.generate(text, seq_len = generate_seq_len, context = av_embeddings)

        if not return_loss:
            return self.decoder(text, context = av_embeddings)

        assert text.shape[-1] > 1

        # flattened combined tokens

        flattened_embeddings = self.to_flattened_combined_tokens(av_embeddings)

        # av autoregressive cosine sim loss

        next_token_predictions = self.to_encoder_next_token_pred(flattened_embeddings)

        past, future = next_token_predictions[:, :-1], next_token_predictions[:, 1:]

        av_autoregressive_loss = cosine_sim_loss(past, future)

        # av autoregressive reconstruction loss (which is also cosine sim, interestingly)

        recon_loss = 0.

        if exists(encoded_video):
            reconstructed_video = self.to_reconstructed_video(flattened_embeddings)

            if self.should_resize_video_for_recon:
                timechunked_video = F.interpolate(video, self.video_recon_shape, mode = self.video_recon_interpolate_mode)
                timechunked_video = rearrange(timechunked_video, 'b d ... -> b ... d')

            timechunked_video = unpack_one(timechunked_video, video_frame_ps, '* f h w d')
            timechunked_video = rearrange(timechunked_video, 'b c f h w d -> b c (f h w d)', b = batch)

            recon_video_loss = cosine_sim_loss(reconstructed_video[:, :-1], timechunked_video[:, 1:num_time_steps])

            recon_loss = recon_loss + recon_video_loss

        if exists(encoded_audio):
            reconstructed_audio = self.to_reconstructed_audio(flattened_embeddings)
            timechunked_audio = unpack_one(timechunked_audio, audio_time_ps, '* f t d')
            timechunked_audio = rearrange(timechunked_audio, 'b c f t d -> b c (f t d)')

            recon_audio_loss = cosine_sim_loss(reconstructed_audio[:, :-1], timechunked_audio[:, 1:num_time_steps])
            recon_loss = recon_loss + recon_audio_loss

        # text autoregressive loss

        text_autoregressive_loss, decoder_outputs = self.wrapped_decoder(
            text,
            context = av_embeddings,
            return_outputs = True
        )

        # similarity regularization loss

        sim_reg_loss = 0.

        if self.has_sim_reg:
            _, decoder_intermediates = decoder_outputs

            text_embed = decoder_intermediates.last_hidden

            if exists(text_mask):
                text_embed_len = text_embed.shape[-2]
                text_mask[:, :text_embed_len]
                text_embed = text_embed.masked_fill(~text_mask[..., None], -torch.finfo(text_embed.dtype).max)

            av_embed = flattened_embeddings

            av_embed, text_embed = map(lambda t: reduce(t, 'b n d -> b d', 'max'), (av_embed, text_embed))
            av_embed, text_embed = map(l2norm, (av_embed, text_embed))

            if get_is_distributed():
                av_embed = all_gather(av_embed, 0, None)
                text_embed = all_gather(text_embed, 0, None)

            av_sim, text_sim = map(lambda t: einsum('i d, j d -> i j', t, t), (av_embed, text_embed))

            sim_reg_loss = F.mse_loss(av_sim, text_sim)

        # total loss

        total_loss = text_autoregressive_loss + \
                     av_autoregressive_loss * self.av_autoregressive_loss_weight + \
                     recon_loss * self.av_reconstruction_loss_weight + \
                     sim_reg_loss * self.sim_reg_loss_weight

        if not return_loss_breakdown:
            return total_loss

        loss_breakdown = Losses(text_autoregressive_loss, av_autoregressive_loss, recon_loss, sim_reg_loss)

        return total_loss, loss_breakdown
