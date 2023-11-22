import torch
import torch.nn.functional as F
from torch import Tensor, nn, einsum
from torch.nn import Module, ModuleList

from beartype import beartype
from beartype.typing import Optional, Union, Tuple, Dict, Any

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

def divisible_by(num, den):
    return (num % den) == 0

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
        video_image_size,
        video_frames_per_timechunk,
        audio_freq_dim,
        audio_time_dim_per_timechunk,
        audio_patch_size: Tuple[int, int],  # (freq, time)
        video_patch_size: Tuple[int, int],  # (spatial, time)
        audio_encoder: Union[Module, Dict[str, Any]],
        video_encoder: Union[Module, Dict[str, Any]],
        text_max_seq_len = 2048,
        encoder_depth = 6,
        decoder_depth = 6,
        combiner_depth = 2,
        combiner_output_num_tokens = 3,
        video_channels = 3,
        attn_dim_head = 64,
        attn_heads = 8,
        attn_layers_kwargs: dict = dict(),
        combiner: Optional[Module] = None,
        combiner_kwargs: dict = dict(),
        autoregressive_wrapper_kwargs: dict = dict(
            pad_value = 0,
            ignore_index = -100
        ),
        av_autoregressive_loss_weight = 1.
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

        self.to_video_tokens = nn.Sequential(
            Rearrange('b c (f pf) (h ph) (w pw) -> b f h w (c pf ph pw)', pf = video_time_patch_size, ph = video_spatial_patch_size, pw = video_spatial_patch_size),
            nn.Linear(video_patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.to_audio_tokens = nn.Sequential(
            Rearrange('b (f pf) (t pt) -> b f t (pf pt)', pf = audio_freq_patch_size, pt = audio_time_patch_size),
            nn.Linear(audio_patch_dim, dim),
            nn.LayerNorm(dim)
        )

        if isinstance(video_encoder, dict):
            video_encoder = Encoder(**{'dim': dim, **video_encoder})

        if isinstance(audio_encoder, dict):
            audio_encoder = Encoder(**{'dim': dim, **audio_encoder})

        self.video_encoder = video_encoder
        self.audio_encoder = audio_encoder

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

        self.text_max_seq_len = text_max_seq_len

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

    @torch.no_grad()
    def generate(
        self,
        *,
        prompt: Tensor,
        seq_len: int,
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
        text: Tensor,
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

            video_tokens = self.to_video_tokens(video)

            video_tokens = rearrange(video_tokens, 'b ... d -> b (...) d')

            encoded_video = self.video_encoder(video_tokens)

            encoded_video = unpack_one(encoded_video, video_frame_ps, '* n d')

        # handle encoding of audio

        if not exists(encoded_audio):
            _, f, t = audio.shape

            assert f == self.audio_freq_dim
            assert divisible_by(t, self.audio_time_dim_per_timechunk)

            audio = rearrange(audio, 'b f (t tc) -> b tc f t', tc = self.audio_time_dim_per_timechunk)
            audio, audio_time_ps = pack_one(audio, '* f t')

            audio_tokens = self.to_audio_tokens(audio)

            audio_tokens = rearrange(audio_tokens, 'b ... d -> b (...) d')

            encoded_audio = self.audio_encoder(audio_tokens)

            encoded_audio = unpack_one(encoded_audio, audio_time_ps, '* n d')

        # ensure audio and video is time aligned

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

        # encode the audio / video tokens autoregressively

        av_embeddings = self.encoder(
            av_encoder_input,
            attn_mask = ~causal_mask,
            rotary_pos_emb = rotary_emb
        )

        if generate:
            generate_seq_len = default(generate_seq_len, self.text_max_seq_len)
            return self.wrapped_decoder.generate(text, seq_len = generate_seq_len, context = av_embeddings)

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
