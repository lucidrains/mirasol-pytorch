<img src="./mirasol.png" width="450px"></img>

## Mirasol - Pytorch (wip)

Implementation of <a href="https://arxiv.org/abs/2311.05698">Mirasol</a>, Multimodal Autoregressive model out of Google Deepmind, in Pytorch

Will simply implement the Transformer Combiner and omit the other variants.

## Appreciation

- <a href="https://stability.ai/">StabilityAI</a>, <a href="https://a16z.com/supporting-the-open-source-ai-community/">A16Z Open Source AI Grant Program</a>, and <a href="https://huggingface.co/">🤗 Huggingface</a> for the generous sponsorships, as well as my other sponsors, for affording me the independence to open source current artificial intelligence research

## Install

```bash
$ pip install mirasol-pytorch
```

## Usage

```python
import torch
from mirasol_pytorch import Mirasol

model = Mirasol(
    dim = 512,
    num_text_tokens = 256,
    video_image_size = 128,
    video_frames_per_timechunk = 2,
    audio_freq_dim = 64,
    audio_time_dim_per_timechunk = 32,
    audio_patch_size = (32, 16),
    video_patch_size = (64, 2),
    audio_encoder = dict(
        dim = 512,
        depth = 2
    ),
    video_encoder = dict(
        dim = 512,
        depth = 2
    )
)

audio = torch.randn(1, 64, 1024)
video = torch.randn(1, 3, 12, 128, 128)

text = torch.randint(0, 256, (1, 1024))

loss = model(
    audio = audio,
    video = video,
    text = text
)

loss.backward()

# after much training

sampled_text = model.generate(
    prompt = text[:, 0],
    audio = audio,
    video = video,
    seq_len = 2
)
```

## Todo

- [ ] positional embeddings for video and audio encoder
- [ ] enable register tokens for both video and audio encoder, inline with new research
- [ ] add audio and video reconstruction losses
- [ ] text generation code
- [ ] add similarity regularization from TTS research
- [ ] auto-handle start token for decoder

## Citations

```bibtex
@article{Piergiovanni2023Mirasol3BAM,
    title   = {Mirasol3B: A Multimodal Autoregressive model for time-aligned and contextual modalities},
    author  = {A. J. Piergiovanni and Isaac Noble and Dahun Kim and Michael S. Ryoo and Victor Gomes and Anelia Angelova},
    journal = {ArXiv},
    year    = {2023},
    volume  = {abs/2311.05698},
    url     = {https://api.semanticscholar.org/CorpusID:265129010}
}
```
