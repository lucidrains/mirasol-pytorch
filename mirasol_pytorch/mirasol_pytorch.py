import torch
import torch.nn.functional as F
from torch import Tensor, nn, einsum
from torch.nn import Module, ModuleList

from x_transformers import Encoder, Decoder

from beartype import beartype

from einops import rearrange, repeat

# helper functions

def exists(v):
    return v is not None

# main class

class Mirasol(Module):
    def __init__(
        self
    ):
        super().__init__()

    def forward(
        self,
        *,
        audio: Tensor,
        video: Tensor,
        text: Tensor
    ):
        return 0.
