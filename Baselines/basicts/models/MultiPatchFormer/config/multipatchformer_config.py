from dataclasses import dataclass, field
from typing import Tuple

from basicts.configs import BasicTSModelConfig


@dataclass
class MultiPatchFormerConfig(BasicTSModelConfig):
    input_len: int = field(default=None)
    output_len: int = field(default=None)
    num_features: int = field(default=None)
    hidden_size: int = field(default=256)
    intermediate_size: int = field(default=512)
    num_layers: int = field(default=1)
    n_heads: int = field(default=8)
    patch_lengths: Tuple[int, ...] = field(default=(8, 16, 24, 32))
    patch_strides: Tuple[int, ...] = field(default=(8, 8, 7, 6))
    prediction_segments: int = field(default=8)
    dropout: float = field(default=0.2)
