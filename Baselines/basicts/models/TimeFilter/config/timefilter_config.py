from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class TimeFilterConfig(BasicTSModelConfig):
    input_len: int = field(default=None)
    output_len: int = field(default=None)
    num_features: int = field(default=None)
    hidden_size: int = field(default=128)
    intermediate_size: int = field(default=256)
    num_layers: int = field(default=2)
    n_heads: int = field(default=4)
    patch_len: int = field(default=4)
    keep_ratio: float = field(default=0.5)
    dropout: float = field(default=0.1)
