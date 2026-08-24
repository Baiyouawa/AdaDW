from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class WPMixerConfig(BasicTSModelConfig):
    input_len: int = field(default=None)
    output_len: int = field(default=None)
    num_features: int = field(default=None)
    hidden_size: int = field(default=256)
    intermediate_size: int = field(default=1024)
    num_layers: int = field(default=2)
    wavelet_level: int = field(default=2)
    patch_len: int = field(default=4)
    patch_stride: int = field(default=2)
    dropout: float = field(default=0.1)
