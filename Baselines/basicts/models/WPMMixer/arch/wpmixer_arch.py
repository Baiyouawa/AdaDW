"""BasicTS adapter for WPMixer's multi-resolution patch-mixing architecture."""

from __future__ import annotations

import math

import torch
from basicts.modules.norm import RevIN
from torch import nn

from ..config.wpmixer_config import WPMixerConfig


def _haar_decompose(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if values.shape[-1] % 2:
        values = torch.cat([values, values[..., -1:]], dim=-1)
    even, odd = values[..., 0::2], values[..., 1::2]
    scale = math.sqrt(2.0)
    return (even + odd) / scale, (even - odd) / scale


def _haar_reconstruct(approximation: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
    scale = math.sqrt(2.0)
    even = (approximation + detail) / scale
    odd = (approximation - detail) / scale
    output = torch.empty(
        *even.shape[:-1], even.shape[-1] * 2, device=even.device, dtype=even.dtype
    )
    output[..., 0::2], output[..., 1::2] = even, odd
    return output


class MixerBlock(nn.Module):
    def __init__(
        self,
        num_patches: int,
        hidden_size: int,
        intermediate_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(hidden_size)
        self.token_mixer = nn.Sequential(
            nn.Linear(num_patches, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, num_patches),
        )
        self.channel_norm = nn.LayerNorm(hidden_size)
        self.channel_mixer = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        mixed = self.token_mixer(self.token_norm(values).transpose(-1, -2)).transpose(-1, -2)
        values = residual + mixed
        return values + self.channel_mixer(self.channel_norm(values))


class ResolutionBranch(nn.Module):
    def __init__(
        self,
        input_length: int,
        output_length: int,
        config: WPMixerConfig,
    ) -> None:
        super().__init__()
        patch_len = min(config.patch_len, input_length)
        stride = min(config.patch_stride, patch_len)
        self.patch_len = patch_len
        self.stride = stride
        self.padding = stride
        self.num_patches = (input_length + self.padding - patch_len) // stride + 1
        self.embedding = nn.Linear(patch_len, config.hidden_size)
        self.blocks = nn.ModuleList(
            [
                MixerBlock(
                    self.num_patches,
                    config.hidden_size,
                    config.intermediate_size,
                    config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.hidden_size)
        self.head = nn.Linear(self.num_patches * config.hidden_size, output_length)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        padding = values[..., -1:].expand(*values.shape[:-1], self.padding)
        patches = torch.cat([values, padding], dim=-1).unfold(
            dimension=-1, size=self.patch_len, step=self.stride
        )
        hidden = self.embedding(patches)
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(self.norm(hidden).flatten(start_dim=-2))


class WPMixerForForecasting(nn.Module):
    """Wavelet decomposition followed by shared multi-resolution patch mixers."""

    def __init__(self, config: WPMixerConfig) -> None:
        super().__init__()
        if config.wavelet_level < 1:
            raise ValueError("wavelet_level must be positive")
        input_lengths = []
        output_lengths = []
        input_length, output_length = config.input_len, config.output_len
        for _ in range(config.wavelet_level):
            input_length = (input_length + 1) // 2
            output_length = (output_length + 1) // 2
            input_lengths.append(input_length)
            output_lengths.append(output_length)
        self.output_len = config.output_len
        self.wavelet_level = config.wavelet_level
        self.revin = RevIN(config.num_features, affine=True)
        self.approximation_branch = ResolutionBranch(
            input_lengths[-1], output_lengths[-1], config
        )
        self.detail_branches = nn.ModuleList(
            [
                ResolutionBranch(input_size, output_size, config)
                for input_size, output_size in zip(input_lengths, output_lengths)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = self.revin(inputs, "norm").transpose(1, 2)
        approximation = normalized
        details = []
        for _ in range(self.wavelet_level):
            approximation, detail = _haar_decompose(approximation)
            details.append(detail)
        predicted_approximation = self.approximation_branch(approximation)
        predicted_details = [
            branch(detail) for branch, detail in zip(self.detail_branches, details)
        ]
        prediction = predicted_approximation
        for detail in reversed(predicted_details):
            prediction = _haar_reconstruct(prediction, detail)
        prediction = prediction[..., : self.output_len].transpose(1, 2)
        return self.revin(prediction, "denorm")
