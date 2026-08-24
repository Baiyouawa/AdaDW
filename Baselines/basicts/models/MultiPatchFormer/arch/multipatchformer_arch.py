"""BasicTS implementation of MultiPatchFormer's multiscale forecasting path."""

from __future__ import annotations

import math

import torch
from basicts.modules.norm import RevIN
from torch import nn

from ..config.multipatchformer_config import MultiPatchFormerConfig


class EncoderBlock(nn.Module):
    def __init__(self, config: MultiPatchFormerConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.attention = nn.MultiheadAttention(
            config.hidden_size,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )

    def forward(self, values: torch.Tensor, causal: bool = False) -> torch.Tensor:
        normalized = self.norm1(values)
        mask = None
        if causal:
            mask = nn.Transformer.generate_square_subsequent_mask(
                values.shape[1], device=values.device, dtype=values.dtype
            )
        attended, _ = self.attention(
            normalized, normalized, normalized, attn_mask=mask, need_weights=False
        )
        values = values + attended
        return values + self.ffn(self.norm2(values))


class MultiPatchFormerForForecasting(nn.Module):
    def __init__(self, config: MultiPatchFormerConfig) -> None:
        super().__init__()
        if config.hidden_size % config.n_heads:
            raise ValueError("hidden_size must be divisible by n_heads")
        branches = [
            (length, stride)
            for length, stride in zip(config.patch_lengths, config.patch_strides)
            if length <= config.input_len
        ]
        if not branches:
            branches = [(config.input_len, config.input_len)]
        self.num_features = config.num_features
        self.output_len = config.output_len
        self.target_patches = max(math.ceil(config.input_len / stride) for _, stride in branches)
        self.revin = RevIN(config.num_features, affine=True)
        self.patch_embeddings = nn.ModuleList(
            [nn.Conv1d(1, config.hidden_size, length, stride=stride) for length, stride in branches]
        )
        self.patch_paddings = [stride for _, stride in branches]
        branch_patch_counts = [
            (config.input_len + stride - length) // stride + 1
            for length, stride in branches
        ]
        self.patch_projections = nn.ModuleList(
            [
                nn.Identity()
                if count == self.target_patches
                else nn.Linear(count, self.target_patches)
                for count in branch_patch_counts
            ]
        )
        self.fusion = nn.Linear(len(branches) * config.hidden_size, config.hidden_size)
        self.position = nn.Parameter(torch.zeros(1, self.target_patches, config.hidden_size))
        nn.init.normal_(self.position, std=0.02)
        self.temporal_blocks = nn.ModuleList(
            [EncoderBlock(config) for _ in range(config.num_layers)]
        )
        self.channel_projection = nn.Linear(
            self.target_patches * config.hidden_size, config.hidden_size
        )
        self.channel_blocks = nn.ModuleList(
            [EncoderBlock(config) for _ in range(config.num_layers)]
        )
        segment_count = min(config.prediction_segments, config.output_len)
        base, remainder = divmod(config.output_len, segment_count)
        segment_sizes = [base + (index < remainder) for index in range(segment_count)]
        self.prediction_heads = nn.ModuleList()
        produced = 0
        for segment_size in segment_sizes:
            self.prediction_heads.append(
                nn.Linear(config.hidden_size + produced, segment_size)
            )
            produced += segment_size

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = self.revin(inputs, "norm").transpose(1, 2)
        batch, channels, _ = normalized.shape
        flattened = normalized.reshape(batch * channels, 1, -1)
        embedded_branches = []
        for embedding, projection, padding in zip(
            self.patch_embeddings, self.patch_projections, self.patch_paddings
        ):
            padded = torch.cat(
                [flattened, flattened[..., -1:].expand(*flattened.shape[:-1], padding)],
                dim=-1,
            )
            branch = embedding(padded)
            branch = projection(branch).transpose(1, 2)
            embedded_branches.append(branch)
        temporal = self.fusion(torch.cat(embedded_branches, dim=-1)) + self.position
        for block in self.temporal_blocks:
            temporal = block(temporal, causal=True)
        channel_tokens = self.channel_projection(temporal.reshape(batch, channels, -1))
        for block in self.channel_blocks:
            channel_tokens = block(channel_tokens)
        segments = []
        for head in self.prediction_heads:
            head_input = torch.cat([channel_tokens, *segments], dim=-1) if segments else channel_tokens
            segments.append(head(head_input))
        prediction = torch.cat(segments, dim=-1).transpose(1, 2)
        return self.revin(prediction[..., : self.num_features], "denorm")
