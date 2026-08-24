"""Paper-faithful BasicTS implementation of patch-specific graph filtration."""

from __future__ import annotations

import math

import torch
from basicts.modules.norm import RevIN
from torch import nn

from ..config.timefilter_config import TimeFilterConfig


class FilteredGraphBlock(nn.Module):
    def __init__(self, config: TimeFilterConfig) -> None:
        super().__init__()
        if config.hidden_size % config.n_heads:
            raise ValueError("hidden_size must be divisible by n_heads")
        self.n_heads = config.n_heads
        self.head_size = config.hidden_size // config.n_heads
        self.keep_ratio = config.keep_ratio
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(hidden)
        batch, nodes, _ = normalized.shape
        query = self.query(normalized).view(batch, nodes, self.n_heads, self.head_size)
        key = self.key(normalized).view(batch, nodes, self.n_heads, self.head_size)
        value = self.value(normalized).view(batch, nodes, self.n_heads, self.head_size)
        scores = torch.einsum("bihd,bjhd->bhij", query, key) / math.sqrt(self.head_size)
        keep = max(1, min(nodes, math.ceil(nodes * self.keep_ratio)))
        threshold = torch.topk(scores, keep, dim=-1).values[..., -1:]
        filtered = scores.masked_fill(scores < threshold, torch.finfo(scores.dtype).min)
        adjacency = self.dropout(torch.softmax(filtered, dim=-1))
        aggregated = torch.einsum("bhij,bjhd->bihd", adjacency, value).reshape(
            batch, nodes, -1
        )
        hidden = hidden + self.output(aggregated)
        return hidden + self.ffn(self.norm2(hidden))


class TimeFilterForForecasting(nn.Module):
    def __init__(self, config: TimeFilterConfig) -> None:
        super().__init__()
        if not 0.0 < config.keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1]")
        self.num_features = config.num_features
        self.output_len = config.output_len
        self.patch_len = min(config.patch_len, config.input_len)
        self.num_patches = math.ceil(config.input_len / self.patch_len)
        padded_length = self.num_patches * self.patch_len
        self.padding = padded_length - config.input_len
        self.revin = RevIN(config.num_features, affine=False)
        self.patch_embedding = nn.Linear(self.patch_len, config.hidden_size)
        self.blocks = nn.ModuleList(
            [FilteredGraphBlock(config) for _ in range(config.num_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size)
        self.head = nn.Linear(self.num_patches * config.hidden_size, config.output_len)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = self.revin(inputs, "norm").transpose(1, 2)
        if self.padding:
            normalized = torch.cat(
                [normalized, normalized[..., -1:].expand(*normalized.shape[:-1], self.padding)],
                dim=-1,
            )
        patches = normalized.unfold(-1, self.patch_len, self.patch_len)
        batch = patches.shape[0]
        hidden = self.patch_embedding(patches).reshape(batch, -1, self.patch_embedding.out_features)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.norm(hidden).reshape(
            batch, self.num_features, self.num_patches, -1
        )
        prediction = self.head(hidden.flatten(start_dim=-2)).transpose(1, 2)
        return self.revin(prediction, "denorm")
