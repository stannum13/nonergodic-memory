"""Deliberately small causal sequence predictors."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class GRUPredictor(nn.Module):
    def __init__(self, vocab_size: int, width: int, layers: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, width)
        self.recurrent = nn.GRU(width, width, num_layers=layers, batch_first=True)
        self.output = nn.Linear(width, vocab_size)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        hidden, _ = self.recurrent(self.embedding(tokens))
        return self.output(hidden), hidden


class TransformerPredictor(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        width: int,
        layers: int = 1,
        heads: int = 2,
        max_length: int = 256,
    ):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Embedding(max_length, width)
        block = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=4 * width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(block, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, vocab_size)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)[None, :, :]
        mask = torch.triu(
            torch.ones(tokens.shape[1], tokens.shape[1], device=tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        hidden = self.norm(self.blocks(hidden, mask=mask, is_causal=True))
        return self.output(hidden), hidden


def build_model(name: str, vocab_size: int, config: dict) -> nn.Module:
    common = {
        "vocab_size": vocab_size,
        "width": int(config["width"]),
        "layers": int(config.get("layers", 1)),
    }
    if name == "gru":
        return GRUPredictor(**common)
    if name == "transformer":
        return TransformerPredictor(
            **common,
            heads=int(config.get("heads", 2)),
            max_length=int(config.get("max_length", 256)),
        )
    raise ValueError(f"unknown model: {name}")

