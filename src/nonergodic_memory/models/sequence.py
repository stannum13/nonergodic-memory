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

    @staticmethod
    def _causal_mask(length: int, device: torch.device) -> Tensor:
        return torch.triu(
            torch.ones(length, length, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def forward_with_layers(self, tokens: Tensor) -> tuple[Tensor, list[Tensor]]:
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)[None, :, :]
        mask = self._causal_mask(tokens.shape[1], tokens.device)
        activations: list[Tensor] = []
        for block in self.blocks.layers:
            hidden = block(hidden, src_mask=mask, is_causal=True)
            activations.append(hidden)
        final_hidden = self.norm(hidden)
        activations.append(final_hidden)
        return self.output(final_hidden), activations

    def logits_from_depth(self, hidden: Tensor, depth: int) -> tuple[Tensor, Tensor]:
        """Resume after a captured depth: block indices first, final norm last."""
        n_blocks = len(self.blocks.layers)
        if not 0 <= depth <= n_blocks:
            raise ValueError("depth outside captured activation range")
        if depth < n_blocks:
            mask = self._causal_mask(hidden.shape[1], hidden.device)
            for block in self.blocks.layers[depth + 1 :]:
                hidden = block(hidden, src_mask=mask, is_causal=True)
            hidden = self.norm(hidden)
        return self.output(hidden), hidden

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        logits, activations = self.forward_with_layers(tokens)
        return logits, activations[-1]


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
