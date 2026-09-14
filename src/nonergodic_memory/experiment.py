"""Shared deterministic training and evaluation utilities."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn

from .data.hmm import HMMMixture, SequenceBatch, make_two_source_mixture
from .models.sequence import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a mapping")
    return config


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def append_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def tensor_sequences(batch: SequenceBatch) -> tuple[Tensor, Tensor]:
    tokens = torch.from_numpy(batch.tokens)
    return tokens[:, :-1], tokens[:, 1:]


@torch.no_grad()
def evaluate_predictions(model: nn.Module, batch: SequenceBatch, mixture: HMMMixture) -> dict:
    model.eval()
    inputs, targets = tensor_sequences(batch)
    logits, _ = model(inputs)
    nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)).item()
    exact = mixture.filter(batch.tokens[:, :-1]).predictive
    exact_tensor = torch.from_numpy(exact).to(dtype=logits.dtype)
    model_log_prob = logits.log_softmax(-1)
    kl = (exact_tensor * (exact_tensor.clamp_min(1e-12).log() - model_log_prob)).sum(-1).mean().item()
    exact_nll = -np.log(
        np.take_along_axis(exact, batch.tokens[:, 1:, None], axis=-1).squeeze(-1).clip(1e-12)
    ).mean()
    return {"nll": float(nll), "kl_exact": float(kl), "bayes_nll": float(exact_nll)}


def train_one(
    config: dict,
    model_name: str,
    seed: int,
    output_dir: str | Path = "checkpoints",
) -> tuple[dict, nn.Module]:
    set_seed(seed)
    data_config = config["data"]
    mixture = make_two_source_mixture(float(data_config["overlap"]))
    train_batch = mixture.sample(
        int(data_config["train_sequences"]), int(data_config["sequence_length"]), seed + 101
    )
    test_batch = mixture.sample(
        int(data_config["test_sequences"]), int(data_config["sequence_length"]), seed + 202
    )
    model = build_model(model_name, mixture.vocab_size, config["model"])
    initial = evaluate_predictions(model, train_batch, mixture)
    train_config = config["train"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_config["learning_rate"]))
    inputs, targets = tensor_sequences(train_batch)
    batch_size = int(train_config["batch_size"])
    generator = torch.Generator().manual_seed(seed + 303)
    model.train()
    for _ in range(int(train_config["epochs"])):
        for indices in torch.randperm(len(inputs), generator=generator).split(batch_size):
            logits, _ = model(inputs[indices])
            loss = F.cross_entropy(
                logits.reshape(-1, mixture.vocab_size), targets[indices].reshape(-1)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    train_metrics = evaluate_predictions(model, train_batch, mixture)
    test_metrics = evaluate_predictions(model, test_batch, mixture)
    result = {
        "record_type": "training",
        "model": model_name,
        "seed": seed,
        "device": "cpu",
        "overlap": float(data_config["overlap"]),
        "sequence_length": int(data_config["sequence_length"]),
        "train_sequences": int(data_config["train_sequences"]),
        "initial_train_nll": initial["nll"],
        "train_nll": train_metrics["nll"],
        "test_nll": test_metrics["nll"],
        "test_kl_exact": test_metrics["kl_exact"],
        "test_bayes_nll": test_metrics["bayes_nll"],
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"config": config, "model_name": model_name, "seed": seed, "state_dict": model.state_dict()},
        destination / f"{model_name}_seed{seed}.pt",
    )
    return result, model


def load_checkpoint(path: str | Path) -> tuple[dict, nn.Module]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    mixture = make_two_source_mixture(float(payload["config"]["data"]["overlap"]))
    model = build_model(payload["model_name"], mixture.vocab_size, payload["config"]["model"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return payload, model

