"""Shared deterministic training and evaluation utilities."""

from __future__ import annotations

import json
import hashlib
import random
import platform
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn

from .data.hmm import HMMMixture, SequenceBatch, make_source_mixture
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


def config_digest(config: dict) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def runtime_provenance() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
    }


def replace_jsonl_runs(
    path: str | Path,
    records: Iterable[dict],
    config_name: str,
    models: Iterable[str],
    seeds: Iterable[int],
) -> None:
    """Atomically replace selected runs while retaining unrelated model/seed cells."""
    destination = Path(path)
    new_records = list(records)
    digests = {record.get("config_sha256") for record in new_records}
    if len(digests) != 1 or None in digests:
        raise ValueError("replacement records must share one config_sha256")
    new_digest = next(iter(digests))
    model_set = set(models)
    seed_set = set(seeds)
    retained: list[dict] = []
    if destination.exists():
        with destination.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                same_name = record.get("config") == config_name
                incompatible_digest = same_name and record.get("config_sha256") != new_digest
                selected = same_name and record.get("model") in model_set and record.get("seed") in seed_set
                if not incompatible_digest and not selected:
                    retained.append(record)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    write_jsonl(temporary, [*retained, *new_records])
    temporary.replace(destination)


def tensor_sequences(batch: SequenceBatch) -> tuple[Tensor, Tensor]:
    tokens = torch.from_numpy(batch.tokens)
    return tokens[:, :-1], tokens[:, 1:]


def mixture_from_config(config: dict) -> HMMMixture:
    data = config["data"]
    return make_source_mixture(
        n_components=int(data.get("components", 2)), overlap=float(data["overlap"])
    )


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
    mixture = mixture_from_config(config)
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
        "training_condition": "trained",
        "model": model_name,
        "seed": seed,
        "device": "cpu",
        "overlap": float(data_config["overlap"]),
        "sequence_length": int(data_config["sequence_length"]),
        "train_sequences": int(data_config["train_sequences"]),
        "components": int(data_config.get("components", 2)),
        "model_width": int(config["model"]["width"]),
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


def validate_checkpoint(payload: dict, config: dict, model_name: str, seed: int) -> None:
    if payload.get("config") != config:
        raise ValueError("checkpoint configuration does not match requested configuration")
    if payload.get("model_name") != model_name:
        raise ValueError("checkpoint model does not match requested model")
    if payload.get("seed") != seed:
        raise ValueError("checkpoint seed does not match requested seed")


def load_checkpoint(
    path: str | Path,
    expected_config: dict | None = None,
    expected_model: str | None = None,
    expected_seed: int | None = None,
) -> tuple[dict, nn.Module]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if expected_config is not None:
        if expected_model is None or expected_seed is None:
            raise ValueError("expected model and seed are required with expected config")
        validate_checkpoint(payload, expected_config, expected_model, expected_seed)
    mixture = mixture_from_config(payload["config"])
    model = build_model(payload["model_name"], mixture.vocab_size, payload["config"]["model"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return payload, model
