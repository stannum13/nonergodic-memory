"""Predictive baselines and matched training diagnostics for Mess3."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from .analysis import collect_transformer_depth_activations, fit_probes
from .context import oracle_window_beliefs
from .data.hmm import HMMMixture, SequenceBatch
from .experiment import (
    generator_name,
    config_digest,
    load_checkpoint,
    mixture_from_config,
    runtime_provenance,
    set_seed,
    tensor_sequences,
)
from .models.sequence import build_model


FloatArray = NDArray[np.float64]


def _probability_rows(values: FloatArray, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f"{name} must be a finite nonnegative probability matrix")
    if not np.allclose(array.sum(axis=1), 1.0):
        raise ValueError(f"{name} rows must sum to one")
    return array


def predictive_kl(exact: FloatArray, candidate: FloatArray) -> float:
    """Mean KL(exact || candidate), in nats."""
    truth = _probability_rows(exact, "exact")
    proposal = _probability_rows(candidate, "candidate")
    if truth.shape != proposal.shape:
        raise ValueError("exact and candidate distributions must have the same shape")
    return float(
        np.mean(
            np.sum(
                truth * (np.log(np.clip(truth, 1e-12, None)) - np.log(np.clip(proposal, 1e-12, None))),
                axis=1,
            )
        )
    )


def competence(kl_exact: float, uniform_kl: float) -> float:
    """Fraction of the uniform-to-Bayes predictive KL gap recovered."""
    if not np.isfinite(uniform_kl) or uniform_kl <= 0:
        raise ValueError("uniform KL must be finite and positive")
    if not np.isfinite(kl_exact) or kl_exact < 0:
        raise ValueError("predictive KL must be finite and nonnegative")
    return float(1.0 - kl_exact / uniform_kl)


def _last_token_table(batch: SequenceBatch, vocab_size: int, smoothing: float = 1.0) -> FloatArray:
    counts = np.full((vocab_size, vocab_size), smoothing, dtype=np.float64)
    current = batch.tokens[:, :-1].reshape(-1)
    following = batch.tokens[:, 1:].reshape(-1)
    np.add.at(counts, (current, following), 1.0)
    return counts / counts.sum(axis=1, keepdims=True)


def _sampled_nll(candidate: FloatArray, targets: NDArray[np.int64]) -> float:
    probabilities = candidate[np.arange(len(targets)), targets]
    return float(-np.log(np.clip(probabilities, 1e-12, None)).mean())


def predictive_baselines(
    mixture: HMMMixture,
    fit_batch: SequenceBatch,
    test_batch: SequenceBatch,
    window: int = 8,
) -> list[dict]:
    """Evaluate simple predictors on positions with a complete context window."""
    inputs = test_batch.tokens[:, :-1]
    if not 1 <= window <= inputs.shape[1]:
        raise ValueError("window outside available input positions")
    full = mixture.filter(inputs).predictive[:, window - 1 :, :].reshape(-1, mixture.vocab_size)
    targets = test_batch.tokens[:, window:].reshape(-1)
    uniform = np.full_like(full, 1.0 / mixture.vocab_size)
    table = _last_token_table(fit_batch, mixture.vocab_size)
    current = inputs[:, window - 1 :].reshape(-1)
    last_token = table[current]
    windowed = oracle_window_beliefs(test_batch, mixture, window).predictive
    if not (len(full) == len(targets) == len(windowed)):
        raise ValueError("baseline predictions are not position-aligned")

    candidates = {
        "uniform": uniform,
        "last_token": last_token,
        f"window_{window}_bayes": windowed,
        "full_bayes": full,
    }
    uniform_kl = predictive_kl(full, uniform)
    records = []
    for name, prediction in candidates.items():
        kl = predictive_kl(full, prediction)
        records.append(
            {
                "record_type": "baseline",
                "predictor": name,
                "window": window,
                "positions": len(targets),
                "kl_exact": kl,
                "nll": _sampled_nll(prediction, targets),
                "uniform_kl": uniform_kl,
                "competence": competence(kl, uniform_kl),
            }
        )
    return records


@torch.no_grad()
def _evaluate_aligned(model, batch: SequenceBatch, mixture: HMMMixture, window: int) -> dict:
    model.eval()
    inputs, targets = tensor_sequences(batch)
    logits, _ = model(inputs)
    exact = mixture.filter(batch.tokens[:, :-1]).predictive[:, window - 1 :, :]
    aligned_logits = logits[:, window - 1 :, :]
    aligned_targets = targets[:, window - 1 :]
    probabilities = aligned_logits.softmax(-1).numpy().reshape(-1, mixture.vocab_size)
    truth = exact.reshape(-1, mixture.vocab_size)
    uniform = np.full_like(truth, 1.0 / mixture.vocab_size)
    uniform_kl = predictive_kl(truth, uniform)
    kl = predictive_kl(truth, probabilities)
    nll = F.cross_entropy(
        aligned_logits.reshape(-1, mixture.vocab_size), aligned_targets.reshape(-1)
    ).item()
    return {
        "nll": float(nll),
        "kl_exact": kl,
        "uniform_kl": uniform_kl,
        "competence": competence(kl, uniform_kl),
        "positions": int(truth.shape[0]),
    }


def _checkpoint_steps(config: dict) -> tuple[int, ...]:
    steps = tuple(sorted({int(step) for step in config["train"]["checkpoint_steps"]}))
    if not steps or steps[0] != 0 or any(step < 0 for step in steps):
        raise ValueError("checkpoint_steps must contain zero and nonnegative integers")
    return steps


def train_diagnostic(
    config: dict,
    seed: int,
    condition: str,
    output_dir: str | Path,
) -> list[dict]:
    """Train a matched fixed-pool or fresh-sequence Mess3 diagnostic run."""
    if condition not in {"reused", "fresh"}:
        raise ValueError("condition must be reused or fresh")
    if generator_name(config) != "mess3":
        raise ValueError("Mess3 diagnosis requires generator=mess3")
    set_seed(seed)
    mixture = mixture_from_config(config)
    data = config["data"]
    train = config["train"]
    window = int(config["diagnosis"]["window"])
    length = int(data["sequence_length"])
    batch_size = int(train["batch_size"])
    train_sequences = int(data["train_sequences"])
    if not 1 <= window < length:
        raise ValueError("diagnosis window must be shorter than sequence length")
    if condition == "reused" and train_sequences % batch_size:
        raise ValueError("reused train_sequences must be divisible by batch_size")
    checkpoint_steps = _checkpoint_steps(config)
    max_step = checkpoint_steps[-1]
    fixed_pool = mixture.sample(train_sequences, length, seed + 101)
    test_batch = mixture.sample(int(data["test_sequences"]), length, seed + 202)
    model = build_model("transformer", mixture.vocab_size, config["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train["learning_rate"]),
        weight_decay=float(train.get("weight_decay", 0.01)),
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    order_generator = torch.Generator().manual_seed(seed + 303)
    fixed_inputs, fixed_targets = tensor_sequences(fixed_pool)
    pending_batches: list[torch.Tensor] = []
    records: list[dict] = []
    checkpoint_set = set(checkpoint_steps)

    def save_record(step: int) -> None:
        metrics = _evaluate_aligned(model, test_batch, mixture, window)
        checkpoint_name = f"transformer_seed{seed}_{condition}_step{step}.pt"
        torch.save(
            {
                "config": config,
                "model_name": "transformer",
                "seed": seed,
                "condition": condition,
                "step": step,
                "state_dict": model.state_dict(),
            },
            destination / checkpoint_name,
        )
        records.append(
            {
                "record_type": "diagnostic_training",
                "generator": "mess3",
                "model": "transformer",
                "seed": seed,
                "condition": condition,
                "step": step,
                "updates": step,
                "batch_size": batch_size,
                "sequence_length": length,
                "training_sequences_seen": step * batch_size,
                "checkpoint": checkpoint_name,
                "device": "cpu",
                **metrics,
                **runtime_provenance(),
            }
        )

    save_record(0)
    for step in range(1, max_step + 1):
        if condition == "fresh":
            batch = mixture.sample(batch_size, length, seed * 1_000_000 + 10_000 + step)
            inputs, targets = tensor_sequences(batch)
        else:
            if not pending_batches:
                pending_batches = list(
                    torch.randperm(len(fixed_inputs), generator=order_generator).split(batch_size)
                )
            indices = pending_batches.pop(0)
            inputs, targets = fixed_inputs[indices], fixed_targets[indices]
        model.train()
        logits, _ = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, mixture.vocab_size), targets.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in checkpoint_set:
            save_record(step)
    return records


def evaluate_checkpoint_geometry(
    config: dict,
    checkpoint: str | Path,
    seed: int,
    condition: str,
) -> list[dict]:
    """Fit held-out belief probes at each Transformer activation site."""
    if condition not in {"reused", "fresh"}:
        raise ValueError("condition must be reused or fresh")
    payload, model = load_checkpoint(checkpoint, config, "transformer", seed)
    if payload.get("condition") != condition:
        raise ValueError("checkpoint condition does not match requested condition")
    step = int(payload.get("step", -1))
    if step < 0:
        raise ValueError("diagnostic checkpoint is missing a valid step")
    mixture = mixture_from_config(config)
    length = int(config["data"]["sequence_length"])
    probe = config["probe"]
    train_batch = mixture.sample(int(probe["train_sequences"]), length, seed + 404)
    test_batch = mixture.sample(int(probe["test_sequences"]), length, seed + 505)
    n_layers = int(config["model"]["layers"])
    records: list[dict] = []
    for depth in range(n_layers + 1):
        site = f"block_{depth + 1}" if depth < n_layers else "final_norm"
        train_table = collect_transformer_depth_activations(
            model, train_batch, mixture, depth, sequence_offset=0
        )
        test_table = collect_transformer_depth_activations(
            model, test_batch, mixture, depth, sequence_offset=1_000_000
        )
        overlap = int(
            len(np.intersect1d(np.unique(train_table.sequence_ids), np.unique(test_table.sequence_ids)))
        )
        for shuffle in (False, True):
            bundle = fit_probes(train_table, test_table, seed, shuffle_labels=shuffle)
            records.append(
                {
                    "record_type": "diagnostic_probe",
                    "generator": "mess3",
                    "model": "transformer",
                    "seed": seed,
                    "condition": condition,
                    "step": step,
                    "depth": depth,
                    "site": site,
                    "control": "shuffled_labels" if shuffle else "none",
                    "probe_sequence_overlap": overlap,
                    "config_sha256": config_digest(config),
                    "device": "cpu",
                    **bundle.metrics,
                    **runtime_provenance(),
                }
            )
    return records
