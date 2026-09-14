#!/usr/bin/env python3
"""Fit held-out linear probes and write quantitative/PCA JSONL records."""

from __future__ import annotations

import argparse
from pathlib import Path

from nonergodic_memory.analysis import collect_activations, fit_probes, pca_records
from nonergodic_memory.data.hmm import make_two_source_mixture
from nonergodic_memory.experiment import load_checkpoint, load_config, set_seed, write_jsonl
from nonergodic_memory.models.sequence import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--models", nargs="+", choices=["gru", "transformer"], default=["gru", "transformer"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--results", default="results/reproduction.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    mixture = make_two_source_mixture(float(config["data"]["overlap"]))
    records: list[dict] = []
    for seed in args.seeds:
        train_batch = mixture.sample(
            int(config["probe"]["train_sequences"]), int(config["data"]["sequence_length"]), seed + 404
        )
        test_batch = mixture.sample(
            int(config["probe"]["test_sequences"]), int(config["data"]["sequence_length"]), seed + 505
        )
        for model_name in args.models:
            for condition in ("trained", "untrained"):
                if condition == "trained":
                    _, model = load_checkpoint(Path(args.checkpoint_dir) / f"{model_name}_seed{seed}.pt")
                else:
                    set_seed(seed)
                    model = build_model(model_name, mixture.vocab_size, config["model"])
                train_table = collect_activations(model, train_batch, mixture, sequence_offset=0)
                test_table = collect_activations(model, test_batch, mixture, sequence_offset=1_000_000)
                bundle = fit_probes(train_table, test_table, seed)
                records.append(
                    {
                        "record_type": "probe",
                        "model": model_name,
                        "seed": seed,
                        "device": "cpu",
                        "training_condition": condition,
                        "control": "none",
                        **bundle.metrics,
                    }
                )
                shuffled = fit_probes(train_table, test_table, seed, shuffle_labels=True)
                records.append(
                    {
                        "record_type": "probe",
                        "model": model_name,
                        "seed": seed,
                        "device": "cpu",
                        "training_condition": condition,
                        "control": "shuffled_labels",
                        **shuffled.metrics,
                    }
                )
                if condition == "trained":
                    points, variance = pca_records(test_table)
                    for point in points:
                        records.append(
                            {
                                "record_type": "pca",
                                "model": model_name,
                                "seed": seed,
                                "training_condition": condition,
                                "explained_variance_pc1": variance[0],
                                "explained_variance_pc2": variance[1],
                                **point,
                            }
                        )
                print(
                    f"{model_name} {condition} seed={seed} "
                    f"component={bundle.metrics['component_accuracy']:.3f} "
                    f"state={bundle.metrics['conditional_state_accuracy']:.3f}"
                )
    write_jsonl(args.results, records)


if __name__ == "__main__":
    main()
