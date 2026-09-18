#!/usr/bin/env python3
"""Fit held-out linear probes and write quantitative/PCA JSONL records."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from nonergodic_memory.analysis import ActivationTable, collect_activations, fit_probes, pca_records
from nonergodic_memory.experiment import (
    config_digest,
    generator_name,
    load_checkpoint,
    load_config,
    mixture_from_config,
    replace_jsonl_runs,
    runtime_provenance,
    set_seed,
)
from nonergodic_memory.models.sequence import build_model


def mess3_geometry_records(table: ActivationTable, joint_regression: object) -> list[dict]:
    """Preserve six joint coordinates for at most 2,000 held-out observations."""
    if table.joint_belief is None or table.joint_belief.shape[1] != 6:
        raise ValueError("Mess3 geometry requires six-coordinate joint beliefs")
    selected = np.linspace(0, len(table.hidden) - 1, min(2000, len(table.hidden)), dtype=int)
    predicted = joint_regression.predict(table.hidden[selected])
    return [
        {
            "component": int(table.components[index]),
            "position": int(table.positions[index]),
            "sequence_id": int(table.sequence_ids[index]),
            **{f"exact_b{i}": float(table.joint_belief[index, i]) for i in range(6)},
            **{f"pred_b{i}": float(predicted[row, i]) for i in range(6)},
        }
        for row, index in enumerate(selected)
    ]


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
    mixture = mixture_from_config(config)
    config_name = Path(args.config).stem
    config_sha256 = config_digest(config)
    source_generator = generator_name(config)
    provenance = runtime_provenance()
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
                    _, model = load_checkpoint(
                        Path(args.checkpoint_dir) / f"{model_name}_seed{seed}.pt",
                        config,
                        model_name,
                        seed,
                    )
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
                        "config": config_name,
                        "config_sha256": config_sha256,
                        "generator": source_generator,
                        **(
                            {"overlap": float(config["data"]["overlap"])}
                            if "overlap" in config["data"]
                            else {}
                        ),
                        "sequence_length": int(config["data"]["sequence_length"]),
                        "components": int(config["data"].get("components", 2)),
                        "model_width": int(config["model"]["width"]),
                        "training_condition": condition,
                        "control": "none",
                        "state_posterior_target": "all_component_conditionals",
                        **bundle.metrics,
                        **provenance,
                    }
                )
                shuffled = fit_probes(train_table, test_table, seed, shuffle_labels=True)
                records.append(
                    {
                        "record_type": "probe",
                        "model": model_name,
                        "seed": seed,
                        "device": "cpu",
                        "config": config_name,
                        "config_sha256": config_sha256,
                        "generator": source_generator,
                        **(
                            {"overlap": float(config["data"]["overlap"])}
                            if "overlap" in config["data"]
                            else {}
                        ),
                        "sequence_length": int(config["data"]["sequence_length"]),
                        "components": int(config["data"].get("components", 2)),
                        "model_width": int(config["model"]["width"]),
                        "training_condition": condition,
                        "control": "shuffled_labels",
                        "state_posterior_target": "all_component_conditionals",
                        **shuffled.metrics,
                        **provenance,
                    }
                )
                if condition == "trained":
                    if source_generator == "mess3":
                        for point in mess3_geometry_records(test_table, bundle.joint_regression):
                            records.append(
                                {
                                    "record_type": "mess3_geometry",
                                    "model": model_name,
                                    "seed": seed,
                                    "training_condition": condition,
                                    "control": "none",
                                    "device": "cpu",
                                    "config": config_name,
                                    "config_sha256": config_sha256,
                                    "generator": source_generator,
                                    "sequence_length": int(config["data"]["sequence_length"]),
                                    "components": len(mixture.components),
                                    "model_width": int(config["model"]["width"]),
                                    **point,
                                    **provenance,
                                }
                            )
                    points, variance = pca_records(test_table)
                    for point in points:
                        records.append(
                            {
                                "record_type": "pca",
                                "model": model_name,
                                "seed": seed,
                                "training_condition": condition,
                                "device": "cpu",
                                "config": config_name,
                                "config_sha256": config_sha256,
                                "generator": source_generator,
                                **(
                                    {"overlap": float(config["data"]["overlap"])}
                                    if "overlap" in config["data"]
                                    else {}
                                ),
                                "sequence_length": int(config["data"]["sequence_length"]),
                                "components": int(config["data"].get("components", 2)),
                                "model_width": int(config["model"]["width"]),
                                "explained_variance_pc1": variance[0],
                                "explained_variance_pc2": variance[1],
                                **point,
                                **provenance,
                            }
                        )
                print(
                    f"{model_name} {condition} seed={seed} "
                    f"component={bundle.metrics['component_accuracy']:.3f} "
                    f"state={bundle.metrics['conditional_state_accuracy']:.3f} "
                    f"joint_r2={bundle.metrics['joint_belief_r2']:.3f}"
                )
    replace_jsonl_runs(args.results, records, config_name, args.models, args.seeds)


if __name__ == "__main__":
    main()
