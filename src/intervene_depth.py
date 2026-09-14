#!/usr/bin/env python3
"""Measure controlled activation erasure across Transformer depth."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from nonergodic_memory.analysis import collect_transformer_depth_activations, fit_probes
from nonergodic_memory.experiment import (
    config_digest,
    load_checkpoint,
    load_config,
    mixture_from_config,
    replace_jsonl_runs,
    runtime_provenance,
    set_seed,
)
from nonergodic_memory.intervention import (
    erase_subspace,
    evaluate_hidden_with_logits,
    intervention_record_with_logits,
    norm_matched_random_erasure,
    probe_basis,
    random_basis,
)
from nonergodic_memory.models.sequence import TransformerPredictor, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/reproduce.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--checkpoint-dir", default="checkpoints/reproduce")
    parser.add_argument("--results", default="results/sweep_depth.jsonl")
    return parser.parse_args()


@torch.no_grad()
def _propagate_logits(
    model: TransformerPredictor,
    flat_hidden: np.ndarray,
    n_sequences: int,
    positions: int,
    depth: int,
) -> np.ndarray:
    hidden = torch.from_numpy(flat_hidden.reshape(n_sequences, positions, -1)).to(torch.float32)
    logits, _ = model.logits_from_depth(hidden, depth)
    return logits.numpy().reshape(-1, logits.shape[-1]).astype(np.float64)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    mixture = mixture_from_config(config)
    config_name = Path(args.config).stem
    config_sha256 = config_digest(config)
    provenance = runtime_provenance()
    n_blocks = int(config["model"]["layers"])
    depth_labels = [f"block_{index + 1}" for index in range(n_blocks)] + ["final_norm"]
    records: list[dict] = []
    for seed in args.seeds:
        basis_batch = mixture.sample(
            int(config["probe"]["train_sequences"]),
            int(config["data"]["sequence_length"]),
            seed + 606,
        )
        evaluator_batch = mixture.sample(
            int(config["probe"]["test_sequences"]),
            int(config["data"]["sequence_length"]),
            seed + 707,
        )
        test_batch = mixture.sample(
            int(config["probe"]["test_sequences"]),
            int(config["data"]["sequence_length"]),
            seed + 808,
        )
        for condition in ("trained", "untrained"):
            if condition == "trained":
                _, loaded = load_checkpoint(
                    Path(args.checkpoint_dir) / f"transformer_seed{seed}.pt",
                    config,
                    "transformer",
                    seed,
                )
            else:
                set_seed(seed)
                loaded = build_model("transformer", mixture.vocab_size, config["model"])
            if not isinstance(loaded, TransformerPredictor):
                raise TypeError("depth intervention requires TransformerPredictor")
            model = loaded
            for depth, depth_label in enumerate(depth_labels):
                basis_table = collect_transformer_depth_activations(
                    model, basis_batch, mixture, depth
                )
                evaluator_table = collect_transformer_depth_activations(
                    model, evaluator_batch, mixture, depth, sequence_offset=1_000_000
                )
                test_table = collect_transformer_depth_activations(
                    model, test_batch, mixture, depth, sequence_offset=2_000_000
                )
                basis_probes = fit_probes(basis_table, evaluator_table, seed)
                shuffled_basis_probes = fit_probes(
                    basis_table, evaluator_table, seed, shuffle_labels=True
                )
                evaluator_probes = fit_probes(evaluator_table, test_table, seed + 1)
                baseline = evaluate_hidden_with_logits(
                    test_table.hidden, test_table.logits, test_table, evaluator_probes
                )
                center = basis_table.hidden.mean(axis=0)
                n_sequences = test_batch.tokens.shape[0]
                positions = test_batch.tokens.shape[1] - 1
                for target_index, target in enumerate(("component", "state")):
                    learned_basis = probe_basis(basis_probes, target)
                    shuffled_basis = probe_basis(shuffled_basis_probes, target)
                    control_basis = random_basis(
                        test_table.hidden.shape[1],
                        len(learned_basis),
                        seed + 800 + 10 * depth + target_index,
                    )
                    interventions = {
                        "baseline": test_table.hidden,
                        "learned": erase_subspace(test_table.hidden, learned_basis, center),
                        "random_subspace": erase_subspace(
                            test_table.hidden, control_basis, center
                        ),
                        "norm_matched_random": norm_matched_random_erasure(
                            test_table.hidden, control_basis, learned_basis, center
                        ),
                        "shuffled_labels": erase_subspace(
                            test_table.hidden, shuffled_basis, center
                        ),
                    }
                    for control, altered in interventions.items():
                        logits = (
                            test_table.logits
                            if control == "baseline"
                            else _propagate_logits(
                                model, altered, n_sequences, positions, depth
                            )
                        )
                        rank = 0
                        if control != "baseline":
                            rank = int(
                                len(shuffled_basis)
                                if control == "shuffled_labels"
                                else len(learned_basis)
                            )
                        records.append(
                            {
                                "record_type": "intervention_depth",
                                "model": "transformer",
                                "seed": seed,
                                "device": "cpu",
                                "config": config_name,
                                "config_sha256": config_sha256,
                                "overlap": float(config["data"]["overlap"]),
                                "sequence_length": int(config["data"]["sequence_length"]),
                                "components": int(config["data"].get("components", 2)),
                                "model_width": int(config["model"]["width"]),
                                "training_condition": condition,
                                "target": target,
                                "control": control,
                                "depth": depth,
                                "depth_label": depth_label,
                                "basis_fit_data_seed": seed + 606,
                                "evaluator_fit_data_seed": seed + 707,
                                "test_data_seed": seed + 808,
                                "independent_evaluator": True,
                                "rank": rank,
                                **(
                                    {
                                        "baseline_component_posterior_r2": evaluator_probes.metrics[
                                            "component_posterior_r2"
                                        ],
                                        "baseline_state_posterior_r2": evaluator_probes.metrics[
                                            "state_posterior_r2"
                                        ],
                                    }
                                    if control == "baseline"
                                    else {}
                                ),
                                **intervention_record_with_logits(
                                    test_table,
                                    evaluator_probes,
                                    altered,
                                    logits,
                                    baseline,
                                ),
                                **provenance,
                            }
                        )
                    learned = records[-4]
                    print(
                        f"transformer {condition} {depth_label} {target} seed={seed} "
                        f"dNLL={learned['delta_nll']:+.4f} "
                        f"dComp={learned['delta_component_accuracy']:+.3f} "
                        f"dState={learned['delta_conditional_state_accuracy']:+.3f}"
                    )
    replace_jsonl_runs(args.results, records, config_name, ["transformer"], args.seeds)


if __name__ == "__main__":
    main()
