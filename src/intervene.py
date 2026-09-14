#!/usr/bin/env python3
"""Erase component or conditional-state activation subspaces with controls."""

from __future__ import annotations

import argparse
from pathlib import Path

from nonergodic_memory.analysis import collect_activations, fit_probes
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
    evaluate_hidden,
    intervention_record,
    norm_matched_random_erasure,
    probe_basis,
    random_basis,
)
from nonergodic_memory.models.sequence import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--models", nargs="+", choices=["gru", "transformer"], default=["gru", "transformer"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--results", default="results/extension.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    mixture = mixture_from_config(config)
    config_name = Path(args.config).stem
    config_sha256 = config_digest(config)
    provenance = runtime_provenance()
    records: list[dict] = []
    for seed in args.seeds:
        basis_fit_batch = mixture.sample(
            int(config["probe"]["train_sequences"]), int(config["data"]["sequence_length"]), seed + 606
        )
        evaluator_fit_batch = mixture.sample(
            int(config["probe"]["test_sequences"]), int(config["data"]["sequence_length"]), seed + 707
        )
        test_batch = mixture.sample(
            int(config["probe"]["test_sequences"]), int(config["data"]["sequence_length"]), seed + 808
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
                basis_fit_table = collect_activations(model, basis_fit_batch, mixture)
                evaluator_fit_table = collect_activations(
                    model, evaluator_fit_batch, mixture, sequence_offset=1_000_000
                )
                test_table = collect_activations(
                    model, test_batch, mixture, sequence_offset=2_000_000
                )
                basis_probes = fit_probes(basis_fit_table, evaluator_fit_table, seed)
                shuffled_basis_probes = fit_probes(
                    basis_fit_table, evaluator_fit_table, seed, shuffle_labels=True
                )
                evaluator_probes = fit_probes(evaluator_fit_table, test_table, seed + 1)
                baseline = evaluate_hidden(
                    test_table.hidden, test_table, evaluator_probes, model.output
                )
                center = basis_fit_table.hidden.mean(axis=0)
                for target_index, target in enumerate(("component", "state")):
                    learned_basis = probe_basis(basis_probes, target)
                    shuffled_basis = probe_basis(shuffled_basis_probes, target)
                    control_basis = random_basis(
                        test_table.hidden.shape[1], len(learned_basis), seed + 800 + target_index
                    )
                    interventions = {
                        "baseline": test_table.hidden,
                        "learned": erase_subspace(test_table.hidden, learned_basis, center),
                        "random_subspace": erase_subspace(test_table.hidden, control_basis, center),
                        "norm_matched_random": norm_matched_random_erasure(
                            test_table.hidden, control_basis, learned_basis, center
                        ),
                        "shuffled_labels": erase_subspace(test_table.hidden, shuffled_basis, center),
                    }
                    for control, altered in interventions.items():
                        record = {
                            "record_type": "intervention",
                            "model": model_name,
                            "seed": seed,
                            "device": "cpu",
                            "config": config_name,
                            "config_sha256": config_sha256,
                            "overlap": float(config["data"]["overlap"]),
                            "sequence_length": int(config["data"]["sequence_length"]),
                            "components": int(config["data"].get("components", 2)),
                            "training_condition": condition,
                            "target": target,
                            "control": control,
                            "layer": "final",
                            "basis_fit_data_seed": seed + 606,
                            "evaluator_fit_data_seed": seed + 707,
                            "test_data_seed": seed + 808,
                            "independent_evaluator": True,
                            "rank": 0 if control == "baseline" else int(
                                len(shuffled_basis) if control == "shuffled_labels" else len(learned_basis)
                            ),
                            **intervention_record(
                                test_table, evaluator_probes, model.output, altered, baseline
                            ),
                            **provenance,
                        }
                        records.append(record)
                    learned = records[-4]
                    print(
                        f"{model_name} {condition} {target} seed={seed} "
                        f"dNLL={learned['delta_nll']:+.4f} "
                        f"dComp={learned['delta_component_accuracy']:+.3f} "
                        f"dState={learned['delta_conditional_state_accuracy']:+.3f}"
                    )
    replace_jsonl_runs(args.results, records, config_name, args.models, args.seeds)


if __name__ == "__main__":
    main()
