#!/usr/bin/env python3
"""Erase component or conditional-state activation subspaces with controls."""

from __future__ import annotations

import argparse
from pathlib import Path

from nonergodic_memory.analysis import collect_activations, fit_probes
from nonergodic_memory.data.hmm import make_two_source_mixture
from nonergodic_memory.experiment import load_checkpoint, load_config, set_seed, write_jsonl
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
    mixture = make_two_source_mixture(float(config["data"]["overlap"]))
    records: list[dict] = []
    for seed in args.seeds:
        fit_batch = mixture.sample(
            int(config["probe"]["train_sequences"]), int(config["data"]["sequence_length"]), seed + 606
        )
        test_batch = mixture.sample(
            int(config["probe"]["test_sequences"]), int(config["data"]["sequence_length"]), seed + 707
        )
        for model_name in args.models:
            for condition in ("trained", "untrained"):
                if condition == "trained":
                    _, model = load_checkpoint(Path(args.checkpoint_dir) / f"{model_name}_seed{seed}.pt")
                else:
                    set_seed(seed)
                    model = build_model(model_name, mixture.vocab_size, config["model"])
                fit_table = collect_activations(model, fit_batch, mixture)
                test_table = collect_activations(model, test_batch, mixture, sequence_offset=1_000_000)
                probes = fit_probes(fit_table, test_table, seed)
                shuffled_probes = fit_probes(fit_table, test_table, seed, shuffle_labels=True)
                baseline = evaluate_hidden(test_table.hidden, test_table, probes, model.output)
                center = fit_table.hidden.mean(axis=0)
                for target_index, target in enumerate(("component", "state")):
                    learned_basis = probe_basis(probes, target)
                    shuffled_basis = probe_basis(shuffled_probes, target)
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
                            "training_condition": condition,
                            "target": target,
                            "control": control,
                            "layer": "final",
                            "rank": 0 if control == "baseline" else int(
                                len(shuffled_basis) if control == "shuffled_labels" else len(learned_basis)
                            ),
                            **intervention_record(
                                test_table, probes, model.output, altered, baseline
                            ),
                        }
                        records.append(record)
                    learned = records[-4]
                    print(
                        f"{model_name} {condition} {target} seed={seed} "
                        f"dNLL={learned['delta_nll']:+.4f} "
                        f"dComp={learned['delta_component_accuracy']:+.3f} "
                        f"dState={learned['delta_conditional_state_accuracy']:+.3f}"
                    )
    write_jsonl(args.results, records)


if __name__ == "__main__":
    main()
