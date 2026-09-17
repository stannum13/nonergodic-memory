#!/usr/bin/env python3
"""Evaluate short-trained Transformers on held-out eight-token windows from length-64 sources."""

from __future__ import annotations

import argparse
from pathlib import Path

from nonergodic_memory.analysis import collect_activations, fit_probes
from nonergodic_memory.context import collect_restart_activations
from nonergodic_memory.experiment import (
    config_digest, load_checkpoint, load_config, mixture_from_config,
    replace_jsonl_runs, runtime_provenance, set_seed,
)
from nonergodic_memory.intervention import evaluate_hidden_with_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-configs", nargs="+", default=[
        "configs/sweeps/interaction_o000_l064.yaml",
        "configs/sweeps/interaction_o035_l064.yaml",
    ])
    parser.add_argument("--short-configs", nargs="+", default=[
        "configs/sweeps/short_o000_l009.yaml",
        "configs/sweeps/short_o035_l009.yaml",
    ])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--checkpoint-root", default="checkpoints/sweeps")
    parser.add_argument("--results", default="results/sweep_short_context.jsonl")
    parser.add_argument("--protocol", choices=["standard", "budget"], default="standard")
    parser.add_argument("--model", choices=["gru", "transformer"], default="transformer")
    return parser.parse_args()


def _check_matched(eval_config: dict, short_config: dict, protocol: str = "standard") -> None:
    if int(eval_config["data"]["sequence_length"]) != 64:
        raise ValueError("evaluation context must have sequence length 64")
    if int(short_config["data"]["sequence_length"]) != 9:
        raise ValueError("short training must have length 9 to supply eight input positions")
    if protocol == "budget":
        if (
            int(eval_config["data"]["train_sequences"]) != 512
            or int(eval_config["train"]["batch_size"]) != 64
            or int(short_config["data"]["train_sequences"]) != 4032
            or int(short_config["train"]["batch_size"]) != 504
            or int(eval_config["train"]["epochs"]) != 12
            or int(short_config["train"]["epochs"]) != 12
        ):
            raise ValueError("budget protocol requires 512/64 long and 4032/504 short batches")
        adjusted = {
            **short_config,
            "data": {**short_config["data"], "sequence_length": 64, "train_sequences": 512},
            "train": {**short_config["train"], "batch_size": 64},
        }
    else:
        adjusted = {**short_config, "data": {**short_config["data"], "sequence_length": 64}}
    if adjusted != eval_config:
        raise ValueError(f"{protocol} short and evaluation configs violate their matched protocol")


def main() -> None:
    args = parse_args()
    if len(args.eval_configs) != len(args.short_configs):
        raise ValueError("evaluation and short config lists must be paired")
    provenance = runtime_provenance()
    for eval_path, short_path in zip(args.eval_configs, args.short_configs):
        config = load_config(eval_path)
        short_config = load_config(short_path)
        _check_matched(config, short_config, args.protocol)
        mixture = mixture_from_config(config)
        eval_name, short_name = Path(eval_path).stem, Path(short_path).stem
        digest, short_digest = config_digest(config), config_digest(short_config)
        records = []
        for seed in args.seeds:
            set_seed(seed)
            train_batch = mixture.sample(int(config["probe"]["train_sequences"]), 64, seed + 909)
            test_batch = mixture.sample(int(config["probe"]["test_sequences"]), 64, seed + 1009)
            _, model = load_checkpoint(
                Path(args.checkpoint_root) / short_name / f"{args.model}_seed{seed}.pt",
                short_config, args.model, seed,
            )
            full_train = collect_activations(model, train_batch, mixture)
            full_test = collect_activations(model, test_batch, mixture, sequence_offset=1_000_000)
            fit_table = collect_restart_activations(model, train_batch, full_train, window=8)
            test_table = collect_restart_activations(model, test_batch, full_test, window=8)
            for control, shuffled in (("none", False), ("shuffled_labels", True)):
                probes = fit_probes(fit_table, test_table, seed, shuffle_labels=shuffled)
                behavior = evaluate_hidden_with_logits(
                    test_table.hidden, test_table.logits, test_table, probes
                )
                record = {
                    "record_type": "budget_context" if args.protocol == "budget" else "short_context",
                    "model": args.model,
                    "training_condition": "budget_short_trained" if args.protocol == "budget" else "short_trained",
                    "context": "restart_8",
                    "control": control, "seed": seed, "device": "cpu",
                    "config": eval_name, "config_sha256": digest,
                    "short_config": short_name, "short_config_sha256": short_digest,
                    "overlap": float(config["data"]["overlap"]),
                    "sequence_length": 64, "short_training_sequence_length": 9,
                    "short_training_input_positions": 8,
                    "window": 8, "positions_evaluated": 56,
                    "probe_fit_sequences": int(config["probe"]["train_sequences"]),
                    "test_sequences": int(config["probe"]["test_sequences"]),
                    "probe_fit_data_seed": seed + 909, "test_data_seed": seed + 1009,
                    "probe_fit_independent": True, "test_sequence_offset": 1_000_000,
                    "observations_evaluated": len(test_table.targets),
                    "state_posterior_target": "all_component_conditionals",
                    "model_width": int(config["model"]["width"]),
                    "components": int(config["data"].get("components", 2)),
                    **({
                        "training_protocol": "token_and_step_matched",
                        "short_training_sequences": 4032,
                        "short_training_batch_size": 504,
                        "supervised_tokens_per_epoch": 32256,
                        "optimizer_steps": 96,
                    } if args.protocol == "budget" else {}),
                    **provenance, **probes.metrics,
                    "nll": behavior["nll"], "kl_exact": behavior["kl_exact"],
                }
                records.append(record)
                if not shuffled:
                    print(
                        f"{short_name} seed={seed} overlap={record['overlap']} "
                        f"R2={record['component_posterior_r2']:.3f} KL={record['kl_exact']:.4f}"
                    )
        replace_jsonl_runs(args.results, records, eval_name, [args.model], args.seeds)


if __name__ == "__main__":
    main()
