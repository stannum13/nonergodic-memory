#!/usr/bin/env python3
"""Compare full-prefix models with eight-token restarts against exact Bayes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

from nonergodic_memory.analysis import collect_activations, fit_probes
from nonergodic_memory.context import (
    aligned_full_table,
    collect_restart_activations,
    oracle_window_beliefs,
)
from nonergodic_memory.experiment import (
    config_digest,
    load_checkpoint,
    load_config,
    mixture_from_config,
    replace_jsonl_runs,
    runtime_provenance,
    set_seed,
)
from nonergodic_memory.intervention import evaluate_hidden_with_logits
from nonergodic_memory.models.sequence import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs", nargs="+", default=[
            "configs/sweeps/interaction_o000_l064.yaml",
            "configs/sweeps/interaction_o035_l064.yaml",
        ]
    )
    parser.add_argument("--models", nargs="+", choices=["gru", "transformer"], default=["gru", "transformer"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--checkpoint-root", default="checkpoints/sweeps")
    parser.add_argument("--results", default="results/sweep_context_restart.jsonl")
    return parser.parse_args()


def _oracle_metrics(batch, mixture, window: int) -> dict[str, float]:
    full = mixture.filter(batch.tokens[:, :-1])
    windowed = oracle_window_beliefs(batch, mixture, window)
    full_predictive = full.predictive[:, window - 1 :].reshape(-1, mixture.vocab_size)
    full_component = full.component_posterior[:, window - 1 :].reshape(
        -1, len(mixture.components)
    )
    full_state = full.state_posterior[:, window - 1 :].reshape(
        -1, len(mixture.components) * mixture.max_states
    )
    window_state = windowed.state_posterior.reshape(-1, full_state.shape[-1])
    targets = batch.tokens[:, 1:][:, window - 1 :].reshape(-1)
    log_full = np.log(np.clip(full_predictive, 1e-12, None))
    log_window = np.log(np.clip(windowed.predictive, 1e-12, None))
    indices = np.arange(len(targets))
    return {
        "oracle_kl_full_to_window": float(np.mean(np.sum(full_predictive * (log_full - log_window), axis=1))),
        "oracle_component_posterior_r2": float(r2_score(full_component, windowed.component_posterior)),
        "oracle_state_posterior_r2": float(r2_score(full_state, window_state)),
        "oracle_nll_full": float(np.mean(-log_full[indices, targets])),
        "oracle_nll_window": float(np.mean(-log_window[indices, targets])),
    }


def main() -> None:
    args = parse_args()
    provenance = runtime_provenance()
    for config_path in args.configs:
        config = load_config(config_path)
        mixture = mixture_from_config(config)
        name = Path(config_path).stem
        digest = config_digest(config)
        length = int(config["data"]["sequence_length"])
        if not 1 <= args.window < length:
            raise ValueError("window must be between one and sequence length minus one")
        records: list[dict] = []
        for seed in args.seeds:
            set_seed(seed)
            train_batch = mixture.sample(int(config["probe"]["train_sequences"]), length, seed + 909)
            test_batch = mixture.sample(int(config["probe"]["test_sequences"]), length, seed + 1009)
            common = {
                "seed": seed,
                "device": "cpu",
                "config": name,
                "config_sha256": digest,
                "overlap": float(config["data"]["overlap"]),
                "sequence_length": length,
                "components": int(config["data"].get("components", 2)),
                "model_width": int(config["model"]["width"]),
                "window": args.window,
                "positions_evaluated": length - args.window,
                "probe_fit_sequences": int(config["probe"]["train_sequences"]),
                "test_sequences": int(config["probe"]["test_sequences"]),
                "probe_fit_data_seed": seed + 909,
                "test_data_seed": seed + 1009,
                **provenance,
            }
            records.append(
                {
                    "record_type": "context_oracle",
                    "model": "exact_bayes",
                    "training_condition": "analytic",
                    "context": f"restart_{args.window}",
                    "control": "none",
                    **common,
                    **_oracle_metrics(test_batch, mixture, args.window),
                }
            )
            for model_name in args.models:
                for condition in ("trained", "untrained"):
                    if condition == "trained":
                        _, model = load_checkpoint(
                            Path(args.checkpoint_root) / name / f"{model_name}_seed{seed}.pt",
                            config, model_name, seed,
                        )
                    else:
                        set_seed(seed)
                        model = build_model(model_name, mixture.vocab_size, config["model"])
                    full_train = collect_activations(model, train_batch, mixture)
                    full_test = collect_activations(
                        model, test_batch, mixture, sequence_offset=1_000_000
                    )
                    contexts = {
                        "full": (
                            aligned_full_table(full_train, args.window),
                            aligned_full_table(full_test, args.window),
                        ),
                        f"restart_{args.window}": (
                            collect_restart_activations(model, train_batch, full_train, args.window),
                            collect_restart_activations(model, test_batch, full_test, args.window),
                        ),
                    }
                    for context_name, (fit_table, test_table) in contexts.items():
                        for control, shuffled in (("none", False), ("shuffled_labels", True)):
                            bundle = fit_probes(fit_table, test_table, seed, shuffle_labels=shuffled)
                            behavior = evaluate_hidden_with_logits(
                                test_table.hidden, test_table.logits, test_table, bundle
                            )
                            record = {
                                "record_type": "context_restart",
                                "model": model_name,
                                "training_condition": condition,
                                "context": context_name,
                                "control": control,
                                "probe_fit_independent": True,
                                "test_sequence_offset": 1_000_000,
                                "observations_evaluated": len(test_table.targets),
                                "state_posterior_target": "all_component_conditionals",
                                **common,
                                **bundle.metrics,
                                "nll": behavior["nll"],
                                "kl_exact": behavior["kl_exact"],
                            }
                            records.append(record)
                            if not shuffled:
                                print(
                                    f"{name} {model_name} {condition} {context_name} seed={seed} "
                                    f"R2={record['component_posterior_r2']:.3f} "
                                    f"KL={record['kl_exact']:.4f}"
                                )
        replace_jsonl_runs(
            args.results, records, name, [*args.models, "exact_bayes"], args.seeds
        )


if __name__ == "__main__":
    main()
