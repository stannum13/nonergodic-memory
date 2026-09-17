"""Raw-only GRU generalization of the token-budget short-context control."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .context_figures import _validated_cells


def _short_cells(records: list[dict], kind: str, long_cells: dict) -> dict:
    budget = kind == "budget"
    expected_type = "budget_context" if budget else "short_context"
    expected_condition = "budget_short_trained" if budget else "short_trained"
    cells = {}
    provenance = None
    for row in records:
        if row.get("record_type") != expected_type:
            continue
        seed, overlap, control = int(row["seed"]), float(row["overlap"]), row.get("control")
        if seed not in (0, 1, 2) or overlap != 0.35 or control not in ("none", "shuffled_labels"):
            raise ValueError("unexpected GRU short context axis")
        long = long_cells[("gru", seed, overlap, "trained", "restart_8", "none")]
        if (row.get("config"), row.get("config_sha256")) != (long["config"], long["config_sha256"]):
            raise ValueError("GRU short and long evaluation provenance must match")
        identity = (row.get("short_config"), row.get("short_config_sha256"))
        if not all(identity) or identity[1] == row["config_sha256"]:
            raise ValueError("missing distinct GRU short checkpoint provenance")
        if provenance is not None and provenance != identity:
            raise ValueError("mixed GRU short checkpoint provenance")
        provenance = identity
        required = (
            row.get("model") == "gru"
            and row.get("training_condition") == expected_condition
            and row.get("context") == "restart_8"
            and row.get("sequence_length") == 64
            and row.get("short_training_sequence_length") == 9
            and row.get("short_training_input_positions") == 8
            and row.get("window") == 8
            and row.get("positions_evaluated") == 56
            and row.get("probe_fit_sequences") == 256
            and row.get("test_sequences") == 192
            and row.get("probe_fit_data_seed") == seed + 909
            and row.get("test_data_seed") == seed + 1009
            and row.get("observations_evaluated") == 192 * 56
            and row.get("probe_fit_independent") is True
        )
        if not required:
            raise ValueError("GRU short context violates registered protocol")
        if budget and (
            row.get("training_protocol") != "token_and_step_matched"
            or row.get("short_training_sequences") != 4032
            or row.get("short_training_batch_size") != 504
            or row.get("supervised_tokens_per_epoch") != 32256
            or row.get("optimizer_steps") != 96
        ):
            raise ValueError("GRU budget context violates matched exposure protocol")
        if not all(math.isfinite(float(row[key])) for key in (
            "component_posterior_r2", "state_posterior_r2", "kl_exact", "nll"
        )):
            raise ValueError("non-finite GRU short metric")
        key = (seed, control)
        if key in cells:
            raise ValueError(f"duplicate GRU short cell: {key}")
        cells[key] = row
    for seed in (0, 1, 2):
        for control in ("none", "shuffled_labels"):
            if (seed, control) not in cells:
                raise ValueError(f"missing GRU short cell: {(seed, control)}")
    return cells


def _joined(standard_records: list[dict], budget_records: list[dict], long_records: list[dict]):
    long, oracle = _validated_cells(long_records)
    standard = _short_cells(standard_records, "standard", long)
    budget = _short_cells(budget_records, "budget", long)
    if standard[(0, "none")]["short_config_sha256"] == budget[(0, "none")]["short_config_sha256"]:
        raise ValueError("standard and budget GRU checkpoints must differ")
    return standard, budget, long, oracle


def gru_budget_effect(
    standard_records: list[dict], budget_records: list[dict],
    long_records: list[dict], metric: str,
) -> list[float]:
    """Per-seed budget improvement: R² gain or KL/NLL reduction."""
    standard, budget, _, _ = _joined(standard_records, budget_records, long_records)
    values = []
    for seed in (0, 1, 2):
        left = float(standard[(seed, "none")][metric])
        right = float(budget[(seed, "none")][metric])
        values.append(right - left if metric.endswith("_r2") else left - right)
    return values


def generate_gru_budget_figure(
    standard_results: str | Path, budget_results: str | Path,
    long_results: str | Path, output: str | Path,
) -> Path:
    def read(path: str | Path) -> list[dict]:
        with Path(path).open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    standard_records, budget_records, long_records = map(
        read, (standard_results, budget_results, long_results)
    )
    standard, budget, long, oracle = _joined(
        standard_records, budget_records, long_records
    )
    specs = (
        ("component_posterior_r2", "component posterior R²", "oracle_component_posterior_r2"),
        ("state_posterior_r2", "conditional-state posterior R²", "oracle_state_posterior_r2"),
        ("kl_exact", "KL(exact full Bayes || model)", "oracle_kl_full_to_window"),
        ("nll", "next-token NLL", None),
    )
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), squeeze=False)
    for column, (metric, ylabel, oracle_key) in enumerate(specs):
        axis = axes[0, column]
        groups = (
            ("standard short", [standard[(s, "none")][metric] for s in (0, 1, 2)], "#CCBB44"),
            ("budget short", [budget[(s, "none")][metric] for s in (0, 1, 2)], "#228833"),
            ("long restart", [long[("gru", s, .35, "trained", "restart_8", "none")][metric] for s in (0, 1, 2)], "#EE6677"),
        )
        if oracle_key:
            groups += (("exact 8-token Bayes", [oracle[(s, .35)][oracle_key] for s in (0, 1, 2)], "black"),)
        xs = np.arange(len(groups))
        axis.bar(xs, [np.mean(g[1]) for g in groups], yerr=[np.std(g[1]) for g in groups],
                 color=[g[2] for g in groups], capsize=3)
        axis.set_xticks(xs, [g[0] for g in groups], rotation=22, ha="right")
        effect = gru_budget_effect(standard_records, budget_records, long_records, metric)
        label = "R² gain" if metric.endswith("_r2") else "reduction"
        axis.set(ylabel=ylabel, title=f"budget {label}: {np.mean(effect):+.3f} ± {np.std(effect):.3f}")
    fig.suptitle("GRU eight-token windows at overlap .35; mean ± seed SD (n=3)")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standard-results", default="results/sweep_gru_short_context.jsonl")
    parser.add_argument("--budget-results", default="results/sweep_gru_budget_context.jsonl")
    parser.add_argument("--long-results", default="results/sweep_context_restart.jsonl")
    parser.add_argument("--output", default="figures/sweep_gru_budget_context.png")
    args = parser.parse_args()
    if not Path(args.budget_results).exists():
        Path(args.output).unlink(missing_ok=True)
        print(f"skipped optional figure: {args.budget_results} has no raw JSONL")
        return
    print(f"generated {generate_gru_budget_figure(args.standard_results, args.budget_results, args.long_results, args.output)}")


if __name__ == "__main__":
    main()
