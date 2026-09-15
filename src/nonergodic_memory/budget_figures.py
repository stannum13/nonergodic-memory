"""Raw-only token-budget-matched short-context comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .short_figures import _short_cells


def _joined(budget_records: list[dict], standard_records: list[dict], long_records: list[dict]):
    budget, long, oracle = _short_cells(
        budget_records, long_records, "budget_context", "budget_short_trained"
    )
    standard, _, _ = _short_cells(standard_records, long_records)
    for overlap in (0.0, 0.35):
        if budget[(0, overlap, "none")]["short_config_sha256"] == standard[(0, overlap, "none")]["short_config_sha256"]:
            raise ValueError("budget and standard checkpoints must differ")
    return budget, standard, long, oracle


def budget_minus_standard(
    budget_records: list[dict], standard_records: list[dict],
    long_records: list[dict], metric: str,
) -> dict[float, list[float]]:
    """Seed-paired budget-short minus fixed-sequence-count short metric."""
    if metric not in {"component_posterior_r2", "state_posterior_r2", "kl_exact", "nll"}:
        raise ValueError(f"unsupported budget metric: {metric}")
    budget, standard, _, _ = _joined(budget_records, standard_records, long_records)
    return {
        overlap: [
            float(budget[(seed, overlap, "none")][metric])
            - float(standard[(seed, overlap, "none")][metric])
            for seed in (0, 1, 2)
        ]
        for overlap in (0.0, 0.35)
    }


def generate_budget_figure(
    budget_results: str | Path, standard_results: str | Path,
    long_results: str | Path, output: str | Path,
) -> Path:
    def read(path: str | Path) -> list[dict]:
        with Path(path).open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    budget_records, standard_records, long_records = (
        read(budget_results), read(standard_results), read(long_results)
    )
    budget, standard, long, oracle = _joined(
        budget_records, standard_records, long_records
    )
    specs = (
        ("component_posterior_r2", "component posterior R²", "oracle_component_posterior_r2"),
        ("state_posterior_r2", "conditional-state posterior R²", "oracle_state_posterior_r2"),
        ("kl_exact", "KL(exact full Bayes || model)", "oracle_kl_full_to_window"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), squeeze=False)
    for column, (metric, ylabel, oracle_key) in enumerate(specs):
        axis = axes[0, column]
        traces = (
            ("budget-matched short", "#228833", lambda seed, overlap: budget[(seed, overlap, "none")]),
            ("standard short", "#CCBB44", lambda seed, overlap: standard[(seed, overlap, "none")]),
            ("long-trained reset", "#EE6677", lambda seed, overlap: long[("transformer", seed, overlap, "trained", "restart_8", "none")]),
            ("long-trained original", "#4477AA", lambda seed, overlap: long[("transformer", seed, overlap, "trained", "restart_8_absolute", "none")]),
        )
        for label, color, source in traces:
            values = [
                [float(source(seed, overlap)[metric]) for seed in (0, 1, 2)]
                for overlap in (0.0, 0.35)
            ]
            axis.errorbar(
                (0.0, 0.35), [np.mean(v) for v in values],
                yerr=[np.std(v) for v in values], marker="o", capsize=3,
                color=color, label=label,
            )
        oracle_values = [
            [float(oracle[(seed, overlap)][oracle_key]) for seed in (0, 1, 2)]
            for overlap in (0.0, 0.35)
        ]
        axis.errorbar(
            (0.0, 0.35), [np.mean(v) for v in oracle_values],
            yerr=[np.std(v) for v in oracle_values], marker="s", linestyle=":",
            capsize=3, color="black", label="exact 8-token Bayes",
        )
        effect = budget_minus_standard(
            budget_records, standard_records, long_records, metric
        )[0.35]
        axis.set(
            xlabel="source overlap", ylabel=ylabel, xticks=(0.0, 0.35),
            title=f"budget − standard short at .35: {np.mean(effect):+.3f} ± {np.std(effect):.3f}",
        )
        axis.legend(frameon=False, fontsize=7)
    fig.suptitle("Matched supervised-token and optimizer-step budgets; mean ± seed SD (n=3)")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-results", default="results/sweep_budget_context.jsonl")
    parser.add_argument("--standard-results", default="results/sweep_short_context.jsonl")
    parser.add_argument("--long-results", default="results/sweep_position_restart.jsonl")
    parser.add_argument("--output", default="figures/sweep_budget_context.png")
    args = parser.parse_args()
    if not Path(args.budget_results).exists():
        Path(args.output).unlink(missing_ok=True)
        print(f"skipped optional figure: {args.budget_results} has no raw JSONL")
        return
    print(f"generated {generate_budget_figure(args.budget_results, args.standard_results, args.long_results, args.output)}")


if __name__ == "__main__":
    main()
