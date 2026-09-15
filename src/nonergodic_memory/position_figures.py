"""Raw-only analysis of original-index versus reset-index Transformer restarts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .context_figures import _damage, _validated_cells


def _cells(records: list[dict]) -> tuple[dict, dict]:
    return _validated_cells(
        records, models=("transformer",),
        contexts=("full", "restart_8", "restart_8_absolute"),
    )


def position_effect(records: list[dict], metric: str) -> dict[tuple[str, float], list[float]]:
    """Per-seed original-index improvement over reset-index at each overlap."""
    cells, _ = _cells(records)
    effects = {}
    for condition in ("trained", "untrained"):
        for overlap in (0.0, 0.35):
            values = []
            for seed in (0, 1, 2):
                reset = cells[("transformer", seed, overlap, condition, "restart_8", "none")]
                absolute = cells[("transformer", seed, overlap, condition, "restart_8_absolute", "none")]
                if metric in {"component_posterior_r2", "state_posterior_r2"}:
                    values.append(float(absolute[metric]) - float(reset[metric]))
                elif metric in {"kl_exact", "nll"}:
                    values.append(float(reset[metric]) - float(absolute[metric]))
                else:
                    raise ValueError(f"unsupported position metric: {metric}")
            effects[(condition, overlap)] = values
    return effects


def position_history_damage(
    records: list[dict], metric: str
) -> dict[tuple[str, str], list[float]]:
    """Per-seed overlap contrast of full-to-window damage by position mode."""
    cells, _ = _cells(records)
    contrasts = {}
    for condition in ("trained", "untrained"):
        for context in ("restart_8", "restart_8_absolute"):
            values = []
            for seed in (0, 1, 2):
                damage = {}
                for overlap in (0.0, 0.35):
                    full = cells[("transformer", seed, overlap, condition, "full", "none")]
                    window = cells[("transformer", seed, overlap, condition, context, "none")]
                    damage[overlap] = _damage(full, window, metric)
                values.append(damage[0.35] - damage[0.0])
            contrasts[(condition, context)] = values
    return contrasts


def generate_position_figure(results: str | Path, output: str | Path) -> Path:
    with Path(results).open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    cells, oracle = _cells(records)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), squeeze=False)
    metric_specs = (
        ("component_posterior_r2", "component R² loss", "oracle_component_posterior_r2"),
        ("state_posterior_r2", "conditional-state R² loss", "oracle_state_posterior_r2"),
        ("kl_exact", "Δ KL(exact full Bayes || model)", "oracle_kl_full_to_window"),
    )
    for row, condition in enumerate(("trained", "untrained")):
        for column, (metric, ylabel, oracle_key) in enumerate(metric_specs):
            axis = axes[row, column]
            for context, label, color in (
                ("restart_8", "reset indices", "#EE6677"),
                ("restart_8_absolute", "original indices", "#4477AA"),
            ):
                values = [
                    [
                        _damage(
                            cells[("transformer", seed, overlap, condition, "full", "none")],
                            cells[("transformer", seed, overlap, condition, context, "none")],
                            metric,
                        )
                        for seed in (0, 1, 2)
                    ]
                    for overlap in (0.0, 0.35)
                ]
                axis.errorbar(
                    (0.0, 0.35), [np.mean(v) for v in values],
                    yerr=[np.std(v) for v in values], marker="o", capsize=3,
                    label=label, color=color,
                )
            oracle_values = [
                [
                    (1.0 - oracle[(seed, overlap)][oracle_key])
                    if metric != "kl_exact" else oracle[(seed, overlap)][oracle_key]
                    for seed in (0, 1, 2)
                ]
                for overlap in (0.0, 0.35)
            ]
            axis.errorbar(
                (0.0, 0.35), [np.mean(v) for v in oracle_values],
                yerr=[np.std(v) for v in oracle_values], marker="s", linestyle=":",
                capsize=3, label="exact 8-token Bayes", color="black",
            )
            axis.axhline(0, color="black", linewidth=0.7)
            effect = position_effect(records, metric)[(condition, 0.35)]
            axis.set(
                title=f"{condition}: index benefit at .35 {np.mean(effect):+.3f} ± {np.std(effect):.3f}",
                xlabel="source overlap", ylabel=ylabel, xticks=(0.0, 0.35),
            )
            axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Transformer last-8-token restarts: position reset vs original indices; mean ± seed SD (n=3)")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/sweep_position_restart.jsonl")
    parser.add_argument("--output", default="figures/sweep_position_restart.png")
    args = parser.parse_args()
    if not Path(args.results).exists():
        Path(args.output).unlink(missing_ok=True)
        print(f"skipped optional figure: {args.results} has no raw JSONL")
        return
    print(f"generated {generate_position_figure(args.results, args.output)}")


if __name__ == "__main__":
    main()
