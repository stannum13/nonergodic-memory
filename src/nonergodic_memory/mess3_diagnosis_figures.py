"""Figures for the Mess3 training-diversity diagnosis."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(path: str | Path, record_type: str) -> list[dict]:
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    selected = [row for row in rows if row.get("record_type") == record_type]
    if not selected:
        raise ValueError(f"missing {record_type} records in {source}")
    return selected


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_diagnosis_figures(
    baseline_results: str | Path,
    training_results: str | Path,
    probe_results: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    baselines = _load(baseline_results, "baseline")
    training = _load(training_results, "diagnostic_training")
    probes = _load(probe_results, "diagnostic_probe")
    digests = {
        row.get("config_sha256")
        for row in [*baselines, *training, *probes]
    }
    if len(digests) != 1 or None in digests:
        raise ValueError("diagnosis figure inputs have incompatible provenance")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    order = ["uniform", "last_token"]
    order.extend(sorted(row["predictor"] for row in baselines if row["predictor"].startswith("window_")))
    order.append("full_bayes")
    cells = {row["predictor"]: row for row in baselines}
    if set(cells) != set(order):
        raise ValueError("baseline records are incomplete or duplicated")
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5))
    axes[0].bar(x, [cells[name]["kl_exact"] for name in order], color="#4477AA")
    axes[0].set(ylabel="KL from full-history Bayes (nats)", title="Available predictive signal")
    axes[1].bar(x, [cells[name]["competence"] for name in order], color="#228833")
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set(ylabel="uniform-to-Bayes gap recovered", title="Predictive competence")
    labels = [name.replace("_", "\n") for name in order]
    for axis in axes:
        axis.set_xticks(x, labels)
    baseline_path = _save(fig, destination / "mess3_predictive_baselines.png")

    colors = {"reused": "#CC6677", "fresh": "#4477AA"}
    markers = {"block_1": "o", "block_2": "s", "final_norm": "^"}
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for condition in ("reused", "fresh"):
        seeds = sorted({int(row["seed"]) for row in training if row["condition"] == condition})
        for seed in seeds:
            rows = sorted(
                (row for row in training if row["condition"] == condition and int(row["seed"]) == seed),
                key=lambda row: int(row["step"]),
            )
            axes[0].plot(
                [row["step"] for row in rows],
                [row["kl_exact"] for row in rows],
                marker="o",
                color=colors[condition],
                alpha=0.7,
                label=condition if seed == seeds[0] else None,
            )
    axes[0].set(xlabel="optimizer updates", ylabel="exact-predictive KL (nats)", title="Prediction")
    axes[0].legend(frameon=False)

    training_cells = {
        (int(row["seed"]), row["condition"], int(row["step"])): row for row in training
    }
    normal_probes = [row for row in probes if row.get("control") == "none"]
    for condition in ("reused", "fresh"):
        for site in markers:
            rows = [row for row in normal_probes if row["condition"] == condition and row["site"] == site]
            axes[1].scatter(
                [training_cells[(int(row["seed"]), condition, int(row["step"]))]["kl_exact"] for row in rows],
                [row["joint_belief_r2"] for row in rows],
                color=colors[condition],
                marker=markers[site],
                alpha=0.75,
                label=f"{condition}, {site}",
            )
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set(
        xlabel="exact-predictive KL (nats)",
        ylabel="held-out joint-belief R²",
        title="Prediction versus geometry",
    )
    axes[1].legend(frameon=False, fontsize=8)
    learning_path = _save(fig, destination / "mess3_learning_geometry.png")
    return [baseline_path, learning_path]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-results", default="results/mess3_diagnosis_baselines.jsonl")
    parser.add_argument("--training-results", default="results/mess3_diagnosis_training.jsonl")
    parser.add_argument("--probe-results", default="results/mess3_diagnosis_probes.jsonl")
    parser.add_argument("--output-dir", default="figures")
    args = parser.parse_args()
    for path in generate_diagnosis_figures(
        args.baseline_results, args.training_results, args.probe_results, args.output_dir
    ):
        print(f"generated {path}")


if __name__ == "__main__":
    main()
