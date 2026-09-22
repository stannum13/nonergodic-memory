"""Figures for the Mess3 training-diversity diagnosis."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .experiment import load_config


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
    *,
    expected_seeds: Iterable[int],
    expected_steps: Iterable[int],
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
    if any(count != 1 for count in Counter(row["predictor"] for row in baselines).values()):
        raise ValueError("baseline cells are duplicated")
    cells = {row["predictor"]: row for row in baselines}
    if set(cells) != set(order):
        raise ValueError("baseline records are incomplete or duplicated")

    training_keys = [
        (int(row["seed"]), row["condition"], int(row["step"])) for row in training
    ]
    if any(count != 1 for count in Counter(training_keys).values()):
        raise ValueError("diagnostic training cells are duplicated")
    training_seeds = {int(seed) for seed in expected_seeds}
    conditions = {cell[1] for cell in training_keys}
    steps = {int(step) for step in expected_steps}
    if not training_seeds or not steps:
        raise ValueError("expected diagnosis seeds and steps cannot be empty")
    expected_training_keys = {
        (seed, condition, step)
        for seed in training_seeds
        for condition in ("reused", "fresh")
        for step in steps
    }
    if conditions != {"reused", "fresh"} or set(training_keys) != expected_training_keys:
        raise ValueError("diagnostic training grid is incomplete or contains unexpected cells")
    probe_keys = [
        (
            int(row["seed"]),
            row["condition"],
            int(row["step"]),
            row["site"],
            row["control"],
        )
        for row in probes
    ]
    if any(count != 1 for count in Counter(probe_keys).values()):
        raise ValueError("diagnostic probe cells are duplicated")
    expected_probe_keys = {
        (*cell, site, control)
        for cell in training_keys
        for site in ("block_1", "block_2", "final_norm")
        for control in ("none", "shuffled_labels")
    }
    if set(probe_keys) != expected_probe_keys:
        raise ValueError("diagnostic probe grid is incomplete or contains unexpected cells")
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
    seed_styles = {
        seed: ("-", "--", ":", "-.")[index % 4]
        for index, seed in enumerate(sorted(training_seeds))
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for condition in ("reused", "fresh"):
        condition_seeds = sorted(
            {int(row["seed"]) for row in training if row["condition"] == condition}
        )
        for seed in condition_seeds:
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
                label=condition if seed == condition_seeds[0] else None,
            )
    axes[0].set(xlabel="optimizer updates", ylabel="exact-predictive KL (nats)", title="Prediction")
    axes[0].legend(frameon=False)

    training_cells = {
        (int(row["seed"]), row["condition"], int(row["step"])): row for row in training
    }
    normal_probes = [row for row in probes if row.get("control") == "none"]
    for condition in ("reused", "fresh"):
        for site in markers:
            seeds = sorted({int(row["seed"]) for row in normal_probes if row["condition"] == condition})
            for seed in seeds:
                rows = sorted(
                    (
                        row
                        for row in normal_probes
                        if row["condition"] == condition
                        and row["site"] == site
                        and int(row["seed"]) == seed
                    ),
                    key=lambda row: int(row["step"]),
                )
                steps = np.asarray([int(row["step"]) for row in rows], dtype=float)
                sizes = 35 + 45 * steps / max(1.0, steps.max())
                xs = [
                    training_cells[(seed, condition, int(row["step"]))]["kl_exact"]
                    for row in rows
                ]
                ys = [row["joint_belief_r2"] for row in rows]
                axes[1].plot(
                    xs,
                    ys,
                    color=colors[condition],
                    linestyle=seed_styles[seed],
                    alpha=0.3,
                    linewidth=0.8,
                )
                axes[1].scatter(
                    xs,
                    ys,
                    s=sizes,
                    color=colors[condition],
                    marker=markers[site],
                    alpha=0.75,
                    label=f"{condition}, {site}" if seed == seeds[0] else None,
                )
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set(
        xlabel="exact-predictive KL (nats)",
        ylabel="held-out joint-belief R²",
        title="Prediction versus geometry (size = update step)",
    )
    handles, labels = axes[1].get_legend_handles_labels()
    handles.extend(
        Line2D([0], [0], color="0.35", linestyle=seed_styles[seed], label=f"seed {seed}")
        for seed in sorted(seed_styles)
    )
    labels.extend(f"seed {seed}" for seed in sorted(seed_styles))
    axes[1].legend(handles, labels, frameon=False, fontsize=8)
    learning_path = _save(fig, destination / "mess3_learning_geometry.png")
    return [baseline_path, learning_path]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-results", default="results/mess3_diagnosis_baselines.jsonl")
    parser.add_argument("--training-results", default="results/mess3_diagnosis_training.jsonl")
    parser.add_argument("--probe-results", default="results/mess3_diagnosis_probes.jsonl")
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--config", default="configs/mess3_diagnosis.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[10, 11])
    args = parser.parse_args()
    config = load_config(args.config)
    for path in generate_diagnosis_figures(
        args.baseline_results,
        args.training_results,
        args.probe_results,
        args.output_dir,
        expected_seeds=args.seeds,
        expected_steps=config["train"]["checkpoint_steps"],
    ):
        print(f"generated {path}")


if __name__ == "__main__":
    main()
