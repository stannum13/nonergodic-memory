"""Reconstruct Mess3 geometry and metric figures exclusively from raw JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SEEDS = (0, 1, 2)
GROUPS = (("trained", "none", "trained"), ("untrained", "none", "untrained"),
          ("trained", "shuffled_labels", "shuffled targets"))
METRICS = (("joint_belief_r2", "Joint-belief R²"),
           ("joint_distance_r2", "Pairwise-distance R²"),
           ("joint_belief_mse", "Joint-belief MSE"))


def _read(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _validate(records: list[dict], training: list[dict], model: str) -> tuple[list[dict], dict]:
    relevant = [row for row in records + training if row.get("model") == model
                and row.get("record_type") in ("probe", "mess3_geometry", "training")]
    identities = {(row.get("config"), row.get("config_sha256")) for row in relevant}
    if len(identities) != 1 or not all(next(iter(identities), ())):
        raise ValueError("Mess3 records require one matching config and config digest")
    if any(row.get("generator") != "mess3" for row in relevant):
        raise ValueError("Mess3 figures require generator=mess3")
    if any(row.get("seed") not in SEEDS for row in relevant):
        raise ValueError("Mess3 figures require seeds 0, 1, 2")
    training_seeds = [row["seed"] for row in training
                      if row.get("model") == model and row.get("record_type") == "training"]
    if sorted(training_seeds) != list(SEEDS):
        raise ValueError("Mess3 figures require exactly one training record per seed")
    cells, geometry = {}, []
    for row in relevant:
        if row["record_type"] == "mess3_geometry":
            if row.get("training_condition") != "trained":
                raise ValueError("mess3_geometry requires trained reconstructions")
            for prefix in ("exact", "pred"):
                for index in range(6):
                    key = f"{prefix}_b{index}"
                    if key not in row:
                        raise ValueError(f"mess3_geometry missing coordinate {key}")
                    if not np.isfinite(float(row[key])):
                        raise ValueError(f"mess3_geometry non-finite coordinate {key}")
            geometry.append(row)
        elif row["record_type"] == "probe":
            key = (row["seed"], row.get("training_condition"), row.get("control"))
            if key in cells:
                raise ValueError(f"duplicate Mess3 probe cell: {key}")
            for metric, _ in METRICS:
                if metric not in row or not np.isfinite(float(row[metric])):
                    raise ValueError(f"Mess3 probe missing or non-finite metric {metric}")
            cells[key] = row
    for seed in SEEDS:
        for condition, control, _ in GROUPS:
            if (seed, condition, control) not in cells:
                raise ValueError(f"missing Mess3 probe cell: {(seed, condition, control)}")
        if not any(row["seed"] == seed for row in geometry):
            raise ValueError(f"missing mess3_geometry for seed {seed}")
    return geometry, cells


def generate_mess3_figures(
    results: str | Path, training_results: str | Path, output_dir: str | Path,
    model: str = "transformer",
) -> list[Path]:
    """Plot both component blocks and mean ± population SD over seeds 0, 1, 2.

    Coordinates are joint beliefs, without renormalizing component blocks or
    clipping linear reconstructions. Both columns use exact posterior mass as
    color, so corresponding points have the same color.
    """
    geometry, cells = _validate(_read(results), _read(training_results), model)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    exact = np.array([[row[f"exact_b{i}"] for i in range(6)] for row in geometry])
    predicted = np.array([[row[f"pred_b{i}"] for i in range(6)] for row in geometry])
    fig = plt.figure(figsize=(12, 10), layout="constrained")
    axes = []
    for component in range(2):
        block = slice(component * 3, component * 3 + 3)
        mass = exact[:, block].sum(axis=1)
        lower = min(0., float(exact[:, block].min()), float(predicted[:, block].min()))
        upper = max(1., float(exact[:, block].max()), float(predicted[:, block].max()))
        for column, (coordinates, label) in enumerate(((exact, "Exact"), (predicted, "Reconstructed"))):
            axis = fig.add_subplot(2, 2, component * 2 + column + 1, projection="3d")
            axes.append(axis)
            points = axis.scatter(*coordinates[:, block].T, c=mass, cmap="viridis",
                                  vmin=0, vmax=1, s=3, alpha=.65, rasterized=True)
            axis.set(title=f"{label}: component {component}", xlabel=f"b{component * 3}",
                     ylabel=f"b{component * 3 + 1}", zlabel=f"b{component * 3 + 2}",
                     xlim=(lower, upper), ylim=(lower, upper), zlim=(lower, upper))
    fig.colorbar(points, ax=axes, shrink=.6, label="Exact posterior mass of plotted component")
    fig.suptitle(f"Mess3 joint-belief geometry · {model} · held-out seeds 0, 1, 2")
    geometry_path = destination / "mess3_geometry.png"
    fig.savefig(geometry_path, dpi=180, bbox_inches="tight", pad_inches=.15)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for axis, (metric, label) in zip(axes, METRICS):
        values = np.array([[cells[(seed, condition, control)][metric] for seed in SEEDS]
                           for condition, control, _ in GROUPS])
        axis.bar(np.arange(3), values.mean(axis=1), yerr=values.std(axis=1, ddof=0),
                 color=("#228833", "#4477AA", "#CCBB44"), capsize=4)
        axis.set_xticks(np.arange(3), [group[2] for group in GROUPS], rotation=20, ha="right")
        axis.set(ylabel=label)
        axis.axhline(0, color="black", linewidth=.5)
    fig.suptitle(f"Mess3 · {model} · mean ± population seed SD (n=3)")
    metrics_path = destination / "mess3_metrics.png"
    fig.savefig(metrics_path, dpi=180)
    plt.close(fig)
    return [geometry_path, metrics_path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/mess3_reproduction.jsonl")
    parser.add_argument("--training-results", default="results/mess3_training.jsonl")
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--model", choices=("transformer", "gru"), default="transformer")
    args = parser.parse_args()
    if not Path(args.results).exists() and not Path(args.training_results).exists():
        print("skipped optional Mess3 figures: no raw Mess3 JSONL")
        return
    for path in generate_mess3_figures(args.results, args.training_results, args.output_dir, args.model):
        print(f"generated {path}")


if __name__ == "__main__":
    main()
