"""Figures for the registered Mess3 geometry-threshold experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .mess3_threshold import validate_threshold_grid


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_threshold_figures(
    config: dict,
    training: list[dict],
    probes: list[dict],
    summary: dict,
    output_dir: str | Path,
) -> list[Path]:
    """Plot paired step trajectories and their competence alignment."""
    validate_threshold_grid(config, training, probes)
    if summary.get("base_config_sha256") != training[0].get("base_config_sha256"):
        raise ValueError("threshold summary has incompatible provenance")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    training_cells = {
        (int(row["seed"]), float(row["learning_rate"]), int(row["step"])): row
        for row in training
    }
    block_rows = [row for row in probes if row["site"] == "block_2"]
    seeds = sorted({int(row["seed"]) for row in training})
    rates = sorted({float(row["learning_rate"]) for row in training}, reverse=True)
    colors = {rates[0]: "#4477AA", rates[1]: "#CC6677"}
    line_styles = ("-", "--", ":", "-.", (0, (5, 1, 1, 1, 1, 1)))
    if len(seeds) > len(line_styles):
        raise ValueError("threshold figure supports at most five traceable seed styles")
    styles = {seed: line_styles[index] for index, seed in enumerate(seeds)}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for rate in rates:
        for seed in seeds:
            curve = sorted(
                (
                    row
                    for row in training
                    if int(row["seed"]) == seed and float(row["learning_rate"]) == rate
                ),
                key=lambda row: int(row["step"]),
            )
            axes[0].plot(
                [row["step"] for row in curve],
                [row["competence"] for row in curve],
                color=colors[rate],
                linestyle=styles[seed],
                alpha=0.7,
            )
            geometry = sorted(
                (
                    row
                    for row in block_rows
                    if row["control"] == "none"
                    and int(row["seed"]) == seed
                    and float(row["learning_rate"]) == rate
                ),
                key=lambda row: int(row["step"]),
            )
            axes[1].plot(
                [row["step"] for row in geometry],
                [row["component_posterior_r2"] for row in geometry],
                color=colors[rate],
                linestyle=styles[seed],
                alpha=0.7,
            )
    axes[0].set(
        xlabel="optimizer updates",
        ylabel="predictive competence",
        title="Prediction learning curves",
    )
    axes[1].set(
        xlabel="optimizer updates",
        ylabel="block-2 component-posterior R²",
        title="Geometry learning curves",
    )
    rate_handles = [
        Line2D([0], [0], color=colors[rate], label=f"lr={rate:g}") for rate in rates
    ]
    seed_handles = [
        Line2D([0], [0], color="0.25", linestyle=styles[seed], label=f"seed {seed}")
        for seed in seeds
    ]
    axes[0].legend(handles=[*rate_handles, *seed_handles], frameon=False, ncol=2)
    learning_path = _save(fig, destination / "mess3_threshold_learning.png")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    max_step = max(int(row["step"]) for row in training)
    sensitivity = summary["posthoc_post_initialization_sensitivity"]
    sensitivity_title = (
        "Post hoc: step > 0\nLOSO MSE ratio "
        f"{sensitivity['competence_to_step_mse_ratio']:.3f}"
        if sensitivity["status"] != "unavailable"
        else "Post hoc: step > 0\nunavailable for this grid"
    )
    panels = (
        (
            axes[0],
            lambda row: True,
            "Registered: all checkpoints\n"
            f"LOSO MSE ratio {summary['competence_to_step_mse_ratio']:.3f}",
        ),
        (
            axes[1],
            lambda row: int(row["step"]) > 0,
            sensitivity_title,
        ),
    )
    for axis, include, title in panels:
        for control, alpha in (("shuffled_labels", 0.18), ("none", 0.72)):
            for rate in rates:
                rows = [
                    row
                    for row in block_rows
                    if row["control"] == control
                    and float(row["learning_rate"]) == rate
                    and include(row)
                ]
                steps = np.asarray([int(row["step"]) for row in rows], dtype=float)
                axis.scatter(
                    [
                        training_cells[
                            (int(row["seed"]), rate, int(row["step"]))
                        ]["competence"]
                        for row in rows
                    ],
                    [row["component_posterior_r2"] for row in rows],
                    s=24 + 50 * steps / max(1, max_step),
                    color=colors[rate],
                    alpha=alpha,
                    marker="x" if control == "shuffled_labels" else "o",
                    label=f"lr={rate:g}" if control == "none" else None,
                )
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set(xlabel="predictive competence", title=title)
    axes[0].set(ylabel="block-2 component-posterior R²")
    axes[0].legend(frameon=False)
    alignment_path = _save(fig, destination / "mess3_threshold_alignment.png")
    return [learning_path, alignment_path]
