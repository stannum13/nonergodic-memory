"""Raw-evidence-validated learning and fixed-forecast audit figures."""

from __future__ import annotations

import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .mess3_rate_aware_clock import (
    _integer, _validate_audit, _validate_grid, analyze_rate_aware_clock, forecast_geometry,
)


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout(rect=(0, .07, 1, .90))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _primary_rows(config, training, probes):
    cells = {(row["seed"], row["learning_rate"], row["step"]): row for row in training}
    return [{**row, "competence": cells[row["seed"], row["learning_rate"], row["step"]]["competence"]}
            for row in probes if row["site"] == "block_2" and row["control"] == "none"]


def _learning_figure(config, rows, destination):
    experiment = config["rate_aware_clock"]
    rates, seeds = experiment["learning_rates"], experiment["seeds"]
    steps = config["train"]["checkpoint_steps"]
    styles = ("-", "--", ":", "-.", (0, (5, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (1, 1)), (0, (5, 2)))
    fig, axes = plt.subplots(2, len(rates), figsize=(max(6.5, 3.5 * len(rates)), 6.5), squeeze=False)
    for column, rate in enumerate(rates):
        for index, seed in enumerate(seeds):
            curve = sorted((row for row in rows if row["seed"] == seed and row["learning_rate"] == rate),
                           key=lambda row: row["step"])
            for axis, metric in zip(axes[:, column], ("competence", "component_posterior_r2")):
                post = [row for row in curve if row["step"] > 0]
                initial = next(row for row in curve if row["step"] == 0)
                axis.plot([row["step"] for row in post], [row[metric] for row in post],
                          linestyle=styles[index], color="#4477AA", marker="o", markersize=3)
                axis.scatter([0], [initial[metric]], marker="D", s=32, facecolor="white",
                             edgecolor="#222222", zorder=3)
        for axis in axes[:, column]:
            separator = steps[1] / 2
            axis.axvspan(0, separator, color=".92", zorder=-1)
            axis.axvline(separator, color=".45", linewidth=.7, linestyle="--")
            axis.grid(axis="y", color=".9", linewidth=.6)
        axes[0, column].set_title(f"learning rate {rate:g}")
        axes[1, column].set_xlabel("optimizer updates")
    axes[0, 0].set_ylabel("predictive competence")
    axes[1, 0].set_ylabel("block-2 component-posterior R²")
    fig.suptitle("Competence and geometry by rate (◇ = untrained step 0)")
    fig.legend(handles=[Line2D([0], [0], color="#4477AA", linestyle=styles[index], label=f"seed {seed}")
                        for index, seed in enumerate(seeds)], loc="lower center", ncol=min(4, len(seeds)), frameon=False)
    return _save(fig, destination / "mess3_rate_aware_clock_learning.png")


def _forecast_figure(config, rows, summary, destination):
    primary = summary["primary"]
    fig, (predictions, errors) = plt.subplots(1, 2, figsize=(10.5, 4.8))
    if primary is None:
        for axis in (predictions, errors):
            axis.set_axis_off()
            axis.text(.5, .5, "Registered forecast analysis unavailable\n" +
                      ", ".join(summary["validity_failures"]), ha="center", va="center", transform=axis.transAxes)
        ratio_text = "unavailable"
    else:
        selected = [row for row in rows if row["step"] in config["rate_aware_clock"]["primary_checkpoints"]]
        observed = np.asarray([row["component_posterior_r2"] for row in selected])
        competence = np.asarray([forecast_geometry(row["competence"], config["forecast"]["competence"])
                                 for row in selected])
        clock = np.asarray([forecast_geometry(np.log1p(row["learning_rate"] * row["step"]),
                                             config["forecast"]["rate_aware_clock"]) for row in selected])
        predictions.scatter(observed, competence, color="#4477AA", alpha=.7, s=24, label="competence")
        predictions.scatter(observed, clock, color="#CC6677", alpha=.65, marker="x", s=28, label="rate-aware clock")
        limits = (min(observed.min(), competence.min(), clock.min()) - .02,
                  max(observed.max(), competence.max(), clock.max()) + .02)
        predictions.plot(limits, limits, color=".2", linestyle="--", linewidth=.8)
        predictions.set(xlim=limits, ylim=limits, xlabel="observed block-2 component-posterior R²",
                        ylabel="fixed forecast R²", title="Post-initialization fixed forecasts")
        predictions.legend(frameon=False)
        per_seed = primary["per_seed"]
        positions = np.arange(len(per_seed))
        errors.bar(positions - .19, [row["competence_mse"] for row in per_seed], .38,
                   color="#4477AA", label="competence")
        errors.bar(positions + .19, [row["clock_mse"] for row in per_seed], .38,
                   color="#CC6677", label="rate-aware clock")
        errors.set(xticks=positions, xticklabels=[str(row["seed"]) for row in per_seed],
                   xlabel="seed", ylabel="fixed forecast MSE", title="Per-seed forecast error")
        errors.legend(frameon=False)
        ratio = primary["clock_to_competence_mse_ratio"]
        ratio_text = "undefined" if ratio is None else f"{ratio:.3f}"
    fig.suptitle(f"Seed-equal clock/competence MSE ratio = {ratio_text}; verdict: {summary['verdict'].upper()}")
    return _save(fig, destination / "mess3_rate_aware_clock_forecasts.png")


def generate_rate_aware_clock_figures(config, training, probes, audit, summary, output_dir) -> list[Path]:
    """Plot only a complete finite grid and its exactly recomputed analysis."""
    _validate_grid(config, training, probes)
    # Preserve overlap as a scientific failure while rejecting corrupted audit identity.
    audit_identity = copy.deepcopy(audit)
    for row in audit_identity:
        if not isinstance(row.get("intersections"), dict):
            raise ValueError("audit intersections must be a mapping")
        for key, value in row["intersections"].items():
            if _integer(value) < 0:
                raise ValueError("audit intersections must be nonnegative")
            row["intersections"][key] = 0
    _validate_audit(config, audit_identity)
    expected = analyze_rate_aware_clock(config, training, probes, audit)
    if not isinstance(summary, dict) or summary != expected:
        raise ValueError("rate-aware summary does not match fresh analysis of raw rows")
    if len(config["rate_aware_clock"]["seeds"]) > 8 or len(config["train"]["checkpoint_steps"]) < 2:
        raise ValueError("figures require at most eight seeds and a post-initialization checkpoint")
    rows = _primary_rows(config, training, probes)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    return [_learning_figure(config, rows, destination), _forecast_figure(config, rows, summary, destination)]
