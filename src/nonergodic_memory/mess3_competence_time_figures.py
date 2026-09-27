"""Auditable figures for the preregistered Mess3 competence--time experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .experiment import config_digest
from .mess3_competence_time import _competence_time_quadratic, validate_competence_time_grid


def _save(fig: plt.Figure, path: Path, *, title_space: float = 0.96) -> Path:
    fig.tight_layout(rect=(0, 0, 1, title_space))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _validate_summary(config: dict, training: list[dict], probes: list[dict], summary: dict) -> None:
    """Reject figures that would combine a valid grid with another run's summary."""
    validate_competence_time_grid(config, training, probes)
    expected_digest = config_digest(config)
    if not isinstance(summary, dict) or summary.get("record_type") != "competence_time_summary":
        raise ValueError("competence-time summary has incompatible identity")
    if (
        summary.get("base_config_sha256") != expected_digest
        or summary.get("base_config_sha256") != training[0].get("base_config_sha256")
    ):
        raise ValueError("competence-time summary has incompatible provenance")
    primary = summary.get("primary")
    if not isinstance(primary, dict) or not isinstance(primary.get("folds"), list):
        raise ValueError("competence-time summary has no plottable LOSO result")


def _primary_rows(config: dict, training: list[dict], probes: list[dict]) -> list[dict]:
    """Pair each registered block-2 geometry observation to raw competence."""
    competence = {
        (int(row["seed"]), float(row["learning_rate"]), int(row["step"])): float(row["competence"])
        for row in training
    }
    site = config["competence_time"]["primary_site"]
    target = f"{config['competence_time']['primary_target']}_r2"
    primary_steps = {int(step) for step in config["competence_time"]["primary_checkpoints"]}
    rows = []
    for row in probes:
        if row["site"] != site or row["control"] != "none":
            continue
        key = (int(row["seed"]), float(row["learning_rate"]), int(row["step"]))
        rows.append({
            "seed": key[0], "learning_rate": key[1], "step": key[2],
            "competence": competence[key], "geometry": float(row[target]),
            "primary": key[2] in primary_steps,
        })
    return rows


def _held_out_predictions(rows: list[dict]) -> list[dict]:
    """Reconstruct the registered LOSO predictions directly from raw primary rows."""
    primary = [row for row in rows if row["primary"]]
    predictions: list[dict] = []
    for seed in sorted({row["seed"] for row in primary}):
        train = [row for row in primary if row["seed"] != seed]
        test = [row for row in primary if row["seed"] == seed]
        train_y = np.asarray([row["geometry"] for row in train])
        competence_prediction, _ = _competence_time_quadratic(
            np.asarray([row["competence"] for row in train]), train_y,
            np.asarray([row["competence"] for row in test]),
        )
        step_prediction, _ = _competence_time_quadratic(
            np.log1p([row["step"] for row in train]), train_y,
            np.log1p([row["step"] for row in test]),
        )
        for row, competence_value, step_value in zip(
            test, competence_prediction, step_prediction, strict=True
        ):
            predictions.append({
                **row,
                "competence_prediction": float(competence_value),
                "step_prediction": float(step_value),
            })
    return predictions


def _fold_mses(summary: dict, seeds: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Read one finite, labelled LOSO error pair for every configured seed."""
    folds = summary["primary"]["folds"]
    by_seed = {int(fold["held_out_seed"]): fold for fold in folds}
    if set(by_seed) != set(seeds) or len(by_seed) != len(folds):
        raise ValueError("competence-time summary has incomplete or duplicate LOSO folds")
    try:
        competence = np.asarray([float(by_seed[seed]["competence_mse"]) for seed in seeds])
        step = np.asarray([float(by_seed[seed]["step_mse"]) for seed in seeds])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("competence-time summary has malformed LOSO folds") from error
    if not np.isfinite(competence).all() or not np.isfinite(step).all():
        raise ValueError("competence-time summary has non-finite LOSO errors")
    return competence, step


def _learning_figure(config: dict, training: list[dict], probes: list[dict], destination: Path) -> Path:
    """Facet paired competence and normal-control block-2 geometry by rate."""
    rows = _primary_rows(config, training, probes)
    rates = sorted(float(rate) for rate in config["competence_time"]["learning_rates"])
    seeds = sorted(int(seed) for seed in config["competence_time"]["seeds"])
    steps = sorted(int(step) for step in config["train"]["checkpoint_steps"])
    styles = ("-", "--", ":", "-.", (0, (5, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (1, 1)), (0, (5, 2)))
    if len(seeds) > len(styles):
        raise ValueError("competence-time figure supports at most eight traceable seed styles")
    fig, axes = plt.subplots(2, len(rates), figsize=(3.1 * len(rates), 6.0), sharex="col")
    axes = np.asarray(axes).reshape(2, len(rates))
    separator = (steps[0] + steps[1]) / 2
    for column, rate in enumerate(rates):
        competence_axis, geometry_axis = axes[:, column]
        for seed_index, seed in enumerate(seeds):
            train_curve = sorted(
                (row for row in training if int(row["seed"]) == seed and float(row["learning_rate"]) == rate),
                key=lambda row: int(row["step"]),
            )
            geometry_curve = sorted(
                (row for row in rows if row["seed"] == seed and row["learning_rate"] == rate),
                key=lambda row: row["step"],
            )
            for axis, curve, metric in (
                (competence_axis, train_curve, "competence"),
                (geometry_axis, geometry_curve, "geometry"),
            ):
                post = [row for row in curve if int(row["step"]) > 0]
                initial = next(row for row in curve if int(row["step"]) == 0)
                axis.plot(
                    [row["step"] for row in post], [row[metric] for row in post],
                    color="#4477AA", linestyle=styles[seed_index], marker="o", markersize=3,
                    alpha=0.8,
                )
                axis.scatter(
                    [initial["step"]], [initial[metric]], marker="D", s=34,
                    facecolor="white", edgecolor="#222222", linewidth=0.8, zorder=3,
                )
        for axis in (competence_axis, geometry_axis):
            axis.axvspan(0, separator, color="0.92", zorder=-1)
            axis.axvline(separator, color="0.45", linewidth=0.7, linestyle="--")
            axis.set_xlim(-max(steps) * 0.035, max(steps) * 1.03)
            axis.grid(axis="y", color="0.9", linewidth=0.6)
        competence_axis.set_title(f"learning rate {rate:g}")
        if column == 0:
            competence_axis.set_ylabel("predictive competence")
            geometry_axis.set_ylabel("block-2 component-posterior R²")
        geometry_axis.set_xlabel("optimizer updates")
    fig.suptitle("Competence and component geometry by rate (◇ = untrained step 0)", y=0.995)
    handles = [
        Line2D([0], [0], color="#4477AA", linestyle=styles[index], label=f"seed {seed}")
        for index, seed in enumerate(seeds)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(4, len(seeds)), frameon=False)
    return _save(fig, destination / "mess3_competence_time_learning.png", title_space=0.90)


def _loso_figure(config: dict, training: list[dict], probes: list[dict], summary: dict, destination: Path) -> Path:
    """Show raw-row LOSO predictions alongside recorded per-seed errors and verdict."""
    rows = _primary_rows(config, training, probes)
    predictions = _held_out_predictions(rows)
    seeds = sorted(int(seed) for seed in config["competence_time"]["seeds"])
    competence_mse, step_mse = _fold_mses(summary, seeds)
    fig, (prediction_axis, error_axis) = plt.subplots(1, 2, figsize=(10.5, 4.5))
    observed = np.asarray([row["geometry"] for row in predictions])
    competence_prediction = np.asarray([row["competence_prediction"] for row in predictions])
    step_prediction = np.asarray([row["step_prediction"] for row in predictions])
    limits = (min(observed.min(), competence_prediction.min(), step_prediction.min()),
              max(observed.max(), competence_prediction.max(), step_prediction.max()))
    padding = max(0.02, (limits[1] - limits[0]) * 0.06)
    limits = (limits[0] - padding, limits[1] + padding)
    prediction_axis.scatter(observed, competence_prediction, color="#4477AA", alpha=0.7,
                            label="competence-only", marker="o", s=24)
    prediction_axis.scatter(observed, step_prediction, color="#CC6677", alpha=0.65,
                            label="log-step-only", marker="x", s=28)
    prediction_axis.plot(limits, limits, color="0.2", linewidth=0.8, linestyle="--")
    prediction_axis.set(
        xlim=limits, ylim=limits, xlabel="held-out observed block-2 component-posterior R²",
        ylabel="LOSO predicted R²", title="Held-out predictions",
    )
    prediction_axis.legend(frameon=False)

    positions = np.arange(len(seeds))
    width = 0.38
    error_axis.bar(positions - width / 2, competence_mse, width, color="#4477AA", label="competence-only")
    error_axis.bar(positions + width / 2, step_mse, width, color="#CC6677", label="log-step-only")
    error_axis.set(
        xticks=positions, xticklabels=[str(seed) for seed in seeds], xlabel="held-out seed",
        ylabel="LOSO MSE", title="Per-seed held-out error",
    )
    error_axis.legend(frameon=False)
    primary = summary["primary"]
    ratio = primary.get("competence_to_step_mse_ratio")
    ratio_text = "undefined" if ratio is None else f"{float(ratio):.3f}"
    verdict = str(summary.get("verdict", "unknown")).upper()
    fig.suptitle(
        f"Registered LOSO comparison — ratio competence/log-step = {ratio_text}; verdict: {verdict}",
        y=0.995,
    )
    return _save(fig, destination / "mess3_competence_time_loso.png", title_space=0.90)


def generate_competence_time_figures(
    config: dict,
    training: list[dict],
    probes: list[dict],
    summary: dict,
    output_dir: str | Path,
) -> list[Path]:
    """Generate only the two preregistered figures after raw and summary validation."""
    _validate_summary(config, training, probes, summary)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    return [
        _learning_figure(config, training, probes, destination),
        _loso_figure(config, training, probes, summary, destination),
    ]
