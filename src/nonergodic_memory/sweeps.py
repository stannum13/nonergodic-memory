"""Dedicated plots for controlled parameter sweeps."""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _mean_sd(records: list[dict], key: str) -> tuple[float, float]:
    if not records:
        return float("nan"), 0.0
    values = [record[key] for record in records]
    return float(np.mean(values)), float(np.std(values))


def _load_paths(paths: Iterable[str | Path]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    return records


def interaction_contrasts(probes: list[dict], metric: str) -> dict[str, list[float]]:
    """Paired 2×2 training-gain interaction for overlap 0/.35 and length 8/64."""
    records = [
        r for r in probes
        if r.get("record_type") == "probe" and r.get("control") == "none"
    ]
    if not records:
        raise ValueError("missing interaction probe records")
    grid = {(0.0, 8), (0.0, 64), (0.35, 8), (0.35, 64)}
    cells: dict[tuple[str, int, float, int, str], dict] = {}
    for record in records:
        axis = (float(record["overlap"]), int(record["sequence_length"]))
        if axis not in grid or record["training_condition"] not in {"trained", "untrained"}:
            raise ValueError(f"unexpected interaction cell: {axis}")
        key = (
            str(record["model"]), int(record["seed"]), *axis,
            str(record["training_condition"]),
        )
        if key in cells:
            raise ValueError(f"duplicate interaction cell: {key}")
        cells[key] = record
    models = sorted({key[0] for key in cells})
    seeds = sorted({key[1] for key in cells})
    contrasts: dict[str, list[float]] = {}
    for model in models:
        values = []
        for seed in seeds:
            gains = {}
            for overlap, length in grid:
                trained_key = (model, seed, overlap, length, "trained")
                untrained_key = (model, seed, overlap, length, "untrained")
                if trained_key not in cells or untrained_key not in cells:
                    raise ValueError(f"missing interaction cell: {(model, seed, overlap, length)}")
                gains[(overlap, length)] = (
                    float(cells[trained_key][metric]) - float(cells[untrained_key][metric])
                )
            values.append(
                (gains[(0.35, 64)] - gains[(0.35, 8)])
                - (gains[(0.0, 64)] - gains[(0.0, 8)])
            )
        contrasts[model] = values
    return contrasts


def _generate_sweep_figure(
    paths: Iterable[str | Path],
    output: str | Path,
    axis_key: str,
    title: str,
    x_label: str,
) -> Path:
    records = _load_paths(paths)
    probes = [
        record
        for record in records
        if record.get("record_type") == "probe" and record.get("control") == "none"
    ]
    interventions = [
        record
        for record in records
        if record.get("record_type") == "intervention"
        and record.get("training_condition") == "trained"
        and record.get("control") == "learned"
    ]
    if not probes or not interventions:
        raise ValueError("sweep figure requires probe and intervention records")
    models = sorted({record["model"] for record in probes})
    axis_values = sorted({float(record[axis_key]) for record in probes})
    fig, axes = plt.subplots(len(models), 2, figsize=(10.0, 3.5 * len(models)), squeeze=False)
    for row, model in enumerate(models):
        probe_axis, intervention_axis = axes[row]
        for metric, label, color in (
            ("component_posterior_r2", "component posterior", "#4477AA"),
            ("state_posterior_r2", "all conditional states", "#EE6677"),
        ):
            gains, errors = [], []
            for axis_value in axis_values:
                trained = [r for r in probes if r["model"] == model and float(r[axis_key]) == axis_value and r["training_condition"] == "trained"]
                untrained = [r for r in probes if r["model"] == model and float(r[axis_key]) == axis_value and r["training_condition"] == "untrained"]
                by_seed = {
                    r["seed"]: r[metric] - next(u[metric] for u in untrained if u["seed"] == r["seed"])
                    for r in trained
                }
                gains.append(float(np.mean(list(by_seed.values()))))
                errors.append(float(np.std(list(by_seed.values()))))
            probe_axis.errorbar(axis_values, gains, yerr=errors, marker="o", capsize=3, label=label, color=color)
        probe_axis.axhline(0, color="black", linewidth=0.7)
        probe_axis.set(title=f"{model}: training gain", xlabel=x_label, ylabel="trained − untrained $R^2$")
        probe_axis.set_xticks(axis_values)
        probe_axis.legend(frameon=False)

        for target, metric, label, color in (
            ("component", "delta_component_accuracy", "component erasure", "#4477AA"),
            ("state", "delta_conditional_state_accuracy", "state erasure", "#EE6677"),
        ):
            means, errors = [], []
            for axis_value in axis_values:
                subset = [r for r in interventions if r["model"] == model and float(r[axis_key]) == axis_value and r["target"] == target]
                mean, sd = _mean_sd(subset, metric)
                means.append(-mean)
                errors.append(sd)
            intervention_axis.errorbar(axis_values, means, yerr=errors, marker="o", capsize=3, label=label, color=color)
        intervention_axis.axhline(0, color="black", linewidth=0.7)
        intervention_axis.set(title=f"{model}: independent-probe damage", xlabel=x_label, ylabel="intended accuracy decrease")
        intervention_axis.set_xticks(axis_values)
        intervention_axis.legend(frameon=False)
    seed_count = len({record["seed"] for record in probes})
    fig.suptitle(f"{title}; mean ± seed SD (n={seed_count})")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return destination


def generate_overlap_figure(paths: Iterable[str | Path], output: str | Path) -> Path:
    return _generate_sweep_figure(
        paths, output, "overlap", "Source-overlap sweep", "source emission overlap"
    )


def generate_length_figure(paths: Iterable[str | Path], output: str | Path) -> Path:
    return _generate_sweep_figure(
        paths, output, "sequence_length", "Sequence-length sweep", "sequence length"
    )


def generate_width_figure(paths: Iterable[str | Path], output: str | Path) -> Path:
    return _generate_sweep_figure(
        paths, output, "model_width", "Model-width sweep", "model width"
    )


def generate_interaction_figure(paths: Iterable[str | Path], output: str | Path) -> Path:
    records = _load_paths(paths)
    probes = [r for r in records if r.get("record_type") == "probe" and r.get("control") == "none"]
    contrasts = interaction_contrasts(probes, "component_posterior_r2")
    interaction_contrasts(probes, "state_posterior_r2")
    models = sorted(contrasts)
    seeds = sorted({int(r["seed"]) for r in probes})
    overlaps = (0.0, 0.35)
    lengths = (8, 64)
    interventions = [
        r for r in records
        if r.get("record_type") == "intervention"
        and r.get("training_condition") == "trained"
        and r.get("target") == "component"
        and r.get("control") in {"learned", "norm_matched_random"}
    ]
    control_cells: dict[tuple[str, int, float, int, str], dict] = {}
    for record in interventions:
        if not record.get("independent_evaluator"):
            raise ValueError("interaction figure requires independent intervention evaluator")
        key = (
            str(record["model"]), int(record["seed"]), float(record["overlap"]),
            int(record["sequence_length"]), str(record["control"]),
        )
        if key in control_cells:
            raise ValueError(f"duplicate interaction intervention cell: {key}")
        control_cells[key] = record
    for model in models:
        for seed in seeds:
            for overlap in overlaps:
                for length in lengths:
                    for control in ("learned", "norm_matched_random"):
                        key = (model, seed, overlap, length, control)
                        if key not in control_cells:
                            raise ValueError(f"missing interaction intervention cell: {key}")

    fig, axes = plt.subplots(len(models), 3, figsize=(15.0, 3.6 * len(models)), squeeze=False)
    for row, model in enumerate(models):
        for column, metric in enumerate(("component_posterior_r2", "state_posterior_r2")):
            axis = axes[row, column]
            for overlap, color in ((0.0, "#4477AA"), (0.35, "#EE6677")):
                means, errors = [], []
                for length in lengths:
                    gains = []
                    for seed in seeds:
                        trained = next(
                            r for r in probes if r["model"] == model and r["seed"] == seed
                            and float(r["overlap"]) == overlap and int(r["sequence_length"]) == length
                            and r["training_condition"] == "trained"
                        )
                        untrained = next(
                            r for r in probes if r["model"] == model and r["seed"] == seed
                            and float(r["overlap"]) == overlap and int(r["sequence_length"]) == length
                            and r["training_condition"] == "untrained"
                        )
                        gains.append(float(trained[metric]) - float(untrained[metric]))
                    means.append(float(np.mean(gains)))
                    errors.append(float(np.std(gains)))
                axis.errorbar(lengths, means, yerr=errors, marker="o", capsize=3,
                              color=color, label=f"overlap {overlap:.2f}")
            axis.axhline(0, color="black", linewidth=0.7)
            if column == 0:
                contrast = contrasts[model]
                title = f"{model}: component gain; I={np.mean(contrast):+.3f} ± {np.std(contrast):.3f}"
            else:
                title = f"{model}: conditional-state gain"
            axis.set(title=title, xlabel="sequence length", ylabel="trained − untrained $R^2$", xticks=lengths)
            axis.legend(frameon=False)

        axis = axes[row, 2]
        for overlap, color in ((0.0, "#4477AA"), (0.35, "#EE6677")):
            means, errors = [], []
            for length in lengths:
                advantages = []
                for seed in seeds:
                    learned = control_cells[(model, seed, overlap, length, "learned")]
                    matched = control_cells[(model, seed, overlap, length, "norm_matched_random")]
                    advantages.append(
                        float(matched["delta_component_accuracy"])
                        - float(learned["delta_component_accuracy"])
                    )
                means.append(float(np.mean(advantages)))
                errors.append(float(np.std(advantages)))
            axis.errorbar(lengths, means, yerr=errors, marker="o", capsize=3,
                          color=color, label=f"overlap {overlap:.2f}")
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set(title=f"{model}: component-erasure advantage", xlabel="sequence length",
                 ylabel="learned − norm-matched damage", xticks=lengths)
        axis.legend(frameon=False)
    fig.suptitle(f"Overlap × context interaction; mean ± seed SD (n={len(seeds)})")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return destination


def generate_depth_figure(paths: Iterable[str | Path], output: str | Path) -> Path:
    records = [r for r in _load_paths(paths) if r.get("record_type") == "intervention_depth"]
    if not records:
        raise ValueError("depth sweep requires intervention_depth records")
    depths = sorted({int(r["depth"]) for r in records})
    labels = {
        int(r["depth"]): str(r["depth_label"]).replace("_", " ")
        for r in records
    }
    fig, (recovery_axis, damage_axis, predictive_axis) = plt.subplots(
        1, 3, figsize=(15.0, 4.0)
    )
    baseline = [
        r for r in records
        if r["control"] == "baseline" and r["target"] == "component"
    ]
    for metric, label, color in (
        ("baseline_component_posterior_r2", "component posterior r2", "#4477AA"),
        ("baseline_state_posterior_r2", "state posterior r2", "#EE6677"),
    ):
        for condition, style in (("trained", "-"), ("untrained", "--")):
            means, errors = [], []
            for depth in depths:
                subset = [
                    r for r in baseline
                    if int(r["depth"]) == depth and r["training_condition"] == condition
                ]
                mean, sd = _mean_sd(subset, metric)
                means.append(mean)
                errors.append(sd)
            recovery_axis.errorbar(
                depths, means, yerr=errors, marker="o", linestyle=style, capsize=3,
                color=color, label=f"{label}: {condition}",
            )
    recovery_axis.set(
        title="Linear recovery by intervention depth",
        xlabel="activation site",
        ylabel="held-out $R^2$",
        xticks=depths,
        xticklabels=[labels[d] for d in depths],
    )
    recovery_axis.legend(frameon=False, fontsize=8)

    for target, metric, color in (
        ("component", "delta_component_accuracy", "#4477AA"),
        ("state", "delta_conditional_state_accuracy", "#EE6677"),
    ):
        for control, style in (("learned", "-"), ("norm_matched_random", "--")):
            means, errors = [], []
            for depth in depths:
                subset = [
                    r for r in records
                    if int(r["depth"]) == depth
                    and r["training_condition"] == "trained"
                    and r["target"] == target
                    and r["control"] == control
                ]
                mean, sd = _mean_sd(subset, metric)
                means.append(-mean)
                errors.append(sd)
            damage_axis.errorbar(
                depths, means, yerr=errors, marker="o", linestyle=style, capsize=3,
                color=color, label=f"{target}: {control.replace('_', ' ')}",
            )
    damage_axis.axhline(0, color="black", linewidth=0.7)
    damage_axis.set(
        title="Intended independent-probe damage",
        xlabel="activation site",
        ylabel="accuracy decrease",
        xticks=depths,
        xticklabels=[labels[d] for d in depths],
    )
    damage_axis.legend(frameon=False, fontsize=8)
    for target, color in (("component", "#4477AA"), ("state", "#EE6677")):
        for control, style in (("learned", "-"), ("norm_matched_random", "--")):
            means, errors = [], []
            for depth in depths:
                subset = [
                    r for r in records
                    if int(r["depth"]) == depth
                    and r["training_condition"] == "trained"
                    and r["target"] == target
                    and r["control"] == control
                ]
                mean, sd = _mean_sd(subset, "delta_kl_exact")
                means.append(mean)
                errors.append(sd)
            predictive_axis.errorbar(
                depths, means, yerr=errors, marker="o", linestyle=style, capsize=3,
                color=color, label=f"{target}: {control.replace('_', ' ')}",
            )
    predictive_axis.axhline(0, color="black", linewidth=0.7)
    predictive_axis.set(
        title="Exact-predictive damage",
        xlabel="activation site",
        ylabel=r"$\Delta$ KL(exact || model)",
        xticks=depths,
        xticklabels=[labels[d] for d in depths],
    )
    predictive_axis.legend(frameon=False, fontsize=8)
    seed_count = len({r["seed"] for r in records})
    fig.suptitle(f"Transformer intervention-depth sweep; mean ± seed SD (n={seed_count})")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return destination


def generate_component_figure(paths: Iterable[str | Path], output: str | Path) -> Path:
    records = _load_paths(paths)
    probes = [r for r in records if r.get("record_type") == "probe" and r.get("control") == "none"]
    interventions = [
        r for r in records
        if r.get("record_type") == "intervention" and r.get("control") == "learned" and r.get("target") == "component"
    ]
    if not probes or not interventions:
        raise ValueError("component sweep requires probe and intervention records")
    models = sorted({r["model"] for r in probes})
    counts = sorted({int(r["components"]) for r in probes})
    fig, axes = plt.subplots(len(models), 2, figsize=(10.0, 3.5 * len(models)), squeeze=False)
    for row, model in enumerate(models):
        recovery_axis, damage_axis = axes[row]
        for metric, color in (("component_accuracy", "#4477AA"), ("component_posterior_r2", "#EE6677")):
            for condition, style in (("trained", "-"), ("untrained", "--")):
                means, errors = [], []
                for count in counts:
                    subset = [r for r in probes if r["model"] == model and r["components"] == count and r["training_condition"] == condition]
                    mean, sd = _mean_sd(subset, metric)
                    means.append(mean)
                    errors.append(sd)
                recovery_axis.errorbar(counts, means, yerr=errors, marker="o", linestyle=style, capsize=3, color=color, label=f"{metric.replace('_', ' ')}: {condition}")
        recovery_axis.set(title=f"{model}: component recovery", xlabel="number of components", ylabel="held-out score", xticks=counts)
        recovery_axis.legend(frameon=False, fontsize=8)

        for condition, style, color in (("trained", "-", "#4477AA"), ("untrained", "--", "#999999")):
            means, errors = [], []
            for count in counts:
                subset = [r for r in interventions if r["model"] == model and r["components"] == count and r["training_condition"] == condition]
                mean, sd = _mean_sd(subset, "delta_component_accuracy")
                means.append(-mean)
                errors.append(sd)
            damage_axis.errorbar(counts, means, yerr=errors, marker="o", linestyle=style, capsize=3, color=color, label=condition)
        damage_axis.axhline(0, color="black", linewidth=0.7)
        damage_axis.set(title=f"{model}: independent-probe component damage", xlabel="number of components", ylabel="accuracy decrease", xticks=counts)
        damage_axis.legend(frameon=False)
    seed_count = len({r["seed"] for r in probes})
    fig.suptitle(f"Component-count sweep; mean ± seed SD (n={seed_count})")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", choices=["overlap", "length", "components", "width", "depth", "interaction"], default="overlap")
    args = parser.parse_args()
    prefix = {"overlap": "sweep_overlap", "length": "sweep_length", "components": "sweep_components", "width": "sweep_width", "depth": "sweep_depth", "interaction": "sweep_interaction"}[args.axis]
    generator = {"overlap": generate_overlap_figure, "length": generate_length_figure, "components": generate_component_figure, "width": generate_width_figure, "depth": generate_depth_figure, "interaction": generate_interaction_figure}[args.axis]
    paths = [f"results/{prefix}.jsonl"] if args.axis == "depth" else [
        f"results/{prefix}_reproduction.jsonl",
        f"results/{prefix}_extension.jsonl",
    ]
    output = generator(paths, f"figures/{prefix}.png")
    print(f"generated {output}")


if __name__ == "__main__":
    main()
