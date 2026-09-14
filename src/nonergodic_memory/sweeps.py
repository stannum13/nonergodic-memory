"""Dedicated plots for controlled parameter sweeps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _mean_sd(records: list[dict], key: str) -> tuple[float, float]:
    values = [record[key] for record in records]
    return float(np.mean(values)), float(np.std(values))


def generate_overlap_figure(paths: Iterable[str | Path], output: str | Path) -> Path:
    records: list[dict] = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
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
        raise ValueError("overlap figure requires probe and intervention records")
    models = sorted({record["model"] for record in probes})
    overlaps = sorted({float(record["overlap"]) for record in probes})
    fig, axes = plt.subplots(len(models), 2, figsize=(10.0, 3.5 * len(models)), squeeze=False)
    for row, model in enumerate(models):
        probe_axis, intervention_axis = axes[row]
        for metric, label, color in (
            ("component_posterior_r2", "component posterior", "#4477AA"),
            ("state_posterior_r2", "all conditional states", "#EE6677"),
        ):
            gains, errors = [], []
            for overlap in overlaps:
                trained = [r for r in probes if r["model"] == model and r["overlap"] == overlap and r["training_condition"] == "trained"]
                untrained = [r for r in probes if r["model"] == model and r["overlap"] == overlap and r["training_condition"] == "untrained"]
                by_seed = {
                    r["seed"]: r[metric] - next(u[metric] for u in untrained if u["seed"] == r["seed"])
                    for r in trained
                }
                gains.append(float(np.mean(list(by_seed.values()))))
                errors.append(float(np.std(list(by_seed.values()))))
            probe_axis.errorbar(overlaps, gains, yerr=errors, marker="o", capsize=3, label=label, color=color)
        probe_axis.axhline(0, color="black", linewidth=0.7)
        probe_axis.set(title=f"{model}: training gain", xlabel="source emission overlap", ylabel="trained − untrained $R^2$")
        probe_axis.legend(frameon=False)

        for target, metric, label, color in (
            ("component", "delta_component_accuracy", "component erasure", "#4477AA"),
            ("state", "delta_conditional_state_accuracy", "state erasure", "#EE6677"),
        ):
            means, errors = [], []
            for overlap in overlaps:
                subset = [r for r in interventions if r["model"] == model and r["overlap"] == overlap and r["target"] == target]
                mean, sd = _mean_sd(subset, metric)
                means.append(-mean)
                errors.append(sd)
            intervention_axis.errorbar(overlaps, means, yerr=errors, marker="o", capsize=3, label=label, color=color)
        intervention_axis.axhline(0, color="black", linewidth=0.7)
        intervention_axis.set(title=f"{model}: independent-probe damage", xlabel="source emission overlap", ylabel="intended accuracy decrease")
        intervention_axis.legend(frameon=False)
    seed_count = len({record["seed"] for record in probes})
    fig.suptitle(f"Source-overlap sweep; mean ± seed SD (n={seed_count})")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return destination


def main() -> None:
    output = generate_overlap_figure(
        [
            "results/sweep_overlap_reproduction.jsonl",
            "results/sweep_overlap_extension.jsonl",
        ],
        "figures/sweep_overlap.png",
    )
    print(f"generated {output}")


if __name__ == "__main__":
    main()

