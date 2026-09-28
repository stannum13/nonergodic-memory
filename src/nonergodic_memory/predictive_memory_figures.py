"""Figures generated only from predictive-memory JSONL evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .experiment import load_config
from .predictive_memory import classify_pilot


def _read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _pooled_score(rows: list[dict]) -> float:
    return float(
        1.0
        - sum(float(row["numerator"]) for row in rows)
        / sum(float(row["denominator"]) for row in rows)
    )


def generate_predictive_memory_figure(
    results_path: str | Path,
    output_path: str | Path = "figures/predictive_memory.png",
    summary_path: str | Path = "results/predictive_memory_summary.jsonl",
) -> Path:
    rows = _read_jsonl(results_path)
    summaries = _read_jsonl(summary_path)
    if len(summaries) != 1 or summaries[0].get("record_type") != "summary":
        raise ValueError("predictive-memory figure requires one terminal summary")
    status = summaries[0].get("status")
    if status not in {
        "actuator_infeasible",
        "invalid_pilot",
        "criterion_not_met",
        "promising_pilot",
    }:
        raise ValueError(f"unsupported predictive-memory terminal status: {status}")
    calibration = [
        row
        for row in rows
        if row.get("record_type") == "calibration" and row.get("step") == 3072
    ]
    if not calibration:
        raise ValueError("predictive-memory results contain no trained calibration rows")
    response = [
        row
        for row in rows
        if row.get("record_type") == "response"
        and row.get("cohort") == "heldout"
        and row.get("step") == 3072
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    calibration.sort(key=lambda row: (row.get("cohort", ""), int(row["seed"])))
    labels = [f"{row.get('cohort', 'dev')[0].upper()}{row['seed']}" for row in calibration]
    positions = np.arange(len(calibration))
    width = 0.36
    axes[0].bar(
        positions - width / 2,
        [row["component_r2"] for row in calibration],
        width,
        label="component $R^2$",
    )
    axes[0].bar(
        positions + width / 2,
        [row["joint_r2"] for row in calibration],
        width,
        label="joint-belief $R^2$",
    )
    axes[0].axhline(0.20, color="C0", linestyle="--", linewidth=1)
    axes[0].axhline(0.50, color="C1", linestyle="--", linewidth=1)
    axes[0].set_xticks(positions, labels, rotation=45, ha="right")
    axes[0].set_ylabel("held-out linear recovery")
    axes[0].set_title("Registered actuator feasibility")
    axes[0].legend(frameon=False, fontsize=8)

    if response:
        if status not in {"criterion_not_met", "promising_pilot"}:
            raise ValueError("response evidence conflicts with terminal summary")
        config_path = Path(__file__).resolve().parents[2] / "configs/predictive_memory.yaml"
        config = load_config(config_path)
        recomputed = classify_pilot(
            response,
            heldout_seeds=tuple(config["models"]["heldout_seeds"]),
            primary_step=int(config["models"]["primary_checkpoint"]),
            doses=tuple(float(dose) for dose in config["experiment"]["doses"] if dose),
            random_controls=int(config["experiment"]["random_controls"]),
            examples_per_cell=int(config["data"]["evaluation"]),
            mean_score_min=float(config["decision"]["mean_score_min"]),
            control_margin_min=float(config["decision"]["control_margin_min"]),
        )
        if recomputed["status"] != status:
            raise ValueError("response evidence does not reproduce terminal verdict")
        seeds = sorted({int(row["seed"]) for row in response})
        controls = ("learned", "shuffled", "random")
        for control in controls:
            scores = []
            for seed in seeds:
                selected = [
                    row
                    for row in response
                    if int(row["seed"]) == seed and row["control"] == control
                ]
                scores.append(_pooled_score(selected))
            axes[1].plot(seeds, scores, marker="o", label=control)
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].axhline(0.20, color="black", linestyle="--", linewidth=1)
        axes[1].set_xlabel("held-out model seed")
        axes[1].set_ylabel("conditional-response score $S$")
        axes[1].set_title("Persistent Bayesian response")
        axes[1].legend(frameon=False)
    else:
        if status not in {"actuator_infeasible", "invalid_pilot"}:
            raise ValueError("complete-outcome summary lacks response evidence")
        errors = [row["displacement_relative_error"] for row in calibration]
        colors = ["C2" if row.get("passed") else "C3" for row in calibration]
        axes[1].bar(positions, errors, color=colors)
        axes[1].axhline(0.50, color="black", linestyle="--", linewidth=1)
        axes[1].set_xticks(positions, labels, rotation=45, ha="right")
        axes[1].set_ylabel("independent displacement error")
        axes[1].set_title("Stopped before causal outcome" if any(not row.get("passed") for row in calibration) else "Displacement audit")

    fig.suptitle("Predictive-memory causal pilot", fontweight="bold")
    fig.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/predictive_memory.jsonl")
    parser.add_argument("--output", default="figures/predictive_memory.png")
    parser.add_argument("--summary", default="results/predictive_memory_summary.jsonl")
    args = parser.parse_args()
    print(generate_predictive_memory_figure(args.results, args.output, args.summary))


if __name__ == "__main__":
    main()
