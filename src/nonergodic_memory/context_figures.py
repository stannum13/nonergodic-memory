"""Paired context-restart analysis and figures generated only from raw JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _validated_cells(
    records: list[dict],
    models: tuple[str, ...] = ("gru", "transformer"),
    contexts: tuple[str, ...] = ("full", "restart_8"),
) -> tuple[dict[tuple[str, int, float, str, str, str], dict],
           dict[tuple[int, float], dict]]:
    seeds = (0, 1, 2)
    overlaps = (0.0, 0.35)
    conditions = ("trained", "untrained")
    controls = ("none", "shuffled_labels")
    model_cells: dict[tuple[str, int, float, str, str, str], dict] = {}
    oracle_cells: dict[tuple[int, float], dict] = {}
    provenance: dict[float, tuple[str, str]] = {}
    seen_models, seen_seeds = set(), set()
    for record in records:
        kind = record.get("record_type")
        if kind not in {"context_restart", "context_oracle"}:
            continue
        seed = int(record["seed"])
        overlap = float(record["overlap"])
        seen_seeds.add(seed)
        if seed not in seeds or overlap not in overlaps:
            raise ValueError(f"unexpected context axis: {(seed, overlap)}")
        if record.get("window") != 8 or record.get("positions_evaluated") != 56:
            raise ValueError("context evidence requires 8-token restart at length 64")
        if record.get("sequence_length") != 64:
            raise ValueError("context evidence requires sequence length 64")
        if (
            record.get("probe_fit_sequences") != 256
            or record.get("test_sequences") != 192
            or record.get("probe_fit_data_seed") != seed + 909
            or record.get("test_data_seed") != seed + 1009
        ):
            raise ValueError("context evidence requires independent registered sample seeds and sizes")
        name, digest = record.get("config"), record.get("config_sha256")
        if not name or not digest:
            raise ValueError("missing context config provenance")
        identity = (str(name), str(digest))
        if overlap in provenance and provenance[overlap] != identity:
            raise ValueError(f"mixed context config provenance at overlap {overlap}")
        provenance[overlap] = identity
        if kind == "context_oracle":
            if record.get("model") != "exact_bayes":
                raise ValueError("context oracle must be exact_bayes")
            key = (seed, overlap)
            if key in oracle_cells:
                raise ValueError(f"duplicate context oracle cell: {key}")
            oracle_cells[key] = record
            continue
        model = str(record["model"])
        seen_models.add(model)
        condition = str(record["training_condition"])
        context = str(record["context"])
        control = str(record["control"])
        if model not in models or condition not in conditions or context not in contexts or control not in controls:
            raise ValueError(f"unexpected context model cell: {(model, condition, context, control)}")
        if not record.get("probe_fit_independent"):
            raise ValueError("context probes must use independent fit/test sequences")
        if record.get("observations_evaluated") != 192 * 56:
            raise ValueError("context test sample count is incomplete")
        key = (model, seed, overlap, condition, context, control)
        if key in model_cells:
            raise ValueError(f"duplicate context model cell: {key}")
        model_cells[key] = record
    if seen_models != set(models):
        raise ValueError(f"missing required models: {seen_models}")
    if seen_seeds != set(seeds):
        raise ValueError(f"missing required seeds: {seen_seeds}")
    for seed in seeds:
        for overlap in overlaps:
            if (seed, overlap) not in oracle_cells:
                raise ValueError(f"missing context oracle cell: {(seed, overlap)}")
            for model in models:
                for condition in conditions:
                    for context in contexts:
                        for control in controls:
                            key = (model, seed, overlap, condition, context, control)
                            if key not in model_cells:
                                raise ValueError(f"missing context model cell: {key}")
    return model_cells, oracle_cells


def _damage(full: dict, restart: dict, metric: str) -> float:
    if metric in {"component_posterior_r2", "state_posterior_r2"}:
        return float(full[metric]) - float(restart[metric])
    if metric in {"kl_exact", "nll"}:
        return float(restart[metric]) - float(full[metric])
    raise ValueError(f"unsupported context-damage metric: {metric}")


def context_damage(records: list[dict], metric: str) -> dict[tuple[str, str], list[float]]:
    """Return per-seed overlap interaction of full-to-restart damage."""
    cells, _ = _validated_cells(records)
    contrasts = {}
    for model in ("gru", "transformer"):
        for condition in ("trained", "untrained"):
            values = []
            for seed in (0, 1, 2):
                damages = {}
                for overlap in (0.0, 0.35):
                    full = cells[(model, seed, overlap, condition, "full", "none")]
                    restart = cells[(model, seed, overlap, condition, "restart_8", "none")]
                    damages[overlap] = _damage(full, restart, metric)
                values.append(damages[0.35] - damages[0.0])
            contrasts[(model, condition)] = values
    return contrasts


def generate_context_figure(results: str | Path, output: str | Path) -> Path:
    """Plot model history dependence and exact eight-token Bayes information loss."""
    with Path(results).open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    cells, oracle = _validated_cells(records)
    component_contrasts = context_damage(records, "component_posterior_r2")
    kl_contrasts = context_damage(records, "kl_exact")
    overlaps = (0.0, 0.35)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), squeeze=False)
    for row, model in enumerate(("gru", "transformer")):
        for column, metric in enumerate(("component_posterior_r2", "kl_exact")):
            axis = axes[row, column]
            for condition, color, style in (
                ("trained", "#4477AA", "-"), ("untrained", "#EE6677", "--")
            ):
                means, errors = [], []
                for overlap in overlaps:
                    values = [
                        _damage(
                            cells[(model, seed, overlap, condition, "full", "none")],
                            cells[(model, seed, overlap, condition, "restart_8", "none")],
                            metric,
                        )
                        for seed in (0, 1, 2)
                    ]
                    means.append(float(np.mean(values)))
                    errors.append(float(np.std(values)))
                axis.errorbar(overlaps, means, yerr=errors, marker="o", linestyle=style,
                              capsize=3, color=color, label=condition)
            oracle_means, oracle_errors = [], []
            for overlap in overlaps:
                oracle_values = [
                    1.0 - oracle[(seed, overlap)]["oracle_component_posterior_r2"]
                    if column == 0 else oracle[(seed, overlap)]["oracle_kl_full_to_window"]
                    for seed in (0, 1, 2)
                ]
                oracle_means.append(float(np.mean(oracle_values)))
                oracle_errors.append(float(np.std(oracle_values)))
            axis.errorbar(overlaps, oracle_means, yerr=oracle_errors, marker="s",
                          linestyle=":", capsize=3, color="black", label="exact 8-token Bayes")
            axis.axhline(0, color="black", linewidth=0.7)
            contrasts = component_contrasts if column == 0 else kl_contrasts
            values = contrasts[(model, "trained")]
            title = f"{model}: trained overlap contrast {np.mean(values):+.3f} ± {np.std(values):.3f}"
            axis.set(
                title=title, xlabel="source overlap", xticks=overlaps,
                ylabel="component $R^2$ loss" if column == 0 else r"$\Delta$ KL(full Bayes || model)",
            )
            axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Length-64 models restarted on last 8 tokens; mean ± seed SD (n=3)")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/sweep_context_restart.jsonl")
    parser.add_argument("--output", default="figures/sweep_context_restart.png")
    args = parser.parse_args()
    if not Path(args.results).exists():
        Path(args.output).unlink(missing_ok=True)
        print(f"skipped optional figure: {args.results} has no raw JSONL")
        return
    print(f"generated {generate_context_figure(args.results, args.output)}")


if __name__ == "__main__":
    main()
