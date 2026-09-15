"""Joined raw-only comparison of short-trained and long-trained eight-token windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .position_figures import _cells


def _short_cells(
    short_records: list[dict], long_records: list[dict],
    expected_type: str = "short_context", expected_condition: str = "short_trained",
) -> tuple[dict, dict, dict]:
    long_cells, oracle = _cells(long_records)
    cells = {}
    short_provenance = {}
    for row in short_records:
        if row.get("record_type") != expected_type:
            continue
        seed = int(row["seed"])
        overlap = float(row["overlap"])
        control = row.get("control")
        if seed not in (0, 1, 2) or overlap not in (0.0, 0.35) or control not in ("none", "shuffled_labels"):
            raise ValueError("unexpected short context axis")
        long = long_cells[("transformer", seed, overlap, "trained", "restart_8", "none")]
        if row.get("config") != long["config"] or row.get("config_sha256") != long["config_sha256"]:
            raise ValueError("short and long evaluation provenance must match")
        short_identity = (row.get("short_config"), row.get("short_config_sha256"))
        if not all(short_identity) or short_identity[1] == row["config_sha256"]:
            raise ValueError("missing distinct short checkpoint provenance")
        if overlap in short_provenance and short_provenance[overlap] != short_identity:
            raise ValueError("mixed short checkpoint provenance")
        short_provenance[overlap] = short_identity
        if (
            row.get("model") != "transformer"
            or row.get("training_condition") != expected_condition
            or row.get("context") != "restart_8"
            or row.get("short_training_sequence_length") != 9
            or row.get("short_training_input_positions") != 8
            or row.get("sequence_length") != 64
            or row.get("window") != 8
            or row.get("positions_evaluated") != 56
            or row.get("probe_fit_sequences") != 256
            or row.get("test_sequences") != 192
            or row.get("probe_fit_data_seed") != seed + 909
            or row.get("test_data_seed") != seed + 1009
            or row.get("observations_evaluated") != 192 * 56
            or not row.get("probe_fit_independent")
        ):
            raise ValueError("short context evidence violates registered protocol")
        if expected_type == "budget_context" and (
            row.get("training_protocol") != "token_and_step_matched"
            or row.get("short_training_sequences") != 4032
            or row.get("short_training_batch_size") != 504
            or row.get("supervised_tokens_per_epoch") != 32256
            or row.get("optimizer_steps") != 96
        ):
            raise ValueError("budget short context violates token-and-step protocol")
        key = (seed, overlap, control)
        if key in cells:
            raise ValueError(f"duplicate short context cell: {key}")
        cells[key] = row
    for seed in (0, 1, 2):
        for overlap in (0.0, 0.35):
            for control in ("none", "shuffled_labels"):
                if (seed, overlap, control) not in cells:
                    raise ValueError(f"missing short context cell: {(seed, overlap, control)}")
    return cells, long_cells, oracle


def short_vs_long(
    short_records: list[dict], long_records: list[dict], metric: str
) -> dict[float, list[float]]:
    """Per-seed short-trained minus long-trained reset-window metric."""
    if metric not in {"component_posterior_r2", "state_posterior_r2", "kl_exact", "nll"}:
        raise ValueError(f"unsupported short context metric: {metric}")
    short, long, _ = _short_cells(short_records, long_records)
    return {
        overlap: [
            float(short[(seed, overlap, "none")][metric])
            - float(long[("transformer", seed, overlap, "trained", "restart_8", "none")][metric])
            for seed in (0, 1, 2)
        ]
        for overlap in (0.0, 0.35)
    }


def generate_short_figure(
    short_results: str | Path, long_results: str | Path, output: str | Path
) -> Path:
    with Path(short_results).open(encoding="utf-8") as handle:
        short_records = [json.loads(line) for line in handle if line.strip()]
    with Path(long_results).open(encoding="utf-8") as handle:
        long_records = [json.loads(line) for line in handle if line.strip()]
    short, long, oracle = _short_cells(short_records, long_records)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), squeeze=False)
    specs = (
        ("component_posterior_r2", "component posterior R²", "oracle_component_posterior_r2"),
        ("state_posterior_r2", "conditional-state posterior R²", "oracle_state_posterior_r2"),
        ("kl_exact", "KL(exact full Bayes || model)", "oracle_kl_full_to_window"),
    )
    for column, (metric, ylabel, oracle_key) in enumerate(specs):
        axis = axes[0, column]
        traces = (
            ("short-trained reset", "#228833", lambda seed, overlap: short[(seed, overlap, "none")]),
            ("long-trained reset", "#EE6677", lambda seed, overlap: long[("transformer", seed, overlap, "trained", "restart_8", "none")]),
            ("long-trained original", "#4477AA", lambda seed, overlap: long[("transformer", seed, overlap, "trained", "restart_8_absolute", "none")]),
            ("long untrained reset", "#CCBB44", lambda seed, overlap: long[("transformer", seed, overlap, "untrained", "restart_8", "none")]),
        )
        for label, color, source in traces:
            values = [
                [float(source(seed, overlap)[metric]) for seed in (0, 1, 2)]
                for overlap in (0.0, 0.35)
            ]
            axis.errorbar(
                (0.0, 0.35), [np.mean(v) for v in values],
                yerr=[np.std(v) for v in values], marker="o", capsize=3,
                color=color, label=label,
            )
        oracle_values = [
            [float(oracle[(seed, overlap)][oracle_key]) for seed in (0, 1, 2)]
            for overlap in (0.0, 0.35)
        ]
        axis.errorbar(
            (0.0, 0.35), [np.mean(v) for v in oracle_values],
            yerr=[np.std(v) for v in oracle_values], marker="s", linestyle=":",
            capsize=3, color="black", label="exact 8-token Bayes",
        )
        effect = short_vs_long(short_records, long_records, metric)[0.35]
        axis.set(
            xlabel="source overlap", ylabel=ylabel, xticks=(0.0, 0.35),
            title=f"short − long reset at .35: {np.mean(effect):+.3f} ± {np.std(effect):.3f}",
        )
        axis.legend(frameon=False, fontsize=7)
    fig.suptitle("Eight-token windows from length-64 sources: training context vs restart; mean ± seed SD (n=3)")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short-results", default="results/sweep_short_context.jsonl")
    parser.add_argument("--long-results", default="results/sweep_position_restart.jsonl")
    parser.add_argument("--output", default="figures/sweep_short_context.png")
    args = parser.parse_args()
    if not Path(args.short_results).exists():
        Path(args.output).unlink(missing_ok=True)
        print(f"skipped optional figure: {args.short_results} has no raw JSONL")
        return
    print(f"generated {generate_short_figure(args.short_results, args.long_results, args.output)}")


if __name__ == "__main__":
    main()
