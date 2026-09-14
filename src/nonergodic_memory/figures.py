"""Regenerate every quantitative figure from JSONL records."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_records(results_dir: Path) -> list[dict]:
    records: list[dict] = []
    paths = sorted(results_dir.glob("*.jsonl"))
    canonical_names = {"training.jsonl", "reproduction.jsonl", "extension.jsonl"}
    central_paths = [path for path in paths if path.name in canonical_names]
    if not central_paths:
        central_paths = [path for path in paths if not path.name.startswith("smoke_")]
    if central_paths:
        paths = central_paths
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    digests_by_name: dict[str, set[str]] = {}
    for record in records:
        if "config" in record and "config_sha256" in record:
            digests_by_name.setdefault(record["config"], set()).add(record["config_sha256"])
    mixed = {name: digests for name, digests in digests_by_name.items() if len(digests) > 1}
    if mixed:
        raise ValueError(f"mixed config digests for the same config name: {mixed}")
    return records


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _std_or_zero(values: list[float]) -> float:
    return float(np.std(values)) if values else 0.0


def generate_figures(results_dir: str | Path = "results", output_dir: str | Path = "figures") -> list[Path]:
    records = _load_records(Path(results_dir))
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("training.png", "probes.png", "pca.png", "intervention.png"):
        (destination / name).unlink(missing_ok=True)
    created: list[Path] = []

    training = [record for record in records if record.get("record_type") == "training"]
    if training:
        models = sorted({record["model"] for record in training})
        means = [np.mean([record["test_nll"] for record in training if record["model"] == model]) for model in models]
        stds = [np.std([record["test_nll"] for record in training if record["model"] == model]) for model in models]
        bayes = np.mean([record["test_bayes_nll"] for record in training])
        fig, axis = plt.subplots(figsize=(5.0, 3.4))
        axis.bar(models, means, yerr=stds, capsize=4, color=["#4477AA", "#EE6677"][: len(models)])
        axis.axhline(bayes, color="black", linestyle="--", label="exact Bayes")
        axis.set_ylabel("held-out NLL (nats/token)")
        axis.set_title("Predictive performance (mean ± seed SD)")
        axis.legend(frameon=False)
        created.append(_save(fig, destination / "training.png"))

    probes = [record for record in records if record.get("record_type") == "probe"]
    if probes:
        metrics = ["component_accuracy", "conditional_state_accuracy", "component_posterior_r2", "state_posterior_r2"]
        labels = ["component\naccuracy", "state | component\naccuracy", "component posterior\n$R^2$", "state posterior\n$R^2$"]
        groups = [
            ("trained", "none", "trained"),
            ("untrained", "none", "untrained"),
            ("trained", "shuffled_labels", "shuffled labels"),
        ]
        models = sorted({record["model"] for record in probes})
        fig, axes = plt.subplots(len(models), 1, figsize=(8.0, 3.4 * len(models)), squeeze=False)
        for axis, model in zip(axes[:, 0], models):
            model_records = [record for record in probes if record["model"] == model]
            x = np.arange(len(metrics))
            width = 0.24
            for group_index, (condition, control, label) in enumerate(groups):
                subset = [r for r in model_records if r["training_condition"] == condition and r["control"] == control]
                values = [_mean_or_nan([r[m] for r in subset]) for m in metrics]
                errors = [_std_or_zero([r[m] for r in subset]) for m in metrics]
                axis.bar(x + (group_index - 1) * width, values, width, yerr=errors, capsize=3, label=label)
            seed_count = len({record["seed"] for record in model_records})
            axis.axhline(0, color="black", linewidth=0.7)
            axis.set_xticks(x, labels)
            axis.set_title(f"{model}: mean ± seed SD (n={seed_count})")
            axis.legend(frameon=False, ncol=3)
        configs = ",".join(sorted({r.get("config", "unspecified") for r in probes}))
        fig.suptitle(f"Held-out linear probes; config={configs}")
        created.append(_save(fig, destination / "probes.png"))

    pca = [record for record in records if record.get("record_type") == "pca"]
    if pca:
        models = sorted({record["model"] for record in pca})
        fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.7), squeeze=False)
        for axis, model in zip(axes[0], models):
            subset = [r for r in pca if r["model"] == model and r.get("seed") == min(x["seed"] for x in pca if x["model"] == model)]
            colors = [r["component"] * 2 + r["state"] for r in subset]
            axis.scatter([r["pc1"] for r in subset], [r["pc2"] for r in subset], c=colors, s=9, alpha=0.55, cmap="viridis")
            axis.set(title=f"{model}: final activations", xlabel="PC1", ylabel="PC2")
        created.append(_save(fig, destination / "pca.png"))

    intervention = [
        record
        for record in records
        if record.get("record_type") == "intervention"
        and record.get("training_condition") == "trained"
        and record.get("control") in {"learned", "norm_matched_random"}
    ]
    if intervention:
        categories = [(target, control) for target in ("component", "state") for control in ("learned", "norm_matched_random")]
        models = sorted({record["model"] for record in intervention})
        fig, axes = plt.subplots(len(models), 1, figsize=(7.2, 3.5 * len(models)), squeeze=False)
        for axis, model in zip(axes[:, 0], models):
            model_records = [record for record in intervention if record["model"] == model]
            comp = [_mean_or_nan([r["delta_component_accuracy"] for r in model_records if (r["target"], r["control"]) == category]) for category in categories]
            state = [_mean_or_nan([r["delta_conditional_state_accuracy"] for r in model_records if (r["target"], r["control"]) == category]) for category in categories]
            comp_error = [_std_or_zero([r["delta_component_accuracy"] for r in model_records if (r["target"], r["control"]) == category]) for category in categories]
            state_error = [_std_or_zero([r["delta_conditional_state_accuracy"] for r in model_records if (r["target"], r["control"]) == category]) for category in categories]
            x = np.arange(len(categories))
            axis.bar(x - 0.18, comp, 0.36, yerr=comp_error, capsize=3, label="component accuracy", color="#4477AA")
            axis.bar(x + 0.18, state, 0.36, yerr=state_error, capsize=3, label="state | component accuracy", color="#EE6677")
            axis.axhline(0, color="black", linewidth=0.7)
            axis.set_xticks(x, [f"{t}\n{c.replace('_', ' ')}" for t, c in categories])
            axis.set_ylabel("post − pre accuracy")
            seed_count = len({record["seed"] for record in model_records})
            axis.set_title(f"{model}: mean ± seed SD (n={seed_count})")
            axis.legend(frameon=False)
        configs = ",".join(sorted({r.get("config", "unspecified") for r in intervention}))
        fig.suptitle(f"Selective causal damage; config={configs}")
        created.append(_save(fig, destination / "intervention.png"))
    return created


def main() -> None:
    created = generate_figures()
    print("generated", ", ".join(str(path) for path in created))


if __name__ == "__main__":
    main()
