import json
from pathlib import Path

import numpy as np
import pytest

from nonergodic_memory.sweeps import (
    generate_component_figure,
    generate_depth_figure,
    generate_interaction_figure,
    generate_length_figure,
    generate_overlap_figure,
    generate_width_figure,
    interaction_contrasts,
)


def _interaction_probe_records() -> list[dict]:
    gains = {
        "gru": {(0.0, 8): 0.10, (0.0, 64): 0.20, (0.35, 8): 0.15, (0.35, 64): 0.45},
        "transformer": {(0.0, 8): 0.20, (0.0, 64): 0.40, (0.35, 8): 0.25, (0.35, 64): 0.35},
    }
    records = []
    for model, grid in gains.items():
        for seed in (0, 1, 2):
            for (overlap, length), gain in grid.items():
                for condition in ("trained", "untrained"):
                    records.append(
                        {
                            "record_type": "probe",
                            "model": model,
                            "seed": seed,
                            "overlap": overlap,
                            "sequence_length": length,
                            "training_condition": condition,
                            "control": "none",
                            "component_posterior_r2": 0.2 + 0.01 * seed + (gain if condition == "trained" else 0.0),
                            "state_posterior_r2": 0.3 + (0.02 if condition == "trained" else 0.0),
                        }
                    )
    return records


def test_interaction_contrast_is_paired_and_complete() -> None:
    records = _interaction_probe_records()
    contrasts = interaction_contrasts(records, "component_posterior_r2")
    np.testing.assert_allclose(contrasts["gru"], [0.20, 0.20, 0.20])
    np.testing.assert_allclose(contrasts["transformer"], [-0.10, -0.10, -0.10])
    with pytest.raises(ValueError, match="missing"):
        interaction_contrasts(records[:-1], "component_posterior_r2")
    with pytest.raises(ValueError, match="duplicate"):
        interaction_contrasts([*records, records[0]], "component_posterior_r2")
    with pytest.raises(ValueError, match="required models"):
        interaction_contrasts([r for r in records if r["model"] != "gru"], "component_posterior_r2")
    with pytest.raises(ValueError, match="required seeds"):
        interaction_contrasts([r for r in records if r["seed"] != 2], "component_posterior_r2")


def _interaction_figure_records() -> tuple[list[dict], list[dict]]:
    probes = _interaction_probe_records()
    for record in probes:
        record["config"] = f"interaction_o{int(record['overlap'] * 100):03d}_l{record['sequence_length']:03d}"
        record["config_sha256"] = f"digest_{int(record['overlap'] * 100):03d}_{record['sequence_length']:03d}"
    probes.extend({**record, "control": "shuffled_labels"} for record in list(probes))
    interventions = []
    for model in ("gru", "transformer"):
        for seed in (0, 1, 2):
            for overlap in (0.0, 0.35):
                for length in (8, 64):
                    for condition in ("trained", "untrained"):
                        for target in ("component", "state"):
                            for control in ("baseline", "learned", "random_subspace", "norm_matched_random", "shuffled_labels"):
                                if condition == "trained" and target == "component" and control == "learned":
                                    damage = 0.10 + 0.1 * overlap + 0.002 * length
                                elif condition == "trained" and target == "component" and control == "norm_matched_random":
                                    damage = 0.01
                                else:
                                    damage = 0.001
                                interventions.append(
                                    {
                                        "record_type": "intervention",
                                        "model": model,
                                        "seed": seed,
                                        "overlap": overlap,
                                        "sequence_length": length,
                                        "config": f"interaction_o{int(overlap * 100):03d}_l{length:03d}",
                                        "config_sha256": f"digest_{int(overlap * 100):03d}_{length:03d}",
                                        "training_condition": condition,
                                        "target": target,
                                        "control": control,
                                        "independent_evaluator": True,
                                        "delta_component_accuracy": -damage,
                                    }
                                )
    return probes, interventions


def _write_interaction_fixture(tmp_path: Path, probes: list[dict], interventions: list[dict]) -> list[Path]:
    probe_path = tmp_path / "probes.jsonl"
    intervention_path = tmp_path / "interventions.jsonl"
    probe_path.write_text("".join(json.dumps(r) + "\n" for r in probes))
    intervention_path.write_text("".join(json.dumps(r) + "\n" for r in interventions))
    return [probe_path, intervention_path]


def test_interaction_figure_uses_four_cell_raw_grid(tmp_path: Path) -> None:
    probes, interventions = _interaction_figure_records()
    paths = _write_interaction_fixture(tmp_path, probes, interventions)
    output = tmp_path / "interaction.png"
    generate_interaction_figure(paths, output)
    assert output.exists() and output.stat().st_size > 1000


def test_interaction_figure_rejects_missing_random_control(tmp_path: Path) -> None:
    probes, interventions = _interaction_figure_records()
    paths = _write_interaction_fixture(
        tmp_path, probes, [r for r in interventions if r["control"] != "random_subspace"]
    )
    with pytest.raises(ValueError, match="missing interaction intervention"):
        generate_interaction_figure(paths, tmp_path / "incomplete.png")


def test_interaction_figure_rejects_spliced_config_digest(tmp_path: Path) -> None:
    probes, interventions = _interaction_figure_records()
    interventions[0]["config_sha256"] = "wrong_digest"
    paths = _write_interaction_fixture(tmp_path, probes, interventions)
    with pytest.raises(ValueError, match="mixed config provenance"):
        generate_interaction_figure(paths, tmp_path / "spliced.png")


def test_overlap_figure_is_generated_from_raw_records(tmp_path: Path) -> None:
    records = []
    for overlap in (0.0, 0.7):
        for model in ("gru", "transformer"):
            for seed in (0, 1):
                for condition, component_r2, state_r2 in (
                    ("trained", 0.9 - overlap / 2, 0.8),
                    ("untrained", 0.5 - overlap / 3, 0.7),
                ):
                    records.append(
                        {
                            "record_type": "probe",
                            "model": model,
                            "seed": seed,
                            "overlap": overlap,
                            "training_condition": condition,
                            "control": "none",
                            "component_posterior_r2": component_r2,
                            "state_posterior_r2": state_r2,
                        }
                    )
                for target in ("component", "state"):
                    records.append(
                        {
                            "record_type": "intervention",
                            "model": model,
                            "seed": seed,
                            "overlap": overlap,
                            "training_condition": "trained",
                            "control": "learned",
                            "target": target,
                            "delta_component_accuracy": -0.2 if target == "component" else -0.01,
                            "delta_conditional_state_accuracy": -0.02 if target == "component" else -0.15,
                        }
                    )
    path = tmp_path / "overlap.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    output = tmp_path / "overlap.png"
    generate_overlap_figure([path], output)
    assert output.exists()
    assert output.stat().st_size > 1000


def test_length_figure_uses_sequence_length_axis(tmp_path: Path) -> None:
    records = []
    for length in (8, 32):
        for condition, component_r2, state_r2 in (
            ("trained", 0.8, 0.75),
            ("untrained", 0.5, 0.70),
        ):
            records.append(
                {
                    "record_type": "probe",
                    "model": "gru",
                    "seed": 0,
                    "sequence_length": length,
                    "training_condition": condition,
                    "control": "none",
                    "component_posterior_r2": component_r2,
                    "state_posterior_r2": state_r2,
                }
            )
        for target in ("component", "state"):
            records.append(
                {
                    "record_type": "intervention",
                    "model": "gru",
                    "seed": 0,
                    "sequence_length": length,
                    "training_condition": "trained",
                    "control": "learned",
                    "target": target,
                    "delta_component_accuracy": -0.2 if target == "component" else -0.01,
                    "delta_conditional_state_accuracy": -0.02 if target == "component" else -0.15,
                }
            )
    path = tmp_path / "length.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    output = tmp_path / "length.png"
    generate_length_figure([path], output)
    assert output.exists()
    assert output.stat().st_size > 1000


def test_component_figure_tracks_absolute_recovery(tmp_path: Path) -> None:
    records = []
    for components in (2, 3):
        for condition in ("trained", "untrained"):
            records.append(
                {
                    "record_type": "probe",
                    "model": "gru",
                    "seed": 0,
                    "components": components,
                    "training_condition": condition,
                    "control": "none",
                    "component_accuracy": 0.8,
                    "component_posterior_r2": 0.7,
                }
            )
        records.append(
            {
                "record_type": "intervention",
                "model": "gru",
                "seed": 0,
                "components": components,
                "training_condition": "trained",
                "control": "learned",
                "target": "component",
                "delta_component_accuracy": -0.2,
            }
        )
    path = tmp_path / "components.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    output = tmp_path / "components.png"
    generate_component_figure([path], output)
    assert output.exists()
    assert output.stat().st_size > 1000


def test_width_figure_uses_model_width_axis(tmp_path: Path) -> None:
    records = []
    for width in (8, 16):
        for condition in ("trained", "untrained"):
            records.append(
                {"record_type": "probe", "model": "gru", "seed": 0, "model_width": width, "training_condition": condition, "control": "none", "component_posterior_r2": 0.8, "state_posterior_r2": 0.7}
            )
        for target in ("component", "state"):
            records.append(
                {"record_type": "intervention", "model": "gru", "seed": 0, "model_width": width, "training_condition": "trained", "control": "learned", "target": target, "delta_component_accuracy": -0.2, "delta_conditional_state_accuracy": -0.1}
            )
    path = tmp_path / "width.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    output = tmp_path / "width.png"
    generate_width_figure([path], output)
    assert output.exists() and output.stat().st_size > 1000


def test_depth_figure_compares_recovery_and_controls(tmp_path: Path) -> None:
    records = []
    for depth, label in enumerate(("block_1", "block_2", "final_norm")):
        for seed in (0, 1):
            for condition in ("trained", "untrained"):
                for target in ("component", "state"):
                    for control in ("baseline", "learned", "norm_matched_random"):
                        records.append(
                            {
                                "record_type": "intervention_depth",
                                "model": "transformer",
                                "seed": seed,
                                "depth": depth,
                                "depth_label": label,
                                "training_condition": condition,
                                "target": target,
                                "control": control,
                                **(
                                    {
                                        "baseline_component_posterior_r2": 0.5 + 0.1 * depth,
                                        "baseline_state_posterior_r2": 0.4 + 0.05 * depth,
                                    }
                                    if control == "baseline"
                                    else {}
                                ),
                                "delta_component_accuracy": -0.2 if control == "learned" and target == "component" else -0.02,
                                "delta_conditional_state_accuracy": -0.15 if control == "learned" and target == "state" else -0.01,
                                "delta_kl_exact": 0.02 if control == "learned" else 0.001,
                            }
                        )
    path = tmp_path / "depth.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    output = tmp_path / "depth.png"
    generate_depth_figure([path], output)
    assert output.exists() and output.stat().st_size > 1000
