import json
from pathlib import Path

import pytest
import torch
import matplotlib.image as mpimg

from nonergodic_memory.mess3_threshold import (
    analyze_threshold,
    replace_threshold_records,
    run_threshold_probes,
    run_threshold_training,
    threshold_checkpoint_path,
    validate_threshold_grid,
)
from nonergodic_memory.mess3_threshold_figures import generate_threshold_figures


def _tiny_threshold_config() -> dict:
    return {
        "data": {
            "generator": "mess3",
            "sampler": "vectorized",
            "sequence_length": 8,
            "train_sequences": 16,
            "test_sequences": 8,
        },
        "model": {"width": 8, "layers": 2, "heads": 2, "max_length": 16},
        "train": {
            "batch_size": 4,
            "learning_rate": 0.01,
            "weight_decay": 0.01,
            "checkpoint_steps": [0, 2],
        },
        "diagnosis": {"window": 3},
        "probe": {"train_sequences": 24, "test_sequences": 16},
        "threshold": {"seeds": [6, 7], "learning_rates": [0.01, 0.005]},
    }


@pytest.fixture(scope="module")
def tiny_threshold_grid(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("threshold")
    config = _tiny_threshold_config()
    training = run_threshold_training(
        config,
        seeds=config["threshold"]["seeds"],
        learning_rates=config["threshold"]["learning_rates"],
        checkpoint_root=root / "checkpoints",
    )
    probes = run_threshold_probes(
        config,
        seeds=config["threshold"]["seeds"],
        learning_rates=config["threshold"]["learning_rates"],
        checkpoint_root=root / "checkpoints",
    )
    return root, config, training, probes


def test_threshold_training_pairs_initialization_and_covers_grid(tiny_threshold_grid) -> None:
    root, config, training, _ = tiny_threshold_grid
    expected = {
        (seed, rate, step)
        for seed in (6, 7)
        for rate in (0.01, 0.005)
        for step in (0, 2)
    }
    assert {(row["seed"], row["learning_rate"], row["step"]) for row in training} == expected
    assert {row["record_type"] for row in training} == {"threshold_training"}
    assert {row["sampler"] for row in training} == {"vectorized"}
    assert len({row["base_config_sha256"] for row in training}) == 1
    assert len({row["config_sha256"] for row in training}) == 2
    for seed in (6, 7):
        fast = torch.load(
            threshold_checkpoint_path(root / "checkpoints", seed, 0.01, 0),
            weights_only=False,
        )
        slow = torch.load(
            threshold_checkpoint_path(root / "checkpoints", seed, 0.005, 0),
            weights_only=False,
        )
        for name, value in fast["state_dict"].items():
            torch.testing.assert_close(value, slow["state_dict"][name], rtol=0, atol=0)
        assert fast["config"]["train"]["learning_rate"] == 0.01
        assert slow["config"]["train"]["learning_rate"] == 0.005


def test_threshold_probe_grid_includes_layers_and_shuffled_controls(tiny_threshold_grid) -> None:
    _, config, training, probes = tiny_threshold_grid
    validate_threshold_grid(config, training, probes)
    assert len(probes) == 2 * 2 * 2 * 3 * 2
    assert {(row["site"], row["control"]) for row in probes} == {
        (site, control)
        for site in ("block_1", "block_2", "final_norm")
        for control in ("none", "shuffled_labels")
    }
    assert {row["probe_sequence_overlap"] for row in probes} == {0}


def test_threshold_grid_rejects_duplicate_and_missing_cells(tiny_threshold_grid) -> None:
    _, config, training, probes = tiny_threshold_grid
    with pytest.raises(ValueError, match="duplicate"):
        validate_threshold_grid(config, [*training, training[0]], probes)
    with pytest.raises(ValueError, match="incomplete"):
        validate_threshold_grid(config, training[1:], probes[6:])


def test_threshold_keyed_replacement_preserves_unselected_cells(tmp_path: Path) -> None:
    path = tmp_path / "training.jsonl"
    existing = [
        {
            "base_config_sha256": "base",
            "seed": seed,
            "learning_rate": rate,
            "step": 0,
            "value": "old",
        }
        for seed in (6, 7)
        for rate in (0.01, 0.005)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in existing))
    replacement = [{**existing[0], "value": "new"}]

    replace_threshold_records(
        path,
        replacement,
        ("base_config_sha256", "seed", "learning_rate", "step"),
    )

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 4
    assert sum(row["value"] == "new" for row in rows) == 1


def _synthetic_threshold_grid() -> tuple[dict, list[dict], list[dict]]:
    config = _tiny_threshold_config()
    config["threshold"] = {
        "seeds": [20, 21, 22, 23, 24],
        "learning_rates": [0.003, 0.0015],
    }
    config["train"]["checkpoint_steps"] = [0, 1, 2, 3]
    from nonergodic_memory.experiment import config_digest

    base_digest = config_digest(config)
    schedules = {0.003: [0.0, 0.35, 0.75, 0.95], 0.0015: [0.0, 0.10, 0.35, 0.75]}
    training = []
    probes = []
    for seed in config["threshold"]["seeds"]:
        for rate in config["threshold"]["learning_rates"]:
            for step, competence in zip(config["train"]["checkpoint_steps"], schedules[rate]):
                geometry = competence**2 + (seed - 22) * 0.001
                training.append(
                    {
                        "record_type": "threshold_training",
                        "base_config_sha256": base_digest,
                        "config_sha256": f"rate-{rate}",
                        "seed": seed,
                        "learning_rate": rate,
                        "step": step,
                        "competence": competence,
                        "kl_exact": 1.0 - competence,
                    }
                )
                for site in ("block_1", "block_2", "final_norm"):
                    for control in ("none", "shuffled_labels"):
                        probes.append(
                            {
                                "record_type": "threshold_probe",
                                "base_config_sha256": base_digest,
                                "config_sha256": f"rate-{rate}",
                                "seed": seed,
                                "learning_rate": rate,
                                "step": step,
                                "site": site,
                                "control": control,
                                "component_posterior_r2": geometry if control == "none" else 0.0,
                                "joint_belief_r2": 0.5 * geometry if control == "none" else 0.0,
                            }
                        )
    return config, training, probes


def test_threshold_analysis_uses_leave_one_seed_out_and_registered_criterion() -> None:
    config, training, probes = _synthetic_threshold_grid()

    summary = analyze_threshold(config, training, probes)

    assert summary["record_type"] == "threshold_summary"
    assert summary["competence_to_step_mse_ratio"] < 0.8
    assert summary["registered_supported"]
    assert summary["shuffled_control_valid"]
    assert {fold["held_out_seed"] for fold in summary["folds"]} == {20, 21, 22, 23, 24}
    for fold in summary["folds"]:
        assert fold["held_out_seed"] not in fold["training_seeds"]
        assert fold["test_cells"] == 2 * 4
    sensitivity = summary["posthoc_post_initialization_sensitivity"]
    assert sensitivity["selection"] == "step > 0"
    assert sensitivity["competence_to_step_mse_ratio"] < 0.8
    assert all(fold["test_cells"] == 2 * 3 for fold in sensitivity["folds"])


def test_threshold_analysis_rejects_out_of_bounds_shuffled_control() -> None:
    config, training, probes = _synthetic_threshold_grid()
    probes[1]["component_posterior_r2"] = 0.021

    summary = analyze_threshold(config, training, probes)

    assert not summary["shuffled_control_valid"]
    assert not summary["registered_supported"]


def test_threshold_analysis_marks_posthoc_unavailable_with_one_trained_step() -> None:
    config, training, probes = _synthetic_threshold_grid()
    config["train"]["checkpoint_steps"] = [0, 1]
    training = [row for row in training if row["step"] in (0, 1)]
    probes = [row for row in probes if row["step"] in (0, 1)]
    from nonergodic_memory.experiment import config_digest

    base_digest = config_digest(config)
    for row in [*training, *probes]:
        row["base_config_sha256"] = base_digest

    summary = analyze_threshold(config, training, probes)

    sensitivity = summary["posthoc_post_initialization_sensitivity"]
    assert sensitivity["status"] == "unavailable"
    assert "two distinct" in sensitivity["reason"]


def test_threshold_figures_are_generated_from_complete_raw_grid(tmp_path: Path) -> None:
    config, training, probes = _synthetic_threshold_grid()
    summary = analyze_threshold(config, training, probes)

    paths = generate_threshold_figures(config, training, probes, summary, tmp_path)

    assert {path.name for path in paths} == {
        "mess3_threshold_learning.png",
        "mess3_threshold_alignment.png",
    }
    assert all(path.stat().st_size > 0 for path in paths)
    alignment = mpimg.imread(tmp_path / "mess3_threshold_alignment.png")
    assert alignment.shape[1] > 2 * alignment.shape[0]
