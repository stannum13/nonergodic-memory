import json
from pathlib import Path

import pytest
import torch

from nonergodic_memory.mess3_threshold import (
    replace_threshold_records,
    run_threshold_probes,
    run_threshold_training,
    threshold_checkpoint_path,
    validate_threshold_grid,
)


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
