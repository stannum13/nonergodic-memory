from pathlib import Path
import json
import pytest

from nonergodic_memory.experiment import (
    mixture_from_config,
    replace_jsonl_runs,
    train_one,
    validate_checkpoint,
)
from nonergodic_memory.checkpoints import checkpoint_set_matches


def tiny_config() -> dict:
    return {
        "data": {"overlap": 0.25, "sequence_length": 8, "train_sequences": 32, "test_sequences": 16},
        "model": {"width": 12, "layers": 1, "heads": 2},
        "train": {"epochs": 4, "batch_size": 16, "learning_rate": 0.02},
        "probe": {"train_sequences": 16, "test_sequences": 16},
    }


def test_mixture_from_config_selects_published_mess3() -> None:
    config = tiny_config()
    config["data"]["generator"] = "mess3"
    config["data"].pop("overlap")
    mixture = mixture_from_config(config)
    assert mixture.vocab_size == 3
    assert [component.n_states for component in mixture.components] == [3, 3]


def test_train_one_is_reproducible(tmp_path: Path) -> None:
    first, _ = train_one(tiny_config(), "gru", seed=7, output_dir=tmp_path / "a")
    second, _ = train_one(tiny_config(), "gru", seed=7, output_dir=tmp_path / "b")
    assert first["train_nll"] == second["train_nll"]
    assert first["test_nll"] == second["test_nll"]


def test_training_reduces_loss(tmp_path: Path) -> None:
    result, _ = train_one(tiny_config(), "gru", seed=3, output_dir=tmp_path)
    assert result["train_nll"] < result["initial_train_nll"]
    assert result["device"] == "cpu"
    assert result["generator"] == "simple"
    assert result["overlap"] == 0.25
    assert (tmp_path / "gru_seed3.pt").exists()
    assert checkpoint_set_matches(tiny_config(), tmp_path, ["gru"], [3])
    changed = tiny_config()
    changed["data"]["overlap"] = 0.9
    assert not checkpoint_set_matches(changed, tmp_path, ["gru"], [3])


def test_partial_result_update_preserves_other_runs(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    existing = [
        {"config": "central", "config_sha256": "same", "model": "gru", "seed": 0, "value": "old"},
        {"config": "central", "config_sha256": "same", "model": "gru", "seed": 1, "value": "keep"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in existing))
    replace_jsonl_runs(
        path,
        [{"config": "central", "config_sha256": "same", "model": "gru", "seed": 0, "value": "new"}],
        "central",
        ["gru"],
        [0],
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert {(row["seed"], row["value"]) for row in rows} == {(0, "new"), (1, "keep")}


def test_changed_config_digest_purges_all_old_same_name_cells(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    old = [
        {"config": "central", "config_sha256": "old", "model": "gru", "seed": seed}
        for seed in (0, 1)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in old))
    replacement = {"config": "central", "config_sha256": "new", "model": "gru", "seed": 0}
    replace_jsonl_runs(path, [replacement], "central", ["gru"], [0])
    assert [json.loads(line) for line in path.read_text().splitlines()] == [replacement]


def test_checkpoint_validation_rejects_stale_config() -> None:
    payload = {"config": tiny_config(), "model_name": "gru", "seed": 2}
    stale = tiny_config()
    stale["model"]["width"] = 99
    with pytest.raises(ValueError, match="configuration"):
        validate_checkpoint(payload, stale, "gru", 2)
