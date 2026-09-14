from pathlib import Path

from nonergodic_memory.experiment import train_one


def tiny_config() -> dict:
    return {
        "data": {"overlap": 0.25, "sequence_length": 8, "train_sequences": 32, "test_sequences": 16},
        "model": {"width": 12, "layers": 1, "heads": 2},
        "train": {"epochs": 4, "batch_size": 16, "learning_rate": 0.02},
        "probe": {"train_sequences": 16, "test_sequences": 16},
    }


def test_train_one_is_reproducible(tmp_path: Path) -> None:
    first, _ = train_one(tiny_config(), "gru", seed=7, output_dir=tmp_path / "a")
    second, _ = train_one(tiny_config(), "gru", seed=7, output_dir=tmp_path / "b")
    assert first["train_nll"] == second["train_nll"]
    assert first["test_nll"] == second["test_nll"]


def test_training_reduces_loss(tmp_path: Path) -> None:
    result, _ = train_one(tiny_config(), "gru", seed=3, output_dir=tmp_path)
    assert result["train_nll"] < result["initial_train_nll"]
    assert result["device"] == "cpu"
    assert (tmp_path / "gru_seed3.pt").exists()
