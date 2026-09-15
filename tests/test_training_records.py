import json
from pathlib import Path

from nonergodic_memory.experiment import config_digest, load_config
from nonergodic_memory.training_records import training_set_matches


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "sweeps" / "short_o035_l009.yaml"


def _rows() -> list[dict]:
    config = load_config(CONFIG)
    return [
        {
            "record_type": "training", "model": "transformer", "seed": seed,
            "config": CONFIG.stem, "config_sha256": config_digest(config),
            "sequence_length": 9, "device": "cpu", "test_nll": 1.2,
            "train_nll": 1.1, "test_kl_exact": 0.1, "test_bayes_nll": 1.0,
        }
        for seed in (0, 1, 2)
    ]


def test_training_grid_validator_rejects_missing_incomplete_and_stale(tmp_path: Path) -> None:
    raw = tmp_path / "training.jsonl"
    assert not training_set_matches(CONFIG, raw, "transformer", [0, 1, 2])
    rows = _rows()
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows[:-1]))
    assert not training_set_matches(CONFIG, raw, "transformer", [0, 1, 2])
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert training_set_matches(CONFIG, raw, "transformer", [0, 1, 2])
    rows[-1]["config_sha256"] = "stale"
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert not training_set_matches(CONFIG, raw, "transformer", [0, 1, 2])
    rows = _rows()
    rows[-1]["test_kl_exact"] = float("nan")
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert not training_set_matches(CONFIG, raw, "transformer", [0, 1, 2])
