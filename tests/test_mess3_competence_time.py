from pathlib import Path

from nonergodic_memory.experiment import config_digest, load_config


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "mess3_competence_time.yaml"


def test_competence_time_config_matches_preregistered_grid_and_lock() -> None:
    config = load_config(CONFIG)

    assert config["data"] == {
        "generator": "mess3",
        "sampler": "vectorized",
        "sequence_length": 64,
        "train_sequences": 2048,
        "test_sequences": 256,
    }
    assert config["model"] == {"width": 32, "layers": 2, "heads": 4, "max_length": 128}
    assert config["train"] == {
        "batch_size": 64,
        "learning_rate": 0.003,
        "weight_decay": 0.01,
        "checkpoint_steps": [0, 384, 768, 1152, 1536, 2048, 2560, 3072, 4096],
    }
    assert config["probe"] == {"train_sequences": 1024, "test_sequences": 512}
    assert config["competence_time"] == {
        "seeds": [30, 31, 32, 33, 34, 35, 36, 37],
        "learning_rates": [0.00075, 0.0015, 0.003, 0.006],
        "primary_site": "block_2",
        "primary_target": "component_posterior",
        "primary_checkpoints": [384, 768, 1152, 1536, 2048, 2560, 3072, 4096],
        "controls": {
            "shuffled_labels": True,
            "untrained_step": 0,
            "seed_grouped_loso": True,
            "require_zero_probe_overlap": True,
        },
        "thresholds": {
            "max_shuffled_component_posterior_r2": 0.02,
            "competence_to_step_mse_ratio": 0.80,
            "minimum_competence_fold_wins": 7,
            "minimum_rate_dissociation_seeds": 6,
            "minimum_rate_competence_difference": 0.10,
        },
    }
    assert not set(config["competence_time"]["seeds"]).intersection(range(20, 25))
    assert config_digest(config) == "a9dae90d3049c220"
