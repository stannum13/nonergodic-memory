from pathlib import Path

from nonergodic_memory.experiment import config_digest, load_config


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "mess3_rate_aware_clock.yaml"


def test_rate_aware_clock_config_matches_preregistered_forecasts_and_grid() -> None:
    config = load_config(CONFIG)

    assert config == {
        "data": {
            "generator": "mess3",
            "sampler": "vectorized",
            "sequence_length": 64,
            "train_sequences": 2048,
            "test_sequences": 256,
        },
        "model": {"width": 32, "layers": 2, "heads": 4, "max_length": 128},
        "train": {
            "batch_size": 64,
            "learning_rate": 0.003,
            "weight_decay": 0.01,
            "checkpoint_steps": [0, 384, 768, 1152, 1536, 2048, 2560, 3072],
        },
        "diagnosis": {"window": 8},
        "probe": {"train_sequences": 1024, "test_sequences": 512},
        "rate_aware_clock": {
            "seeds": [40, 41, 42, 43, 44, 45, 46, 47],
            "learning_rates": [0.003, 0.006],
            "primary_site": "block_2",
            "primary_target": "component_posterior",
            "primary_control": "none",
            "primary_checkpoints": [384, 768, 1152, 1536, 2048, 2560, 3072],
            "controls": {
                "shuffled_labels": True,
                "untrained_step": 0,
                "seed_grouped_scoring": True,
                "require_zero_token_overlap": True,
            },
            "thresholds": {
                "clock_to_competence_mse_ratio": 0.80,
                "minimum_clock_seed_wins": 7,
                "max_shuffled_component_posterior_r2": 0.02,
                "minimum_rate_dissociation_seeds": 6,
                "minimum_rate_competence_difference": 0.10,
            },
        },
        "forecast": {
            "provenance": {
                "training_jsonl_sha256": (
                    "2c90e72389db3a98e0c1196fffaf6bdf24f3492009460bfbe0c99f417315420f"
                ),
                "probe_jsonl_sha256": (
                    "9a7aa242f267bddc2064c8ddf93c3163891167630d7dd3f0ea4e603993e3abd3"
                ),
                "seeds": [30, 31, 32, 33, 34, 35, 36, 37],
                "learning_rates": [0.00075, 0.0015, 0.003, 0.006],
                "post_initialization_steps": [
                    384,
                    768,
                    1152,
                    1536,
                    2048,
                    2560,
                    3072,
                    4096,
                ],
                "site": "block_2",
                "control": "none",
                "metric": "component_posterior_r2",
            },
            "competence": {
                "center": 0.4428598689138331,
                "scale": 0.3380605976966924,
                "coefficients": [
                    0.03302589694739769,
                    0.14699110421172318,
                    0.09640601399613305,
                ],
                "support": [-0.24055542481822356, 0.8940463808722273],
                "in_sample_mse": 0.00761248,
            },
            "rate_aware_clock": {
                "transform": "log1p(learning_rate * step)",
                "center": 1.5596899070965387,
                "scale": 0.7723241912698406,
                "coefficients": [
                    0.08426303744815684,
                    0.13373197023956584,
                    0.045168873495373824,
                ],
                "support": [0.2530906276821619, 3.2416544117575405],
                "in_sample_mse": 0.00543398,
            },
        },
    }
    assert config_digest(config) == "59f938bbff48f300"
