import json
from pathlib import Path

import numpy as np
import pytest

from test_position_figures import _rows as _position_rows
from nonergodic_memory.short_figures import generate_short_figure, short_vs_long


def _short_rows() -> list[dict]:
    rows = []
    for overlap in (0.0, 0.35):
        for seed in (0, 1, 2):
            for control in ("none", "shuffled_labels"):
                rows.append({
                    "record_type": "short_context", "model": "transformer",
                    "training_condition": "short_trained", "context": "restart_8",
                    "seed": seed, "overlap": overlap, "control": control,
                    "config": f"o{overlap}_l064", "config_sha256": f"digest_{overlap}",
                    "short_config": f"short_o{overlap}_l009",
                    "short_config_sha256": f"short_digest_{overlap}",
                    "sequence_length": 64, "short_training_sequence_length": 9,
                    "short_training_input_positions": 8,
                    "window": 8, "positions_evaluated": 56,
                    "probe_fit_sequences": 256, "test_sequences": 192,
                    "probe_fit_data_seed": seed + 909, "test_data_seed": seed + 1009,
                    "probe_fit_independent": True, "observations_evaluated": 192 * 56,
                    "component_posterior_r2": 0.7,
                    "state_posterior_r2": 0.65,
                    "kl_exact": 0.02, "nll": 1.22,
                })
    return rows


def test_short_vs_long_is_paired_and_requires_complete_cells() -> None:
    short, long = _short_rows(), _position_rows()
    values = short_vs_long(short, long, "kl_exact")
    np.testing.assert_allclose(values[0.35], [-0.02] * 3)
    with pytest.raises(ValueError, match="missing short"):
        short_vs_long(short[:-1], long, "kl_exact")


def test_short_figure_regenerates_only_from_two_raw_files(tmp_path: Path) -> None:
    short_path = tmp_path / "short.jsonl"
    long_path = tmp_path / "long.jsonl"
    short_path.write_text("".join(json.dumps(row) + "\n" for row in _short_rows()))
    long_path.write_text("".join(json.dumps(row) + "\n" for row in _position_rows()))
    output = tmp_path / "joined.png"
    generate_short_figure(short_path, long_path, output)
    assert output.exists() and output.stat().st_size > 1000
