import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from nonergodic_memory.position_figures import (
    generate_position_figure,
    position_effect,
    position_history_damage,
)


def _rows() -> list[dict]:
    rows = []
    for overlap in (0.0, 0.35):
        for seed in (0, 1, 2):
            common = {
                "seed": seed, "overlap": overlap, "config": f"o{overlap}_l064",
                "config_sha256": f"digest_{overlap}", "window": 8,
                "positions_evaluated": 56, "sequence_length": 64,
                "probe_fit_sequences": 256, "test_sequences": 192,
                "probe_fit_data_seed": seed + 909, "test_data_seed": seed + 1009,
            }
            rows.append({
                **common, "record_type": "context_oracle", "model": "exact_bayes",
                "oracle_component_posterior_r2": 0.8,
                "oracle_state_posterior_r2": 0.9,
                "oracle_kl_full_to_window": 0.02,
            })
            for condition in ("trained", "untrained"):
                for context in ("full", "restart_8", "restart_8_absolute"):
                    for control in ("none", "shuffled_labels"):
                        r2 = {"full": 0.9, "restart_8": 0.6, "restart_8_absolute": 0.65}[context]
                        state = {"full": 0.8, "restart_8": 0.5, "restart_8_absolute": 0.6}[context]
                        kl = {"full": 0.01, "restart_8": 0.04, "restart_8_absolute": 0.03}[context]
                        if overlap == 0:
                            r2 += 0.1
                        rows.append({
                            **common, "record_type": "context_restart", "model": "transformer",
                            "training_condition": condition, "context": context,
                            "control": control, "probe_fit_independent": True,
                            "observations_evaluated": 192 * 56,
                            "component_posterior_r2": r2,
                            "state_posterior_r2": state, "kl_exact": kl, "nll": 1.2 + kl,
                        })
    return rows


def test_position_effect_and_history_damage_require_complete_grid() -> None:
    rows = _rows()
    np.testing.assert_allclose(position_effect(rows, "component_posterior_r2")[("trained", 0.35)], [0.05] * 3)
    np.testing.assert_allclose(position_effect(rows, "state_posterior_r2")[("untrained", 0.0)], [0.1] * 3)
    np.testing.assert_allclose(position_effect(rows, "kl_exact")[("trained", 0.35)], [0.01] * 3)
    np.testing.assert_allclose(position_history_damage(rows, "component_posterior_r2")[("trained", "restart_8_absolute")], [0] * 3)
    with pytest.raises(ValueError, match="missing context"):
        position_effect(rows[:-1], "kl_exact")


def test_position_figure_regenerates_from_raw(tmp_path: Path) -> None:
    raw = tmp_path / "position.jsonl"
    raw.write_text("".join(json.dumps(row) + "\n" for row in _rows()))
    output = tmp_path / "position.png"
    generate_position_figure(raw, output)
    assert output.exists() and output.stat().st_size > 1000


def test_optional_position_figure_skips_missing_raw_and_removes_stale(tmp_path: Path) -> None:
    raw = tmp_path / "missing.jsonl"
    stale = tmp_path / "stale.png"
    stale.write_bytes(b"old")
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / "src"))
    result = subprocess.run(
        [sys.executable, "-m", "nonergodic_memory.position_figures", "--results", str(raw), "--output", str(stale)],
        env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0 and "skipped" in result.stdout
    assert not stale.exists()
