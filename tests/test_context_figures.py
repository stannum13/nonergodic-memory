import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from nonergodic_memory.context_figures import context_damage, generate_context_figure


def _context_fixture() -> list[dict]:
    rows = []
    for overlap in (0.0, 0.35):
        config = f"interaction_o{int(overlap * 100):03d}_l064"
        digest = f"digest_{int(overlap * 100):03d}"
        for seed in (0, 1, 2):
            rows.append(
                {
                    "record_type": "context_oracle", "model": "exact_bayes", "seed": seed,
                    "overlap": overlap, "config": config, "config_sha256": digest,
                    "window": 8, "positions_evaluated": 56,
                    "sequence_length": 64, "probe_fit_sequences": 256, "test_sequences": 192,
                    "probe_fit_data_seed": seed + 909, "test_data_seed": seed + 1009,
                    "oracle_component_posterior_r2": 0.95 if overlap == 0 else 0.8,
                    "oracle_kl_full_to_window": 0.005 if overlap == 0 else 0.02,
                }
            )
            for model in ("gru", "transformer"):
                for condition in ("trained", "untrained"):
                    for context in ("full", "restart_8"):
                        for control in ("none", "shuffled_labels"):
                            r2_full = 0.8 if model == "gru" else 0.9
                            r2_loss = (
                                (0.10 if overlap == 0 else 0.30)
                                if condition == "trained"
                                else (0.05 if overlap == 0 else 0.10)
                            )
                            kl_damage = (
                                (0.01 if overlap == 0 else 0.03)
                                if condition == "trained"
                                else (0.002 if overlap == 0 else 0.004)
                            )
                            rows.append(
                                {
                                    "record_type": "context_restart", "model": model, "seed": seed,
                                    "overlap": overlap, "config": config, "config_sha256": digest,
                                    "window": 8, "positions_evaluated": 56,
                                    "sequence_length": 64, "probe_fit_sequences": 256, "test_sequences": 192,
                                    "probe_fit_data_seed": seed + 909, "test_data_seed": seed + 1009,
                                    "observations_evaluated": 192 * 56,
                                    "training_condition": condition, "context": context,
                                    "control": control, "probe_fit_independent": True,
                                    "component_posterior_r2": r2_full - (r2_loss if context == "restart_8" else 0),
                                    "state_posterior_r2": 0.7,
                                    "kl_exact": 0.01 + (kl_damage if context == "restart_8" else 0),
                                }
                            )
    return rows


def test_context_damage_is_paired_and_requires_complete_grid() -> None:
    rows = _context_fixture()
    r2 = context_damage(rows, "component_posterior_r2")
    kl = context_damage(rows, "kl_exact")
    np.testing.assert_allclose(r2[("gru", "trained")], [0.20] * 3)
    np.testing.assert_allclose(kl[("gru", "trained")], [0.02] * 3)
    np.testing.assert_allclose(r2[("gru", "untrained")], [0.05] * 3)
    with pytest.raises(ValueError, match="required seeds"):
        context_damage([r for r in rows if r["seed"] != 2], "component_posterior_r2")
    with pytest.raises(ValueError, match="missing context"):
        context_damage(rows[:-1], "component_posterior_r2")


def test_context_figure_is_generated_only_from_raw_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "restart.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in _context_fixture()))
    output = tmp_path / "restart.png"
    generate_context_figure(path, output)
    assert output.exists() and output.stat().st_size > 1000


def test_missing_optional_raw_skips_and_removes_stale_figure(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"
    stale = tmp_path / "stale.png"
    stale.write_bytes(b"old figure")
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / "src"))
    result = subprocess.run(
        [sys.executable, "-m", "nonergodic_memory.context_figures", "--results", str(missing), "--output", str(stale)],
        capture_output=True, text=True, env=environment, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "skipped" in result.stdout
    assert not stale.exists()


def test_context_figure_rejects_misreported_length_and_sample_seeds() -> None:
    rows = _context_fixture()
    rows[0]["sequence_length"] = 32
    with pytest.raises(ValueError, match="length"):
        context_damage(rows, "kl_exact")
    rows = _context_fixture()
    rows[-1]["test_data_seed"] = 1009
    with pytest.raises(ValueError, match="sample"):
        context_damage(rows, "kl_exact")
