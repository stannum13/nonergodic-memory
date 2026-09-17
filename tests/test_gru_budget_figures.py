import json
from pathlib import Path

import numpy as np
import pytest

from test_context_figures import _context_fixture
from nonergodic_memory.gru_budget_figures import generate_gru_budget_figure, gru_budget_effect


def _gru_short(kind: str) -> list[dict]:
    budget = kind == "budget"
    rows = []
    for seed in (0, 1, 2):
        for control in ("none", "shuffled_labels"):
            rows.append({
                "record_type": "budget_context" if budget else "short_context",
                "model": "gru", "training_condition": "budget_short_trained" if budget else "short_trained",
                "context": "restart_8", "seed": seed, "overlap": 0.35, "control": control,
                "config": "interaction_o035_l064", "config_sha256": "digest_035",
                "short_config": ("budget" if budget else "short") + "_o035_l009",
                "short_config_sha256": ("budget" if budget else "short") + "_digest",
                "sequence_length": 64, "short_training_sequence_length": 9,
                "short_training_input_positions": 8, "window": 8, "positions_evaluated": 56,
                "probe_fit_sequences": 256, "test_sequences": 192,
                "probe_fit_data_seed": seed + 909, "test_data_seed": seed + 1009,
                "probe_fit_independent": True, "observations_evaluated": 192 * 56,
                "component_posterior_r2": 0.8 if budget else 0.7,
                "state_posterior_r2": 0.9 if budget else 0.85,
                "kl_exact": 0.02 if budget else 0.04, "nll": 1.2 if budget else 1.22,
                **({
                    "training_protocol": "token_and_step_matched",
                    "short_training_sequences": 4032, "short_training_batch_size": 504,
                    "supervised_tokens_per_epoch": 32256, "optimizer_steps": 96,
                } if budget else {}),
            })
    return rows


def _long_rows() -> list[dict]:
    rows = _context_fixture()
    for row in rows:
        if row["record_type"] == "context_oracle":
            row["oracle_state_posterior_r2"] = 0.95
        else:
            row["nll"] = 1.2 + row["kl_exact"]
    return rows


def test_gru_budget_effect_is_paired_and_complete() -> None:
    standard, budget, long = _gru_short("standard"), _gru_short("budget"), _long_rows()
    np.testing.assert_allclose(gru_budget_effect(standard, budget, long, "kl_exact"), [0.02] * 3)
    with pytest.raises(ValueError, match="missing GRU short"):
        gru_budget_effect(standard[:-1], budget, long, "kl_exact")


def test_gru_budget_figure_reads_complete_raw_files(tmp_path: Path) -> None:
    paths = [tmp_path / name for name in ("standard.jsonl", "budget.jsonl", "long.jsonl")]
    for path, rows in zip(paths, (_gru_short("standard"), _gru_short("budget"), _long_rows())):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "gru.png"
    generate_gru_budget_figure(*paths, output)
    assert output.exists() and output.stat().st_size > 1000
