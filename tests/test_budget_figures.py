import json
from pathlib import Path

import numpy as np
import pytest

from test_position_figures import _rows as _position_rows
from test_short_figures import _short_rows
from nonergodic_memory.budget_figures import budget_minus_standard, generate_budget_figure


def _budget_rows() -> list[dict]:
    rows = _short_rows()
    for row in rows:
        row["record_type"] = "budget_context"
        row["training_condition"] = "budget_short_trained"
        row["short_config"] = "budget_" + row["short_config"]
        row["short_config_sha256"] = "budget_" + row["short_config_sha256"]
        row["training_protocol"] = "token_and_step_matched"
        row["short_training_sequences"] = 4032
        row["short_training_batch_size"] = 504
        row["supervised_tokens_per_epoch"] = 32256
        row["optimizer_steps"] = 96
        row["kl_exact"] = 0.015
    return rows


def test_budget_contrast_is_seed_paired_and_complete() -> None:
    budget, short, long = _budget_rows(), _short_rows(), _position_rows()
    np.testing.assert_allclose(budget_minus_standard(budget, short, long, "kl_exact")[0.35], [-0.005] * 3)
    with pytest.raises(ValueError, match="missing short"):
        budget_minus_standard(budget[:-1], short, long, "kl_exact")


def test_budget_figure_is_joined_only_from_complete_raw(tmp_path: Path) -> None:
    paths = [tmp_path / name for name in ("budget.jsonl", "short.jsonl", "long.jsonl")]
    for path, rows in zip(paths, (_budget_rows(), _short_rows(), _position_rows())):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "budget.png"
    generate_budget_figure(*paths, output)
    assert output.exists() and output.stat().st_size > 1000
