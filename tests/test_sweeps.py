import json
from pathlib import Path

from nonergodic_memory.sweeps import generate_overlap_figure


def test_overlap_figure_is_generated_from_raw_records(tmp_path: Path) -> None:
    records = []
    for overlap in (0.0, 0.7):
        for model in ("gru", "transformer"):
            for seed in (0, 1):
                for condition, component_r2, state_r2 in (
                    ("trained", 0.9 - overlap / 2, 0.8),
                    ("untrained", 0.5 - overlap / 3, 0.7),
                ):
                    records.append(
                        {
                            "record_type": "probe",
                            "model": model,
                            "seed": seed,
                            "overlap": overlap,
                            "training_condition": condition,
                            "control": "none",
                            "component_posterior_r2": component_r2,
                            "state_posterior_r2": state_r2,
                        }
                    )
                for target in ("component", "state"):
                    records.append(
                        {
                            "record_type": "intervention",
                            "model": model,
                            "seed": seed,
                            "overlap": overlap,
                            "training_condition": "trained",
                            "control": "learned",
                            "target": target,
                            "delta_component_accuracy": -0.2 if target == "component" else -0.01,
                            "delta_conditional_state_accuracy": -0.02 if target == "component" else -0.15,
                        }
                    )
    path = tmp_path / "overlap.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    output = tmp_path / "overlap.png"
    generate_overlap_figure([path], output)
    assert output.exists()
    assert output.stat().st_size > 1000
