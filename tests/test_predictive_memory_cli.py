from pathlib import Path

import predictive_memory as cli
from nonergodic_memory.data.hmm import make_mess3_mixture
from nonergodic_memory.experiment import load_config


def _tiny_registered_config(tmp_path: Path) -> Path:
    config = load_config("configs/predictive_memory.yaml")
    config["data"] = {
        "actuator_fit": 3,
        "decoder_fit": 3,
        "calibration": 2,
        "evaluation": 2,
    }
    path = tmp_path / "config.yaml"
    import yaml

    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _mock_loader(tmp_path: Path, events: list[tuple]):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    mixture = make_mess3_mixture()
    parent = load_config("configs/mess3_diagnosis.yaml")

    def load(seed, step, cohort):
        events.append(("load", cohort, seed, step))
        return checkpoint, {"config": parent}, object()

    return load, mixture


def test_failed_development_gate_never_opens_heldout(tmp_path: Path, monkeypatch):
    events = []
    loader, mixture = _mock_loader(tmp_path, events)
    monkeypatch.setattr(cli, "validate_registered_config", lambda config: None)
    monkeypatch.setattr(cli, "analyze_evidence", lambda *args, **kwargs: {"valid": True, "validity_errors": []})
    monkeypatch.setattr(cli, "_load_registered_checkpoint", loader)

    def calibrate(experiment, seed, step, cohort, splits):
        events.append(("calibrate", cohort, seed, step))
        row = {"record_type": "calibration", "seed": seed, "step": step, "cohort": cohort, "passed": False}
        return object(), row, (mixture, object())

    monkeypatch.setattr(cli, "_calibrate", calibrate)
    summary = cli.run_registered(
        _tiny_registered_config(tmp_path),
        tmp_path / "results.jsonl",
        tmp_path / "summary.jsonl",
    )
    assert summary["status"] == "actuator_infeasible"
    assert not any(event[1] == "heldout" for event in events)


def test_all_heldout_calibrations_finish_before_any_heldout_response(
    tmp_path: Path, monkeypatch
):
    events = []
    loader, mixture = _mock_loader(tmp_path, events)
    monkeypatch.setattr(cli, "validate_registered_config", lambda config: None)
    monkeypatch.setattr(cli, "analyze_evidence", lambda *args, **kwargs: {"valid": True, "validity_errors": []})
    monkeypatch.setattr(cli, "_load_registered_checkpoint", loader)

    def calibrate(experiment, seed, step, cohort, splits):
        passed = not (cohort == "heldout" and seed == 24 and step == 3072)
        events.append(("calibrate", cohort, seed, step))
        row = {"record_type": "calibration", "seed": seed, "step": step, "cohort": cohort, "passed": passed}
        return object(), row, (mixture, object())

    def evaluate(experiment, seed, step, cohort, splits, calibration, loaded_mixture, model):
        events.append(("evaluate", cohort, seed, step))
        return []

    monkeypatch.setattr(cli, "_calibrate", calibrate)
    monkeypatch.setattr(cli, "_evaluate", evaluate)
    summary = cli.run_registered(
        _tiny_registered_config(tmp_path),
        tmp_path / "results.jsonl",
        tmp_path / "summary.jsonl",
    )
    assert summary["status"] == "invalid_pilot"
    heldout_calibrations = [event for event in events if event[:2] == ("calibrate", "heldout")]
    heldout_evaluations = [event for event in events if event[:2] == ("evaluate", "heldout")]
    assert {event[2] for event in heldout_calibrations} == {20, 21, 22, 23, 24}
    assert heldout_evaluations == []


def test_success_path_calibrates_all_heldout_models_before_first_response(
    tmp_path: Path, monkeypatch
):
    events = []
    loader, mixture = _mock_loader(tmp_path, events)
    monkeypatch.setattr(cli, "validate_registered_config", lambda config: None)
    monkeypatch.setattr(cli, "_load_registered_checkpoint", loader)
    monkeypatch.setattr(cli, "analyze_evidence", lambda *args, **kwargs: {"valid": True, "validity_errors": []})
    monkeypatch.setattr(cli, "validate_complete_evidence", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        cli,
        "classify_pilot",
        lambda *args, **kwargs: {
            "status": "criterion_not_met",
            "seed_scores": {},
            "mean_learned_score": 0.0,
            "mean_shuffled_score": 0.0,
            "mean_random_score": 0.0,
            "shuffled_margin": 0.0,
            "random_margin": 0.0,
            "all_heldout_positive": False,
        },
    )

    def calibrate(experiment, seed, step, cohort, splits):
        events.append(("calibrate", cohort, seed, step))
        row = {"record_type": "calibration", "seed": seed, "step": step, "cohort": cohort, "passed": True}
        return object(), row, (mixture, object())

    def evaluate(experiment, seed, step, cohort, splits, calibration, loaded_mixture, model):
        events.append(("evaluate", cohort, seed, step))
        return []

    monkeypatch.setattr(cli, "_calibrate", calibrate)
    monkeypatch.setattr(cli, "_evaluate", evaluate)
    summary = cli.run_registered(
        _tiny_registered_config(tmp_path),
        tmp_path / "results.jsonl",
        tmp_path / "summary.jsonl",
    )
    assert summary["status"] == "criterion_not_met"
    first_heldout_evaluation = next(
        index
        for index, event in enumerate(events)
        if event[:2] == ("evaluate", "heldout")
    )
    trained_heldout_calibrations = [
        index
        for index, event in enumerate(events)
        if event[:2] == ("calibrate", "heldout") and event[3] == 3072
    ]
    assert len(trained_heldout_calibrations) == 5
    assert max(trained_heldout_calibrations) < first_heldout_evaluation
