"""Post-result safeguards; all generated data stay in pytest temporary paths."""

import copy
import math
from argparse import Namespace
from pathlib import Path

import pytest
import torch

import mess3_competence_time as cli
from nonergodic_memory.experiment import write_jsonl
from nonergodic_memory.mess3_competence_time import (
    analyze_competence_time,
    competence_time_checkpoint_path,
    validate_competence_time_grid,
)
from nonergodic_memory.mess3_competence_time_figures import generate_competence_time_figures
from test_mess3_competence_time import _synthetic_analysis_grid, _tiny_competence_time_config


def _args(tmp_path, mode):
    return Namespace(
        config="unused", mode=mode, seeds=None, learning_rates=None,
        checkpoint_dir=tmp_path / "checkpoints",
        training_results=tmp_path / "training.jsonl",
        probe_results=tmp_path / "probes.jsonl",
        summary_results=tmp_path / "summary.jsonl",
        output_dir=tmp_path / "figures",
    )


def _prepare_cli(monkeypatch, config, args):
    monkeypatch.setattr(cli, "parse_args", lambda: args)
    monkeypatch.setattr(cli, "load_config", lambda _: config)


def _canonical_paths(training, probes, root):
    for row in [*training, *probes]:
        row["checkpoint_path"] = str(competence_time_checkpoint_path(
            root, row["seed"], row["learning_rate"], row["step"]
        ))
        row["learning_rate_label"] = cli._rate_label(row["learning_rate"])


@pytest.mark.parametrize("mode", ["train", "probe", "all"])
@pytest.mark.parametrize("output", ["training", "probe", "summary"])
@pytest.mark.parametrize("corruption", ["digest", "identity"])
def test_preflight_rejects_existing_output_before_side_effects(
    tmp_path, monkeypatch, mode, output, corruption,
):
    config, training, probes = _synthetic_analysis_grid()
    args = _args(tmp_path, mode)
    _canonical_paths(training, probes, args.checkpoint_dir)
    _prepare_cli(monkeypatch, config, args)
    row = copy.deepcopy({
        "training": training[0], "probe": probes[0],
        "summary": analyze_competence_time(config, training, probes),
    }[output])
    row["base_config_sha256" if corruption == "digest" else "record_type"] = "incompatible"
    destination = getattr(args, f"{output}_results")
    write_jsonl(destination, [row])
    original = destination.read_bytes()
    sentinel = competence_time_checkpoint_path(args.checkpoint_dir, 30, 0.00075, 0)
    sentinel.parent.mkdir(parents=True)
    sentinel.write_bytes(b"frozen checkpoint")

    def would_overwrite(*_):
        sentinel.write_bytes(b"overwritten")
        raise RuntimeError("computation started before compatibility check")

    monkeypatch.setattr(cli, "run_competence_time_training", would_overwrite)
    monkeypatch.setattr(cli, "run_competence_time_probes", would_overwrite)
    monkeypatch.setattr(cli, "_checkpoint_cache_complete", lambda *_: mode == "probe")
    with pytest.raises((SystemExit, ValueError), match="incompatible"):
        cli.main()
    assert sentinel.read_bytes() == b"frozen checkpoint"
    assert destination.read_bytes() == original


def test_completed_trajectory_is_persisted_and_reused_after_interruption(tmp_path, monkeypatch):
    config = _tiny_competence_time_config()
    args = _args(tmp_path, "train")
    _prepare_cli(monkeypatch, config, args)
    real_train = cli.run_competence_time_training
    calls = []

    def interrupt_second(config, seeds, rates, root):
        calls.append((seeds[0], rates[0]))
        if len(calls) == 2:
            raise InterruptedError("simulated interruption")
        return real_train(config, seeds, rates, root)

    monkeypatch.setattr(cli, "run_competence_time_training", interrupt_second)
    with pytest.raises(InterruptedError):
        cli.main()
    assert args.training_results.exists(), "completed first trajectory was lost"
    saved = cli._read_jsonl(args.training_results)
    assert len(saved) == 2
    checkpoint = competence_time_checkpoint_path(args.checkpoint_dir, 30, 0.01, 2)
    before = checkpoint.stat().st_mtime_ns
    assert not args.training_results.with_suffix(".jsonl.tmp").exists()
    calls.clear()

    def resume(config, seeds, rates, root):
        calls.append((seeds[0], rates[0]))
        return real_train(config, seeds, rates, root)

    monkeypatch.setattr(cli, "run_competence_time_training", resume)
    cli.main()
    assert (30, 0.01) not in calls
    assert checkpoint.stat().st_mtime_ns == before
    rows = cli._read_jsonl(args.training_results)
    assert len(rows) == 8
    assert all(row in rows for row in saved)


@pytest.mark.parametrize("output", ["training", "probe"])
@pytest.mark.parametrize("corruption", ["rate_digest", "seed", "checkpoint_root"])
def test_preflight_rejects_incompatible_raw_identity(tmp_path, monkeypatch, output, corruption):
    config, training, probes = _synthetic_analysis_grid()
    args = _args(tmp_path, "train")
    _canonical_paths(training, probes, args.checkpoint_dir)
    _prepare_cli(monkeypatch, config, args)
    row = copy.deepcopy(training[0] if output == "training" else probes[0])
    row["checkpoint_path"] = str(competence_time_checkpoint_path(
        args.checkpoint_dir, row["seed"], row["learning_rate"], row["step"]
    ))
    if corruption == "rate_digest":
        row["rate_config_sha256"] = "wrong"
    elif corruption == "seed":
        row["seed"] = 999
    else:
        row["checkpoint_path"] = str(competence_time_checkpoint_path(
            tmp_path / "different-run", row["seed"], row["learning_rate"], row["step"]
        ))
    destination = getattr(args, f"{output}_results")
    write_jsonl(destination, [row])
    original = destination.read_bytes()

    def computation_started(*_):
        raise RuntimeError("training started before raw identity check")

    monkeypatch.setattr(cli, "run_competence_time_training", computation_started)
    with pytest.raises((SystemExit, ValueError), match="incompatible"):
        cli.main()
    assert destination.read_bytes() == original
    assert not args.checkpoint_dir.exists()


@pytest.mark.parametrize("mode", ["analyze", "all"])
@pytest.mark.parametrize("failure", ["nonfinite_training", "nonfinite_probe", "leakage"])
def test_analysis_replaces_stale_summary_with_inconclusive(tmp_path, monkeypatch, mode, failure):
    config, training, probes = _synthetic_analysis_grid()
    args = _args(tmp_path, mode)
    _canonical_paths(training, probes, args.checkpoint_dir)
    _prepare_cli(monkeypatch, config, args)
    stale = analyze_competence_time(config, training, probes)
    assert stale["verdict"] == "supported"
    write_jsonl(args.training_results, training)
    assert cli._training_records_complete(
        args.training_results, config, args.seeds or config["competence_time"]["seeds"],
        config["competence_time"]["learning_rates"], args.checkpoint_dir,
    )
    # Real cache-valid checkpoint identities isolate the NaN record-cache failure.
    for row in training:
        checkpoint = Path(row["checkpoint_path"])
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "config": cli._rate_config(config, row["learning_rate"]),
            "seed": row["seed"], "step": row["step"], "condition": "fresh",
        }, checkpoint)
    checkpoint_bytes = {row["checkpoint_path"]: Path(row["checkpoint_path"]).read_bytes()
                        for row in training}
    if failure == "nonfinite_training":
        training[-1]["nll"] = float("nan")
    elif failure == "nonfinite_probe":
        probes[-1]["joint_belief_mse"] = float("inf")
    else:
        probes[-1]["probe_sequence_overlap"] = 1
    write_jsonl(args.training_results, training)
    write_jsonl(args.probe_results, probes)
    write_jsonl(args.summary_results, [stale])
    originals = {path: path.read_bytes() for path in (args.training_results, args.probe_results)}
    validate_competence_time_grid(config, training, probes, validate_science=False)
    if failure == "nonfinite_training":
        assert not cli._training_records_complete(
            args.training_results, config, config["competence_time"]["seeds"],
            config["competence_time"]["learning_rates"], args.checkpoint_dir,
        )

    def producer_must_not_run(*_):
        pytest.fail("complete invalid raw evidence must not be regenerated")

    monkeypatch.setattr(cli, "run_competence_time_training", producer_must_not_run)
    monkeypatch.setattr(cli, "run_competence_time_probes", producer_must_not_run)
    try:
        cli.main()
    except (SystemExit, ValueError):
        # Figures may reject unplottable invalid raw data after summary persistence.
        pass
    saved = cli._read_jsonl(args.summary_results)[0]
    assert saved == analyze_competence_time(config, training, probes)
    assert saved["verdict"] == "inconclusive"
    assert saved["validity_failures"] == ["probe_leakage" if failure == "leakage" else "invalid_grid"]
    for path, original in originals.items():
        assert path.read_bytes() == original
    for path, original in checkpoint_bytes.items():
        assert Path(path).read_bytes() == original
    if failure == "nonfinite_training":
        assert math.isnan(cli._read_jsonl(args.training_results)[-1]["nll"])


@pytest.mark.parametrize("missing", ["training_cell", "probe_file"])
def test_all_mode_repairs_incomplete_raw_work(tmp_path, monkeypatch, missing):
    config = _tiny_competence_time_config()
    registered, _, _ = _synthetic_analysis_grid()
    config["competence_time"].update({
        "primary_site": "block_2", "primary_target": "component_posterior",
        "primary_checkpoints": [2],
        "thresholds": registered["competence_time"]["thresholds"],
    })
    args = _args(tmp_path, "all")
    _prepare_cli(monkeypatch, config, args)
    seeds = config["competence_time"]["seeds"]
    rates = config["competence_time"]["learning_rates"]
    training = cli.run_competence_time_training(config, seeds, rates, args.checkpoint_dir)
    probes = cli.run_competence_time_probes(config, seeds, rates, args.checkpoint_dir)
    write_jsonl(args.training_results, training[:-1] if missing == "training_cell" else training)
    if missing != "probe_file":
        write_jsonl(args.probe_results, probes)
    real_train, real_probe = cli.run_competence_time_training, cli.run_competence_time_probes
    trained = []
    probed = []

    def train(*arguments):
        trained.append((arguments[1], arguments[2]))
        return real_train(*arguments)

    def probe(*arguments):
        probed.append(True)
        return real_probe(*arguments)

    monkeypatch.setattr(cli, "run_competence_time_training", train)
    monkeypatch.setattr(cli, "run_competence_time_probes", probe)
    cli.main()
    assert len(trained) == (1 if missing == "training_cell" else 0)
    assert len(probed) == 1
    repaired_training = cli._read_jsonl(args.training_results)
    repaired_probes = cli._read_jsonl(args.probe_results)
    validate_competence_time_grid(config, repaired_training, repaired_probes)
    assert cli._read_jsonl(args.summary_results) == [
        analyze_competence_time(config, repaired_training, repaired_probes)
    ]


@pytest.mark.parametrize("field", ["verdict", "ratio", "folds"])
def test_figures_reject_same_digest_stale_analysis(tmp_path, field):
    config, training, probes = _synthetic_analysis_grid()
    summary = analyze_competence_time(config, training, probes)
    if field == "verdict":
        summary["verdict"] = "falsified"
    elif field == "ratio":
        summary["primary"]["competence_to_step_mse_ratio"] = 0.9
    else:
        summary["primary"]["folds"][0]["competence_mse"] += 0.1
    with pytest.raises(ValueError, match="summary.*raw"):
        generate_competence_time_figures(config, training, probes, summary, tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("field,value", [
    ("condition", "reused"), ("condition", None), ("checkpoint_path", None),
    ("checkpoint_path", "synthetic/lr_0p00075/transformer_seed31_fresh_step0.pt"),
    ("checkpoint_path", "synthetic/lr_0p003/transformer_seed30_fresh_step0.pt"),
    ("checkpoint_path", "synthetic/lr_0p00075/transformer_seed30_fresh_step384.pt"),
])
def test_raw_grid_rejects_wrong_training_condition_or_checkpoint(field, value):
    config, training, probes = _synthetic_analysis_grid()
    training[0][field] = value
    with pytest.raises(ValueError, match="condition|checkpoint|identity"):
        validate_competence_time_grid(config, training, probes)


def test_retained_frozen_grid_and_summary_still_match():
    root = Path(__file__).resolve().parents[1]
    config = cli.load_config(root / "configs/mess3_competence_time.yaml")
    training, probes, summaries = [
        cli._read_jsonl(root / f"results/mess3_competence_time_{name}.jsonl")
        for name in ("training", "probes", "summary")
    ]
    validate_competence_time_grid(config, training, probes)
    assert analyze_competence_time(config, training, probes) == summaries[0]
