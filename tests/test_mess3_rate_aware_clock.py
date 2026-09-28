from pathlib import Path

import copy
import hashlib
import importlib
import json

import numpy as np
import pytest

from nonergodic_memory.experiment import config_digest, load_config
import torch


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "mess3_rate_aware_clock.yaml"


def _tiny_storage_config():
    config = load_config(CONFIG)
    config["data"].update(sequence_length=24, train_sequences=16, test_sequences=12)
    config["model"].update(width=8, heads=2, max_length=32)
    config["train"].update(batch_size=8, checkpoint_steps=[0, 1])
    config["diagnosis"]["window"] = 4
    config["probe"].update(train_sequences=24, test_sequences=16)
    config["rate_aware_clock"].update(seeds=[3, 4], primary_checkpoints=[1])
    return config


def _storage_run(tmp_path, seeds=(3,), rates=(.003, .006)):
    config = _tiny_storage_config()
    root, output = tmp_path / "checkpoints", tmp_path / "training.jsonl"
    rows = _api("run_rate_aware_training")(config, seeds, rates, root, results_path=output)
    return config, root, output, rows


def test_training_pairs_initialization_and_records_exact_checkpoint_identity(tmp_path):
    config, root, output, rows = _storage_run(tmp_path)
    assert len(rows) == 4
    assert _api("read_rate_aware_jsonl")(output) == rows
    initial = [row for row in rows if row["step"] == 0]
    assert initial[0]["parameter_sha256"] == initial[1]["parameter_sha256"]
    states = []
    for row in rows:
        path = _api("rate_aware_checkpoint_path")(root, 3, row["learning_rate"], row["step"])
        assert row["checkpoint_path"] == str(path)
        payload = torch.load(path, weights_only=False)
        assert payload["config"]["train"]["learning_rate"] == row["learning_rate"]
        assert payload["seed"] == row["seed"] == 3
        assert payload["step"] == row["step"]
        assert payload["condition"] == row["condition"] == "fresh"
        assert row["record_type"] == "rate_aware_clock_training"
        assert row["base_config_sha256"] == config_digest(config)
        assert row["config_sha256"] == row["rate_config_sha256"] == config_digest(payload["config"])
        assert row["parameters_finite"] is True
        assert len(row["parameter_sha256"]) == 64
        if row["step"] == 0:
            states.append(payload["state_dict"])
    assert all(torch.equal(states[0][key], states[1][key]) for key in states[0])


@pytest.mark.parametrize("kind", ["training", "probe"])
def test_storage_completed_trajectory_persists_and_restarts_after_interruption(tmp_path, monkeypatch, kind):
    module = importlib.import_module("nonergodic_memory.mess3_rate_aware_clock")
    runner = _api(f"run_rate_aware_{'probes' if kind == 'probe' else 'training'}")
    config = _tiny_storage_config()
    root, output = tmp_path / "checkpoints", tmp_path / f"{kind}.jsonl"
    if kind == "probe":
        _api("run_rate_aware_training")(config, [3, 4], [.003], root)
    producer = "train_diagnostic" if kind == "training" else "evaluate_checkpoint_geometry"
    real = getattr(module, producer)
    calls = []
    def interrupt(config, *args, **kwargs):
        seed = args[0] if kind == "training" else args[1]
        calls.append(seed)
        if seed == 4:
            raise InterruptedError("simulated interruption")
        return real(config, *args, **kwargs)
    monkeypatch.setattr(module, producer, interrupt)
    with pytest.raises(InterruptedError):
        runner(config, [3, 4], [.003], root, results_path=output)
    saved = _api("read_rate_aware_jsonl")(output)
    assert len(saved) == (2 if kind == "training" else 12)
    assert {row["seed"] for row in saved} == {3}
    checkpoint = _api("rate_aware_checkpoint_path")(root, 3, .003, 1)
    original = checkpoint.read_bytes(), checkpoint.stat().st_mtime_ns
    calls.clear()
    def resume(config, *args, **kwargs):
        calls.append(args[0] if kind == "training" else args[1])
        return real(config, *args, **kwargs)
    monkeypatch.setattr(module, producer, resume)
    rows = runner(config, [3, 4], [.003], root, results_path=output)
    assert 3 not in calls
    assert len(rows) == 2 * len(saved)
    assert all(row in rows for row in saved)
    assert original == (checkpoint.read_bytes(), checkpoint.stat().st_mtime_ns)


def test_probe_complete_keys_and_cache_reuse(tmp_path, monkeypatch):
    config, root, _, training = _storage_run(tmp_path)
    output = tmp_path / "probes.jsonl"
    runner = _api("run_rate_aware_probes")
    probes = runner(config, [3], [.003, .006], root, results_path=output)
    assert len(probes) == 24
    expected = {(3, rate, step, site, control) for rate in (.003, .006) for step in (0, 1)
                for site in ("block_1", "block_2", "final_norm") for control in ("none", "shuffled_labels")}
    assert {(r["seed"], r["learning_rate"], r["step"], r["site"], r["control"]) for r in probes} == expected
    assert all(r["record_type"] == "rate_aware_clock_probe" for r in probes)
    assert all(any(r["checkpoint_path"] == t["checkpoint_path"] for t in training) for r in probes)
    module = importlib.import_module("nonergodic_memory.mess3_rate_aware_clock")
    monkeypatch.setattr(module, "evaluate_checkpoint_geometry", lambda *a, **kw: pytest.fail("recomputed cached probes"))
    before = output.read_bytes(), output.stat().st_mtime_ns
    assert runner(config, [3], [.003, .006], root, results_path=output) == probes
    assert before == (output.read_bytes(), output.stat().st_mtime_ns)


@pytest.mark.parametrize("corruption", ["digest", "nonfinite", "partial", "duplicate", "path", "parameter_digest"])
def test_training_cache_rejects_invalid_evidence_before_side_effects(tmp_path, monkeypatch, corruption):
    config, root, output, rows = _storage_run(tmp_path, rates=(.003,))
    if corruption == "digest":
        rows[0]["base_config_sha256"] = "wrong"
    elif corruption == "nonfinite":
        rows[0]["competence"] = float("nan")
    elif corruption == "partial":
        rows.pop()
    elif corruption == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif corruption == "path":
        rows[0]["checkpoint_path"] = "wrong.pt"
    else:
        rows[0]["parameter_sha256"] = "a" * 64
    output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    module = importlib.import_module("nonergodic_memory.mess3_rate_aware_clock")
    monkeypatch.setattr(module, "train_diagnostic", lambda *a, **kw: pytest.fail("producer ran before preflight"))
    with pytest.raises(ValueError):
        _api("run_rate_aware_training")(config, [3, 4], [.003], root, results_path=output)
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("corruption", ["config", "seed", "step", "condition", "model", "nonfinite", "state"])
def test_training_cache_rejects_checkpoint_payload_corruption(tmp_path, monkeypatch, corruption):
    config, root, output, rows = _storage_run(tmp_path, rates=(.003,))
    path = Path(rows[-1]["checkpoint_path"])
    payload = torch.load(path, weights_only=False)
    if corruption == "config":
        payload["config"]["train"]["learning_rate"] = .999
    elif corruption in ("seed", "step"):
        payload[corruption] = 999
    elif corruption == "condition":
        payload["condition"] = "reused"
    elif corruption == "model":
        payload["model_name"] = "other"
    else:
        tensor = next(iter(payload["state_dict"].values()))
        tensor.flatten()[0] = float("nan") if corruption == "nonfinite" else tensor.flatten()[0] + 1
    torch.save(payload, path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        _api("run_rate_aware_training")(config, [3, 4], [.003], root, results_path=output)
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_probe_preflight_checks_all_selected_checkpoints_before_computing(tmp_path, monkeypatch):
    config, root, _, rows = _storage_run(tmp_path)
    Path(rows[-1]["checkpoint_path"]).unlink()
    output = tmp_path / "probes.jsonl"
    module = importlib.import_module("nonergodic_memory.mess3_rate_aware_clock")
    monkeypatch.setattr(module, "evaluate_checkpoint_geometry", lambda *a, **kw: pytest.fail("probe ran before preflight"))
    with pytest.raises(ValueError):
        _api("run_rate_aware_probes")(config, [3], [.003, .006], root, results_path=output)
    assert not output.exists()


def test_audit_hashes_actual_deterministic_disjoint_token_arrays():
    from nonergodic_memory.experiment import mixture_from_config
    from nonergodic_memory.mess3_diagnosis import sample_from_config
    config = _tiny_storage_config()
    audit = _api("audit_token_isolation")(config, [3, 4])
    assert audit == _api("audit_token_isolation")(config, [3, 4])
    assert len(audit) == 2
    mixture = mixture_from_config(config)
    for row in audit:
        assert row["record_type"] == "rate_aware_clock_audit"
        assert row["base_config_sha256"] == config_digest(config)
        assert row["intersections"] == {"evaluation__probe_fit": 0, "evaluation__probe_test": 0,
                                         "probe_fit__probe_test": 0}
        for name, count, offset in (("evaluation", 12, 202), ("probe_fit", 24, 404), ("probe_test", 16, 505)):
            tokens = sample_from_config(mixture, config, count, 24, row["seed"] + offset).tokens
            canonical = np.asarray(tokens, dtype="<i8", order="C")
            digest = hashlib.sha256(np.asarray(canonical.shape, dtype="<u8").tobytes() + canonical.tobytes()).hexdigest()
            assert row["datasets"][name] == {"sha256": digest, "n_rows": count, "n_unique_rows": count}


def test_audit_detects_real_duplicate_sequences_across_namespaces(monkeypatch):
    audit = _api("audit_token_isolation")
    module = importlib.import_module("nonergodic_memory.mess3_rate_aware_clock")
    real = module.sample_from_config
    batches = []
    def duplicate(*args, **kwargs):
        batch = real(*args, **kwargs)
        if batches:
            batch.tokens[0] = batches[0].tokens[0]
        else:
            batch.tokens[1] = batch.tokens[0]
        batches.append(batch)
        return batch
    monkeypatch.setattr(module, "sample_from_config", duplicate)
    row = audit(_tiny_storage_config(), [3])[0]
    assert row["datasets"]["evaluation"]["n_unique_rows"] == 11
    assert row["intersections"] == {"evaluation__probe_fit": 1, "evaluation__probe_test": 1,
                                     "probe_fit__probe_test": 1}


def test_audit_canonical_hash_is_independent_of_dtype_byteorder_and_layout():
    identity = _api("_token_array_hashes")
    tokens = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int32)
    assert identity(tokens) == identity(np.array(tokens, dtype=">i8", order="F"))
    assert identity(tokens) != identity(tokens[::-1])


def test_training_infrastructure_retry_is_logged_and_scientific_failure_is_not_retried(tmp_path, monkeypatch):
    runner = _api("run_rate_aware_training")
    module = importlib.import_module("nonergodic_memory.mess3_rate_aware_clock")
    config = _tiny_storage_config()
    root, output = tmp_path / "checkpoints", tmp_path / "training.jsonl"
    real = module.train_diagnostic
    def fail(*args, **kwargs):
        raise InterruptedError("infrastructure interruption")
    monkeypatch.setattr(module, "train_diagnostic", fail)
    with pytest.raises(InterruptedError):
        runner(config, [3], [.003], root, results_path=output)
    log = root / "rate_aware_clock_attempts.jsonl"
    assert log.exists(), "infrastructure attempts must be logged"
    events = _api("read_rate_aware_jsonl")(log)
    assert [e["status"] for e in events] == ["started", "failed"]
    assert all(e["base_config_sha256"] == config_digest(config) for e in events)
    monkeypatch.setattr(module, "train_diagnostic", real)
    runner(config, [3], [.003], root, results_path=output)
    events = _api("read_rate_aware_jsonl")(log)
    assert [e["status"] for e in events] == ["started", "failed", "started", "completed"]
    def nonfinite(*args, **kwargs):
        raise ValueError("nonfinite optimization")
    monkeypatch.setattr(module, "train_diagnostic", nonfinite)
    with pytest.raises(ValueError, match="nonfinite"):
        runner(config, [4], [.003], root, results_path=output)
    before = log.read_bytes()
    monkeypatch.setattr(module, "train_diagnostic", lambda *a, **kw: pytest.fail("retried scientific failure"))
    with pytest.raises(ValueError, match="scientific"):
        runner(config, [4], [.003], root, results_path=output)
    assert log.read_bytes() == before


def test_training_rejects_new_unpaired_initialization_before_publishing(tmp_path, monkeypatch):
    runner = _api("run_rate_aware_training")
    config, root, output, rows = _storage_run(tmp_path, rates=(.003,))
    module = importlib.import_module("nonergodic_memory.mess3_rate_aware_clock")
    real = module.train_diagnostic
    def changed_initialization(config, seed, condition, destination):
        run = real(config, seed, condition, destination)
        path = destination / f"transformer_seed{seed}_fresh_step0.pt"
        payload = torch.load(path, weights_only=False)
        next(iter(payload["state_dict"].values())).flatten()[0] += 1
        torch.save(payload, path)
        return run
    monkeypatch.setattr(module, "train_diagnostic", changed_initialization)
    original = output.read_bytes()
    with pytest.raises(ValueError, match="pairing"):
        runner(config, [3], [.006], root, results_path=output)
    assert output.read_bytes() == original
    assert not _api("rate_aware_checkpoint_path")(root, 3, .006, 0).exists()


@pytest.mark.parametrize("case", ["zero_heads", "zero_batch", "short_context", "fractional_count", "confirmation_config"])
def test_training_invalid_configuration_fails_before_any_side_effect(tmp_path, monkeypatch, case):
    config = _tiny_storage_config()
    if case == "zero_heads":
        config["model"]["heads"] = 0
    elif case == "zero_batch":
        config["train"]["batch_size"] = 0
    elif case == "short_context":
        config["model"]["max_length"] = 2
    elif case == "fractional_count":
        config["data"]["test_sequences"] = 1.5
    else:
        config["rate_aware_clock"]["seeds"] = [40]
    module = importlib.import_module("nonergodic_memory.mess3_rate_aware_clock")
    monkeypatch.setattr(module, "train_diagnostic", lambda *a, **kw: pytest.fail("producer called"))
    with pytest.raises(ValueError):
        _api("run_rate_aware_training")(config, config["rate_aware_clock"]["seeds"], [.003],
                                        tmp_path / "checkpoints", results_path=tmp_path / "training.jsonl")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("kind", ["training", "probe"])
def test_storage_empty_producer_result_is_rejected(tmp_path, monkeypatch, kind):
    config, root, output, _ = _storage_run(tmp_path, rates=(.003,))
    module = importlib.import_module("nonergodic_memory.mess3_rate_aware_clock")
    if kind == "training":
        monkeypatch.setattr(module, "train_diagnostic", lambda *a, **kw: [])
        runner, seeds = _api("run_rate_aware_training"), [4]
    else:
        monkeypatch.setattr(module, "evaluate_checkpoint_geometry", lambda *a, **kw: [])
        runner, seeds, output = _api("run_rate_aware_probes"), [3], tmp_path / "probes.jsonl"
    before = output.read_bytes() if output.exists() else None
    with pytest.raises(ValueError, match="complete"):
        runner(config, seeds, [.003], root, results_path=output)
    assert (output.read_bytes() if output.exists() else None) == before


def test_cache_parameter_digest_uses_sorted_tensors_and_detects_any_change():
    identity = _api("_parameter_identity")
    state = {"b": torch.tensor([1., 2.]), "a": torch.tensor([[3.]])}
    assert identity(state) == identity(dict(reversed(list(state.items()))))
    changed = {key: value.clone() for key, value in state.items()}
    changed["a"][0, 0] += 1
    assert identity(state)[0] != identity(changed)[0]
    changed["a"][0, 0] = float("nan")
    assert identity(changed)[1] is False


def test_cache_atomic_writer_preserves_original_when_rename_fails(tmp_path, monkeypatch):
    writer = _api("atomic_write_rate_aware_jsonl")
    output = tmp_path / "output.jsonl"
    writer(output, [{"original": True}])
    before = output.read_bytes()
    def fail(*args, **kwargs):
        raise OSError("simulated rename interruption")
    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(OSError):
        writer(output, [{"new": True}])
    assert output.read_bytes() == before
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("kind", ["audit", "summary"])
def test_cache_preflight_rejects_other_output_identity_before_writes(tmp_path, kind):
    config = _tiny_storage_config()
    path = tmp_path / f"{kind}.jsonl"
    path.write_text(json.dumps({"record_type": f"rate_aware_clock_{kind}", "base_config_sha256": "wrong"}) + "\n")
    before = path.read_bytes()
    with pytest.raises(ValueError):
        _api("preflight_rate_aware_outputs")(config, tmp_path / "checkpoints", **{f"{kind}_path": path})
    assert path.read_bytes() == before
    assert not (tmp_path / "checkpoints").exists()


def test_training_missing_measurement_is_recorded_as_scientific_failure(tmp_path, monkeypatch):
    module = importlib.import_module("nonergodic_memory.mess3_rate_aware_clock")
    real = module.train_diagnostic
    def missing(*args, **kwargs):
        rows = real(*args, **kwargs)
        del rows[-1]["competence"]
        return rows
    monkeypatch.setattr(module, "train_diagnostic", missing)
    root = tmp_path / "checkpoints"
    with pytest.raises(ValueError):
        _api("run_rate_aware_training")(_tiny_storage_config(), [3], [.003], root,
                                        results_path=tmp_path / "training.jsonl")
    events = _api("read_rate_aware_jsonl")(root / "rate_aware_clock_attempts.jsonl")
    assert events[-1]["scientific_failure"] is True
    assert not (tmp_path / "training.jsonl").exists()
    assert list(root.glob(".rate-aware-*/lr_0p003/*.pt")), "failed evidence must remain available"


@pytest.mark.parametrize("kind", ["training", "probe"])
def test_storage_missing_producer_step_is_validated_before_enrichment(tmp_path, monkeypatch, kind):
    module = importlib.import_module("nonergodic_memory.mess3_rate_aware_clock")
    config = _tiny_storage_config()
    root, output = tmp_path / "checkpoints", tmp_path / f"{kind}.jsonl"
    if kind == "probe":
        _api("run_rate_aware_training")(config, [3], [.003], root)
    producer = "train_diagnostic" if kind == "training" else "evaluate_checkpoint_geometry"
    real = getattr(module, producer)
    def missing(*args, **kwargs):
        rows = real(*args, **kwargs)
        del rows[-1]["step"]
        return rows
    monkeypatch.setattr(module, producer, missing)
    monkeypatch.setattr(module, "_enrich_record", lambda *a, **kw: pytest.fail("enriched unvalidated producer output"))
    runner = _api("run_rate_aware_training" if kind == "training" else "run_rate_aware_probes")
    with pytest.raises(ValueError, match="producer"):
        runner(config, [3], [.003], root, results_path=output)
    log = root / "rate_aware_clock_attempts.jsonl"
    assert _api("read_rate_aware_jsonl")(log)[-1]["scientific_failure"] is True
    before = log.read_bytes()
    with pytest.raises(ValueError, match="scientific"):
        runner(config, [3], [.003], root, results_path=output)
    assert log.read_bytes() == before
    assert not output.exists()


@pytest.mark.parametrize("kind", ["training", "probe"])
@pytest.mark.parametrize("error_type", [KeyError, RuntimeError, Exception, OSError, TimeoutError])
def test_storage_unknown_producer_exception_never_allows_retry(tmp_path, monkeypatch, kind, error_type):
    module = importlib.import_module("nonergodic_memory.mess3_rate_aware_clock")
    config = _tiny_storage_config()
    root, output = tmp_path / "checkpoints", tmp_path / f"{kind}.jsonl"
    if kind == "probe":
        _api("run_rate_aware_training")(config, [3], [.003], root)
    producer = "train_diagnostic" if kind == "training" else "evaluate_checkpoint_geometry"
    def fail(*args, **kwargs):
        raise error_type("unclassified producer failure")
    monkeypatch.setattr(module, producer, fail)
    runner = _api("run_rate_aware_training" if kind == "training" else "run_rate_aware_probes")
    with pytest.raises(error_type):
        runner(config, [3], [.003], root, results_path=output)
    log = root / "rate_aware_clock_attempts.jsonl"
    assert _api("read_rate_aware_jsonl")(log)[-1]["scientific_failure"] is True
    before = log.read_bytes()
    monkeypatch.setattr(module, producer, lambda *a, **kw: pytest.fail("unclassified failure was retried"))
    with pytest.raises(ValueError, match="scientific"):
        runner(config, [3], [.003], root, results_path=output)
    assert log.read_bytes() == before
    assert not output.exists()


@pytest.mark.parametrize("corruption", ["missing_hash", "malformed_hash", "missing_finite", "nonfinite",
                                        "checkpoint_changed", "digest_mismatch", "unbound_changed_checkpoint"])
def test_probe_cache_requires_verified_parameter_binding(tmp_path, monkeypatch, corruption):
    config, root, _, _ = _storage_run(tmp_path, rates=(.003,))
    output = tmp_path / "probes.jsonl"
    runner = _api("run_rate_aware_probes")
    rows = runner(config, [3], [.003], root, results_path=output)
    row = next(row for row in rows if row["step"] == 1)
    if corruption in ("missing_hash", "unbound_changed_checkpoint"):
        # Removing all hashes reproduces the old optional-binding bypass.
        for candidate in rows:
            del candidate["parameter_sha256"]
    elif corruption == "malformed_hash":
        row["parameter_sha256"] = "not-a-sha256"
    elif corruption == "missing_finite":
        del row["parameters_finite"]
    elif corruption == "nonfinite":
        row["parameters_finite"] = False
    elif corruption == "digest_mismatch":
        row["parameter_sha256"] = "a" * 64
    if corruption in ("checkpoint_changed", "unbound_changed_checkpoint"):
        path = Path(row["checkpoint_path"])
        payload = torch.load(path, weights_only=False)
        next(iter(payload["state_dict"].values())).flatten()[0] += 1
        torch.save(payload, path)
    output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    module = importlib.import_module("nonergodic_memory.mess3_rate_aware_clock")
    monkeypatch.setattr(module, "evaluate_checkpoint_geometry", lambda *a, **kw: pytest.fail("invalid probe cache reached producer"))
    with pytest.raises(ValueError, match="parameter"):
        runner(config, [3], [.003], root, results_path=output)
    assert before == {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}


def _api(name):
    try:
        module = importlib.import_module("nonergodic_memory.mess3_rate_aware_clock")
    except ModuleNotFoundError:
        pytest.fail("frozen forecast module is missing")
    function = getattr(module, name, None)
    assert callable(function), f"{name} API is missing"
    return function


def test_forecast_provenance_independently_refits_both_frozen_models():
    config = load_config(CONFIG)
    root = CONFIG.parents[1]
    result = _api("verify_forecast_provenance")(config, root)
    assert result["rows"] == 256
    assert result["training_jsonl_sha256"] == config["forecast"]["provenance"]["training_jsonl_sha256"]
    assert result["probe_jsonl_sha256"] == config["forecast"]["provenance"]["probe_jsonl_sha256"]
    for name in ("competence", "rate_aware_clock"):
        for key in ("center", "scale", "coefficients", "support"):
            np.testing.assert_allclose(result["refit"][name][key], config["forecast"][name][key], atol=1e-12, rtol=0)


@pytest.mark.parametrize("change", ["hash", "target", "coefficient", "center", "transform", "old_rows", "support"])
def test_forecast_provenance_rejects_changed_frozen_spec(change):
    config = load_config(CONFIG)
    if change == "hash":
        value = config["forecast"]["provenance"]["training_jsonl_sha256"]
        config["forecast"]["provenance"]["training_jsonl_sha256"] = ("3" if value[0] == "2" else "2") + value[1:]
    elif change == "target":
        config["forecast"]["provenance"]["metric"] = "joint_belief_r2"
    elif change == "coefficient":
        values = config["forecast"]["competence"]["coefficients"]
        values[0] = float(np.nextafter(values[0], np.inf))
    elif change == "center":
        config["forecast"]["rate_aware_clock"]["center"] = float(np.nextafter(config["forecast"]["rate_aware_clock"]["center"], np.inf))
    elif change == "transform":
        config["forecast"]["rate_aware_clock"]["transform"] = "log1p(step)"
    elif change == "support":
        config["forecast"]["competence"]["support"][0] -= 1e-10
    else:
        config["forecast"]["provenance"]["seeds"][0] = 29
    with pytest.raises(ValueError, match="frozen"):
        _api("verify_forecast_provenance")(config, CONFIG.parents[1])


def test_forecast_provenance_rejects_one_bit_source_change(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    for name in ("training", "probes"):
        filename = f"mess3_competence_time_{name}.jsonl"
        data = (CONFIG.parents[1] / "results" / filename).read_bytes()
        if name == "training":
            data = bytes([data[0] ^ 1]) + data[1:]
        (results / filename).write_bytes(data)
    with pytest.raises(ValueError, match="hash"):
        _api("verify_forecast_provenance")(load_config(CONFIG), tmp_path)


@pytest.mark.parametrize("name", ["competence", "rate_aware_clock"])
def test_forecast_exact_unclipped_scalar_polynomial(name):
    spec = load_config(CONFIG)["forecast"][name]
    forecast = _api("forecast_geometry")
    assert forecast(spec["center"], spec) == spec["coefficients"][0]
    value = spec["center"] + 10 * spec["scale"]
    z = (value - spec["center"]) / spec["scale"]
    a, b, c = spec["coefficients"]
    assert forecast(value, spec) == a + b * z + c * z**2
    assert forecast(value, spec) > 1


def _synthetic_analysis_grid(kind="clock"):
    """Pure in-memory fixtures; no confirmation data or checkpoints generated."""
    config = load_config(CONFIG)
    forecast = _api("forecast_geometry")
    training, probes, audit = [], [], []
    for seed in config["rate_aware_clock"]["seeds"]:
        audit.append({
            "record_type": "rate_aware_clock_audit", "seed": seed,
            "base_config_sha256": config_digest(config),
            "datasets": {name: {"sha256": hashlib.sha256(f"{seed}:{name}".encode()).hexdigest(),
                                "n_rows": count, "n_unique_rows": count}
                         for name, count in (("evaluation", 256), ("probe_fit", 1024), ("probe_test", 512))},
            "intersections": {"evaluation__probe_fit": 0, "evaluation__probe_test": 0,
                              "probe_fit__probe_test": 0},
        })
        for index, rate in enumerate(config["rate_aware_clock"]["learning_rates"]):
            rate_config = copy.deepcopy(config)
            rate_config["train"]["learning_rate"] = rate
            for step in config["train"]["checkpoint_steps"]:
                competence = 0.2 + 0.2 * index + 0.01 * (seed - 40)
                c = forecast(competence, config["forecast"]["competence"])
                r = forecast(np.log1p(rate * step), config["forecast"]["rate_aware_clock"])
                geometry = r if kind == "clock" else c
                common = {
                    "seed": seed, "learning_rate": rate, "step": step,
                    "base_config_sha256": config_digest(config),
                    "rate_config_sha256": config_digest(rate_config),
                    "config_sha256": config_digest(rate_config),
                    "sampler": "vectorized", "condition": "fresh",
                    "checkpoint_path": f"synthetic/lr_{str(rate).replace('.', 'p')}/transformer_seed{seed}_fresh_step{step}.pt",
                }
                training.append({
                    **common, "record_type": "rate_aware_clock_training",
                    "competence": competence, "kl_exact": 1 - competence,
                    "uniform_kl": 1.0, "nll": 1.2 - competence,
                    "parameters_finite": True,
                    "parameter_sha256": hashlib.sha256(f"{seed}:{step}".encode()).hexdigest(),
                })
                for site in ("block_1", "block_2", "final_norm"):
                    for control in ("none", "shuffled_labels"):
                        probes.append({
                            **common, "record_type": "rate_aware_clock_probe", "site": site,
                            "control": control, "component_posterior_r2": geometry if control == "none" else 0.,
                            "probe_sequence_overlap": 0,
                            "component_accuracy": 0.5, "conditional_state_accuracy": 0.5,
                            "state_posterior_r2": 0., "joint_belief_mse": 0.1,
                            "joint_belief_r2": 0., "joint_distance_r2": 0.,
                        })
    return config, training, probes, audit


def _analyze(fixture):
    return _api("analyze_rate_aware_clock")(*fixture)


def test_analysis_supported_seed_equal_frozen_forecasts():
    fixture = _synthetic_analysis_grid()
    original = copy.deepcopy(fixture)
    summary = _analyze(fixture)
    assert fixture == original
    assert summary["verdict"] == "supported"
    assert summary["validity_failures"] == []
    assert summary["excluded_initialization_count"] == 16
    assert summary["provenance"]["rows"] == 256
    primary = summary["primary"]
    assert primary["observations"] == 112
    assert primary["clock_mse"] == 0
    assert primary["clock_to_competence_mse_ratio"] == 0
    assert primary["clock_seed_wins"] == 8
    assert len(primary["per_seed"]) == 8
    assert primary["competence_mse"] == np.mean([row["competence_mse"] for row in primary["per_seed"]])
    assert all(row["observations"] == 14 for row in primary["per_seed"])


def test_analysis_valid_falsified_and_zero_competence_denominator():
    summary = _analyze(_synthetic_analysis_grid("competence"))
    assert summary["verdict"] == "falsified"
    assert summary["validity_failures"] == []
    assert summary["primary"]["competence_mse"] == 0
    assert summary["primary"]["clock_to_competence_mse_ratio"] is None
    assert summary["primary"]["ratio_reason"] == "zero_competence_mse"
    assert summary["primary"]["competence_seed_wins"] == 8
    json.dumps(summary, allow_nan=False)


def test_analysis_ratio_pass_but_only_six_seed_wins_is_falsified():
    fixture = _synthetic_analysis_grid()
    other = _synthetic_analysis_grid("competence")
    for probe, competence_probe in zip(fixture[2], other[2]):
        if probe["seed"] in (46, 47):
            probe["component_posterior_r2"] = competence_probe["component_posterior_r2"]
    summary = _analyze(fixture)
    assert summary["primary"]["clock_to_competence_mse_ratio"] < .8
    assert summary["primary"]["clock_seed_wins"] == 6
    assert summary["verdict"] == "falsified"


@pytest.mark.parametrize("change,failure", [
    ("shuffled", "shuffled_labels"), ("dissociation", "rate_dissociation"),
    ("support", "forecast_support"), ("missing", "invalid_grid"),
    ("duplicate", "invalid_grid"), ("nonfinite", "invalid_grid"),
    ("secondary_nonfinite", "invalid_grid"), ("wrong_token", "token_isolation"),
    ("missing_audit", "token_isolation"), ("duplicate_audit", "token_isolation"),
    ("audit_hash", "token_isolation"), ("audit_count", "token_isolation"),
    ("audit_digest", "token_isolation"), ("audit_unique", "token_isolation"),
    ("parameters", "invalid_grid"), ("pairing", "initialization_pairing"),
    ("digest", "invalid_grid"), ("probe_path", "invalid_grid"),
    ("condition", "invalid_grid"), ("fractional_seed", "invalid_grid"),
])
def test_analysis_validity_failures_override_favorable_forecasts(change, failure):
    fixture = _synthetic_analysis_grid()
    _, training, probes, audit = fixture
    if change == "shuffled":
        next(row for row in probes if row["step"] > 0 and row["site"] == "block_2" and row["control"] == "shuffled_labels")["component_posterior_r2"] = -.021
    elif change == "dissociation":
        for row in training:
            row["competence"] = .3
    elif change == "support":
        training[1]["competence"] = .9
    elif change == "missing":
        training.pop()
    elif change == "duplicate":
        probes.append(copy.deepcopy(probes[-1]))
    elif change == "nonfinite":
        training[1]["competence"] = float("nan")
    elif change == "secondary_nonfinite":
        probes[0]["joint_belief_mse"] = float("inf")
    elif change == "wrong_token":
        audit[0]["intersections"]["evaluation__probe_fit"] = .5
    elif change == "missing_audit":
        audit.pop()
    elif change == "duplicate_audit":
        audit.append(copy.deepcopy(audit[0]))
    elif change == "audit_hash":
        audit[0]["datasets"]["evaluation"]["sha256"] = "namespace_only"
    elif change == "audit_count":
        audit[0]["datasets"]["evaluation"]["n_rows"] = 255
    elif change == "audit_digest":
        audit[0]["base_config_sha256"] = "wrong"
    elif change == "audit_unique":
        audit[0]["datasets"]["evaluation"]["n_unique_rows"] = 257
    elif change == "parameters":
        training[-1]["parameters_finite"] = False
    elif change == "pairing":
        training[8]["parameter_sha256"] = "a" * 64
    elif change == "digest":
        probes[0]["rate_config_sha256"] = "wrong"
    elif change == "probe_path":
        probes[0]["checkpoint_path"] = "wrong.pt"
    elif change == "condition":
        probes[0]["condition"] = "repeated"
    else:
        training[0]["seed"] = 40.5
    summary = _analyze(fixture)
    assert summary["verdict"] == "inconclusive"
    assert failure in summary["validity_failures"]


def test_analysis_extreme_finite_initialization_excluded_from_support_and_scoring():
    fixture = _synthetic_analysis_grid()
    for row in fixture[1]:
        if row["step"] == 0:
            row["competence"] = -1e308
    for row in fixture[2]:
        if row["step"] == 0:
            row["component_posterior_r2"] = 1e308
    summary = _analyze(fixture)
    assert summary["verdict"] == "supported"
    assert summary["primary"]["clock_mse"] == 0


def test_analysis_extra_observations_cannot_reweight_seeds():
    fixture = _synthetic_analysis_grid()
    fixture[1].extend(copy.deepcopy([row for row in fixture[1] if row["seed"] == 40]))
    summary = _analyze(fixture)
    assert summary["verdict"] == "inconclusive"
    assert summary["primary"] is None


@pytest.mark.parametrize("case", ["ratio_boundary", "ties", "both_perfect"])
def test_analysis_exact_arithmetic_decision_edges(monkeypatch, case):
    fixture = _synthetic_analysis_grid()
    # Isolate decision arithmetic from polynomial roundoff with exact forecasts.
    def exact_forecast(value, spec):
        if case == "both_perfect":
            return 0.
        if case == "ties":
            return 1.
        if "transform" in spec:
            return 2.
        return 1. if value < .4 else 3.
    module = importlib.import_module("nonergodic_memory.mess3_rate_aware_clock")
    monkeypatch.setattr(module, "forecast_geometry", exact_forecast)
    for row in fixture[2]:
        row["component_posterior_r2"] = 0.
    summary = _analyze(fixture)
    assert summary["verdict"] == "falsified"
    assert summary["validity_failures"] == []
    if case == "ratio_boundary":
        assert summary["primary"]["clock_to_competence_mse_ratio"] == .8
        assert summary["primary"]["clock_seed_wins"] == 8
    else:
        assert summary["primary"]["clock_seed_wins"] == 0
        assert summary["primary"]["competence_seed_wins"] == 0
    if case == "both_perfect":
        assert summary["primary"]["clock_to_competence_mse_ratio"] is None
        assert summary["primary"]["ratio_reason"] == "zero_competence_mse"


def test_analysis_rejects_modified_forecast_provenance():
    fixture = _synthetic_analysis_grid()
    fixture[0]["forecast"]["competence"]["coefficients"][0] += 1e-15
    summary = _analyze(fixture)
    assert summary["verdict"] == "inconclusive"
    assert "forecast_provenance" in summary["validity_failures"]


@pytest.mark.parametrize("value", ["0.3", True, None])
def test_analysis_malformed_numeric_metrics_are_invalid_grid(value):
    fixture = _synthetic_analysis_grid()
    fixture[1][1]["competence"] = value
    summary = _analyze(fixture)
    assert summary["verdict"] == "inconclusive"
    assert "invalid_grid" in summary["validity_failures"]


def test_analysis_scoring_helper_weights_seeds_equally_even_with_unequal_counts():
    config, training, probes, _ = _synthetic_analysis_grid()
    cells = {(row["seed"], row["learning_rate"], row["step"]): row for row in training}
    selected = [row for row in probes if row["step"] > 0 and row["site"] == "block_2"
                and row["control"] == "none" and (row["seed"] != 40 or row["step"] == 384)]
    scores = _api("_score_forecasts")(config, cells, selected)
    seed_equal = np.mean([row["competence_mse"] for row in scores["per_seed"]])
    pooled = np.average([row["competence_mse"] for row in scores["per_seed"]],
                        weights=[row["observations"] for row in scores["per_seed"]])
    assert scores["competence_mse"] == seed_equal
    assert scores["competence_mse"] != pooled


def test_analysis_predeclared_thresholds_cannot_be_changed():
    fixture = _synthetic_analysis_grid()
    fixture[0]["rate_aware_clock"]["thresholds"]["minimum_clock_seed_wins"] = 1
    summary = _analyze(fixture)
    assert summary["verdict"] == "inconclusive"
    assert "invalid_configuration" in summary["validity_failures"]


@pytest.mark.parametrize("row_type", ["training", "probe"])
@pytest.mark.parametrize("rate", ["0.003", True, 0.003 + 0j, None])
def test_analysis_malformed_rate_types_return_inconclusive(row_type, rate):
    fixture = _synthetic_analysis_grid()
    rows = fixture[1] if row_type == "training" else fixture[2]
    row = next(row for row in rows if row["step"] > 0 and row["learning_rate"] == .003
               and (row_type == "training" or (row["site"] == "block_2" and row["control"] == "none")))
    row["learning_rate"] = rate
    summary = _analyze(fixture)
    assert summary["verdict"] == "inconclusive"
    assert "invalid_grid" in summary["validity_failures"]
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("field", ["datasets", "intersections"])
def test_analysis_audit_collections_must_be_mappings(field):
    fixture = _synthetic_analysis_grid()
    fixture[3][0][field] = list(fixture[3][0][field])
    summary = _analyze(fixture)
    assert summary["verdict"] == "inconclusive"
    assert "token_isolation" in summary["validity_failures"]
    assert "mapping" in summary["audit_error"]
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("dataset", ["evaluation", "probe_fit", "probe_test"])
def test_analysis_nonempty_audit_dataset_requires_positive_unique_count(dataset):
    fixture = _synthetic_analysis_grid()
    fixture[3][0]["datasets"][dataset]["n_unique_rows"] = 0
    summary = _analyze(fixture)
    assert summary["verdict"] == "inconclusive"
    assert "token_isolation" in summary["validity_failures"]
    json.dumps(summary, allow_nan=False)


def test_analysis_extreme_finite_competence_arithmetic_is_json_safe():
    fixture = _synthetic_analysis_grid()
    for row in fixture[1]:
        if row["seed"] == 40 and row["step"] == 384:
            row["competence"] = -1e308 if row["learning_rate"] == .003 else 1e308
    summary = _analyze(fixture)
    assert summary["verdict"] == "inconclusive"
    assert "forecast_support" in summary["validity_failures"]
    assert "nonfinite_analysis" in summary["validity_failures"]
    json.dumps(summary, allow_nan=False)
    difference = next(row for row in summary["rate_dissociation"]["per_seed"] if row["seed"] == 40)
    assert difference["max_competence_difference"] is None
    assert difference["reason"] == "nonfinite_competence_difference"
    assert 40 not in summary["rate_dissociation"]["passing_seeds"]


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
