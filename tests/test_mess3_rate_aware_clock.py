from pathlib import Path

import copy
import hashlib
import importlib
import json

import numpy as np
import pytest

from nonergodic_memory.experiment import config_digest, load_config


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "mess3_rate_aware_clock.yaml"


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
