#!/usr/bin/env python3
"""Run the isolated, preregistered Mess3 rate-aware clock experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from nonergodic_memory.experiment import config_digest, load_config
from nonergodic_memory.mess3_rate_aware_clock import (
    _checked_checkpoint, _storage_selection, _validate_audit, _validate_grid, _validate_storage_paths,
    _validate_trajectory_rows, analyze_rate_aware_clock, atomic_write_rate_aware_jsonl,
    audit_token_isolation, preflight_rate_aware_outputs, rate_aware_checkpoint_path,
    read_rate_aware_jsonl, run_rate_aware_probes, run_rate_aware_training,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mess3_rate_aware_clock.yaml")
    parser.add_argument("--mode", choices=("train", "probe", "audit", "analyze", "figures", "all"), default="all")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--learning-rates", nargs="+", type=float)
    parser.add_argument("--checkpoint-dir", default="checkpoints/mess3_rate_aware_clock")
    for kind, filename in (("training", "training"), ("probe", "probes"),
                           ("audit", "audit"), ("summary", "summary")):
        parser.add_argument(f"--{kind}-results", default=f"results/mess3_rate_aware_clock_{filename}.jsonl")
    parser.add_argument("--output-dir", default="figures")
    return parser.parse_args()


def _validate_paths(config: dict, args: argparse.Namespace) -> None:
    """Reject equal, linked, or nested file targets before creating anything."""
    outputs = [Path(getattr(args, f"{kind}_results")) for kind in ("training", "probe", "audit", "summary")]
    outputs += [Path(args.output_dir) / f"mess3_rate_aware_clock_{name}.png"
                for name in ("learning", "forecasts")]
    _validate_storage_paths(config, args.checkpoint_dir, outputs, protected=[Path(args.config)],
                            directories=[Path(args.output_dir)])


def _preflight(config: dict, args: argparse.Namespace) -> dict:
    """Check immutable evidence identity without treating science failure as recovery."""
    _validate_paths(config, args)
    rows = {kind: read_rate_aware_jsonl(path) if path.exists() else []
            for kind in ("training", "probe", "audit", "summary")
            for path in [Path(getattr(args, f"{kind}_results"))]}
    rows["terminal_scientific_failure"] = False
    log = Path(args.checkpoint_dir) / "rate_aware_clock_attempts.jsonl"
    for event in read_rate_aware_jsonl(log) if log.exists() else []:
        if (event.get("record_type") != "rate_aware_clock_attempt"
                or event.get("base_config_sha256") != config_digest(config)
                or event.get("kind") not in ("training", "probe")
                or event.get("status") not in ("started", "failed", "completed")
                or event.get("seed") not in config["rate_aware_clock"]["seeds"]
                or event.get("learning_rate") not in config["rate_aware_clock"]["learning_rates"]):
            raise ValueError("incompatible attempt log")
        if event.get("scientific_failure"):
            rows["terminal_scientific_failure"] = True
    _validate_grid(config, rows["training"], rows["probe"], require_complete=False, check_science=False)
    identities = {}
    for seed in config["rate_aware_clock"]["seeds"]:
        for rate in config["rate_aware_clock"]["learning_rates"]:
            for step in config["train"]["checkpoint_steps"]:
                if rate_aware_checkpoint_path(args.checkpoint_dir, seed, rate, step).exists():
                    identities[seed, rate, step] = _checked_checkpoint(config, args.checkpoint_dir, seed, rate, step)
    for row in rows["training"] + rows["probe"]:
        cell = row["seed"], row["learning_rate"], row["step"]
        if row["checkpoint_path"] != str(rate_aware_checkpoint_path(args.checkpoint_dir, *cell)):
            raise ValueError("raw checkpoint root mismatch")
        if cell not in identities or row["parameter_sha256"] != identities[cell]["parameter_sha256"]:
            raise ValueError("raw evidence does not match checkpoint parameters")
    if rows["audit"]:
        # Overlap is a scientific outcome; malformed hashes/counts are corruption.
        _validate_audit(config, rows["audit"], require_isolation=False)
    summaries = rows["summary"]
    if Path(args.summary_results).exists() and (len(summaries) != 1
            or summaries[0].get("record_type") != "rate_aware_clock_summary"
            or summaries[0].get("base_config_sha256") != config_digest(config)):
        raise ValueError("summary identity mismatch")
    if Path(args.audit_results).exists() and not rows["audit"]:
        raise ValueError("existing audit must contain one complete seed snapshot")
    return rows


def _preserve_scientific_evidence(config, rows) -> bool:
    """Stop only for a full grid or terminal execution/trajectory evidence.

    Finite scientific outcomes never select which remaining trajectories run.
    """
    if rows["terminal_scientific_failure"]:
        return True
    for kind in ("training", "probe"):
        try:
            _validate_trajectory_rows(config, rows[kind], kind)
        except ValueError:
            return True
    try:
        _validate_grid(config, rows["training"], rows["probe"], require_isolation=False)
    except ValueError:
        return False
    return True


def main() -> None:
    args = parse_args()
    try:
        config = load_config(args.config)
        experiment = config["rate_aware_clock"]
        seeds, rates = _storage_selection(config, args.seeds or experiment["seeds"],
                                           args.learning_rates or experiment["learning_rates"])
        if args.mode in ("all", "analyze", "figures", "audit") and (
                set(seeds) != set(experiment["seeds"])
                or set(rates) != set(experiment["learning_rates"])):
            raise ValueError(f"{args.mode} requires the full configured seed/rate grid; use train/probe for subsets")
        rows = _preflight(config, args)
        terminal_scientific_failure = rows["terminal_scientific_failure"]
        preserve = args.mode == "all" and _preserve_scientific_evidence(config, rows)
        if args.mode in ("train", "probe", "all") and not preserve:
            preflight_rate_aware_outputs(config, args.checkpoint_dir,
                training_path=args.training_results, probe_path=args.probe_results,
                audit_path=args.audit_results, summary_path=args.summary_results)
            if args.mode in ("train", "all"):
                run_rate_aware_training(config, seeds, rates, args.checkpoint_dir,
                                        results_path=args.training_results)
            if args.mode in ("probe", "all"):
                run_rate_aware_probes(config, seeds, rates, args.checkpoint_dir,
                                      results_path=args.probe_results)
        if args.mode in ("audit", "all") and not rows["audit"]:
            audit_ready = not preserve
            if preserve and not terminal_scientific_failure:
                # Completed trajectories need no producer, but their fixed audit
                # is still required, independently of finite scientific outcomes.
                try:
                    _validate_grid(config, rows["training"], rows["probe"], require_isolation=False)
                except ValueError:
                    pass
                else:
                    audit_ready = True
            if audit_ready:
                audit = audit_token_isolation(config, experiment["seeds"])
                atomic_write_rate_aware_jsonl(args.audit_results, audit)
        if args.mode in ("analyze", "all"):
            rows = {kind: read_rate_aware_jsonl(path) if Path(path).exists() else []
                    for kind, path in (("training", args.training_results),
                                       ("probe", args.probe_results), ("audit", args.audit_results))}
            summary = analyze_rate_aware_clock(config, rows["training"], rows["probe"], rows["audit"],
                                               terminal_scientific_failure=terminal_scientific_failure)
            atomic_write_rate_aware_jsonl(args.summary_results, [summary])
            print(f"rate-aware clock verdict: {summary['verdict']}")
        if args.mode in ("figures", "all"):
            from nonergodic_memory.mess3_rate_aware_clock_figures import generate_rate_aware_clock_figures
            if args.mode == "all":
                try:
                    _validate_grid(config, rows["training"], rows["probe"])
                    _validate_audit(config, rows["audit"], require_isolation=False)
                except ValueError:
                    print("figures unavailable: raw grid is incomplete or invalid")
                    return
            training, probes, audit, summaries = [read_rate_aware_jsonl(path) for path in
                (args.training_results, args.probe_results, args.audit_results, args.summary_results)]
            for path in generate_rate_aware_clock_figures(config, training, probes, audit,
                    summaries[0], args.output_dir, terminal_scientific_failure=terminal_scientific_failure):
                print(f"generated {path}")
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
        raise SystemExit(f"rate-aware clock preflight or execution failed: {error}") from error


if __name__ == "__main__":
    main()
