#!/usr/bin/env python3
"""Run the isolated, preregistered Mess3 rate-aware clock experiment."""

from __future__ import annotations

import argparse
import copy
from itertools import combinations
from pathlib import Path

from nonergodic_memory.experiment import config_digest, load_config
from nonergodic_memory.mess3_rate_aware_clock import (
    _checked_checkpoint, _integer, _storage_selection, _validate_audit, _validate_grid,
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
    checkpoints = [rate_aware_checkpoint_path(args.checkpoint_dir, seed, rate, step)
                   for seed in config["rate_aware_clock"]["seeds"]
                   for rate in config["rate_aware_clock"]["learning_rates"]
                   for step in config["train"]["checkpoint_steps"]]
    targets = [*outputs, Path(args.config), *checkpoints,
               Path(args.checkpoint_dir) / "rate_aware_clock_attempts.jsonl"]
    for left, right in combinations(targets, 2):
        a, b = left.resolve(), right.resolve()
        if (a == b or a in b.parents or b in a.parents
                or (left.exists() and right.exists() and left.samefile(right))):
            raise ValueError(f"artifact paths alias or nest: {left} and {right}")
    for path in outputs:
        if path.exists() and not path.is_file():
            raise ValueError(f"output must be a file: {path}")
        for parent in path.parents:
            if parent.exists() and not parent.is_dir():
                raise ValueError(f"output parent must be a directory: {parent}")
    for directory in (Path(args.checkpoint_dir), Path(args.output_dir)):
        if directory.exists() and not directory.is_dir():
            raise ValueError(f"artifact root must be a directory: {directory}")
        if any(directory.resolve() == path.resolve() or path.resolve() in directory.resolve().parents
               for path in targets):
            raise ValueError("artifact directory aliases a file")


def _preflight(config: dict, args: argparse.Namespace) -> dict:
    """Check immutable evidence identity without treating science failure as recovery."""
    _validate_paths(config, args)
    rows = {kind: read_rate_aware_jsonl(path) if path.exists() else []
            for kind in ("training", "probe", "audit", "summary")
            for path in [Path(getattr(args, f"{kind}_results"))]}
    log = Path(args.checkpoint_dir) / "rate_aware_clock_attempts.jsonl"
    for event in read_rate_aware_jsonl(log) if log.exists() else []:
        if (event.get("record_type") != "rate_aware_clock_attempt"
                or event.get("base_config_sha256") != config_digest(config)
                or event.get("kind") not in ("training", "probe")
                or event.get("status") not in ("started", "failed", "completed")
                or event.get("seed") not in config["rate_aware_clock"]["seeds"]
                or event.get("learning_rate") not in config["rate_aware_clock"]["learning_rates"]):
            raise ValueError("incompatible attempt log")
    for kind in ("training", "probe"):
        _validate_trajectory_rows(config, rows[kind], kind)
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
        audit_identity = copy.deepcopy(rows["audit"])
        for row in audit_identity:
            if not isinstance(row.get("intersections"), dict):
                raise ValueError("audit intersections must be a mapping")
            for key, value in row["intersections"].items():
                if _integer(value) < 0:
                    raise ValueError("audit intersections must be nonnegative")
                row["intersections"][key] = 0
        _validate_audit(config, audit_identity)
    summaries = rows["summary"]
    if Path(args.summary_results).exists() and (len(summaries) != 1
            or summaries[0].get("record_type") != "rate_aware_clock_summary"
            or summaries[0].get("base_config_sha256") != config_digest(config)):
        raise ValueError("summary identity mismatch")
    if Path(args.audit_results).exists() and not rows["audit"]:
        raise ValueError("existing audit must contain one complete seed snapshot")
    return rows


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
        complete = False
        try:
            _validate_grid(config, rows["training"], rows["probe"])
            complete = True
        except ValueError:
            pass
        preserve = args.mode == "all" and complete and bool(rows["audit"])
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
            # Audits are one full-config snapshot even when training selects a subset.
            audit = audit_token_isolation(config, experiment["seeds"])
            atomic_write_rate_aware_jsonl(args.audit_results, audit)
        if args.mode in ("analyze", "all"):
            rows = {kind: read_rate_aware_jsonl(path) if Path(path).exists() else []
                    for kind, path in (("training", args.training_results),
                                       ("probe", args.probe_results), ("audit", args.audit_results))}
            summary = analyze_rate_aware_clock(config, rows["training"], rows["probe"], rows["audit"])
            atomic_write_rate_aware_jsonl(args.summary_results, [summary])
            print(f"rate-aware clock verdict: {summary['verdict']}")
        if args.mode in ("figures", "all"):
            from nonergodic_memory.mess3_rate_aware_clock_figures import generate_rate_aware_clock_figures
            training, probes, audit, summaries = [read_rate_aware_jsonl(path) for path in
                (args.training_results, args.probe_results, args.audit_results, args.summary_results)]
            if args.mode == "all":
                try:
                    _validate_grid(config, training, probes)
                except ValueError:
                    print("figures unavailable: raw grid is incomplete or invalid")
                    return
            for path in generate_rate_aware_clock_figures(config, training, probes, audit,
                                                          summaries[0], args.output_dir):
                print(f"generated {path}")
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
        raise SystemExit(f"rate-aware clock preflight or execution failed: {error}") from error


if __name__ == "__main__":
    main()
