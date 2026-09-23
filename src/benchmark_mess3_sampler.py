#!/usr/bin/env python3
"""Benchmark the reference and vectorized Mess3 samplers."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from nonergodic_memory.data import HMMMixture, make_mess3_mixture
from nonergodic_memory.experiment import runtime_provenance, write_jsonl


def benchmark_sampler(
    mixture: HMMMixture,
    sequences: int,
    length: int,
    repeats: int,
) -> dict:
    """Return median wall times for equal-sized exact sampling calls."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    mixture.sample(sequences, length, seed=0)
    mixture.sample_vectorized(sequences, length, seed=0)
    timings: dict[str, list[float]] = {"reference": [], "vectorized": []}
    for repeat in range(repeats):
        for name, sampler in (
            ("reference", mixture.sample),
            ("vectorized", mixture.sample_vectorized),
        ):
            started = time.perf_counter()
            sampler(sequences, length, seed=10_000 + repeat)
            timings[name].append(time.perf_counter() - started)
    reference = statistics.median(timings["reference"])
    vectorized = statistics.median(timings["vectorized"])
    return {
        "record_type": "mess3_sampler_benchmark",
        "samplers": ["reference", "vectorized"],
        "sequences": sequences,
        "length": length,
        "repeats": repeats,
        "reference_median_seconds": reference,
        "vectorized_median_seconds": vectorized,
        "speedup": reference / vectorized,
        **runtime_provenance(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", type=int, default=64)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", default="results/mess3_sampler_benchmark.jsonl")
    args = parser.parse_args()
    row = benchmark_sampler(make_mess3_mixture(), args.sequences, args.length, args.repeats)
    write_jsonl(Path(args.output), [row])
    print(
        f"reference={row['reference_median_seconds']:.6f}s "
        f"vectorized={row['vectorized_median_seconds']:.6f}s "
        f"speedup={row['speedup']:.2f}x"
    )


if __name__ == "__main__":
    main()
