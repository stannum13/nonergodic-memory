# Frozen competence–time confirmation result

The preregistered verdict is **`supported`**. The competence-only quadratic
has LOSO MSE 0.008194242898, compared with 0.017921418299 for the log-step-only
quadratic: a ratio of **0.457231830704**, below the frozen 0.80 threshold.
Competence wins in **8/8 held-out seeds**, exceeding the required 7/8. Every
registered validity condition passes; the frozen summary reports
`validity_failures: []`.

This supports the registered post-initialization comparison for this model,
generator, and grid. It does not establish causal use of decoded information,
universality across architectures, or an exact reproduction of the target
study's compute scale. The separate v1.0 all-checkpoint prediction remains
falsified. No secondary analysis changes either verdict.

## Frozen provenance and execution

- Public pre-data lock: `667d96213ff1140bd624276febfac9180c3fb3be`.
- Execution HEAD: `2c24040419ce263b69774ee73e8bb7b97ae95f56`, whose experiment
  configuration, training/probe code, analysis, and figure pipeline match the lock.
- Base configuration digest: `f75dd20f93eb8827`.
- Execution date: 2026-09-27, starting approximately 11:44:43 UTC.
- Command: `make competence-time`, measured with `/usr/bin/time -p`; console
  output was captured with `tee`.
- Wall time: **8,559.58 seconds (2 h 22 min 39.58 s)**; user CPU 7,669.11 s;
  system CPU 814.97 s. These times cover training, probes, analysis, and the
  original figure generation, excluding the subsequent audit and tests.
- Runtime: CPU, Python 3.14.2, NumPy 2.4.1, PyTorch 2.11.0; the committed
  trainer enables deterministic algorithms and one PyTorch thread.
- One uninterrupted run; zero infrastructure reruns, seed replacements,
  dropped trajectories, or changes to thresholds, checkpoints, or exclusions.

Complete captured command log:

```text
bash scripts/mess3_competence_time.sh
wrote competence-time training records
wrote competence-time probe records
wrote competence-time summary
generated figures/mess3_competence_time_learning.png
generated figures/mess3_competence_time_loso.png
validated complete competence-time grid
real 8559.58
user 7669.11
sys 814.97
```

## Complete grid and primary comparison

All 32 trajectories (seeds 30–37 × rates 0.00075/0.0015/0.003/0.006) reached
step 4,096. The retained raw files contain 288 training rows, 1,728 probe rows,
and one summary row. All 288 checkpoints remain in the ignored local
`checkpoints/mess3_competence_time/` directory; checkpoints are not committed.

The primary comparison uses exactly 256 normal-control, block-2,
post-initialization component-posterior R² observations. Each fold fits 224
observations from seven seeds and evaluates all 32 observations from the
remaining seed. Both quadratic models standardize their predictor using
training seeds only. The 32 step-0 training cells and their 192 probe rows
are retained as controls and excluded from the primary comparison.

| Held-out seed | Competence MSE | Log-step MSE | Competence wins |
| --- | ---: | ---: | --- |
| 30 | 0.014699227587 | 0.025375386321 | yes |
| 31 | 0.010042999052 | 0.025663498171 | yes |
| 32 | 0.007051815322 | 0.015143690213 | yes |
| 33 | 0.004706650900 | 0.010145801527 | yes |
| 34 | 0.001872824628 | 0.010888027654 | yes |
| 35 | 0.003345922454 | 0.012091495943 | yes |
| 36 | 0.018758546889 | 0.035019100480 | yes |
| 37 | 0.005075956349 | 0.009044346082 | yes |
| Pooled | 0.008194242898 | 0.017921418299 | 8/8 |

## Every registered validity condition

| Condition | Observed check | Status |
| --- | --- | --- |
| Shuffled labels | Maximum absolute block-2 component-posterior R² over all 256 primary shuffled cells is 0.011410204186, at most 0.02. | pass |
| Untrained control | All 32 step-0 cells retained, visibly separated in the learning figure, and excluded from both primary regressions. | pass |
| Seed grouping | Eight folds, each holding out all four rates and eight primary checkpoints for one seed; 224 fit / 32 test observations per fold. | pass |
| No probe leakage | Every one of 1,728 raw probe rows reports zero overlap; the supplemental exact-token audit below also finds zero overlap. | pass |
| Complete finite grid | Exact expected keys, no duplicate/missing/unexpected cells; all recorded training and probe metrics finite; all 288 checkpoint state dictionaries finite. | pass |
| Rate dissociation | Fastest–slowest competence difference reaches at least 0.10 in 8/8 seeds, exceeding the required 6/8. | pass |
| Immutable provenance | Every raw row has the locked base digest, correct rate digest, seed, step, vectorized sampler, CPU device, and matching checkpoint path; all checkpoint configurations and identities match. | pass |

The initial model tensors are also exactly equal across all four rates within
each of the eight seeds. Data pairing follows the unchanged step-indexed
sampler and evaluation/probe seed offsets in the locked implementation.
Recomputing the frozen analysis from the retained raw rows reproduces the
entire saved summary exactly.

The manipulation check takes each seed's maximum absolute competence
difference between learning rates 0.006 and 0.00075 over shared registered
post-initialization checkpoints:

| Seed | Maximum competence difference |
| --- | ---: |
| 30 | 0.655654538500 |
| 31 | 0.681725081631 |
| 32 | 0.785649827456 |
| 33 | 0.611784259660 |
| 34 | 0.713542606609 |
| 35 | 0.710190830436 |
| 36 | 0.671703732523 |
| 37 | 0.709745862904 |

### Supplemental exact-sequence audit

The frozen `probe_sequence_overlap` field compares disjoint sequence-ID
namespaces. To verify the protocol's stronger exact-sequence requirement,
the post-run audit regenerated evaluation, probe-fit, and probe-test inputs
using the unchanged vectorized sampler, configuration, and seed offsets
`seed + 202`, `seed + 404`, and `seed + 505`. Each 64-token sequence was
encoded as little-endian int64 bytes and hashed with SHA-256.

For every seed 30–37, all 256 evaluation sequences, 1,024 probe-fit sequences,
and 512 probe-test sequences were unique within their respective sets. Each
of the three within-seed intersections—fit/test, evaluation/fit, and
evaluation/test—contained **zero** hashes. Thus all 24 pairwise checks pass
across 14,336 regenerated sequences. The four rates use these same seed-paired
sets. This audit changes neither the frozen analysis nor its verdict.

## Retained artifacts and figures

- [Training rows](../../results/mess3_competence_time_training.jsonl)
- [Probe rows](../../results/mess3_competence_time_probes.jsonl)
- [Frozen summary](../../results/mess3_competence_time_summary.jsonl)
- [Learning curves](../../figures/mess3_competence_time_learning.png)
- [LOSO comparison](../../figures/mess3_competence_time_loso.png)

Full-file SHA-256 checksums:

```text
training  2c90e72389db3a98e0c1196fffaf6bdf24f3492009460bfbe0c99f417315420f
probes    9a7aa242f267bddc2064c8ddf93c3163891167630d7dd3f0ea4e603993e3abd3
summary   0cc55ffba7a37f161f845b750a6ce1484e88904ac2499d08205b379c9d3fc029
```

Both figures were generated solely by the committed pipeline from these raw
rows and the matching frozen summary. The learning figure's seed legend
overlaps some x-axis labels, and the negative initialization range compresses
the trained competence curves. Those presentation limitations are retained
without editing the frozen plotting code. The LOSO figure is readable and
displays the registered ratio and supported verdict.

## Repository verification

`pytest -q`: **171 passed in 89.71 seconds** (92.64 s command wall time).
`make smoke`: **passed**, exit 0, in 11.83 s command wall time. It exercised
training, probing, interventions, and the existing repository figure pipeline.
The two unrelated smoke rewrites (`results/smoke_training.jsonl` and
`results/smoke_reproduction.jsonl`) were restored to their pre-run committed
contents. No v1.0 result or figure change is included.

Verification log tails:

```text
171 passed in 89.71s (0:01:29)
real 92.64
user 79.56
sys 9.34

generated figures/training.png, figures/probes.png, figures/pca.png, figures/intervention.png
real 11.83
user 9.81
sys 1.15
```

## Post-result maintenance

After freezing this result at `9f3c264`, maintenance added output-compatibility
checks before checkpoint writes, atomic persistence of each completed
rate–seed trajectory, and checks for fresh-condition canonical training paths.
The CLI now lets the existing analyzer record scientific validity failures
as inconclusive, and figure generation checks the supplied summary against
analysis recomputed from raw rows. These safeguards do not change the
registered decision rules or the frozen result. The three retained JSONLs,
both figures, configuration, protocol, and v1.0 artifacts remain unchanged;
reanalysis still reproduces the complete saved summary exactly.

Maintenance verification: 100 focused tests passed, the full suite passed
212 tests, and `make smoke` exited successfully. Its two smoke JSONL rewrites
were restored. SHA-256 checks for all three frozen JSONLs and both figures
match their pre-maintenance values.
