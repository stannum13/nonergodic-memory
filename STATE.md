# Experimental state

## Hypothesis

For the exact published two-Mess3 mixture, component geometry emerges when the model becomes predictively competent, rather than at a fixed optimizer step. Across paired fresh-data learning curves, predictive competence should therefore explain held-out block-2 component-posterior R² better than update count does.

## Last experiment

Registered Mess3 geometry-threshold test, committed at `2bd1072` before any threshold checkpoint, result, or figure. Confirmation seeds 20–24 were trained on paired fresh batches at learning rates 0.003/0.0015 and probed at seven checkpoints from initialization through 3,072 updates. The primary leave-one-seed-out comparison predicts block-2 component-posterior R² from either competence or `log1p(step)`.

## Result

- The registered primary prediction fails in every held-out seed. Competence-only LOSO MSE is 0.013170 versus 0.005537 for log-step, giving a ratio of 2.378 rather than the predicted `< 0.80`.
- Shuffled-label component-posterior R² remains within [−0.01074, 0.01069], satisfying the registered ±0.02 interpretability control. Probe sequence overlap is zero.
- At learning rate 0.003, mean block-2 component R² grows from 0.005 at step 768 to 0.363 at step 3,072; at 0.0015 it grows from −0.002 to 0.238. Predictive competence at the final checkpoint is 0.850/0.814.
- A post-hoc sensitivity analysis excluding initialization reverses the comparison in every seed: competence/log-step LOSO MSE ratio is 0.477. This was not registered and does not rescue the primary result.
- The complete grid contains 70 training and 420 probe records. It completed in 30:57 wall time after reusing one 136.65-second pilot.

## Interpretation

The proposed global competence threshold is falsified as specified. Across initialization and training, optimizer step generalizes better to held-out seeds than predictive competence. However, initialization occupies an extreme competence range (mean about −11.8), while trained checkpoints lie near 0–0.85; one quadratic across both regimes is scale-sensitive. The post-hoc reversal after removing initialization suggests a two-regime account: prediction becomes nontrivial first, then component geometry grows with competence during training. That account is a new hypothesis, not a confirmed reinterpretation. The experiment remains far below the target study's architecture and compute budget.

## Next smallest experiment

Do not weaken the failed global criterion. The next smallest falsification is a new-seed preregistered two-regime analysis: treat initialization as a distinct categorical regime and compare competence versus step only among post-initialization checkpoints, using a model family fixed before examining new seeds. No additional training should begin until that specification is committed.

## Competence–time dissociation: implementation lock

This is a new preregistered confirmation experiment, not a revision of the
completed v1.0 threshold loop above. The complete protocol, configuration,
implementation, tests, and audit-figure code were locked at commit
`667d96213ff1140bd624276febfac9180c3fb3be` (`667d962`) before any confirmation
data generation. Its base configuration digest, calculated with the
repository's `config_digest(load_config(...))`, is `f75dd20f93eb8827`.

The prospective grid is seeds 30–37 across the four preregistered learning
rates. At the lock, the repository tree contained no
`checkpoints/mess3_competence_time/` checkpoint, no
`results/mess3_competence_time_{training,probes,summary}.jsonl` result row, and
no `figures/mess3_competence_time_{learning,loso}.png` figure. This lock does
not alter the v1.0 result: v1.0 remains falsified as recorded above. At the
lock, the new experiment had no generated confirmation data or verdict.

## Competence–time dissociation: frozen confirmation result

The new experiment completed on 2026-09-27 with the frozen verdict
**supported**. Competence-only LOSO MSE is 0.008194242898 versus 0.017921418299
for log-step, giving a ratio of **0.457231830704 < 0.80**. Competence wins in
**8/8 held-out seeds**, exceeding the registered 7/8 requirement. The
comparison contains exactly 256 post-initialization normal-control block-2
component-posterior R² observations; each fold trains on 224 observations and
holds out all 32 observations from one seed.

Every registered validity check passes:

- Shuffled-label control: maximum absolute primary component-posterior R²
  is 0.011410204186, within the frozen 0.02 limit in every primary cell.
- Initialization: all 32 step-0 training cells and 192 associated probe rows
  are retained as controls and excluded from both primary regressions.
- Seed grouping: all four rates and all eight primary checkpoints for each
  held-out seed stay together; standardization uses training seeds only.
- No probe leakage: all 1,728 raw overlap fields are zero. A supplemental
  deterministic SHA-256 audit of the actual 64-token sequences also finds
  zero fit/test, evaluation/fit, and evaluation/test intersections in every
  seed (24 pairwise checks; 14,336 regenerated sequences).
- Complete finite grid: 288 training rows, 1,728 probe rows, one summary,
  and all 288 checkpoints; no missing or duplicate cells, non-finite recorded
  metrics, or non-finite checkpoint tensors.
- Rate dissociation: 8/8 seeds exceed the 0.10 fastest–slowest competence
  difference at a shared checkpoint, meeting the required 6/8. Per-seed
  maximum differences range from 0.611784259660 to 0.785649827456.
- Provenance: every raw row and checkpoint matches the frozen base/rate
  configuration, identity, CPU device, sampler, and checkpoint path. All
  four same-seed initializations match exactly. Reanalysis reproduces the
  saved summary exactly; `validity_failures` is empty.

The unchanged `make competence-time` run took 8,559.58 seconds
(2 h 22 min 39.58 s) on CPU, with no interruption, rerun, replacement,
post-data tuning, or additional exclusions. Raw JSONLs, the exact log,
per-fold metrics, every validity check, and figure limitations are recorded
in [the experiment results](experiments/competence_time/RESULTS.md).

This supports the preregistered post-initialization comparison for the tested
grid. It does not establish causal use or generalization across architectures.
The earlier v1.0 all-checkpoint prediction remains falsified; no secondary
analysis rescues or revises it.

Post-result maintenance after `9f3c264` hardens output compatibility,
interruption recovery, invalid-data summary handling, and figure/raw-data
consistency. It leaves the registered decision rules and all frozen result
artifacts unchanged; see the post-result maintenance note in
[RESULTS.md](experiments/competence_time/RESULTS.md).

## Rate-aware clock external validation: configuration lock

This is a separately preregistered external validation on new seeds 40–47,
not a revision of the completed competence–time verdict. The pre-data primary
hypothesis is that the frozen quadratic rate-aware clock
`log1p(learning_rate * step)` forecasts post-initialization normal-control
block-2 component-posterior R² better than the frozen quadratic predictive
competence forecast: aggregate clock/competence MSE must be strictly below
0.80 and the clock must have lower MSE in at least 7/8 seeds. All validity
conditions, including the 0.02 shuffled-label bound, a 0.10 rate-dissociation
in at least six seeds, and frozen forecast-support bounds, are fixed before
any confirmation data are generated.

The configuration digest is `59f938bbff48f300`. Its competence and rate-aware
forecast constants, centers, scales, coefficients, and support ranges are
frozen from retained seeds 30–37, rates 0.00075/0.0015/0.003/0.006, and
post-initialization steps 384/768/1152/1536/2048/2560/3072/4096. Provenance is
locked to training JSONL SHA-256
`2c90e72389db3a98e0c1196fffaf6bdf24f3492009460bfbe0c99f417315420f` and probe
JSONL SHA-256
`9a7aa242f267bddc2064c8ddf93c3163891167630d7dd3f0ea4e603993e3abd3`; neither
forecast may be refit or selected using confirmation geometry.

At this configuration lock, no checkpoint, training/probe/audit/summary JSONL
record, or figure exists for the rate-aware-clock experiment, and no seed
40–47 data have been generated.

## Rate-aware clock external validation: implementation lock

The final approved code-lock commit is
`933f50fc22e72ce4f264b811fc09e039bfe10cad` (`933f50fc`), before any confirmation
execution. The documentation commit titled
`docs: approve rate-aware implementation lock` records this boundary,
superseding the initial lock recorded in `5f02d75`. Independent methods,
integrity, and code reviewers all **APPROVE**, with no remaining Critical or
Important findings. Publication of the approved locked branch remains pending
and must be recorded before seeds 40–47 are executed. No confirmation result
or verdict exists.

Council findings were repaired in `ec368508` and `933f50fc`, then independently
re-reviewed. The repairs bind probe rows to checkpoint parameters, verify cached
audits against deterministic token reconstruction, reject output aliases of
protected evidence, publish figures atomically, preserve scientific failure
evidence through analysis, and serialize derived ratios safely. Follow-up
regressions enforce outcome-independent completion of every untouched
registered trajectory, completion of a missing fixed audit for intact finite
grids, and propagation of terminal execution failure into the verdict and
figure reanalysis. The scientific rules, forecast constants, and grid remain
unchanged.

Final repair verification and fresh lock checks:

- `PYTHONPATH=src pytest -q`: **470 passed in 263.67s (0:04:23)**, exit 0,
  on the final repaired implementation. The covering resumption, failure
  evidence, audit-completion, overlap, and cache regressions passed **60 tests**
  (224 deselected) in **122.81s (0:02:02)**, exit 0.
- `make smoke`: exit 0 at `ec368508`; training, probing, interventions, and
  four legacy figures completed. Unrelated generated rewrites of
  `results/smoke_training.jsonl` and `results/smoke_reproduction.jsonl` were
  inspected and restored. The `933f50fc` follow-up changed only rate-aware
  orchestration, validation, analysis/figure evidence handling, and associated
  tests; numerical producers and generic smoke paths were unchanged, so smoke
  was not repeated. The tree was clean before the final lock-documentation
  edits.
- `config_digest(load_config("configs/mess3_rate_aware_clock.yaml"))`:
  **`59f938bbff48f300`**, unchanged from the configuration lock.
- `verify_forecast_provenance(config, Path("."))`: exit 0; all **256** eligible
  retained source cells reproduce the frozen centers, population scales,
  coefficients, and support bounds within absolute tolerance `1e-12`.
  Training SHA-256 is
  `2c90e72389db3a98e0c1196fffaf6bdf24f3492009460bfbe0c99f417315420f`;
  probe SHA-256 is
  `9a7aa242f267bddc2064c8ddf93c3163891167630d7dd3f0ea4e603993e3abd3`.

The pre-execution scan enumerated the worktree including ignored files and
every tree in the 11-commit history from preregistration `37d5201` through
`933f50fc`, inclusive. The worktree contained 580 checkpoint/JSONL/image
artifacts, including 513 checkpoints with identifiable nonconfirmation seeds
and 44 JSONLs containing 74,576 rows. Historical trees contained 737 artifact
entries and 44 distinct JSONL blobs (74,576 rows). Artifact paths were checked
for the rate-aware experiment or seeds 40–47; every JSONL row was parsed and
checked recursively for confirmation-seed identity and rate-aware identity.
There were **zero matching checkpoint, JSONL, or figure artifacts** in either
scope. Synthetic in-memory test fixtures are not generated confirmation data.

`git ls-files .superpowers` and `git log 933f50fc -- .superpowers` both returned
no entries: neither the tracked tree nor this branch's reachable history
contains private `.superpowers` content. A broader `git log --all` check found
four legacy commits on other local refs; those are outside this branch's
history and were not modified. The frozen competence-time source JSONLs,
summary, figures, result narrative, and rate-aware protocol are unchanged
since preregistration, and the YAML is unchanged since `3bf4a83`. Fresh SHA-256
and byte comparisons of these eight protected files match those baselines.

Remaining implementation limitations are explicit: execution assumes one
writer; orphan checkpoints without a completed raw trajectory require explicit
recovery and are never overwritten automatically; alias validation can cover
only supplied or otherwise known result paths; and figure publication is
atomic per image, not across both images. None changes the scientific rules
or permits adaptive recovery.

The remaining publication step and the unstarted execution checklist are tracked in
[IMPLEMENTATION.md](experiments/rate_aware_clock/IMPLEMENTATION.md).

## Registered Mess3 geometry-threshold prediction

Registered before creating any `checkpoints/mess3_threshold/`, `results/mess3_threshold_*.jsonl`, or `figures/mess3_threshold_*.png` artifact. Configuration digest `d2423ea9f3b44075` uses confirmation seeds 20–24, learning rates 0.003 and 0.0015, fresh vectorized samples, and checkpoints 0/384/768/1,152/1,536/2,304/3,072. Same-seed rate conditions share initialization, the seed-indexed fresh batch at every step, held-out evaluation data, probe-fit data, and probe-test data.

The primary prediction is: a quadratic competence-only regression will have at least 20% lower leave-one-seed-out MSE for normal-control block-2 component-posterior R² than a quadratic `log1p(step)`-only regression. Equivalently, `MSE_competence / MSE_step < 0.80`. Each fold holds out both learning-rate trajectories for one seed. Shuffled-label component-posterior R² must remain within ±0.02 in every cell for the primary result to be interpretable. Onset locations, joint-belief R², pairwise-distance R², and other activation sites are secondary with no directional success criterion.

Seeds 20–24 and the 20% criterion are confirmatory and cannot be replaced or weakened after inspection. A positive raw correlation, a ratio between 0.80 and 1.00, or an improvement confined to a subset of folds does not satisfy the registered prediction.

## Registered Mess3 fidelity prediction

For the exact published two-Mess3 source, trained Transformer joint-belief R² will exceed its same-seed untrained control in all three seeds. Pairwise-distance R² is secondary. The experiment is a direct data/process reproduction but not an exact compute reproduction: width 32, two layers, absolute positions, LayerNorm, and CPU training differ from the published width-128 four-layer TransformerLens model with rotary positions, RMSNorm, gated GELU, and 45,000 optimization steps.

## Registered exploratory training-diversity prediction

Registered before creating any diagnosis checkpoint, JSONL record, or figure. Configuration digest `aa2de784730aa352` compares `reused` and `fresh` sequence conditions for exploratory seeds 10 and 11 at steps 0, 768, and 3,072. Both conditions use the same initialized width-32 two-layer Transformer, sequence length 64, batch size 64, AdamW settings, and supervised tokens per update. The reused condition traverses a deterministic fixed pool of 2,048 sequences; the fresh condition samples a new batch of 64 sequences at every update.

The primary prediction is: at step 3,072, fresh-data training has lower held-out exact-predictive KL than reused-data training in both seeds 10 and 11. Layerwise component-posterior, conditional-state, and six-coordinate weighted joint-belief recovery are secondary outcomes with no directional success criterion. Seeds 10 and 11 are exploratory and cannot be reused for later confirmation.
