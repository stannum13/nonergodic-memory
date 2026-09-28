# Rate-aware clock external validation

Status: **preregistered design; no seeds 40–47 data generated**

Parent result: `experiments/competence_time/RESULTS.md`

Branch: `experiment/rate-aware-clock`

## Why this experiment exists

The competence–time experiment prospectively established that a quadratic
model of predictive competence forecasts post-initialization block-2 component
geometry better than an equal-capacity quadratic model of rate-unaware
`log1p(step)`.
Its registered ratio was 0.457 and competence won all eight grouped
leave-one-seed-out folds. That result remains valid for the comparator that was
registered.

After the result was frozen, an expert methods audit identified a stronger
alternative clock. On the retained seeds 30–37, an equal-capacity quadratic in
`log1p(learning_rate * step)` has grouped LOSO MSE 0.006086, compared with
0.008194 for competence, and wins six of eight seed folds. A six-coefficient
rate/step surface performs still better, but it has twice the capacity and is
not the primary comparator here.

Those calculations are exploratory and cannot alter the completed verdict.
They motivate this independent external validation on new seeds. The question
is no longer whether competence beats raw step. It is whether a rate-aware
optimization clock transports better than competence to new training runs.

## Literature and methodological basis

The experiment remains grounded in the nonergodic source/conditional-state
factorization of [Ray, Riechers, and Shai](https://belief-updates.pub/nonergodic-geometry/)
and the exact belief-state account of transformer activations developed by
[Shai et al.](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8936fa1691764912d9519e1b5673ea66-Abstract-Conference.html).
Controlled synthetic training can exhibit substantial separation between
optimization time and held-out behavior, as emphasized by
[Power et al.](https://arxiv.org/abs/2201.02177). Consequently, neither update
count nor behavioral competence is assumed to be the privileged developmental
coordinate.

The old-data comparison selected the new hypothesis. All coefficients,
standardization constants, outcomes, exclusions, and validity rules below are
therefore fixed before new-seed data, following the confirmatory/exploratory
separation advocated by [Nosek et al.](https://doi.org/10.1073/pnas.1708274114).

## Frozen forecasts

Predictive competence is fixed to the existing definition

\[
C=1-\frac{\operatorname{mean}_{i,t}D_{KL}(p^*_{it}\|q_{it})}
              {\operatorname{mean}_{i,t}D_{KL}(p^*_{it}\|u)},
\]

using zero-based prediction positions 7–62 for the independent 256-sequence
evaluation set at length 64 (`diagnosis.window=8`). The ratio is taken after
both KL quantities are averaged; it is not a mean of per-position ratios. The
primary geometry probe continues to pool zero-based input positions 0–62 from
the independent probe sets, uses the existing StandardScaler followed by
multi-output Ridge regression with `alpha=1.0`, and reports the existing
uniform-average multi-output R² (the scikit-learn default). These measurement
definitions are inherited unchanged from commit
`2219d3737b9913a2e6e4995e96440279f9545716`.

The population mismatch is retained for direct comparability; the
aligned-position analysis is secondary and cannot change the verdict.

Let `G` be normal-control block-2 linear-probe R² for the exact component
posterior at a post-initialization checkpoint. Two three-coefficient forecasts
are fitted once using all 256 eligible rows from seeds 30–37, four learning
rates, and eight post-initialization checkpoints in the completed experiment.
They are never refitted on seeds 40–47.

For predictive competence `C`,

\[
z_C = \frac{C-0.4428598689138331}{0.3380605976966924},
\]

\[
\widehat G_C =
0.03302589694739769
+0.14699110421172318 z_C
+0.09640601399613305 z_C^2.
\]

For learning rate `η` and optimizer step `t`, define the rate-aware clock

\[
R=\log(1+\eta t),\qquad
z_R=\frac{R-1.5596899070965387}{0.7723241912698406},
\]

\[
\widehat G_R =
0.08426303744815684
+0.13373197023956584 z_R
+0.045168873495373824 z_R^2.
\]

Both forecasts return their raw polynomial values without clipping or
calibration. The old-data in-sample MSEs are 0.00761248 and 0.00543398
respectively.
They are descriptive only. Forecast provenance is locked to:

- training JSONL SHA-256:
  `2c90e72389db3a98e0c1196fffaf6bdf24f3492009460bfbe0c99f417315420f`;
- probe JSONL SHA-256:
  `9a7aa242f267bddc2064c8ddf93c3163891167630d7dd3f0ea4e603993e3abd3`;
- old seeds 30–37;
- old post-initialization steps 384/768/1152/1536/2048/2560/3072/4096;
- old rates 0.00075/0.0015/0.003/0.006;
- target `block_2`, control `none`, metric `component_posterior_r2`.

Configuration constants and source hashes must match exactly as serialized.
The independent numerical refit must reproduce centers, population standard
deviations, and coefficients within absolute tolerance `1e-12`.

## Primary hypothesis and verdict

For each new seed `s`, calculate mean squared forecast error across its 14
primary cells: two learning rates × seven post-initialization checkpoints.
Each seed receives equal weight in the aggregate regardless of token or
checkpoint count.

Define

\[
Q=\frac{\frac18\sum_s \operatorname{MSE}_s(\widehat G_R)}
        {\frac18\sum_s \operatorname{MSE}_s(\widehat G_C)}.
\]

If the aggregate competence MSE in the denominator is zero, serialize `Q` as
JSON `null`, add `ratio_reason: "zero_competence_mse"`, and classify the valid
experiment as `falsified`, including when both forecasts are perfect. The
operational effect-size check is the strict inequality
`MSE_clock < 0.80 * MSE_competence`, so an undefined ratio can never count as
support.

The primary prediction is that the rate-aware clock transports better as a
predictor of contemporaneous geometry than competence. Even a supported result
would not show that `learning_rate * step` is the causal optimization mechanism
or that the decoded geometry is causally used by the network.

- `supported`: `Q < 0.80`, the rate-aware forecast has lower MSE in at least
  7/8 seeds, and every validity check passes.
- `falsified`: the experiment is valid, but either directional criterion
  fails.
- `inconclusive`: any predeclared validity check fails.

The symmetric competence/rate ratio and competence seed wins are reported, but
cannot redefine the primary hypothesis after inspection. Failure does not
establish equivalence or universal competence superiority.

Seed wins are descriptive clustered replications, not an exact independent
LOSO sign test. Unlike the earlier overlapping LOSO folds, the new seed
outcomes are conditionally independent given the frozen old-data forecasts,
but eight seeds still provide limited power. Seven-of-eight wins alone has
50.3% power if the true clock-win probability is 0.8 and 81.3% if it is 0.9;
the additional 20% aggregate-effect rule lowers joint power further. This is a
small external falsification/validation study, not strong population-level
assurance. No p-value is a success rule.

## Prospective experimental set

### Generator and model

- Exact two-Mess3 mixture used by the completed experiments.
- Vectorized sampler and sequence length 64.
- Decoder-only transformer: width 32, two blocks, four heads, maximum length
  128.
- Fresh batch of 64 sequences per update.
- Weight decay 0.01.
- Diagnostic window 8.
- Probe fit/test sets: 1,024/512 sequences.
- Model evaluation set: 256 sequences.

### Confirmation grid

- Seeds: `40, 41, 42, 43, 44, 45, 46, 47`.
- Learning rates: `0.003, 0.006`.
- Checkpoints: `0, 384, 768, 1152, 1536, 2048, 2560, 3072`.
- Primary cells exclude step 0.
- Primary target/site/control:
  `component_posterior_r2` / `block_2` / `none`.

This produces 16 trajectories, 128 training rows, 768 probe rows, and 112
primary forecast observations. Within a seed, both rates share initialization,
step-indexed fresh batches, evaluation data, probe-fit data, and probe-test
data. Different seeds generate independent versions of those objects.

## Validity conditions

All conditions below must pass:

1. **Complete finite grid:** every registered training/probe cell exists once;
   all required metrics and model parameters are finite. No seed replacement.
2. **Forecast provenance:** source hashes, coefficients, transformations,
   training rows, and target identity exactly match the frozen specification.
   Confirmation geometry is never used to fit or select a forecast.
3. **Shuffled-label control:** absolute shuffled component-posterior R² is at
   most 0.02 in every primary cell.
4. **Actual token isolation:** SHA-256 hashes of token arrays—not only sequence
   ID namespaces—show zero intersection among evaluation, probe-fit, and
   probe-test sets for every seed. Dataset hashes and intersection counts are
   stored in an audit JSONL.
5. **Rate manipulation:** in at least 6/8 seeds, rates 0.003 and 0.006 differ in
   competence by at least 0.10 at one or more shared post-initialization steps.
   Applied retrospectively to the old grid, this exact check passes 7/8 seeds;
   seed 33 is near the boundary and seed 36 fails. An inconclusive confirmation
   outcome is therefore plausible and the rule will not be relaxed.
6. **Forecast support:** every **primary post-initialization** confirmation
   competence and rate-clock value is within the frozen old-data ranges
   `[-0.24055542481822356,
   0.8940463808722273]` and `[0.2530906276821619,
   3.2416544117575405]`. No extrapolation is interpreted confirmatorily.
7. **Pairing and identity:** step-0 parameters match exactly across rates within
   seed; raw rows record fresh condition, canonical checkpoint path, sampler,
   base digest, and rate-specific digest.
8. **No adaptive recovery:** infrastructure retries use the identical seed and
   configuration and are logged. Nonfinite optimization, missing primary rows,
   or an irrecoverable run makes the experiment inconclusive.

The rate-manipulation and forecast-support checks apply only to the registered
post-initialization cells. Step zero is retained as an untrained control and
excluded from both forecast errors. It cannot be added after inspection.

## Secondary analyses

The following are exploratory, direction-free descriptions and cannot rescue
the primary verdict:

- the original raw-step forecast comparison;
- joint-belief, conditional-state, component-accuracy, and distance metrics;
- block 1 and final-normalization sites;
- fixed-step contrasts discussed as observed, without data-dependent pair
  selection being treated as an inferential test;
- predictor correlations and residuals by rate and checkpoint;
- position-aligned competence/geometry sensitivity analysis;
- step-zero descriptive controls.

No causal intervention is part of this experiment. It is a separately bounded
research question with its own feasibility and confirmation protocol.

Both models forecast geometry at previously unseen seeds, but the competence
forecast consumes contemporaneously measured checkpoint behavior while the
rate-aware clock consumes predetermined training metadata. Success transports
only to new random realizations of this selected Mess3 architecture, rates, and
steps—not to unseen rates, architectures, or generators. Neither forecast
comparison identifies a causal mechanism or makes competence causally
irrelevant.

## Stopping and program boundary

- Run all 16 trajectories to step 3,072 and retain all registered checkpoints.
- Do not stop on competence, probe performance, or apparent forecast error.
- Do not add seeds, rates, steps, model families, predictors, or thresholds
  after confirmation data are observed.
- Report and publish the predetermined verdict regardless of direction.
- After this experiment, one separately bounded causal-predictive-memory pilot
  may begin only after its own protocol commits a numerical compute budget,
  development/confirmation data allocation, intervention-validity criteria,
  and stopping rule. Pilot outcomes remain exploratory and can never be
  relabeled confirmation. If that pilot cannot validate its intervention
  within its fixed budget, the autonomous research program stops and reports
  the limitation.

## Planned artifacts

- `configs/mess3_rate_aware_clock.yaml`
- `src/nonergodic_memory/mess3_rate_aware_clock.py`
- `src/nonergodic_memory/mess3_rate_aware_clock_figures.py`
- `src/mess3_rate_aware_clock.py`
- `scripts/mess3_rate_aware_clock.sh`
- `results/mess3_rate_aware_clock_training.jsonl`
- `results/mess3_rate_aware_clock_probes.jsonl`
- `results/mess3_rate_aware_clock_audit.jsonl`
- `results/mess3_rate_aware_clock_summary.jsonl`
- `figures/mess3_rate_aware_clock_learning.png`
- `figures/mess3_rate_aware_clock_forecasts.png`
- `experiments/rate_aware_clock/RESULTS.md`
