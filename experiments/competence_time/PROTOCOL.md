# Competence–time dissociation in Mess3 geometry

Status: **preregistered design; no confirmation data generated**  
Parent release: [`v1.0`](https://github.com/stannum13/nonergodic-memory/releases/tag/v1.0)  
Branch: `experiment/competence-time-dissociation`

## Pre-data amendment — 2026-09-27

**Explicit diagnostic window.** The competence evaluation position window
inherited from v1.0 (`diagnosis.window: 8`) was omitted from the initial YAML.
It is now frozen explicitly in `configs/mess3_competence_time.yaml` and is
included in its configuration digest. No seed 30–37 checkpoint, raw result,
or figure existed when this amendment was made. This is a documentation and
configuration-lock correction, not a change to the planned evaluation window.

## Why this experiment exists

The v1.0 threshold experiment asked whether predictive competence explains the
emergence of component geometry better than optimizer step. Its registered,
all-checkpoint analysis failed: the leave-one-seed-out (LOSO) competence/step
MSE ratio was 2.378. A post-hoc analysis that excluded initialization reversed
the ordering (ratio 0.477 in favor of competence). That reversal is useful for
forming a hypothesis, but it is not confirmatory evidence.

This experiment tests the resulting two-regime hypothesis on new random seeds.
Initialization is specified in advance as a separate control regime. The
confirmatory analysis contains only checkpoints after at least one optimizer
update and deliberately varies learning rate so that predictive competence and
optimizer step do not remain interchangeable.

## Literature basis

The target nonergodic-composition work derives a factorization of prediction
into posterior mass over persistent source components and conditional belief
within a component, then reports the corresponding geometry in transformer
activations ([Ray, Riechers, and Shai, 2026](https://belief-updates.pub/nonergodic-geometry/)).
It extends the mixed-state account in which next-token models linearly encode
beliefs over hidden states ([Shai et al., NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8936fa1691764912d9519e1b5673ea66-Abstract-Conference.html)).

Three methodological lessons determine the present design:

1. Probe performance must be interpreted against a label control; otherwise a
   flexible probe can create apparent structure
   ([Hewitt and Liang, 2019](https://aclanthology.org/D19-1275/)).
2. Decodability alone is not behavioral or causal evidence. Projection-based
   erasure is a complementary test of whether decoded information is used
   ([Elazar et al., 2021](https://aclanthology.org/2021.tacl-1.10/)). The present
   experiment makes a geometric claim; causal use remains a separately labeled
   secondary analysis and cannot rescue the primary result.
3. Optimization time and held-out behavior can separate sharply on controlled
   synthetic problems ([Power et al., 2022](https://arxiv.org/abs/2201.02177)).
   Consequently, learning rate is manipulated and entire seeds—not individual
   checkpoints—are held out during model comparison.

The protocol is committed before confirmatory training. This preserves the
distinction between the exploratory v1.0 observation and the new confirmatory
test, following the basic rationale for preregistration
([Nosek et al., 2018](https://doi.org/10.1073/pnas.1708274114)).

## Formal hypothesis

For confirmation seed \(s\), learning rate \(\eta\), and checkpoint \(t>0\),
define predictive competence

\[
C_{s,\eta,t}
= 1 -
\frac{D_{\mathrm{KL}}(p^*_{s,t}\,\|\,q_{s,\eta,t})}
     {D_{\mathrm{KL}}(p^*_{s,t}\,\|\,u)},
\]

where \(p^*\) is the exact Bayesian Mess3 next-token predictor, \(q\) is the
transformer predictor, and \(u\) is uniform over the vocabulary. Let
\(G_{s,\eta,t}\) be held-out linear-probe \(R^2\) for the exact component
posterior at transformer block 2.

Two fixed-capacity models are compared:

\[
\mathcal M_C: G = \beta_0 + \beta_1 z(C) + \beta_2 z(C)^2,
\qquad
\mathcal M_T: G = \gamma_0 + \gamma_1 z(\log(1+t))
                         + \gamma_2 z(\log(1+t))^2.
\]

Standardization \(z(\cdot)\) is fitted using training seeds only. In each LOSO
fold, every rate and checkpoint from one seed is excluded from fitting. Define

\[
R = \frac{\operatorname{MSE}_{\mathrm{LOSO}}(\mathcal M_C)}
         {\operatorname{MSE}_{\mathrm{LOSO}}(\mathcal M_T)}.
\]

**Primary prediction.** The two-regime hypothesis is supported only if
\(R<0.80\), competence has lower fold MSE in at least 7 of 8 held-out seeds,
and all validity controls pass. Otherwise the prediction is falsified. If a
predeclared validity condition fails, the result is *inconclusive*, not
supported or falsified.

The effect-size threshold is unchanged from the v1.0 registration. Requiring
7/8 seed-level wins prevents a pooled result from being driven by one seed; an
exact one-sided sign test gives \(P(X\geq7)=9/256\) under equal win probability.
No null-hypothesis p-value is used as the primary decision rule.

## Experimental set

### Generator and model

- Exact published two-Mess3 mixture already implemented and tested in v1.0.
- Sequence length: 64 tokens.
- Model: decoder-only transformer, width 32, two blocks, four heads.
- Objective: next-token cross-entropy.
- Fresh independently generated training batches at every update.
- Weight decay 0.01 and batch size 64, unchanged from v1.0.

### Prospective grid

- Confirmation seeds: `30, 31, 32, 33, 34, 35, 36, 37`.
- Learning rates: `0.00075, 0.0015, 0.003, 0.006`.
- Checkpoints: `0, 384, 768, 1152, 1536, 2048, 2560, 3072, 4096`.
- Primary checkpoints: every listed checkpoint except step 0.
- Primary activation site: output of transformer block 2.
- Primary target: exact component posterior.

Within each seed, all four rates share the same initialization, step-indexed
training batches, evaluation sequences, probe-fit sequences, and probe-test
sequences. This pairing isolates the optimizer-rate manipulation. Different
seeds generate independent versions of all of those objects. Seeds 20–24 from
the exploratory threshold experiment are never included in confirmation.

The probe fit set contains 1,024 sequences and the probe test set contains 512.
They are disjoint from each other and from model evaluation sequences; exact
sequence-hash overlap must be zero. Hyperparameters and checkpoints are fixed
before observing any seed 30–37 result.

### Why this grid identifies the question

At a fixed step, changing learning rate changes how much predictive competence
has been acquired. At approximately matched competence, different rates reach
that level at different steps. The resulting crossed trajectories supply the
variation needed to distinguish a competence-indexed account from a clock-like
step-indexed account. Merely adding more seeds at the original two rates would
reduce variance but leave the central predictors too correlated.

## Controls and validity conditions

The following are mandatory:

1. **Shuffled labels:** the absolute component-posterior \(R^2\) must be at most
   0.02 in every primary grid cell.
2. **Untrained model:** step 0 is retained and plotted as a distinct control,
   but excluded from both confirmatory regressions.
3. **Seed grouping:** a fold holds out all four learning-rate trajectories for
   its seed.
4. **No probe leakage:** probe-fit/test sequence overlap is exactly zero.
5. **Complete grid:** no failed or divergent run is silently replaced. If any
   rate–seed cell has non-finite loss, competence, or geometry, the primary
   result is inconclusive; the failure is reported.
6. **Rate dissociation:** within at least six seeds, the fastest and slowest
   rates must differ in competence by at least 0.10 at one or more shared
   post-initialization checkpoints. Otherwise the manipulation check fails and
   the primary result is inconclusive.
7. **Immutable provenance:** raw rows carry the base configuration digest,
   rate-specific digest, seed, checkpoint, sampler, and checkpoint path.

The v1.0 random-subspace and norm-matched causal-erasure controls remain the
standard for any secondary intervention. They are not part of the primary
geometry decision and cannot change its verdict.

## Analysis hierarchy

### Confirmatory

Exactly one result is confirmatory: the LOSO comparison above on normal-control,
block-2, post-initialization component-posterior \(R^2\).

Possible verdicts are:

- `supported`: ratio below 0.80, at least 7/8 competence fold wins, all controls
  and the manipulation check pass;
- `falsified`: valid experiment, but either directional criterion fails;
- `inconclusive`: incomplete/non-finite grid, failed shuffled-label control,
  probe leakage, or failed rate-dissociation check.

### Secondary, direction-free

- Joint-belief and pairwise-distance \(R^2\).
- Block 1 and final normalization sites.
- Component classification accuracy.
- Conditional-state targets.
- Per-rate and per-seed learning curves.
- Descriptive competence-matched checkpoint pairs.
- Step-0 comparison with trained checkpoints.
- Causal component erasure at predeclared early and final checkpoints, if run.

These analyses are reported with effect sizes and uncertainty but have no
success criterion and cannot rescue a failed primary prediction.

## Stopping, exclusions, and amendments

- All 32 rate–seed trajectories are run through step 4,096 unless non-finite
  optimization makes a trajectory unusable.
- There is no accuracy-based early stopping and no seed replacement.
- Infrastructure failures may be rerun only with the identical configuration
  and seed; both the failure and rerun are logged.
- Any protocol change after this commit is an amendment with a timestamp,
  rationale, and commit hash. It applies prospectively and never silently
  overwrites this specification.
- No confirmatory result may be inspected before the protocol commit is pushed.

## Planned artifacts

- `configs/mess3_competence_time.yaml`
- `results/mess3_competence_time_training.jsonl`
- `results/mess3_competence_time_probes.jsonl`
- `results/mess3_competence_time_summary.jsonl`
- `figures/mess3_competence_time_learning.png`
- `figures/mess3_competence_time_loso.png`
- a result section appended to this directory after the frozen analysis runs
