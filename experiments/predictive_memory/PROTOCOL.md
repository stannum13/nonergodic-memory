# Predictive-memory causal pilot: locked protocol

Status: preregistered before implementation and before any pilot result was generated.

## Question and scope

In a two-component Mess3 mixture, does a trained transformer's retained prefix
memory support a causal, component-level belief edit whose downstream response
matches the exact Bayesian counterfactual after one additional observed token?

This is a bounded pilot, not a confirmatory study. It does not claim the first
activation intervention on an HMM belief state. Closely related work already
establishes linear belief-state geometry, state forcing, and steering in HMM
settings. The narrower contribution tested here is **counterfactual persistence**:
edit the source posterior in the stored prefix, append a real token, and compare
the model's conditional second-token response with an analytic Bayesian target.

Relevant boundaries are Ray, Riechers, and Shai, *The Geometry of Nonergodic
Composition* (2026); Shai et al., *Transformers learn factored representations*
(arXiv:2602.02385); Balcells et al., *Large Language Models Develop Belief State
Geometry In-Context* (arXiv:2609.17376); and *Markovian Circuit Tracing for
Transformer State Dynamic* (arXiv:2605.20824).

## Exact counterfactual

After a length-32 prefix, let the exact joint belief be

\[
q(c,s)=w(c)r(s\mid c), \qquad c\in\{0,1\},\ s\in\{0,1,2\}.
\]

For a registered log-odds dose \(\lambda\), define

\[
w_\lambda(1)=\frac{e^\lambda w(1)}{w(0)+e^\lambda w(1)},\qquad
w_\lambda(0)=1-w_\lambda(1),
\]

and \(q_\lambda(c,s)=w_\lambda(c)r(s\mid c)\). Thus the edit changes
the component posterior while exactly preserving both normalized within-component
state posteriors. Primary doses are \(\lambda=\pm\log 2\), which double or halve
the posterior odds. Dose zero is an identity check.

For each possible next token \(a\), exact HMM operators produce
\(P_q(a)\), \(P_q(b\mid a)\), and \(P_{q_\lambda}(b\mid a)\). The analytic target
is

\[
d_*(b\mid a)=P_{q_\lambda}(b\mid a)-P_q(b\mid a).
\]

The model target \(d_M(b\mid a)\) is the difference between edited and baseline
model predictions after appending \(a\). Branches are weighted by the unedited
exact \(P_q(a)\). The primary score is

\[
S=1-\frac{\sum_a P_q(a)\lVert d_M(\cdot\mid a)-d_*(\cdot\mid a)\rVert_2^2}
{\sum_a P_q(a)\lVert d_*(\cdot\mid a)\rVert_2^2}.
\]

Zero means no better than predicting no conditional response; one is exact.
Negative scores are permitted and retained.

## Intervention site and actuator

Models are the existing width-32, two-block, learning-rate 0.003 Mess3
transformers. No network is trained for this pilot. The intervention captures
the complete 32-position activation tensor after block 1, flattens its
\(32\times32=1024\) coordinates, and fits a `StandardScaler` plus ridge
regression (fixed \(\alpha=1\)) from activations to the first five
component-major joint-belief coordinates; the sixth is inferred by normalization.
The minimum standardized-norm activation displacement that realizes the requested
decoded displacement is computed with a fixed pseudoinverse cutoff of
\(10^{-3}\). This is a diagonal-covariance metric induced by feature
standardization, not a full-covariance Mahalanobis metric.

The edited block-1 prefix is passed through block 2. For every possible appended
token \(a\), block 1 is run on the original prefix plus \(a\); the first 32
positions are then replaced with the same saved edited prefix tensor, while the
new token's position is left untouched, and block 2 produces the distribution
over \(b\). This tests a persistent intervention on block 2's attention memory.
It does not eliminate the unedited-token bypass through the appended position,
which is an explicit limitation.

## Data boundaries

All synthetic sets are fresh, deterministic length-32 Mess3 prefixes. The three
possible 33rd tokens are enumerated during evaluation. Sets are disjoint by both
RNG seed namespace and exact prefix.

- Development model seeds: 10 and 11, existing fresh step-0 and step-3072
  checkpoints from `configs/mess3_diagnosis.yaml`.
- Held-out pilot model seeds: 20--24, existing learning-rate 0.003 fresh step-0
  and step-3072 checkpoints from `configs/mess3_threshold.yaml`.
- Seeds 30--47 are excluded because earlier studies used them.
- Any later confirmation must use untouched model seeds 50--57 and a new lock.
- Per model seed: 2,048 actuator-fit prefixes, 1,024 independent-decoder prefixes,
  128 calibration prefixes, and 128 evaluation prefixes.

No result from seeds 20--24 may alter the actuator, thresholds, doses, site,
metrics, or controls.

## Feasibility gate

The causal outcome is evaluated only if both development seeds pass all gates at
step 3072:

1. held-out whole-prefix component-posterior \(R^2\ge0.20\);
2. held-out five-coordinate joint-belief \(R^2\ge0.50\);
3. the frozen pseudoinverse retains all five decoder directions and its ratio of
   summed squared construction residual to summed squared requested displacement
   is at most \(10^{-10}\);
4. independent-decoder relative displacement error (ratio of summed squared
   errors to summed squared requests, maximized over doses) is \(\le0.50\);
5. mean per-prefix analytic conditional-response denominator, minimized over
   doses, is \(\ge10^{-6}\);
6. the 95th percentile intervention RMS is no larger than the 95th percentile
   natural standardized RMS distance from the actuator-fit mean;
7. all records are finite, dose zero reconstructs baseline probabilities within
   \(10^{-6}\), exact operators agree with direct enumeration within \(10^{-10}\),
   and all split-overlap counts are zero.

If either development seed fails, the registered outcome is **actuator
infeasible**. Held-out causal effects are not run and no alternate layer, local
site, regularizer, cutoff, or target is searched.

## Controls and decision rule

The learned actuator is compared with:

- eight deterministic random directions, each norm-matched per example;
- a shuffled-target actuator fit after permuting whole-prefix target rows;
- dose zero (identity);
- the corresponding untrained step-0 checkpoints.

Step 3072 is primary; step 0 is a negative control. All finite results, including
failed controls and negative scores, are written to JSONL. The pilot is labeled
**promising**, not confirmed, only if every held-out seed 20--24 has \(S>0\), the
equal-seed mean is at least 0.20, and the learned mean exceeds both the random
control mean and shuffled mean by at least 0.10. Otherwise it is falsified or
inconclusive according to the registered validity checks. With five seeds this
rule is descriptive and underpowered; no population-level claim is made.

The fixed random direction reverses sign with the registered signed dose and is
matched to the learned edit's norm in standardized feature coordinates. For each
seed and control, prefix-and-dose numerators and denominators are pooled before
forming \(S\). Each random direction is pooled separately and the eight scores
are averaged. Model seeds then receive equal weight.

## Pre-data implementation clarification (2026-09-28)

This clarification was added after independent code audit and before any
registered checkpoint was evaluated or any `predictive_memory` result existed.
All five trained held-out checkpoints must pass the same feasibility and
numerical gates after both development checkpoints pass and before any held-out
evaluation split is opened. Failure maps to `invalid_pilot`; development failure
maps to `actuator_infeasible`; complete valid evidence below the behavioral rule
maps to `criterion_not_met`; a pass maps to `promising_pilot`; timeout or an
execution/integrity failure maps to an explicit inconclusive terminal status.
The 0.20 threshold denotes partial response alignment, not near-exact Bayesian
implementation. Exact grids, finite evidence, positive denominators, and
agreement between per-prefix contributions and aggregate scores are mandatory.

## Execution and stopping rule

One public command will run validation, development gates, the frozen held-out
pilot if eligible, JSONL generation, figures, and a machine-readable summary.
The wall-time cap is 60 minutes on CPU. The command must terminate without
substitution if the cap is reached. Existing checkpoints are inputs and their
SHA-256 hashes are recorded. Implementation tests and an independent audit occur
before the result-bearing command. No pilot result may be deleted or overwritten
after inspection; corrections require an append-only erratum and a new protocol.
