# Exact beliefs and causal erasure in small nonergodic sequence models

## Summary

This artifact tests whether small GRU and decoder-only Transformer predictors linearly represent the two levels of Bayesian inference required by a fixed-component mixture of HMMs: component identity and every latent-state distribution conditional on a component. Across three CPU seeds, exact component posteriors were linearly recoverable from held-out final activations and training improved substantially over same-architecture untrained controls. The full vector of conditional-state posteriors was also recoverable, but training did not reliably improve it: the untrained GRU was better and the Transformer gain was small. This is a partial reproduction, not blanket agreement.

The causal extension uses separate direction-fit, evaluator-fit, and test sequences. Learned erasures were selective on average, but effects varied sharply by architecture and seed and also occurred in untrained networks. Next-token loss changed little except for Transformer component erasure. The extension therefore falsifies the strong claim that training consistently creates stable, selectively necessary final-layer subspaces in this small setting.

## Relation to the target result

Ray, Riechers, and Shai derive a telescoping belief geometry for nonergodic compositions and report that a linear map from Transformer residual activations recovers weighted beliefs for two Mess3 sources with held-out R² near 0.985, versus about 0.45 for an untrained network. Their model and process are substantially larger than those used here. This is a conceptual reproduction: conventional two-state HMMs replace Mess3, and final activations of width-32 models replace a width-128, four-layer Transformer. Detailed correspondences and non-equivalences are recorded in `paper_notes.md`.

## Analytic ground truth

A component `c` is sampled once per sequence. For each observed token, the filter updates joint mass

`q_t(c,s) ∝ p(x_t | s,c) Σ_s' q_{t-1}(c,s') p(s | s',c)`.

It then records `p(c | x_0:t) = Σ_s q_t(c,s)`, `p(s_t | c,x_0:t) = q_t(c,s)/p(c | x_0:t)`, and the exact next-token distribution obtained by one transition and emission step. Normalization and a hand-computed one-state mixture are tested. Neural position `t` is evaluated against the same `x_(t+1)` predicted by the filter.

## Methods

The central dataset uses two equally weighted, two-state HMMs over four tokens. Emissions interpolate between distinct and identical sources; overlap is fixed at 0.35. Each seed trains on 1,024 sequences of length 32 and evaluates on 256 separate sequences. The models are a two-layer width-32 GRU and a two-layer width-32 causal Transformer with four heads. Both train for 20 epochs with AdamW.

Probe fitting and evaluation use separately sampled datasets. Component classification uses one logistic probe. Conditional-state accuracy uses one state probe per true component, so component prediction errors are not counted as state errors. Ridge regression targets the exact component posterior and the flattened `K×S` vector containing every component-conditional state posterior. All metrics are scored on held-out sequences. Shuffled labels and freshly initialized networks are controls. PCA points are saved to JSONL and plotted descriptively.

For causal erasure, the effective probe coefficient rows are mapped back through feature standardization, reduced to an orthonormal row-space basis, and projected out around the direction-fit activation mean. The state subspace combines rows from component-specific state probes. A second probe family, fitted on an independent dataset, evaluates altered activations on a third test dataset; it is never used to define the removed basis. Controls use a random subspace of equal rank, a random intervention rescaled per example to match the learned intervention’s removed-vector norm, and directions from shuffled-label probes. The intervention is at the final activation immediately before the output head.

The depth follow-up captures activations after each Transformer block and after final normalization. Interventions before the final layer are propagated through the actual remaining attention blocks and normalization before NLL and exact-predictive KL are computed. Classification metrics are evaluated at the edited site; explicitly named baseline posterior-R² fields describe pre-intervention linear recovery at that site. A continuation test verifies that unedited captured activations reproduce the original logits at every site.

## Reproduction results

Values are mean ± population standard deviation across seeds 0, 1, and 2.

| model | held-out NLL | exact Bayes NLL | component posterior R² | untrained R² | conditional-state posterior R² | untrained R² |
|---|---:|---:|---:|---:|---:|---:|
| GRU | 1.2084 ± 0.0119 | 1.2039 ± 0.0112 | 0.9901 ± 0.0021 | 0.8265 ± 0.0084 | 0.9632 ± 0.0035 | 0.9762 ± 0.0040 |
| Transformer | 1.2383 ± 0.0141 | 1.2039 ± 0.0112 | 0.9344 ± 0.0041 | 0.6452 ± 0.0383 | 0.7439 ± 0.0209 | 0.7325 ± 0.0122 |

Trained component classification was 0.9521 ± 0.0017 for the GRU and 0.9407 ± 0.0015 for the Transformer. Conditional-state classification was 0.8093 ± 0.0058 and 0.7847 ± 0.0004. Shuffled-label posterior regression stayed near zero. The component-belief result reproduces linear recoverability and a trained-over-untrained gain. The full conditional-state result does not: training reduced GRU R² by 0.013 and improved Transformer R² by only 0.011. None of these numbers are an exact numerical replication of the larger Mess3 experiment.

## Causal-erasure extension

| model | target erased | Δ component accuracy | Δ conditional-state accuracy | Δ NLL | norm-matched largest mean accuracy change |
|---|---|---:|---:|---:|---:|
| GRU | component | −0.0309 ± 0.0341 | −0.0014 ± 0.0036 | +0.0001 ± 0.0004 | 0.0085 |
| GRU | state | −0.0081 ± 0.0022 | −0.1113 ± 0.0835 | +0.0008 ± 0.0008 | 0.0046 |
| Transformer | component | −0.4097 ± 0.2948 | −0.0079 ± 0.0028 | +0.0143 ± 0.0064 | 0.0014 |
| Transformer | state | −0.0023 ± 0.0020 | −0.1686 ± 0.0388 | +0.0018 ± 0.0040 | 0.0008 |

Cross-damage is smaller than intended damage on average, and equal-rank random, norm-matched, and shuffled-label directions are near zero. However, independent-evaluator effects are not stable: GRU component damage ranges from −0.079 to −0.001, Transformer component damage from −0.795 to −0.080, and GRU state damage from −0.225 to −0.026. Untrained networks also show selective damage, including mean Transformer component/state-target damage of −0.372/−0.176 on their intended metrics. The extension therefore finds selective linear organization but fails to establish that it is a consistent training-induced mechanism.

## Source-overlap sweep

### Registered prediction and result

After the central experiment, overlap was swept through 0.00, 0.35, 0.70, and 0.90 with fixed length 32 and smaller fixed training compute. The registered prediction was a monotonic decline in trained-over-untrained component-posterior R² as source emissions became less distinguishable. It failed:

| model | metric | overlap 0.00 | 0.35 | 0.70 | 0.90 |
|---|---|---:|---:|---:|---:|
| GRU | component R² gain | 0.074 | 0.153 | 0.193 | −0.004 |
| Transformer | component R² gain | 0.170 | 0.301 | 0.287 | 0.027 |
| GRU | conditional-state R² gain | −0.010 | −0.009 | 0.004 | 0.012 |
| Transformer | conditional-state R² gain | 0.019 | 0.027 | 0.048 | 0.063 |

Component gain is largest at intermediate overlap and collapses only at 0.90. A post-hoc explanation is that component identity is trivially available from recent tokens at zero overlap, benefits from trained integration at intermediate overlap, and becomes nearly unidentifiable at high overlap. Independent-probe damage is also nonmonotonic: mean intended component damage across the same overlap values is 0.013/0.053/0.017/0.042 for the GRU and 0.133/0.115/0.221/0.038 for the Transformer. The raw three-seed distributions and error bars are retained; this sweep strengthens the negative conclusion about a simple, stable causal geometry.

## Sequence-length follow-up

The overlap interpretation made a new, falsifiable prediction: if training adds temporal integration, trained-over-untrained component-posterior R² gain should grow with context length at fixed overlap 0.35. The prediction was registered in `STATE.md` before sweeping lengths 8, 16, 32, and 64.

| model | metric | length 8 | 16 | 32 | 64 |
|---|---|---:|---:|---:|---:|
| GRU | component R² gain | 0.040 | 0.099 | 0.153 | 0.181 |
| Transformer | component R² gain | 0.035 | 0.138 | 0.301 | 0.442 |
| GRU | conditional-state R² gain | −0.010 | −0.013 | −0.009 | −0.006 |
| Transformer | conditional-state R² gain | −0.011 | −0.001 | 0.027 | 0.036 |

The component result is monotonic for both architectures and supports learned long-context integration. The conditional-state result remains negative for the GRU and small for the Transformer. Intended Transformer component-erasure damage grows from 0.010 at length 8 to 0.462 at length 64, but the latter has seed SD 0.309; causal alignment is much less stable than correlational recovery. The complete sweep took 146.5 seconds on CPU.

## Component-count sweep

The generator was extended to three and four sources using distinct emission permutations while retaining the exact original two-source case. At fixed width 32, the registered prediction was that expanding the component simplex would reduce component recovery.

| model | metric | 2 components | 3 | 4 |
|---|---|---:|---:|---:|
| GRU | trained component accuracy | 0.949 | 0.836 | 0.759 |
| Transformer | trained component accuracy | 0.944 | 0.807 | 0.729 |
| GRU | trained component posterior R² | 0.984 | 0.931 | 0.877 |
| Transformer | trained component posterior R² | 0.936 | 0.762 | 0.630 |
| GRU | untrained component posterior R² | 0.831 | 0.698 | 0.616 |
| Transformer | untrained component posterior R² | 0.636 | 0.415 | 0.360 |

The decline is monotonic and the registered prediction is supported. Training retains substantial posterior-regression advantage at every count. Intended trained component-erasure damage rises from 0.053/0.115 at two sources to 0.195/0.356 at four sources for GRU/Transformer, but untrained damage at four sources is 0.266/0.324. Thus, larger identity spaces amplify probe-defined sensitivity without making it uniquely training-induced. The sweep took 136.5 seconds on CPU.

## Model-width sweep

Widths 8, 16, 32, and 64 were tested at fixed two-source data and depth two. The registered prediction was that trained recovery would saturate while relative seed variability of independent-erasure damage decreased.

| model | metric | width 8 | 16 | 32 | 64 |
|---|---|---:|---:|---:|---:|
| GRU | trained component posterior R² | 0.936 | 0.964 | 0.984 | 0.990 |
| GRU | untrained component posterior R² | 0.693 | 0.778 | 0.831 | 0.857 |
| Transformer | trained component posterior R² | 0.831 | 0.922 | 0.936 | 0.918 |
| Transformer | untrained component posterior R² | 0.136 | 0.434 | 0.636 | 0.794 |

GRU recovery saturates; Transformer recovery peaks at width 32. The training advantage shrinks with width because untrained random features improve more rapidly. Component-erasure damage CV is 0.78/0.95/0.62/0.77 for GRU and 0.48/0.47/0.74/0.71 for Transformer, so the registered stability prediction is falsified. The same nonmonotonicity appears for state erasure. The sweep took 147.6 seconds on CPU.

## Transformer intervention-depth sweep

The final registered prediction was that component-posterior recovery and behavioral damage would increase from block 1 through block 2 to final normalization, with learned selectivity exceeding matched controls. The result is mixed:

| metric | block 1 | block 2 | final norm |
|---|---:|---:|---:|
| trained component posterior R² | 0.733 ± 0.009 | 0.893 ± 0.006 | 0.928 ± 0.009 |
| untrained component posterior R² | 0.493 ± 0.068 | 0.627 ± 0.042 | 0.633 ± 0.041 |
| trained conditional-state posterior R² | 0.773 ± 0.015 | 0.774 ± 0.024 | 0.746 ± 0.022 |
| component erasure: intended accuracy decrease | 0.589 ± 0.179 | 0.438 ± 0.224 | 0.410 ± 0.295 |
| component erasure: Δ exact-predictive KL | 0.0050 | 0.0074 | 0.0135 |
| state erasure: intended accuracy decrease | 0.152 ± 0.040 | 0.149 ± 0.021 | 0.169 ± 0.039 |
| state erasure: Δ exact-predictive KL | 0.0076 | 0.0026 | 0.0019 |

Component recovery and component-target predictive damage rise with depth, supporting that portion of the prediction. Component-decoding damage instead decreases, conditional-state recovery is flat then lower, and state-target predictive damage decreases. Intended learned accuracy damage remains much larger than norm-matched damage at every site (component matched means at most 0.001; state at most 0.003), while cross-target accuracy damage stays at most 0.020. Thus selective linear organization exists throughout the network, but deeper layers do not uniformly make both belief types more behaviorally necessary. The sweep took about 25 seconds on CPU.

## Negative results and limitations

- Untrained networks are surprisingly decodable: recent-token features alone expose much of component and state information. Classification accuracy without the untrained and shuffled controls would overstate the result.
- Full conditional-state posterior regression is worse after GRU training (0.976 untrained versus 0.963 trained) and barely better after Transformer training (0.733 versus 0.744).
- Independent-evaluator erasure effects have large seed variation, particularly for component information, and similarly selective effects can exist before training.
- Final-layer erasure barely changes GRU NLL, and only Transformer component erasure produces a clearly nontrivial mean NLL increase. Decoder damage does not imply equivalent behavioral necessity.
- Three seeds quantify run variability but are insufficient for strong population-level inference; no p-values are reported.
- The simple state-emission HMMs do not recreate Mess3’s fractal reachable-state geometry. PCA separation is not evidence for telescoping cones.
- The central result still covers only overlap 0.35, two components, length 32, and width 32. The exploratory overlap, length, component-count, width, and Transformer-depth sweeps each change one variable; interactions between these axes remain untested.
- Erasure is based on a single linear probe fit. Iterative nullspace projection or nonlinear adversaries could find residual information not measured here.

## Reproducibility

The checked-in central run used CPU only. In the observed environment, six training runs took about 39 seconds, cached-checkpoint reproduction analysis 14.1 seconds, and the three-split intervention analysis 17.4 seconds. The complete overlap, length, component-count, width, and depth sweeps took 151.7, 146.5, 136.5, 147.6, and about 25 seconds. `make smoke` runs the complete one-seed pipeline. `make train`, `make reproduce`, `make extension`, `make figures`, and the five `make sweep-*` commands regenerate the artifact. Checkpoints are validated against the full requested configuration, model, and seed. Partial CLI reruns atomically replace only matching result cells. Records carry a configuration digest and runtime library versions. Figures read only JSONL records, discard stale outputs, facet architectures, state seed sample sizes, and keep central aggregation separate from sweep records.
