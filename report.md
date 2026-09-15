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

## Overlap-by-context interaction

The next prediction was registered in `STATE.md` and committed as `8e27881` before running a fresh 2×2 grid. Overlap 0.00 versus 0.35 and sequence length 8 versus 64 varied; model size, 512 training sequences, 12 epochs, probe sizes, and seeds 0/1/2 were matched. The primary per-seed contrast was `I = [G(0.35,64) − G(0.35,8)] − [G(0,64) − G(0,8)]`, where `G` is trained-minus-untrained held-out component-posterior R². The registered prediction was `I > 0` for both models.

| model | component gain: overlap 0, length 8/64 | component gain: overlap .35, length 8/64 | paired component interaction I | paired conditional-state interaction I |
|---|---:|---:|---:|---:|
| GRU | 0.063 / 0.076 | 0.040 / 0.181 | +0.127 ± 0.024 | +0.003 ± 0.012 |
| Transformer | 0.056 / 0.249 | 0.035 / 0.442 | +0.214 ± 0.020 | +0.003 ± 0.023 |

Every seed has a positive primary interaction: GRU +0.113/+0.160/+0.108 and Transformer +0.204/+0.241/+0.196. This supports the registered prediction that training’s component-belief advantage grows more with context at intermediate overlap than at disjoint emissions. The same specificity is absent for the full conditional-state posterior. No p-value or population-level claim is made from three seeds.

The causal diagnostic is less stable. Learned-minus-norm-matched component-accuracy damage at overlap 0/.35 and length 8/64 is GRU 0.183/0.157 and 0.046/0.108; Transformer 0.003/0.027 and 0.010/0.462. The final Transformer cell has seed SD 0.309 and seed values 0.502/0.065/0.818. Thus the large correlational interaction is not accompanied by a consistent, architecture-independent erasure interaction.

An alternative explanation is a recovery ceiling at zero overlap. Trained component R² is already 0.988/0.996 for GRU and 0.961/0.993 for Transformer at zero-overlap lengths 8/64, leaving less room for raw R² gain. A post-hoc error-closure diagnostic, `(R²_trained−R²_untrained)/(1−R²_untrained)`, still shows a larger length change at overlap .35 than zero for both models (GRU +0.374 versus +0.103; Transformer +0.696 versus +0.380), but this was not the registered outcome. Exact component-posterior entropy is also higher at overlap .35 than zero at both lengths (0.355 versus 0.170 nats at length 8; 0.057 versus 0.018 at length 64). This indicates a harder inference problem, not proof that the model uses earlier tokens causally. Equal sequence counts imply a larger token budget at length 64; the difference-in-differences controls a shared length effect but not every possible difficulty-by-compute interaction.

## Eight-token context-restart control

Registered in `STATE.md` and committed as `a0207a1` before any result, this control reused the length-64 interaction checkpoints. At held-out prediction positions 7–62, each model was scored with its full prefix and separately after restarting on only the most recent eight observed tokens. The full-history exact Bayes belief and predictive distribution remained the scoring targets. The oracle eight-token Bayes filter uses the unconditional HMM state prior propagated to the window's elapsed starting position, not the time-zero prior. Probe fitting used seed+909 data, testing used seed+1009 data, and full/restart probes were fitted separately. Three seeds, both models, both overlaps, trained/untrained controls, and shuffled-label probes yielded 96 model and six oracle raw records, all on CPU.

The registered paired overlap contrast is `[damage at overlap .35] − [damage at overlap 0]`, with component damage defined as full-minus-restart held-out posterior R² and predictive damage as restart-minus-full KL from the exact full-history next-token distribution. Values are mean ± population seed SD (n=3).

| contrast | exact eight-token Bayes | trained GRU | untrained GRU | trained Transformer | untrained Transformer |
|---|---:|---:|---:|---:|---:|
| component-posterior R² loss | +0.156 ± 0.010 | +0.155 ± 0.003 | +0.045 ± 0.000 | +0.150 ± 0.004 | −0.046 ± 0.023 |
| exact-predictive KL increase | +0.0148 ± 0.0014 | +0.0148 ± 0.0016 | +0.0007 ± 0.0005 | +0.0157 ± 0.0057 | +0.0093 ± 0.0403 |
| conditional-state-posterior R² loss | not primary | −0.001 ± 0.001 | −0.000 ± 0.001 | −0.048 ± 0.009 | −0.052 ± 0.011 |

Both registered trained-model contrasts are positive in all three seeds. At overlap .35 alone, component R² loss is 0.189/0.185 for trained GRU/Transformer versus 0.069/−0.188 for untrained models. The untrained Transformer actually becomes more component-decodable after restart; absolute R² at the two contexts is therefore not a measure of learned history use. Shuffled-label component R² never exceeds 0.016 in absolute value. The exact oracle has a component R² loss of 0.027 at overlap 0 and 0.183 at .35, and a predictive KL penalty of 0.0076/0.0225. The trained model overlap contrasts are close to the oracle contrasts, consistent with useful remote-history information at intermediate overlap. This does not prove that a particular final-activation subspace is causally necessary.

Conditional-state results are not selective evidence for training: the GRU interaction is negligible and Transformer state R² falls under restart in both trained and untrained networks. Transformer restarts position indices at zero and re-encodes the same recent tokens at new positions; this distribution shift can alter probe recoverability and logits independently of losing remote tokens. Full and restarted probes are separately refitted, preventing a single-probe transfer artifact but not eliminating this positional confound. This motivated the position-preserving control below.

## Position-preserving Transformer restart

This control was registered in `STATE.md` at `07f799b` and planned in `b0dec1f` before running. It reused the exact same length-64 Transformer checkpoints and held-out sequences, comparing full prefixes, eight-token windows with position indices reset to 0–7, and the same windows indexed at their original absolute positions. The only difference between the two window conditions is the positional embedding assigned to each of the same eight tokens. Full-history Bayes targets and elapsed-prior eight-token oracle rows are unchanged. The 72 Transformer model cells plus six oracle cells cover both overlaps, all three seeds, trained/untrained networks, and normal/shuffled-label probes. Full/reset/oracle metrics reproduce the earlier JSONL exactly (maximum absolute numerical difference 0).

The preregistered simple positional-benefit prediction failed. At overlap .35, original-minus-reset R² is −0.034 ± 0.002 for trained component belief and −0.284 ± 0.035 for untrained component belief; for conditional-state belief it is −0.137 ± 0.030 and −0.221 ± 0.030. These signs are negative in every seed. Thus preserving indices does not improve absolute decodability. Resetting positions can make recent-window random features unusually easy to decode, particularly in the untrained model. Predictive KL benefit (`KL_reset − KL_original`) is −0.0051 ± 0.0063 for trained and +0.0161 ± 0.0643 for untrained at overlap .35; it is not a stable absolute behavioral improvement either.

The remote-history component prediction survived the control. With original indices, the registered full-to-window overlap contrast in component R² loss is +0.164 ± 0.006 for trained Transformer versus +0.034 ± 0.004 untrained, positive for every seed. The predictive KL overlap contrast is +0.0157 ± 0.0025 trained versus −0.0006 ± 0.0021 untrained, again positive for every trained seed. These compare with +0.150 and +0.0157 under reset indices and +0.156 and +0.0148 for the exact eight-token Bayes oracle. Keeping original absolute indices therefore does not erase the trained model's overlap-dependent full-history advantage; it makes the untrained predictive control closer to zero. The state R² overlap contrast shifts from −0.048 to −0.020 trained and from −0.052 to −0.005 untrained, so the overlap-specific state artifact is attenuated, even though absolute state R² is worse under original indices at both overlaps. The state result remains nonselective and does not support the strong subspace claim.

This is still a window-restart intervention: removing older tokens changes attention length and the input distribution. Re-fitting probes per context prevents a probe-transfer artifact but cannot make window-only representations identical to full-prefix representations. The small three-seed sample supports a falsifiable within-artifact contrast, not a population claim.

## Matched eight-input-token training control

The final diagnostic registered in `STATE.md` at `b28f940` corrected an off-by-one issue before running: length-nine next-token training sequences give a model eight input positions. Six width-32, two-layer Transformer checkpoints were trained for 12 epochs on 512 length-nine HMM sequences at overlap 0/.35 and seeds 0/1/2. The short and length-64 configs differ only in training sequence length; architecture, optimizer settings, initial seed, and training sequence count are otherwise matched. On the exact same held-out length-64 test batches and positions 7–62 used above, the short-trained model received the same reset-index last-eight-token windows as the long-trained reset control. Independently fitted normal/shuffled probes and exact full-history Bayes targets were unchanged. Twelve short-model raw cells and six training records are saved with both evaluation and checkpoint config hashes; the joined figure reads those records plus the published position-restart JSONL.

The registered prediction that short training would *lower* eight-token exact-predictive KL at overlap .35 failed in all three seeds. Paired short-minus-long-reset KL is +0.0434/+0.0289/+0.0176 nats (mean +0.0300 ± 0.0106), and NLL is +0.0422/+0.0267/+0.0183 (mean +0.0291 ± 0.0099). At overlap 0, the KL difference is also positive, +0.0216 ± 0.0027. Component-posterior R² is lower by 0.048 ± 0.014 at overlap .35 and 0.016 ± 0.003 at overlap 0; conditional-state R² differs by only +0.005 ± 0.017 at overlap .35. The shuffled-label component R² is within ±0.014. The exact eight-token Bayes oracle KL at overlap .35 is 0.0225 versus mean model KL about 0.0701 short-trained and 0.0401 long-trained reset. The short model does not recover older-token information, and it also predicts the eight-token windows less accurately than the restarted long model.

This rejects the *simple* window-distribution explanation that short training alone would rescue restarted prediction under the specified fixed-sequence-count protocol. It does not isolate sequence-length distribution from training token budget: 512 length-nine sequences provide 8 input tokens each, while 512 length-64 sequences provide 63, so the long model receives substantially more supervised tokens at the same 12 epochs and 96 optimizer steps. This motivated the token- and step-matched follow-up below.

## Token- and optimizer-step-matched short training

Registered in `STATE.md` at `b115b52` and planned in `403162a` before running, this follow-up trained length-nine Transformers on 4,032 sequences in batches of 504 at overlap 0/.35, seeds 0/1/2, for 12 epochs. Both short and long training therefore see exactly 32,256 supervised input/next-token pairs per epoch and eight optimizer batches per epoch: `4,032×8 = 512×63` and `4,032/504 = 512/64 = 8`. Width, depth, learning rate, initialization seed, held-out eight-token windows, and Bayes targets are fixed. The larger short-model batch and greater sequence diversity are required confounds of matching both budgets. Six CPU training records and 12 independent-probe budget cells are saved with distinct checkpoint and evaluation config hashes; the figure joins these raw files to the standard-short and position-restart raw files.

The registered `KL_budget_short − KL_standard_short < 0` prediction at overlap .35 is supported in every seed: −0.0611/−0.0283/−0.0189 nats, mean −0.0361 ± 0.0181. NLL falls by 0.0347 ± 0.0149 and component-posterior R² rises by 0.053 ± 0.021. At overlap 0, KL falls by 0.0291 ± 0.0026 and component R² rises by 0.022 ± 0.007. Conditional-state R² rises by 0.028 ± 0.021 at overlap .35, secondary to the predictive test. Shuffled-label component R² stays within ±0.017. The budget-matched short model's mean KL at overlap .35 is 0.0340, versus 0.0701 for standard short training, 0.0401 for the long-trained reset-index restart, and the exact eight-token Bayes information-loss floor of 0.0225. Budget-minus-long-reset KL is −0.0177/+0.0006/−0.0012 across seeds; the short model now matches or slightly improves on the long model when both see only the same eight tokens.

This falsifies the temptation to interpret the fixed-sequence-count short-model failure as evidence that long-input training alone is necessary for eight-token prediction. Equalized training exposure (jointly with batch size and sequence diversity) removes most of that failure. It does not weaken the earlier full-prefix versus last-eight-token contrast: full-length trained models still exploit remote history, and even an accurately trained eight-token model cannot beat the exact information-loss floor when older tokens are absent. The floor is a Bayes full-to-window KL, whereas model KL additionally includes approximation error, so their numerical gap is not a model-only causal effect.

## Negative results and limitations

- Untrained networks are surprisingly decodable: recent-token features alone expose much of component and state information. Classification accuracy without the untrained and shuffled controls would overstate the result.
- Full conditional-state posterior regression is worse after GRU training (0.976 untrained versus 0.963 trained) and barely better after Transformer training (0.733 versus 0.744).
- Independent-evaluator erasure effects have large seed variation, particularly for component information, and similarly selective effects can exist before training.
- Final-layer erasure barely changes GRU NLL, and only Transformer component erasure produces a clearly nontrivial mean NLL increase. Decoder damage does not imply equivalent behavioral necessity.
- Three seeds quantify run variability but are insufficient for strong population-level inference; no p-values are reported.
- The overlap-by-context component interaction is positive, but a zero-overlap R² ceiling and unequal per-sequence token budgets across lengths remain alternative explanations. The direct context-restart control supports remote-history dependence for component belief but does not isolate an activation-level causal mechanism.
- Transformer restart changes absolute positional embeddings and sequence-length distribution. The untrained Transformer's negative component-damage contrast and nonselective state damage show why restart effects cannot be attributed solely to learned memory.
- Preserving original position indices made absolute Transformer component and state posterior recovery worse in every seed, falsifying the predicted positional benefit. The trained component/KL overlap contrasts nonetheless remain positive, while state effects remain nonselective.
- Length-nine-trained Transformers had worse eight-window KL and component R² than the restarted length-64 models in every seed, falsifying the predicted short-training rescue under fixed sequence count. Their training token budgets are not equal.
- Matching short training to the long model's supervised-token and optimizer-step budgets improves short-window KL in every seed, overturning the fixed-count inference. Batch size and training-sequence diversity move with budget, so the improvement is not uniquely attributable to token count.
- The simple state-emission HMMs do not recreate Mess3’s fractal reachable-state geometry. PCA separation is not evidence for telescoping cones.
- The central result still covers only overlap 0.35, two components, length 32, and width 32. The exploratory one-axis sweeps and one matched 2×2 overlap-by-context grid leave most cross-axis interactions untested.
- Erasure is based on a single linear probe fit. Iterative nullspace projection or nonlinear adversaries could find residual information not measured here.

## Reproducibility

The checked-in central run used CPU only. In the observed environment, six training runs took about 39 seconds, cached-checkpoint reproduction analysis 14.1 seconds, and the three-split intervention analysis 17.4 seconds. The complete overlap, length, component-count, width, and depth sweeps took 151.7, 146.5, 136.5, 147.6, and about 25 seconds; the matched interaction grid took about three minutes. `make smoke` runs the complete one-seed pipeline. `make train`, `make reproduce`, `make extension`, `make figures`, and the ten `make sweep-*` commands regenerate the artifact. Checkpoints are validated against the full requested configuration, model, and seed. Partial CLI reruns atomically replace only matching result cells. Records carry a configuration digest and runtime library versions. Figures read only JSONL records, discard stale outputs, facet architectures, state seed sample sizes, and keep central aggregation separate from sweep records.
