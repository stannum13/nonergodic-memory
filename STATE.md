# Experimental state

## Hypothesis

For the exact published two-Mess3 mixture, component geometry emerges when the model becomes predictively competent, rather than at a fixed optimizer step. Across paired fresh-data learning curves, predictive competence should therefore explain held-out block-2 component-posterior R² better than update count does.

## Last experiment

Registered Mess3 geometry-threshold test, committed at `a8491b5` before any threshold checkpoint, result, or figure. Confirmation seeds 20–24 were trained on paired fresh batches at learning rates 0.003/0.0015 and probed at seven checkpoints from initialization through 3,072 updates. The primary leave-one-seed-out comparison predicts block-2 component-posterior R² from either competence or `log1p(step)`.

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

## Registered Mess3 geometry-threshold prediction

Registered before creating any `checkpoints/mess3_threshold/`, `results/mess3_threshold_*.jsonl`, or `figures/mess3_threshold_*.png` artifact. Configuration digest `d2423ea9f3b44075` uses confirmation seeds 20–24, learning rates 0.003 and 0.0015, fresh vectorized samples, and checkpoints 0/384/768/1,152/1,536/2,304/3,072. Same-seed rate conditions share initialization, the seed-indexed fresh batch at every step, held-out evaluation data, probe-fit data, and probe-test data.

The primary prediction is: a quadratic competence-only regression will have at least 20% lower leave-one-seed-out MSE for normal-control block-2 component-posterior R² than a quadratic `log1p(step)`-only regression. Equivalently, `MSE_competence / MSE_step < 0.80`. Each fold holds out both learning-rate trajectories for one seed. Shuffled-label component-posterior R² must remain within ±0.02 in every cell for the primary result to be interpretable. Onset locations, joint-belief R², pairwise-distance R², and other activation sites are secondary with no directional success criterion.

Seeds 20–24 and the 20% criterion are confirmatory and cannot be replaced or weakened after inspection. A positive raw correlation, a ratio between 0.80 and 1.00, or an improvement confined to a subset of folds does not satisfy the registered prediction.

## Registered Mess3 fidelity prediction

For the exact published two-Mess3 source, trained Transformer joint-belief R² will exceed its same-seed untrained control in all three seeds. Pairwise-distance R² is secondary. The experiment is a direct data/process reproduction but not an exact compute reproduction: width 32, two layers, absolute positions, LayerNorm, and CPU training differ from the published width-128 four-layer TransformerLens model with rotary positions, RMSNorm, gated GELU, and 45,000 optimization steps.

## Registered exploratory training-diversity prediction

Registered before creating any diagnosis checkpoint, JSONL record, or figure. Configuration digest `aa2de784730aa352` compares `reused` and `fresh` sequence conditions for exploratory seeds 10 and 11 at steps 0, 768, and 3,072. Both conditions use the same initialized width-32 two-layer Transformer, sequence length 64, batch size 64, AdamW settings, and supervised tokens per update. The reused condition traverses a deterministic fixed pool of 2,048 sequences; the fresh condition samples a new batch of 64 sequences at every update.

The primary prediction is: at step 3,072, fresh-data training has lower held-out exact-predictive KL than reused-data training in both seeds 10 and 11. Layerwise component-posterior, conditional-state, and six-coordinate weighted joint-belief recovery are secondary outcomes with no directional success criterion. Seeds 10 and 11 are exploratory and cannot be reused for later confirmation.
