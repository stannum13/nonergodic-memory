# Experimental state

## Hypothesis

For the exact published two-Mess3 mixture, component geometry emerges when the model becomes predictively competent, rather than at a fixed optimizer step. Across paired fresh-data learning curves, predictive competence should therefore explain held-out block-2 component-posterior R² better than update count does.

## Last experiment

Exploratory Mess3 training-diversity diagnosis, registered at `7cac6cb` before any diagnosis output. For seeds 10/11, compared a fixed pool of 2,048 sequences with fresh batches at every update, holding initialization, architecture, batch size, sequence length, optimizer, and checkpoints fixed. Exact baselines quantify the available predictive signal; independent probes measure all three activation sites at steps 0/768/3,072.

## Result

- The registered primary prediction is supported in both exploratory seeds. At step 3,072, fresh/reused exact-predictive KL is 0.001649/0.203223 for seed 10 and 0.001301/0.179832 for seed 11.
- Uniform predictive KL is 0.008175; exact eight-token Bayes is 0.003313. Fresh training recovers 79.8%/84.4% of the uniform-to-full-Bayes gap, while reused training becomes much worse than uniform.
- At the original 768-update budget, fresh training improves predictive KL but joint-belief R² is only 0.378–0.387 and component-posterior R² remains near zero.
- At step 3,072, fresh block-2 joint-belief R² is 0.726/0.616 versus reused 0.251/0.250. Fresh final-norm R² is 0.677/0.628. Shuffled-target joint R² stays near zero.
- The uncached CPU command reported 3:02:13 wall time, almost consuming the four-hour exploratory cap.

## Interpretation

Fixed-pool sequence reuse explains a large part of the initial predictive-generalization failure: reused training generalizes poorly and eventually damages prediction. It does not by itself explain the geometry failure. At 768 updates, fresh data improves prediction without recovering component identity; the substantial deeper-layer weighted-belief representation appears only with fresh data and a fourfold larger 3,072-update budget. The result remains exploratory with two seeds and does not match the paper's architecture, training budget, or reported R². It narrows the failure mechanism without establishing an exact numerical reproduction or a diversity-only causal explanation.

## Next smallest experiment

Run the registered five-seed, two-learning-rate competence-versus-step experiment below. The distribution-exact vectorized sampler is 23.9× faster than the reference sampler on 64 length-64 sequences in the recorded 20-repeat benchmark. Before launching the full grid, time one 3,072-update run and stop if ten training runs plus checkpoint probes project beyond two hours.

## Registered Mess3 geometry-threshold prediction

Registered before creating any `checkpoints/mess3_threshold/`, `results/mess3_threshold_*.jsonl`, or `figures/mess3_threshold_*.png` artifact. Configuration digest `d2423ea9f3b44075` uses confirmation seeds 20–24, learning rates 0.003 and 0.0015, fresh vectorized samples, and checkpoints 0/384/768/1,152/1,536/2,304/3,072. Same-seed rate conditions share initialization, the seed-indexed fresh batch at every step, held-out evaluation data, probe-fit data, and probe-test data.

The primary prediction is: a quadratic competence-only regression will have at least 20% lower leave-one-seed-out MSE for normal-control block-2 component-posterior R² than a quadratic `log1p(step)`-only regression. Equivalently, `MSE_competence / MSE_step < 0.80`. Each fold holds out both learning-rate trajectories for one seed. Shuffled-label component-posterior R² must remain within ±0.02 in every cell for the primary result to be interpretable. Onset locations, joint-belief R², pairwise-distance R², and other activation sites are secondary with no directional success criterion.

Seeds 20–24 and the 20% criterion are confirmatory and cannot be replaced or weakened after inspection. A positive raw correlation, a ratio between 0.80 and 1.00, or an improvement confined to a subset of folds does not satisfy the registered prediction.

## Registered Mess3 fidelity prediction

For the exact published two-Mess3 source, trained Transformer joint-belief R² will exceed its same-seed untrained control in all three seeds. Pairwise-distance R² is secondary. The experiment is a direct data/process reproduction but not an exact compute reproduction: width 32, two layers, absolute positions, LayerNorm, and CPU training differ from the published width-128 four-layer TransformerLens model with rotary positions, RMSNorm, gated GELU, and 45,000 optimization steps.

## Registered exploratory training-diversity prediction

Registered before creating any diagnosis checkpoint, JSONL record, or figure. Configuration digest `aa2de784730aa352` compares `reused` and `fresh` sequence conditions for exploratory seeds 10 and 11 at steps 0, 768, and 3,072. Both conditions use the same initialized width-32 two-layer Transformer, sequence length 64, batch size 64, AdamW settings, and supervised tokens per update. The reused condition traverses a deterministic fixed pool of 2,048 sequences; the fresh condition samples a new batch of 64 sequences at every update.

The primary prediction is: at step 3,072, fresh-data training has lower held-out exact-predictive KL than reused-data training in both seeds 10 and 11. Layerwise component-posterior, conditional-state, and six-coordinate weighted joint-belief recovery are secondary outcomes with no directional success criterion. Seeds 10 and 11 are exploratory and cannot be reused for later confirmation.
