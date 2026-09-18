# Experimental state

## Hypothesis

Small next-token predictors trained on a fixed-component HMM mixture should expose linearly separable component identity and within-component predictive state. Component and state erasures should selectively damage their intended targets beyond rank- and norm-matched random controls. The context follow-ups ask whether learned component belief uses remote history rather than only recent tokens.

## Last experiment

GRU architecture-generalization of the token/step-matched short-training control, registered at `534747d` and planned at `97805c5` before running. At overlap .35, seeds 0/1/2, compared standard length-nine GRU training (512 sequences/batch 64) with budget matching (4,032/batch 504), then evaluated identical held-out length-64 last-eight-token windows against full-history Bayes targets. Six CPU training and 12 independent-probe evaluation cells were saved across the two protocols.

## Result

- The registered GRU `KL_standard_short − KL_budget_short > 0` contrast is positive in every seed: +0.00511/+0.00487/+0.00635 nats, mean +0.00544 ± 0.00065. NLL reduction is +0.00594 ± 0.00175 and component R² gain +0.00765 ± 0.00213.
- Budget-short GRU KL is 0.0269 ± 0.0014 versus standard-short 0.0323 ± 0.0019, long restart 0.0290 ± 0.0018, and exact eight-token Bayes information loss 0.0225 ± 0.0016. Conditional-state R² gain is −0.00013 ± 0.00041; shuffled-label component R² is within ±0.018.
- The matched-exposure effect generalizes across both architectures, with a smaller GRU magnitude. Token count, batch size, and sequence diversity remain bundled.

## Interpretation

The registered exposure-matching effect generalizes from Transformer to GRU. Fixed-count short-model failure cannot be attributed to long-context training alone; equal token/step budgets, larger batches, and more diverse data jointly improve window prediction. The full-prefix versus window history contrast still follows the exact Bayesian information-loss oracle, so remote tokens matter for component belief. State-probe effects and subspace erasure remain nonselective/seed-sensitive; no stable selectively necessary activation subspace is established.

## Next smallest experiment

No additional scientific result is registered. The operational fresh-clone audit is complete: all required commands, eleven sweeps, and 65 tests passed without checkpoints; numerical records matched exactly, figures were byte-identical, and nine schema-only provenance drifts were corrected. Further work should begin with a new preregistered scientific question rather than extending the current loop post hoc.

## Registered Mess3 fidelity prediction

For the exact published two-Mess3 source, trained Transformer joint-belief R² will exceed its same-seed untrained control in all three seeds. Pairwise-distance R² is secondary. The experiment is a direct data/process reproduction but not an exact compute reproduction: width 32, two layers, absolute positions, LayerNorm, and CPU training differ from the published width-128 four-layer TransformerLens model with rotary positions, RMSNorm, gated GELU, and 45,000 optimization steps.
