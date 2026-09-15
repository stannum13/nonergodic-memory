# Experimental state

## Hypothesis

Small next-token predictors trained on a fixed-component mixture will expose linearly separable component identity and within-component predictive state. Probe-row-space erasure will cause selective damage: component erasure will reduce component accuracy more than conditional-state accuracy, and state erasure will do the converse, beyond rank- and norm-matched random controls.

## Last experiment

Fresh overlap-by-context 2×2 grid at overlap 0.00/0.35 and length 8/64, both models, seeds 0/1/2, on CPU with identical 512-sequence/12-epoch settings per cell. The primary paired contrast was registered and committed before running; all four cells included untrained and shuffled probe controls plus three-split causal interventions.

## Result

- Component-posterior training-gain interaction `I` is positive for all three seeds in both models: GRU +0.113/+0.160/+0.108 (mean +0.127 ± 0.024); Transformer +0.204/+0.241/+0.196 (mean +0.214 ± 0.020).
- Full conditional-state-posterior interaction is near zero and inconsistent: GRU +0.003 ± 0.012; Transformer +0.003 ± 0.023.
- Learned-minus-norm-matched component-erasure accuracy damage at overlap .35/length 64 is 0.108 ± 0.088 for GRU and 0.462 ± 0.309 for Transformer; the latter ranges 0.065–0.818 across seeds.
- The zero-overlap trained component R² is already near ceiling at lengths 8/64; a post-hoc remaining-error-closure diagnostic still has a larger length change at overlap .35, but it was not preregistered.

## Interpretation

The registered interaction prediction is supported for component belief in this small grid: training adds more length-related recoverability at intermediate overlap than at disjoint emissions. It is not supported for conditional-state belief, and erasure effects remain seed-sensitive. The positive R² contrast does not by itself establish use of remote history; zero-overlap saturation and difficulty-by-compute interactions remain alternatives.

## Next smallest experiment

Registered before running: reuse the length-64 interaction checkpoints at overlap 0.00 and 0.35. For every held-out position 7–62, reset the model to its most recent eight observed tokens, but keep the full-history exact Bayesian belief/prediction as the scoring target. Fit separate held-out probes for full and restarted activations. Primary predictions, for both trained models: (1) full-minus-restart component-posterior R² loss and (2) restart-minus-full exact-predictive KL increase are larger at overlap .35 than 0.00; (3) at overlap .35 these damages exceed those of untrained models. The exact Bayesian eight-token predictor is an information-loss oracle control. Full conditional-state R² is secondary. Probe-fit/test data seeds are `seed+909`/`seed+1009`, with seeds 0/1/2 and restart window eight. No result is claimed for this unrun control.
