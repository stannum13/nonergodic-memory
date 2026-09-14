# Experimental state

## Hypothesis

Small next-token predictors trained on a fixed-component mixture will expose linearly separable component identity and within-component predictive state. Probe-row-space erasure will cause selective damage: component erasure will reduce component accuracy more than conditional-state accuracy, and state erasure will do the converse, beyond rank- and norm-matched random controls.

## Last experiment

Sequence-length sweep at 8, 16, 32, and 64 tokens, seeds 0/1/2, on CPU. Each cell fixed source overlap 0.35, width/depth 32/2, 512 training sequences, 12 epochs, and three disjoint intervention datasets. The prediction was registered after the overlap sweep and before this run.

## Result

- GRU trained-minus-untrained component-posterior R² gain across lengths 8/16/32/64: 0.040/0.099/0.153/0.181. Transformer: 0.035/0.138/0.301/0.442.
- Full conditional-state R² gain remained small: GRU −0.010/−0.013/−0.009/−0.006; Transformer −0.011/−0.001/0.027/0.036.
- Intended component-erasure accuracy damage was GRU 0.046/0.012/0.053/0.108 and Transformer 0.010/0.180/0.115/0.462. The length-64 Transformer SD was 0.309, so the apparent causal increase is much less stable than the regression gain.
- Intended state-erasure damage was GRU 0.051/0.028/0.040/0.024 and Transformer 0.045/0.053/0.174/0.195. Every value is a mean across three seeds; seed SDs are plotted in `figures/sweep_length.png`.

## Interpretation

The registered length prediction is supported: trained-over-untrained component-posterior R² gain increases monotonically with context length for both models. This supports the overlap-sweep interpretation that training adds temporal integration where evidence accumulates over context, rather than merely making current-token features linearly accessible. It does not generalize to the full conditional-state posterior. Independent-probe component damage also becomes large at length 64 for the Transformer, but its seed variance is too high for a stable causal claim.

## Next smallest experiment

At fixed overlap 0.35 and length 32, sweep component count through 2, 3, and 4 using distinct emission permutations. The falsifiable prediction is that per-component classification and component-posterior R² fall as the identity simplex expands at fixed model width. Width and intervention depth follow after this test.
