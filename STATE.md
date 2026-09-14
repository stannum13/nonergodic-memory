# Experimental state

## Hypothesis

Small next-token predictors trained on a fixed-component mixture will expose linearly separable component identity and within-component predictive state. Probe-row-space erasure will cause selective damage: component erasure will reduce component accuracy more than conditional-state accuracy, and state erasure will do the converse, beyond rank- and norm-matched random controls.

## Last experiment

Model-width sweep at 8, 16, 32, and 64, seeds 0/1/2, on CPU. Each cell fixed two sources, overlap 0.35, length 32, depth 2, 512 training sequences, 12 epochs, and three disjoint intervention datasets.

## Result

- Trained component-posterior R² at width 8/16/32/64: GRU 0.936/0.964/0.984/0.990; Transformer 0.831/0.922/0.936/0.918.
- Untrained R² rises more sharply with width: GRU 0.693/0.778/0.831/0.857; Transformer 0.136/0.434/0.636/0.794. Consequently, trained-minus-untrained gain decreases rather than increases.
- Intended component-erasure damage and coefficient of variation: GRU 0.024 (0.78), 0.080 (0.95), 0.053 (0.62), 0.018 (0.77); Transformer 0.306 (0.48), 0.128 (0.47), 0.115 (0.74), 0.043 (0.71).
- Intended state-erasure CV is also nonmonotonic for the GRU (0.11/0.69/0.86/0.44) and Transformer (0.33/0.16/0.20/0.12).

## Interpretation

The saturation part of the width prediction is broadly supported for GRU and only through width 32 for Transformer. The causal-stability part is falsified: relative seed variability does not decrease monotonically with width. The trained-over-untrained regression gap shrinks because random wide features recover far more belief information. Width improves representational capacity but does not stabilize the learned erasure basis.

## Next smallest experiment

For the two-layer Transformer at the central setting, compare intervention after block 1, after block 2, and after final normalization. The falsifiable prediction is that component-posterior recovery and behavioral damage increase with depth, while independent-evaluator selectivity remains larger than matched controls. This is the final planned sweep.
