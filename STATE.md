# Experimental state

## Hypothesis

Small next-token predictors trained on a fixed-component mixture will expose linearly separable component identity and within-component predictive state. Probe-row-space erasure will cause selective damage: component erasure will reduce component accuracy more than conditional-state accuracy, and state erasure will do the converse, beyond rank- and norm-matched random controls.

## Last experiment

Transformer intervention-depth sweep after block 1, block 2, and final normalization, seeds 0/1/2, on CPU at the central setting. Each intervention used disjoint direction-fit, evaluator-fit, and test sequences and was propagated through the actual remaining Transformer layers.

## Result

- Trained component-posterior R² rises with depth: 0.733/0.893/0.928; untrained recovery is 0.493/0.627/0.633.
- Component-erasure Δ exact-predictive KL also rises: 0.0050/0.0074/0.0135. Its intended component-accuracy decrease instead falls: 0.589/0.438/0.410, with large seed SD 0.179/0.224/0.295.
- Conditional-state posterior R² is 0.773/0.774/0.746. State-erasure intended accuracy decrease is 0.152/0.149/0.169, while Δ exact-predictive KL falls 0.0076/0.0026/0.0019.
- Learned intended damage exceeds norm-matched controls at all depths; cross-target accuracy decrease is at most 0.020.

## Interpretation

The prediction is partially supported for component belief: linear recovery and predictive KL damage increase with depth, and selectivity beats matched controls. It is falsified as a general account of both belief types. Component-decoding damage does not increase, conditional-state recovery does not improve, and state-target predictive damage decreases. Depth changes how component information affects prediction, but does not create a uniform hierarchy of increasing causal necessity.

## Next smallest experiment

The planned one-axis sweeps are complete. The next smallest discriminating experiment, if extending the artifact, is a preregistered 2×2 overlap-by-context grid to test whether the observed temporal-integration gain is specifically largest at intermediate source overlap rather than an additive artifact. No result is claimed for that unrun interaction.
