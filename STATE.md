# Experimental state

## Hypothesis

Small next-token predictors trained on a fixed-component mixture will expose linearly separable component identity and within-component predictive state. Probe-row-space erasure will cause selective damage: component erasure will reduce component accuracy more than conditional-state accuracy, and state erasure will do the converse, beyond rank- and norm-matched random controls.

## Last experiment

Component-count sweep at 2, 3, and 4 sources, seeds 0/1/2, on CPU. Each cell fixed overlap 0.35, length 32, width/depth 32/2, 512 training sequences, 12 epochs, and three disjoint intervention datasets. The generalized generator preserves the original two-source emissions and adds distinct emission permutations for sources three and four.

## Result

- Trained component accuracy for 2/3/4 sources: GRU 0.949/0.836/0.759; Transformer 0.944/0.807/0.729.
- Trained component-posterior R²: GRU 0.984/0.931/0.877; Transformer 0.936/0.762/0.630. Untrained R² was 0.831/0.698/0.616 and 0.636/0.415/0.360 respectively.
- Intended trained component-erasure damage: GRU 0.053/0.227/0.195; Transformer 0.115/0.366/0.356. Untrained damage was 0.111/0.280/0.266 and 0.271/0.339/0.324, so increased damage is not training-specific.

## Interpretation

The registered component-count prediction is supported: both absolute classification and posterior R² decline as the component simplex expands at fixed width. Training retains meaningful advantage over random features, especially for the Transformer. Independent component-erasure damage grows from two to three components but does not increase further at four and is comparable in untrained networks. This continues to separate the robust correlational result from the unstable training-specific causal claim.

## Next smallest experiment

At fixed overlap 0.35, length 32, and two components, sweep model width through 8, 16, 32, and 64. The falsifiable prediction is that trained component-posterior R² saturates while independent-erasure stability (lower relative seed SD) improves with width. Intervention depth follows after this test.
