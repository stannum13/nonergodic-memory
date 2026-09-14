# Experimental state

## Hypothesis

Small next-token predictors trained on a fixed-component mixture will expose linearly separable component identity and within-component predictive state. Probe-row-space erasure will cause selective damage: component erasure will reduce component accuracy more than conditional-state accuracy, and state erasure will do the converse, beyond rank- and norm-matched random controls.

## Last experiment

End-to-end smoke configuration (`configs/smoke.yaml`), seed 0, on CPU. Two 2-state/four-token HMMs at emission overlap 0.35 generated 96 training sequences of length 12. Width-16, one-layer GRU and Transformer models trained for eight epochs. Probes used 64 independent fit sequences and 48 held-out sequences; interventions used a second disjoint pair of datasets.

## Result

- Held-out NLL: GRU 1.2087, Transformer 1.2313, exact Bayes approximately 1.175.
- Trained component/state accuracies: GRU 0.848/0.822; Transformer 0.826/0.790.
- Untrained component/state accuracies remained high: GRU 0.830/0.801; Transformer 0.703/0.797. Shuffled-label controls were near chance and posterior R² values were negative.
- Learned component erasure changed component/state accuracies by −0.470/−0.062 (GRU) and −0.451/−0.062 (Transformer).
- Learned state erasure changed component/state accuracies by −0.055/−0.210 (GRU) and −0.044/−0.167 (Transformer). Norm-matched random changes were much smaller in this seed.

## Interpretation

The one-seed smoke run supports selective linear-subspace damage but is not confirmatory. Decodability is clearly not unique to training: finite context features are recoverable from random recurrent and attention features. The relevant reproduction statistic is therefore the trained-versus-untrained posterior-regression improvement, not trained accuracy alone. Causal effects must be summarized across seeds and compared directly with norm-matched controls.

## Next smallest experiment

Run the central configuration for seeds 0, 1, and 2, regenerate all figures from the central JSONL files, and quantify means plus seed standard deviations. If selective damage survives, the first follow-up sweep should increase source overlap because it directly weakens component evidence without changing model size.
