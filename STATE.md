# Experimental state

## Hypothesis

Small next-token predictors trained on a fixed-component mixture will expose linearly separable component identity and within-component predictive state. Probe-row-space erasure will cause selective damage: component erasure will reduce component accuracy more than conditional-state accuracy, and state erasure will do the converse, beyond rank- and norm-matched random controls.

## Last experiment

Central configuration (`configs/reproduce.yaml`), seeds 0/1/2, on CPU. Two 2-state/four-token HMMs at emission overlap 0.35 generated 1,024 training sequences of length 32. Width-32, two-layer GRU and Transformer models trained for 20 epochs. Probes used 512 independent fit sequences and 256 held-out sequences; interventions used three disjoint direction-fit, evaluator-fit, and test datasets.

## Result

- Mean ± seed SD held-out NLL: GRU 1.2084 ± 0.0119, Transformer 1.2383 ± 0.0141, exact Bayes 1.2039 ± 0.0112.
- Trained component/full-conditional-state posterior R²: GRU 0.9901 ± 0.0021 / 0.9632 ± 0.0035; Transformer 0.9344 ± 0.0041 / 0.7439 ± 0.0209. Untrained values were 0.8265/0.9762 and 0.6452/0.7325 respectively. Shuffled-label R² was near zero.
- With independent evaluator probes, learned component erasure changed component/state accuracy by −0.0309 ± 0.0341 / −0.0014 ± 0.0036 (GRU) and −0.4097 ± 0.2948 / −0.0079 ± 0.0028 (Transformer).
- Learned state erasure changed component/state accuracy by −0.0081 ± 0.0022 / −0.1113 ± 0.0835 (GRU) and −0.0023 ± 0.0020 / −0.1686 ± 0.0388 (Transformer). Norm-matched controls were near zero, but intended damage varied substantially across seeds.
- Predictive damage remained small except Transformer component erasure at mean ΔNLL +0.0143 ± 0.0064. Untrained networks also displayed selective erasure effects.

## Interpretation

The central run reproduces training-enhanced linear recovery of component belief. It does not reproduce a training-specific conditional-state result: random GRU features perform better and Transformer improvement is marginal. Independent-evaluator erasure reveals selective organization on average, especially in the Transformer, but high seed variance and strong untrained effects falsify the stronger hypothesis of stable, training-created selective subspaces. Output behavior is mostly robust to final-layer erasure.

## Next smallest experiment

Sweep source overlap through 0.0, 0.35, 0.7, and 0.9 at fixed width/length. The falsifiable prediction is that trained-over-untrained component-posterior R² and component-erasure selectivity shrink as sources become observationally indistinguishable. Only after that should sequence length, component count, width, and intervention depth be varied.
