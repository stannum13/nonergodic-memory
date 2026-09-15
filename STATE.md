# Experimental state

## Hypothesis

Small next-token predictors trained on a fixed-component HMM mixture should expose linearly separable component identity and within-component predictive state. Component and state erasures should selectively damage their intended targets beyond rank- and norm-matched random controls. The context follow-ups ask whether learned component belief uses remote history rather than only recent tokens.

## Last experiment

Preregistered eight-token context restart on overlap 0.00/0.35 length-64 checkpoints, positions 7–62, seeds 0/1/2, GRU and Transformer, trained/untrained and shuffled-label controls, separate probe-fit/test sequences, plus an exact Bayesian eight-token information-loss oracle. All measurements ran on CPU and were saved as 102 raw JSONL records. The registered plan was committed as `a0207a1` before measuring.

## Result

- The registered overlap contrast in full-minus-restart component R² loss is +0.155 ± 0.003 for trained GRU and +0.150 ± 0.004 for trained Transformer, positive in all seeds; exact Bayes oracle contrast +0.156 ± 0.010. Untrained contrasts are +0.045 ± 0.000 and −0.046 ± 0.023.
- Registered restart-minus-full predictive KL contrasts are +0.0148 ± 0.0016 for trained GRU and +0.0157 ± 0.0057 for trained Transformer, positive in all seeds; exact oracle +0.0148 ± 0.0014. Untrained KL contrasts are +0.0007 ± 0.0005 and +0.0093 ± 0.0403.
- Secondary conditional-state R² contrast is approximately zero for GRU and −0.048 ± 0.009 for trained Transformer; the untrained Transformer is similarly −0.052 ± 0.011. Shuffled-label component R² is within ±0.016.

## Interpretation

The primary context-damage predictions are supported for trained component belief and predictive KL; their overlap contrasts are close to the analytic eight-token information-loss oracle. This strengthens evidence that full-length models use remote tokens at intermediate overlap. It does not establish a stable, selective activation subspace. The Transformer state effect is not training-specific, and restarting resets absolute positions and re-encodes recent tokens at a different sequence length.

## Next smallest experiment

Register before running: repeat the Transformer eight-token restart with the original absolute position indices (window starts at full-prefix position `t−7`) rather than resetting them to zero, using identical checkpoints, tokens, and held-out splits. Compare original-index and reset-index windows to the same full-prefix target and exact oracle. Prediction: preserving indices reduces the trained and untrained conditional-state R² distortion if the current state effect is primarily a positional confound; the trained component-belief and predictive KL overlap contrasts should remain positive if remote-token loss is the main source of their damage. This is a diagnostic, not an already obtained result.
