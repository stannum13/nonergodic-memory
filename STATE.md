# Experimental state

## Hypothesis

Small next-token predictors trained on a fixed-component HMM mixture should expose linearly separable component identity and within-component predictive state. Component and state erasures should selectively damage their intended targets beyond rank- and norm-matched random controls. The context follow-ups ask whether learned component belief uses remote history rather than only recent tokens.

## Last experiment

Preregistered position-preserving Transformer restart on the same overlap 0.00/0.35 length-64 checkpoints and held-out batches, positions 7–62, seeds 0/1/2. Compared full prefixes with identical last-eight-token windows using either reset or original absolute position indices. Independent probes, trained/untrained and shuffled-label controls, and exact eight-token Bayes oracle yielded 78 CPU raw records. The prediction was registered in `07f799b` before measuring.

## Result

- The predicted absolute-index recovery benefit failed: at overlap .35, original-minus-reset component R² is −0.034 ± 0.002 trained and −0.284 ± 0.035 untrained; conditional-state R² is −0.137 ± 0.030 and −0.221 ± 0.030. All seeds are negative.
- With original indices, trained full-to-window component R² loss overlap contrast is +0.164 ± 0.006, versus untrained +0.034 ± 0.004; trained exact-predictive KL contrast is +0.0157 ± 0.0025, versus untrained −0.0006 ± 0.0021. All trained seed contrasts remain positive. The exact eight-token oracle contrasts are +0.156 ± 0.010 and +0.0148 ± 0.0014.
- Original indices attenuate the overlap-specific conditional-state R² interaction from −0.048 to −0.020 trained and −0.052 to −0.005 untrained, although absolute state recoverability is worse. Full/reset/oracle rows match the previous sweep numerically exactly.

## Interpretation

Original indices do not simply rescue restart probes; the registered positional-benefit prediction is falsified. However, trained component and predictive overlap contrasts survive without positional reset, strengthening remote-history evidence at intermediate overlap and eliminating the untrained predictive contrast. State-probe distortion is not training-specific and is sensitive to index policy. Window restart still changes attention length and re-encodes recent tokens; no stable selectively necessary activation subspace is established.

## Next smallest experiment

Corrected before running: a length-eight next-token training sequence supplies only seven input tokens, so train matched length-nine Transformer checkpoints (eight input positions) at overlap 0.00/.35. Score these models on the same reset-index eight-token windows and full-history Bayesian targets as the length-64 reset-index restarts; original-index long restarts remain a secondary reference. Hold architecture width/depth, seeds, probe-fit/test batches, and window tokens fixed; vary training sequence length only. Prediction: the short-trained model has lower eight-token predictive KL than the length-64 restarted model at overlap .35 if part of the latter penalty is window-input distribution shift; neither model can recover older-token Bayes information from the eight-token window. This is a diagnostic, not an already obtained result.
