# Experimental state

## Hypothesis

Small next-token predictors trained on a fixed-component HMM mixture should expose linearly separable component identity and within-component predictive state. Component and state erasures should selectively damage their intended targets beyond rank- and norm-matched random controls. The context follow-ups ask whether learned component belief uses remote history rather than only recent tokens.

## Last experiment

Matched eight-input-token training control, corrected and registered at `b28f940` before running. Trained width-32, two-layer Transformers at overlap 0/.35 and seeds 0/1/2 on length-nine sequences (eight model input positions), then evaluated the same reset-index last-eight-token windows, held-out length-64 batches, and full-history Bayes targets as the length-64 restart control. Twelve short-model cells plus six training records ran on CPU; normal/shuffled probes used independent fit/test sequences.

## Result

- The registered `KL_short − KL_long_reset < 0` prediction at overlap .35 failed in every seed: +0.0434/+0.0289/+0.0176 nats, mean +0.0300 ± 0.0106. NLL is also worse by +0.0291 ± 0.0099. At overlap 0, KL is worse by +0.0216 ± 0.0027.
- Component-posterior R² is lower for short training by 0.048 ± 0.014 at overlap .35 and 0.016 ± 0.003 at overlap 0; conditional-state R² difference is +0.005 ± 0.017 at overlap .35. Shuffled-label component R² is within ±0.014.
- The short-trained/long-trained reset pair uses identical held-out windows, positions, seeds, Bayesian scoring targets, model width/depth, and optimizer steps; short training sees far fewer total next-token targets (8 versus 63 per sequence at fixed 512 sequences/12 epochs).

## Interpretation

Short-context training does not rescue prediction under the fixed-sequence-count protocol; the registered prediction is falsified. That weakens a simple window-distribution-only explanation of the long-model restart penalty, but the short model has a smaller supervised-token budget, so this control cannot distinguish learning longer context from learning more data. Earlier original-index restarts retain positive trained component and predictive overlap contrasts; state-probe effects remain nonselective. No stable selectively necessary activation subspace is established.

## Next smallest experiment

Register before running: train a length-nine Transformer with 4,032 source sequences and batch size 504, giving eight optimizer batches per epoch, 96 steps in 12 epochs, and 4,032×8 = 32,256 supervised tokens per epoch versus the long model's 512×63 = 32,256. Keep overlap 0/.35, seeds, architecture, learning rate, eight-token evaluation windows, and exact targets fixed. The larger short-model batch and greater sequence diversity are new confounds required to match both supervised-token and step budgets. Prediction: if the short model's current KL disadvantage is mainly token-budget-driven, token-matched training will reduce its short-minus-long-reset KL difference at overlap .35; it cannot remove the exact eight-token Bayes information-loss floor. This is a diagnostic, not an already obtained result.
