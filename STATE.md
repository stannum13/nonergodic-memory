# Experimental state

## Hypothesis

Small next-token predictors trained on a fixed-component HMM mixture should expose linearly separable component identity and within-component predictive state. Component and state erasures should selectively damage their intended targets beyond rank- and norm-matched random controls. The context follow-ups ask whether learned component belief uses remote history rather than only recent tokens.

## Last experiment

Token- and optimizer-step-matched length-nine Transformer training control, registered at `b115b52` and planned at `403162a` before any result. Used 4,032 sequences/batch 504 versus long training's 512 sequences/batch 64, giving exactly 32,256 supervised tokens and eight optimizer steps per epoch for both models across 12 epochs. Reused the same length-64 held-out reset-index eight-token windows, full-history Bayes targets, seeds 0/1/2, and overlap 0/.35. Six CPU training and 12 independent-probe budget cells were saved.

## Result

- The registered budget-minus-standard-short KL contrast at overlap .35 is negative for all seeds: −0.0611/−0.0283/−0.0189 nats, mean −0.0361 ± 0.0181. NLL improves by −0.0347 ± 0.0149 and component R² by +0.053 ± 0.021. At overlap 0, KL improves by −0.0291 ± 0.0026.
- Budget-minus-long-reset KL at overlap .35 is −0.0177/+0.0006/−0.0012 nats; mean budget-short KL 0.0340 versus standard-short 0.0701, long reset 0.0401, exact eight-token Bayes information-loss floor 0.0225. Shuffled-label component R² remains within ±0.017.
- The previous fixed-sequence-count short-training prediction failed, but matching supervised tokens and steps removes most of its predictive penalty. Larger batch size and greater training-sequence diversity change with budget and remain alternative explanations.

## Interpretation

The budget-matched follow-up supports the registered prediction and overturns the temptation to attribute fixed-count short-model failure to long-context training alone. Equal token/step budgets, larger short-model batches, and more diverse short training data jointly explain its improvement; no single factor is isolated. The full-prefix versus window history contrast still follows the exact Bayesian information-loss oracle, so remote tokens matter for component belief. State-probe effects and subspace erasure remain nonselective/seed-sensitive; no stable selectively necessary activation subspace is established.

## Next smallest experiment

Register before running: train both fixed-count (512 sequences/batch 64) and token/step-matched (4,032 sequences/batch 504) length-nine GRUs at overlap .35, then compare them to the published length-64 GRU restarts on the same seeds, held-out windows, and exact Bayes targets. Prediction: `KL_standard_short − KL_budget_short > 0` in all seeds if the exposure effect is architecture-general; conditional-state probe behavior is secondary. This is an unrun architecture check, not a claimed result.
