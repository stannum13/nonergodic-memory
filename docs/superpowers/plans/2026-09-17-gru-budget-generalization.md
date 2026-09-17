# GRU Token-Budget Generalization Control

The prediction is registered in `STATE.md` at commit `534747d` before any result. This closes the remaining architecture-generalization question rather than leaving it as an unrun footnote.

## Protocol

- At overlap .35 and seeds 0/1/2, train width-32 two-layer GRUs on length-nine sources under both standard-short (512 sequences, batch 64) and budget-short (4,032 sequences, batch 504) protocols, 12 epochs, learning rate .003.
- Reuse the same held-out length-64 batches, reset-index last-eight-token windows, and full-history exact Bayes targets used by the published GRU restart control. Fit normal/shuffled probes independently on seed+909 and test on seed+1009 sequences.
- Primary registered contrast: `KL_standard_short − KL_budget_short > 0` in every seed. Compare budget-short to the published long-trained GRU restart as a secondary diagnostic; component/state posterior R² and NLL are secondary.
- Save six training cells per protocol and six evaluation cells per protocol, a raw-only joined figure, limitations, and exact config/checkpoint provenance. Batch size and training-sequence diversity remain bundled with budget matching.

## Verification

1. Extend the tested short-window evaluator to accept GRU without changing Transformer records.
2. Validate complete standard/budget GRU training grids and checkpoints, then run CPU training/evaluation.
3. Require complete raw grids and published GRU restart controls in the figure; audit signed paired metrics and inspect output.
4. Update README/report/STATE, run full tests and figures, independently review, commit, push, then run the required commands and sweeps from a fresh clone.
