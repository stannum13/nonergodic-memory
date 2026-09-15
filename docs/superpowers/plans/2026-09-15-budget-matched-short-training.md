# Token- and Step-Matched Short-Training Control

The prediction was registered in `STATE.md` at `b115b52` before any run. This control targets the unresolved supervised-token budget confound in the fixed-sequence-count short-training negative result.

## Protocol

- Use overlap 0/.35, seeds 0/1/2, width-32 two-layer Transformer, learning rate .003, 12 epochs. Train on 4,032 length-nine HMM sequences with batch size 504. There are exactly eight optimizer batches/epoch and 4,032×8 = 32,256 supervised input/next-token pairs/epoch, matching 512 length-64 sequences with batch size 64 and 512×63 = 32,256 tokens.
- Score the exact same reset-index eight-token held-out windows and full-history Bayes targets as standard-short and long-trained restart controls. Fit normal/shuffled probes on independent `seed+909` batches and test on `seed+1009` batches.
- Primary registered contrast at overlap .35: `KL_budget_short − KL_standard_short < 0` per-seed/mean if the previous disadvantage was mostly token-budget-driven. Also report `KL_budget_short − KL_long_reset`; component/state R² and NLL are secondary. The exact eight-token Bayes oracle is a floor only for Bayes information loss, not model approximation error.
- Batch size and training-sequence diversity must change to match tokens and optimizer steps; neither explanation is separately identifiable. No result is claimed yet.

## Execution

1. Test first: budget config validator requires 4,032/504/9 and equivalence to length-64 eval config on all other fields. Extend the short-window evaluator with a separate budget protocol and raw provenance; preserve standard-short behavior.
2. Test first: joined figure requires all 12 budget cells plus complete standard-short and position-control raw files; plot paired seed means±SD and reject incomplete/mixed provenance.
3. Validate/retrain six budget checkpoints and regenerate six raw training cells when absent/incomplete/stale; evaluate and save 12 held-out budget cells on CPU.
4. Audit, inspect plot, report positive and negative findings with batch/diversity limitations, update README/STATE, run full tests/figures/shell checks, commit and fast-forward GitHub branches.
