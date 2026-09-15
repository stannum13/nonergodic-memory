# Matched Eight-Input-Token Training Control

The corrected prediction is registered in `STATE.md` before running. Because next-token training uses `tokens[:, :-1]`, length-nine source sequences are required to train on all eight input positions of the test windows.

## Protocol

- Train width-32 two-layer Transformers at overlap 0/.35, seeds 0/1/2, on 512 length-nine source sequences for 12 epochs; configs differ from the length-64 interaction checkpoints only in sequence length.
- Reuse exactly the same length-64 held-out probe-fit/test batches, seeds+909/+1009, and last-eight-token windows at positions 7–62. Both short-trained and long-trained reset-index models see window positions 0–7; compare logits and separately fitted held-out posterior probes to the same full-history exact Bayes targets.
- Keep untrained and length-64 original-index results, shuffled labels, and exact eight-token Bayes as controls/references. Save a new self-describing JSONL of short-trained cells with both evaluation and checkpoint config hashes; figures may join this raw file with the already published position-restart raw file.
- Registered primary outcome: paired per-seed `KL_short − KL_long_reset` at overlap .35 should be negative if long-trained restart penalty partly reflects window-input distribution shift. Quantify component/state R², NLL, and both overlaps as secondary diagnostics. Do not assume agreement.

## Execution

1. Test first: evaluator accepts validated short checkpoints and emits one short-trained normal/shuffled pair per seed/overlap, with correct aligned observations and config provenance.
2. Test first: figure joins only complete short and published long raw grids, computes signed paired outcomes, skips when optional short raw is absent, and fails on malformed present data.
3. Train/run on CPU; audit 12 short cells, finite values, independent samples, and checkpoint config/seed validation; plot and interpret versus exact oracle and prior long-window controls.
4. Update README, report, and `STATE.md`; run full tests/figures/shell checks, commit the raw/figure/report loop, and fast-forward both GitHub branches.
