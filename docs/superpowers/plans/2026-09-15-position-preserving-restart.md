# Position-Preserving Restart Control

The prediction was registered in `STATE.md` at commit `07f799b` before any run. The existing length-64 Transformer restart resets absolute position embeddings to zero; this control isolates that intervention from removal of remote tokens.

## Protocol

- Reuse overlap 0/.35 length-64 Transformer checkpoints, seeds 0/1/2, trained/untrained controls, and the same independent fit/test batches (seed+909/+1009) as the eight-token restart.
- At positions 7–62, compare full prefix, last-eight-token reset-position restart, and last-eight-token original-position restart. The latter assigns position IDs `t−7,...,t` to the window. The exact full-history Bayesian targets, token windows, model weights, and observation counts are identical.
- Fit probes independently for each context, with normal and shuffled-label controls. Score component and full conditional-state posterior R², NLL, and KL to exact full-history prediction. Keep the existing exact eight-token Bayes oracle rows.
- Primary diagnostic: original-index minus reset-index component and state R², and reset-index minus original-index predictive KL, evaluated per overlap and condition. Prediction: original positions reduce untrained and trained Transformer state distortion if positional reset caused it. Trained component and predictive KL full-to-original-index overlap contrasts should remain positive if remote history is the dominant cause. No result is claimed yet.

## Execution

1. Test first: Transformer per-window position IDs change later-window outputs but preserve the first-window output; restart tables remain aligned. Add optional per-example offsets to the model and restart collector.
2. Test first: CLI emits the 78 required cells and reuses the exact same data seeds/configs; script validates matching checkpoints and saves JSONL. Existing reset-only CLI behavior and outputs remain unchanged.
3. Test first: raw-only figure validator enforces complete 3-context grid and computes signed seed-paired contrasts; absent optional result skips figure, malformed present result errors.
4. Run on CPU; audit 72 Transformer + 6 oracle records, all axes/provenance/finite metrics; compare registered predictions and existing reset-only raw cells; plot, inspect, report negatives/limitations, update `STATE.md`, test, commit, push to GitHub.
