# Eight-Token Context Restart Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The user requested autonomous inline work and excluded brainstorming.

**Goal:** Test whether trained length-64 models at overlap 0.35 actually depend more on tokens older than eight than at overlap 0.00.

**Architecture:** Reuse validated length-64 checkpoints from the matched interaction grid, but sample fresh probe-fit and test sequences. At positions 7–62, compare full-prefix activations/logits with model activations/logits after restarting on the last eight observations. Keep full-history exact Bayesian targets fixed, and compute the exact eight-token Bayes predictor as an information-loss reference. Fit probes separately for each context on disjoint sequences; score NLL and KL from model logits, not probe outputs.

**Tech Stack:** Python 3.11+, NumPy, PyTorch, scikit-learn, Matplotlib, PyYAML, pytest, Make.

## Global Constraints

- Do not use the brainstorming extension.
- CPU-only honest results; seeds 0/1/2, both GRU and Transformer, trained and untrained controls.
- Validate checkpoints against complete config, model, and seed; retrain only the two required length-64 configurations if unavailable.
- Use overlap 0.00 and 0.35, sequence length 64, restart window 8, probe-fit data seed `seed+909`, test seed `seed+1009`.
- All scores compare the same positions 7–62 and the same full-history exact Bayesian targets.
- Save raw JSONL, figure regenerated from raw, technical interpretation, `STATE.md`, and a commit/push.

---

### Task 1: Register and implement aligned extraction, test-first

**Files:** Modify `STATE.md`; create `src/nonergodic_memory/context.py` and `tests/test_context.py`.

**Interfaces:** `aligned_full_table(table: ActivationTable, window: int) -> ActivationTable`; `collect_restart_activations(model: nn.Module, batch: SequenceBatch, full_table: ActivationTable, window: int, batch_windows: int = 512) -> ActivationTable`; `oracle_window_beliefs(batch: SequenceBatch, mixture: HMMMixture, window: int) -> FilterResult` with final-window rows in flattened sequence-major order.

- [ ] Register primary predictions before run: for trained models, overlap .35 has a larger full-minus-restart component-posterior R² drop and larger restart-minus-full KL increase than overlap 0; the same damage is larger in trained than untrained models at overlap .35. Exact Bayes eight-token KL is the oracle reference, not a model result.
- [ ] Write a failing test: sample four length-10 HMM sequences, collect a full GRU table, and assert `aligned_full_table(full,8)` has eight rows, positions 7/8 for each sequence, and unchanged target/belief labels.
- [ ] Run the targeted test; observe missing-function failure. Implement fieldwise row masking with a `window` range check, then re-run.
- [ ] Write a failing test: `collect_restart_activations` returns the same aligned labels/positions and its first logits equal a manual model run on `tokens[0,:8]` at the final position. Implement `np.lib.stride_tricks.sliding_window_view` over input tokens, batched no-grad model inference, and the aligned `ActivationTable`; re-run.
- [ ] Write a failing test: `oracle_window_beliefs` first row equals `mixture.filter(tokens[0,:8]).predictive[0,-1]` and its component posterior matches the full filter at position 7. Implement batched analytic filtering of flattened windows and take final-window rows; re-run `pytest tests/test_context.py -q`.

### Task 2: Raw evaluator and controls

**Files:** Create `src/context_restart.py`, `scripts/sweep_context_restart.sh`; modify `Makefile`, `tests/test_cli.py`.

**Interfaces:** CLI accepts `--configs` (the two interaction length-64 YAMLs), `--seeds`, `--checkpoint-root`, `--results`. It writes `record_type=context_restart` for each model/seed/overlap/training-condition/context/full-or-restart/probe-control cell, plus `record_type=context_oracle` for each overlap/seed. `replace_jsonl_runs` is called per config with model names `gru`, `transformer`, `exact_bayes`.

- [ ] Add `context_restart.py` to the entrypoint-help test and run red, then implement argparse with those exact options.
- [ ] For each config/seed, sample 256 probe-fit and 192 test sequences with seeds `+909`/`+1009`; collect full/aligned/restart tables for trained and freshly initialized untrained models. Fit held-out normal and shuffled probes separately for full/restart, and evaluate logits with `evaluate_hidden_with_logits` against full exact predictive targets.
- [ ] Save each cell’s component/state R², component/state accuracy, NLL, full-history exact-predictive KL, config digest, library versions, sample seeds, window, aligned position count, and CPU device. Add exact Bayes eight-token component R² and KL/NLL versus full-history targets as oracle rows.
- [ ] The shell script checks the two checkpoint sets, retrains missing/stale ones with existing `src/train.py`, clears only `results/sweep_context_restart.jsonl`, invokes the CLI, and plots from raw. Add `make sweep-context-restart` and run shell syntax/help tests.

### Task 3: Paired analysis and raw-only figure, test-first

**Files:** Create `src/nonergodic_memory/context_figures.py`, `tests/test_context_figures.py`; modify `scripts/figures.sh`.

**Interfaces:** `context_damage(records: list[dict], metric: str) -> dict[tuple[str,str], list[float]]` returns ordered per-seed overlap interaction contrasts for each model and training condition, validating all required context/control cells. `generate_context_figure(path, output) -> Path` reads raw JSONL and plots mean ± seed SD of full-minus-restart component R² and restart-minus-full predictive KL, with trained/untrained model traces and exact-Bayes reference.

- [ ] Write synthetic three-seed/two-model/two-overlap/full-restart fixture with exact known signed contrasts; run the targeted test red, implement paired cell validation and computation, run green.
- [ ] Write a figure fixture test that requires a generated PNG and rejects a removed whole seed; run red, implement raw-only plotting/validation, run green.
- [ ] Add `python -m nonergodic_memory.context_figures` to `scripts/figures.sh`, run `pytest -q` before any real result.

### Task 4: Run, audit, interpret, publish

**Files:** Create `results/sweep_context_restart.jsonl`, `figures/sweep_context_restart.png`; modify `README.md`, `report.md`, `STATE.md`.

- [ ] Run `make sweep-context-restart` on CPU.
- [ ] Audit 96 model records and six oracle records, all three seeds, both models, both training conditions, full/restart contexts, normal/shuffled controls, two matching config digests, finite metrics, and held-out data offsets.
- [ ] Compute registered paired component-R² and KL contrasts, untrained comparisons, state-R² secondary diagnostic, and oracle penalties; show every seed and mean ± population SD.
- [ ] Regenerate/inspect PNG and run `make figures`; update report with positive, negative, and OOD-restart limitations. Update README command/map and `STATE.md` experiment ledger.
- [ ] Run full tests, `git diff --check`, shell checks, and raw audit; commit the raw/result/report loop; fast-forward both GitHub branches only if remote SHAs match the expected prior commit.

## Self-Review

- Full and restarted outputs align to exactly the same test positions and full-history Bayes targets.
- Regression probes are fitted on disjoint sequences; they are not reused across the shifted activation distributions.
- Predictive KL from logits and exact Bayes truncated-history KL provide behavioral and information-loss checks beyond classification.
- The restart mechanism may induce distribution shift; that alternative is named in the report instead of treated as proof of mechanistic memory.
