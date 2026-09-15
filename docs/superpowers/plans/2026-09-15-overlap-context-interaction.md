# Overlap–Context Interaction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This project’s user requested autonomous inline execution and explicitly excluded brainstorming; no subagents are assigned.

**Goal:** Test whether length-related trained-over-untrained component-belief recovery is larger at intermediate source overlap than at zero overlap.

**Architecture:** Run a fresh, matched 2×2 grid: overlaps 0.00/0.35 and sequence lengths 8/64, with all other parameters identical to the existing 12-epoch sweep protocol. The existing train, probe, and intervention CLIs write raw JSONL; a small analysis function computes paired per-seed difference-in-differences, and a figure reads only the new raw files.

**Tech Stack:** Python 3.11+, NumPy, PyTorch, scikit-learn, Matplotlib, PyYAML, pytest, Make.

## Global Constraints

- Do not use the brainstorming extension for this work.
- Keep CPU runs honest; never fabricate GPU results.
- Use seeds 0/1/2, both GRU and Transformer, and matching 512-training-sequence/12-epoch settings in every cell.
- Include the existing shuffled-label and untrained probe controls plus learned, random-subspace, norm-matched, shuffled-label, and untrained intervention controls.
- Keep raw JSONL, figure, interpretation, and `STATE.md` in the inspect → test → run → plot → interpret → commit loop.
- A negative or ambiguous interaction is valid; do not manufacture agreement.

---

### Task 1: Register grid and run contract

**Files:** Modify `STATE.md`; create four `configs/sweeps/interaction_o000_l008.yaml`, `interaction_o000_l064.yaml`, `interaction_o035_l008.yaml`, `interaction_o035_l064.yaml`; create `scripts/sweep_interaction.sh`; modify `Makefile`.

**Interfaces:** The script invokes the existing `src/train.py`, `src/probe.py`, and `src/intervene.py` with `--config`, `--seeds 0 1 2`, matching per-cell checkpoint directories, and shared `results/sweep_interaction_{training,reproduction,extension}.jsonl` outputs.

- [ ] Register primary prediction in `STATE.md` before the grid is run: for each model, `I = [G(0.35,64)−G(0.35,8)]−[G(0,64)−G(0,8)] > 0`, where `G` is trained-minus-untrained component-posterior R² on held-out sequences. State that no result exists yet.
- [ ] Create the four YAML files with only `data.overlap` and `data.sequence_length` differing: `train_sequences: 512`, `test_sequences: 192`, model width 32/layers 2/heads 4/max_length 64, epochs 12/batch 64/learning rate .003, probe train 256/test 192.
- [ ] Write `scripts/sweep_interaction.sh` to clear only its three specific JSONL targets, run all four configs over seeds 0/1/2 and both models, and call `python -m nonergodic_memory.sweeps --axis interaction`. Add `make sweep-interaction`.
- [ ] Run `bash -n scripts/sweep_interaction.sh` and a YAML/digest grid audit before training.

### Task 2: Paired contrast and figure, test-first

**Files:** Modify `tests/test_sweeps.py` and `src/nonergodic_memory/sweeps.py`; modify `scripts/figures.sh`.

**Interfaces:** `interaction_contrasts(probes: list[dict], metric: str) -> dict[str, list[float]]` returns one ordered per-seed contrast list per model, validating that trained and untrained probe records exist for every model/seed/overlap/length cell. `generate_interaction_figure(paths, output) -> Path` plots the 2×2 recovery and causal-control data from raw JSONL.

- [ ] Add a failing test with synthetic four-cell, two-seed probe records where the exact component contrast is +0.20 for GRU and −0.10 for Transformer. Assert paired lists, and assert missing cells raise `ValueError`.
- [ ] Run `pytest tests/test_sweeps.py::test_interaction_contrast_is_paired_and_complete -q`; observe the expected missing-function failure.
- [ ] Implement `interaction_contrasts` using exact `(model,seed,overlap,length,condition)` keys; reject duplicate or missing cells rather than silently averaging.
- [ ] Re-run the targeted test and the whole sweeps test file.
- [ ] Add a failing fixture test for `generate_interaction_figure` containing probe controls and learned/norm-matched intervention records at all four cells; assert a nonempty PNG is generated.
- [ ] Run the targeted figure test; observe the expected missing-function failure.
- [ ] Implement the figure: one model per row; component-posterior and full conditional-state posterior training gains versus length, with separate overlap traces and seed SD; third panel shows learned-minus-norm-matched component-erasure intended accuracy decrease. Put the primary per-seed interaction mean ± seed SD in each model’s component panel title. Add `interaction` to the sweeps CLI and `scripts/figures.sh`.
- [ ] Re-run targeted tests, all sweeps tests, and `pytest -q`.

### Task 3: Execute and interpret without post-hoc selection

**Files:** Create `results/sweep_interaction_*.jsonl`, `figures/sweep_interaction.png`; modify `README.md`, `report.md`, `STATE.md`.

- [ ] Run `make sweep-interaction` on CPU; no GPU claim.
- [ ] Audit exactly 24 training records, 12,096 reproduction records, and 480 intervention records (four configs × both models × three seeds); assert one digest per config, finite numbers, all controls, and independent evaluator flags.
- [ ] Run the paired contrast analysis for component and state R²; print each seed’s values and mean ± population SD for both architectures. Quantify intended learned-minus-norm-matched component-erasure damage as a secondary diagnostic.
- [ ] Regenerate and visually inspect `figures/sweep_interaction.png`; run `make figures` to prove all figures are rebuilt from raw files.
- [ ] Update the technical report with the registered prediction, exact results, controls, any negative or ambiguous findings, and limitations. Update `STATE.md` with last experiment/result/interpretation/next smallest experiment and README’s commands/artifact map.
- [ ] Run `pytest -q`, `git diff --check`, shell syntax checks, and raw-grid audits. Commit the loop atomically, then push the branch and `main` fast-forward if both still point to the previous artifact commit.

## Self-Review

- Spec coverage: the 2×2 interaction is preregistered, newly trained, held-out, multi-seed, controlled, plotted from raw JSONL, interpreted, documented, and committed.
- No unrun result is asserted in this plan.
- The contrast function and figure use the same exact four `(overlap,length)` cells and paired seed keys.
