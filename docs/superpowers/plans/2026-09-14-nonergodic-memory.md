# Nonergodic Memory Research Artifact Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CPU-reproducible experiment that compares GRU and decoder-only Transformer representations with exact Bayesian filtering in finite HMM mixtures, then causally erases learned component and conditional-state subspaces with controls.

**Architecture:** A NumPy data layer samples fixed-component HMM sequences and computes exact filtering targets. Small PyTorch sequence models expose final-layer activations; shared experiment utilities train models, fit held-out linear probes, apply final-representation subspace projections, append tidy JSONL records, and regenerate figures. YAML configs separate a seconds-scale smoke run from the central reproduction.

**Tech Stack:** Python 3.11+, NumPy, PyTorch, scikit-learn, matplotlib, PyYAML, pytest, GNU Make.

## Global Constraints

- The artifact must contain a reproducible training pipeline, analytic ground truth, quantitative figures, a falsifiable causal-erasure extension, and a concise technical report.
- Ground truth at every observed position must include component posterior, latent-state posterior conditional on component, and exact next-token predictive distribution.
- Every conclusion must use held-out sequences and multiple seeds; PCA is descriptive only.
- Intervention controls must include random subspaces, norm-matched random interventions, shuffled labels, and untrained models.
- Never fabricate GPU results; all checked-in results identify their device and configuration.
- Every experimental loop ends with raw JSONL, regenerated plots, interpretation in `STATE.md`, and a commit.

---

### Task 1: Exact finite-HMM mixture data layer

**Files:**
- Create: `pyproject.toml`
- Create: `src/nonergodic_memory/__init__.py`
- Create: `src/nonergodic_memory/data/hmm.py`
- Create: `src/data/__init__.py`
- Create: `tests/test_hmm.py`

**Interfaces:**
- Produces: `HMM`, `HMMMixture.sample(n_sequences, length, seed)`, `HMMMixture.filter(tokens)`, and `make_two_source_mixture(overlap)`.
- `filter` returns arrays `component_posterior [N,T,K]`, `state_posterior [N,T,K,S]`, and `predictive [N,T,V]`.

- [ ] **Step 1: Write tests for normalized analytic outputs and a hand-computed one-state mixture.**
- [ ] **Step 2: Run `pytest tests/test_hmm.py -q` and confirm failure because the package is absent.**
- [ ] **Step 3: Implement validated HMM parameters, seeded sampling, log-stable mixture filtering, and overlap-controlled sources.**
- [ ] **Step 4: Run `pytest tests/test_hmm.py -q` and confirm all analytic tests pass.**
- [ ] **Step 5: Commit the exact-data slice.**

### Task 2: Reproducible neural training

**Files:**
- Create: `src/nonergodic_memory/models/sequence.py`
- Create: `src/nonergodic_memory/experiment.py`
- Create: `src/models/__init__.py`
- Create: `src/train.py`
- Create: `configs/smoke.yaml`
- Create: `configs/reproduce.yaml`
- Create: `tests/test_models.py`
- Create: `tests/test_training.py`

**Interfaces:**
- Produces: `GRUPredictor`, `TransformerPredictor`, `build_model`, `train_one`, `evaluate_predictions`, and checkpoint dictionaries containing config plus weights.
- Both models return `(logits, final_hidden)` for token prefixes.

- [ ] **Step 1: Write shape, causal-prefix invariance, reproducibility, and loss-improvement tests.**
- [ ] **Step 2: Run the focused tests and confirm missing-interface failures.**
- [ ] **Step 3: Implement deterministic CPU model construction, next-token batches, evaluation, checkpointing, and JSONL output.**
- [ ] **Step 4: Run focused tests and the complete suite.**
- [ ] **Step 5: Commit the training slice.**

### Task 3: Quantitative representation probes

**Files:**
- Create: `src/nonergodic_memory/analysis.py`
- Create: `src/probe.py`
- Create: `tests/test_analysis.py`

**Interfaces:**
- Produces: flattened held-out activation tables, train/test-split linear classification, Bayesian-posterior regression, shuffled-label controls, and PCA coordinates.
- Appends one tidy record per seed/model/control to `results/reproduction.jsonl`.

- [ ] **Step 1: Write failing tests for leak-free splitting, shuffled controls, metric bounds, and PCA shapes.**
- [ ] **Step 2: Run `pytest tests/test_analysis.py -q` and verify expected failures.**
- [ ] **Step 3: Implement probe fitting only on probe-train sequences and scoring only on held-out probe-test sequences.**
- [ ] **Step 4: Run analysis tests and the complete suite.**
- [ ] **Step 5: Run the smoke reproduction and commit raw records plus probe code.**

### Task 4: Causal erasure with matched controls

**Files:**
- Create: `src/nonergodic_memory/intervention.py`
- Create: `src/intervene.py`
- Create: `tests/test_intervention.py`

**Interfaces:**
- Produces: orthonormal row-space bases from learned probes, learned/random/norm-matched activation erasure, and pre/post metrics for NLL, exact-predictive KL, component accuracy, and conditional-state accuracy.
- Appends one record per target/control/seed/model to `results/extension.jsonl`.

- [ ] **Step 1: Write failing tests proving exact removal, orthogonal preservation, rank matching, and norm matching.**
- [ ] **Step 2: Run `pytest tests/test_intervention.py -q` and verify expected failures.**
- [ ] **Step 3: Implement projection and evaluation using probes trained on disjoint sequences from the intervention test set.**
- [ ] **Step 4: Run intervention tests and the complete suite.**
- [ ] **Step 5: Run the smoke extension and commit raw records plus intervention code.**

### Task 5: Reproducible commands, figures, and report

**Files:**
- Create: `Makefile`
- Create: `scripts/reproduce.sh`
- Create: `scripts/extension.sh`
- Create: `scripts/figures.sh`
- Create: `src/nonergodic_memory/figures.py`
- Create: `README.md`
- Create: `paper_notes.md`
- Create: `STATE.md`
- Create: `report.md`
- Create: `tests/test_cli.py`
- Create: `figures/.gitkeep`
- Create: `results/.gitkeep`

**Interfaces:**
- Produces working `make smoke`, `make train`, `make reproduce`, `make extension`, and `make figures` commands.
- Figures are derived solely from JSONL files and report captions identify sample sizes and uncertainty aggregation.

- [ ] **Step 1: Write CLI/help and figure-from-fixture tests, then run them to verify failure.**
- [ ] **Step 2: Implement orchestration scripts, aggregation plots, README quick explanation, paper notes, state ledger, and technical report.**
- [ ] **Step 3: Run `make smoke` from a clean result target and inspect every emitted JSON record and figure.**
- [ ] **Step 4: Run `pytest -q`, `make train`, `make reproduce`, `make extension`, and `make figures`; document actual CPU runtime and results.**
- [ ] **Step 5: Self-review every mission, control, command, and done criterion; record limitations and negative findings without overstating evidence.**
- [ ] **Step 6: Commit the final reproducible artifact.**

## Plan Self-Review

- Spec coverage: Tasks 1–5 cover analytic targets, both models, held-out quantitative probing, every named intervention metric/control, raw results, figures, state tracking, commands, and reporting.
- Deferred sweeps: overlap, sequence length, component count, width, and intervention depth are config fields or documented follow-ups; the contract explicitly schedules sweeps only after the minimal experiment works.
- Type consistency: training exposes final hidden states consumed by both analysis and intervention; both use the same flattened sequence-position indexing and exact Bayesian targets.
- No GPU claim is permitted by the output schema: each result records `device`, config name, model, and seed.
