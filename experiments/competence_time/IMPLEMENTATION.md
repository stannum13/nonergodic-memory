# Competence–time experiment implementation plan

The experiment must remain additive: v1.0 files and recorded results are not
rewritten. Each task is completed with tests before implementation and committed
separately where practical.

## Task 1: Freeze the prospective configuration

- [ ] Add `configs/mess3_competence_time.yaml` with exactly the model, data,
  seeds, rates, checkpoints, controls, and thresholds in `PROTOCOL.md`.
- [ ] Add a test asserting that the checked-in configuration matches the
  preregistered grid and excludes seeds 20–24.
- [ ] Record and expose the base configuration digest.
- [ ] Run: `pytest -q tests/test_mess3_competence_time.py -k config`.

## Task 2: Add isolated training and probe storage

- [ ] Add `src/nonergodic_memory/mess3_competence_time.py`.
- [ ] Reuse the tested Mess3 trainer and evaluator without modifying the v1.0
  threshold result schema or paths.
- [ ] Store checkpoints below `checkpoints/mess3_competence_time/` and include
  rate, seed, step, base digest, rate digest, and sampler in every raw row.
- [ ] Test same-seed paired initialization, complete keys, and digest rejection.
- [ ] Run: `pytest -q tests/test_mess3_competence_time.py -k 'training or grid'`.

## Task 3: Implement the frozen confirmatory analysis

- [ ] Exclude step 0 before either model is fitted.
- [ ] Fit quadratic competence-only and log-step-only models with scaling learned
  inside each training fold.
- [ ] Hold out all four rate trajectories of one seed per fold.
- [ ] Aggregate squared error by observations and count per-seed fold wins.
- [ ] Implement the `supported`, `falsified`, and `inconclusive` state machine
  exactly as specified in `PROTOCOL.md`.
- [ ] Add synthetic tests for support, falsification, shuffled-control failure,
  incomplete/non-finite grids, leakage, and failed rate dissociation.
- [ ] Run: `pytest -q tests/test_mess3_competence_time.py -k analysis`.

## Task 4: Add a resumable command and Make target

- [ ] Add `src/mess3_competence_time.py` with `train`, `probe`, `analyze`,
  `figures`, and `all` modes.
- [ ] Cache only checkpoints whose full configuration, seed, condition, and step
  match; reject partial or incompatible JSONL grids.
- [ ] Add `scripts/mess3_competence_time.sh` and `make competence-time`.
- [ ] Test a tiny end-to-end grid, including a second invocation that reuses
  valid cells without duplicating rows.
- [ ] Run: `pytest -q tests/test_mess3_competence_time.py tests/test_cli.py`.

## Task 5: Generate auditable figures from raw rows

- [ ] Add `src/nonergodic_memory/mess3_competence_time_figures.py`.
- [ ] Figure 1: competence and component geometry versus step, faceted by rate,
  with step 0 visually separated.
- [ ] Figure 2: held-out predictions and per-seed competence-versus-step MSE,
  including the aggregate ratio and final verdict.
- [ ] Validate the complete raw grid and matching summary digest before plotting.
- [ ] Test file creation, nonempty pixels, and rejection of mismatched summaries.

## Task 6: Lock the implementation before data generation

- [ ] Run the complete test suite: `pytest -q`.
- [ ] Run repository smoke checks: `make smoke`.
- [ ] Commit the implementation and protocol, then push the experiment branch.
- [ ] Record the lock commit and configuration digest in `STATE.md`.
- [ ] Confirm that no seed 30–37 checkpoint, result row, or figure exists before
  the lock commit.

## Task 7: Execute and report without changing the rules

- [ ] Run `make competence-time`, retaining raw JSONL and elapsed-time logs.
- [ ] Regenerate figures only from the committed raw JSONL files.
- [ ] Append the frozen verdict and all validity checks to this experiment
  directory and update `STATE.md`.
- [ ] Report negative and inconclusive outcomes as such; do not tune thresholds,
  discard seeds, or reclassify secondary analyses as confirmatory.
- [ ] Run `pytest -q && make smoke` and commit the result artifacts.

