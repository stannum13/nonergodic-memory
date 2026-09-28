# Rate-Aware Clock External Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run an externally validated, new-seed comparison of frozen
competence and rate-aware-clock forecasts for Mess3 component geometry.

**Architecture:** Reuse the tested competence-time trainer and probe evaluator
behind new experiment-specific paths and record types. Fit no model on
confirmation geometry: forecasts are immutable constants whose provenance is
verified against the retained old JSONLs. A separate analyzer joins complete
raw grids, performs seed-equal scoring and validity checks, and emits one of
`supported`, `falsified`, or `inconclusive`.

**Tech Stack:** Python 3, PyTorch, NumPy, Matplotlib, PyYAML, pytest, Make.

## Public implementation-lock boundary

Tasks 1–4 and the Task 5 independent audit are complete through approved code
commit `933f50fc22e72ce4f264b811fc09e039bfe10cad` (`933f50fc`).
The checked RED/GREEN steps below reflect the recorded task reports and commit
history. Fresh whole-suite and smoke verification, immutable-source checks,
and worktree/history zero-artifact evidence are recorded in `STATE.md`.

The documentation commit titled `docs: approve rate-aware implementation lock`
records this final boundary, superseding the initial documentation lock
`5f02d75`. Independent methods, integrity, and code reviewers all APPROVE
after repairs `ec368508` and `933f50fc`, with no remaining Critical or Important
findings. Repairs preserve parameter/audit identity, output safety, strict JSON,
terminal execution evidence, and outcome-independent registered completion;
they do not change scientific rules. Final verification is 470 full-suite tests
and 60 covering regressions passing; smoke passed at `ec368508`, with its
numerical and generic paths unchanged by the follow-up. Fresh artifact/history,
config, source-hash, and protected-file checks through `933f50fc` passed.

Task 5's approved documentation lock
`d1b388054a2cf5ebbac3040a8346f80faf1082a3` was published to
`origin/experiment/rate-aware-clock` before Task 6 execution. A fresh remote-head
query, clean synchronized branch, exact source/config/protocol hashes, frozen
forecast refit, and zero-artifact scan passed before the single registered run.
The published boundary was recorded in the private execution report before
starting. Task 6 execution and raw-evidence validation are complete: the frozen
verdict is **falsified** (clock/competence MSE ratio 1.1795463140097937;
clock wins 4/8 seeds), with every validity condition passing. The full report
is [RESULTS.md](RESULTS.md). Final verification passed 470 tests in 260.77 s
and `make smoke`, both exit 0; unrelated smoke schema rewrites were inspected
and restored. Independent reviews and publication are tracked below.
The single-writer, explicit orphan-checkpoint recovery, known-path alias
validation, and per-image publication
limitations remain recorded in `STATE.md`.

## Global constraints

- No seed 40–47 checkpoint or metric may exist before the protocol/config lock.
- Never modify the frozen competence-time JSONLs, figures, summary, or verdict.
- Forecasts are fixed from seeds 30–37 and never refitted on confirmation data.
- Primary grid is exactly seeds 40–47 × rates 0.003/0.006 × steps
  0/384/768/1152/1536/2048/2560/3072.
- Primary scoring uses only post-initialization block-2 normal-control
  component-posterior R² and weights eight seeds equally.
- No seed replacement, adaptive stopping, threshold change, or added predictor.
- Checkpoints remain ignored; raw JSONL results and generated figures are
  committed.
- Internal `.superpowers/` files remain private and absent from git history.

---

### Task 1: Lock configuration and frozen forecast provenance

**Files:**
- Create: `configs/mess3_rate_aware_clock.yaml`
- Create: `tests/test_mess3_rate_aware_clock.py`
- Modify: `STATE.md`

**Interfaces:**
- Consumes: `load_config(path)` and `config_digest(config)` from
  `nonergodic_memory.experiment`.
- Produces: immutable config mapping containing `forecast.competence`,
  `forecast.rate_aware_clock`, source hashes, confirmation grid, decision
  thresholds, and validity bounds.

- [x] Write a failing config-lock test that asserts exact seeds, rates, steps,
  model/data/probe settings, source hashes, centers, scales, coefficients,
  support ranges, ratio threshold 0.80, minimum wins 7, shuffled bound 0.02,
  dissociation threshold 0.10, and six-seed minimum.
- [x] Run `pytest -q tests/test_mess3_rate_aware_clock.py -k config`; verify it
  fails because the config does not exist.
- [x] Add the exact YAML values from `PROTOCOL.md` and update `STATE.md` with
  the pre-data hypothesis, frozen forecast provenance, and zero-artifact state;
  Task 5 records the later implementation-lock commit before execution.
- [x] Run the config test and verify it passes.
- [x] Commit with `git commit -m "experiment: lock rate-aware clock config"`.

### Task 2: Verify and evaluate immutable forecasts

**Files:**
- Create: `src/nonergodic_memory/mess3_rate_aware_clock.py`
- Modify: `tests/test_mess3_rate_aware_clock.py`

**Interfaces:**
- Produces `verify_forecast_provenance(config: dict, root: Path) -> dict`.
- Produces `forecast_geometry(value: float, spec: dict) -> float`.
- Produces `analyze_rate_aware_clock(config, training, probes, audit) -> dict`.
- Analyzer summary includes `verdict`, `validity_failures`, seed-equal aggregate
  MSEs, clock/competence ratio, strict clock-win count, per-seed errors,
  excluded initialization count, and provenance hashes.

- [x] Write failing tests that recompute both frozen coefficient sets from the
  retained old rows, reject one-bit source-hash/target/coefficient changes, and
  verify exact scalar forecasts.
- [x] Run the provenance tests; verify missing functions fail.
- [x] Implement SHA-256 verification, exact old-row selection, standardized
  quadratic fitting with NumPy least squares, exact serialized config-constant
  checks, and absolute tolerance `1e-12` only for the independent numerical
  refit.
- [x] Run provenance tests and verify they pass.
- [x] Write synthetic failing analyzer tests for supported, valid falsified,
  ratio-pass/win-fail, shuffled failure, dissociation failure, out-of-support
  input, missing/duplicate/nonfinite cells, wrong token audit, accidental seed
  weighting by observations, exact `Q == 0.80`, strict per-seed ties, zero
  competence MSE, and extreme finite step-zero values excluded from support
  checks.
- [x] Implement structural validation, seed-equal scoring, manipulation checks,
  forecast-support checks, the registered zero-denominator ratio rule, and the
  validity-first three-way verdict.
- [x] Run `pytest -q tests/test_mess3_rate_aware_clock.py -k 'forecast or analysis'`
  and verify all tests pass.
- [x] Commit with `git commit -m "feat: add frozen forecast analysis"`.

### Task 3: Add isolated resumable training, probing, and token audit

**Files:**
- Modify: `src/nonergodic_memory/mess3_rate_aware_clock.py`
- Modify: `tests/test_mess3_rate_aware_clock.py`

**Interfaces:**
- Produces `rate_aware_checkpoint_path(root, seed, rate, step) -> Path`.
- Produces `run_rate_aware_training(...) -> list[dict]` and
  `run_rate_aware_probes(...) -> list[dict]` using new record types and paths.
- Produces `audit_token_isolation(config, seeds) -> list[dict]` with actual
  token-array SHA-256 dataset hashes, set sizes, and pairwise intersections.
- Produces strict cache/preflight helpers that persist each completed trajectory
  and never overwrite incompatible/scientifically invalid evidence.

- [x] Write failing tiny-grid tests for paired initialization, exact checkpoint
  identity, complete training/probe keys, per-trajectory persistence on
  interruption, restart reuse, and rejection before side effects.
- [x] Run focused tests and observe the missing storage API failures.
- [x] Reuse the competence-time trainer/evaluator through experiment-specific
  enrichment and paths; implement atomic per-trajectory JSONL replacement.
- [x] Run storage tests and verify they pass.
- [x] Write failing audit tests with a deliberately duplicated token sequence
  and with the normal deterministic disjoint sets.
- [x] Implement byte-stable token-row hashing, whole-dataset hashing, and all
  three pairwise intersection counts per seed.
- [x] Run `pytest -q tests/test_mess3_rate_aware_clock.py -k 'training or probe or audit or cache'`
  and verify all tests pass.
- [x] Commit with `git commit -m "feat: add rate-aware experiment storage"`.

### Task 4: Add CLI, Make target, and audit figures

**Files:**
- Create: `src/mess3_rate_aware_clock.py`
- Create: `src/nonergodic_memory/mess3_rate_aware_clock_figures.py`
- Create: `scripts/mess3_rate_aware_clock.sh`
- Modify: `Makefile`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_mess3_rate_aware_clock.py`

**Interfaces:**
- CLI modes: `train`, `probe`, `audit`, `analyze`, `figures`, `all`.
- Make target: `rate-aware-clock` invokes the isolated shell script.
- Figure generator consumes validated raw rows and a summary exactly equal to a
  fresh analysis, then writes the two protocol filenames.

- [x] Write a failing tiny CLI test using nonconfirmation seeds 6/7 that runs
  `all` twice, verifies row counts and byte-stable reuse, and confirms that an
  invalid complete grid writes `inconclusive` without producer calls.
- [x] Implement argparse modes, preflight-before-side-effects, per-trajectory
  persistence, strict subset selection, and separate default artifact paths.
- [x] Run CLI tests and verify they pass.
- [x] Write failing figure tests for exact filenames, nonwhite pixels, step-zero
  separation, per-seed forecast errors, aggregate ratio/verdict, and stale
  same-digest summary rejection.
- [x] Implement the learning and fixed-forecast figures with complete-grid and
  raw-reanalysis checks before plotting.
- [x] Run focused CLI/figure tests and verify they pass.
- [x] Add the shell script and Make target; run the tiny CLI twice again.
- [x] Commit with `git commit -m "feat: add rate-aware experiment command"`.

### Task 5: Lock and independently audit the implementation

**Files:**
- Modify: `STATE.md`
- Modify: `experiments/rate_aware_clock/IMPLEMENTATION.md`

**Interfaces:**
- Produces a public lock commit, configuration digest, old-source hashes, test
  evidence, and zero-artifact scan before confirmation execution.

- [x] Run `PYTHONPATH=src pytest -q` and `make smoke`; restore any unrelated
  generated result rewrites and record exact counts/exit status.
- [x] Scan the working tree and git history for seed 40–47 rate-aware
  checkpoints, JSONL rows, or figures; require zero before lock.
- [x] Record the implementation lock commit and config digest in `STATE.md`.
- [x] Have an independent reviewer audit protocol/code/test agreement and fix
  every Critical or Important finding without changing the scientific rules.
- [x] Commit checklist evidence with
  `git commit -m "docs: lock rate-aware experiment"`.
- [x] Refresh the approved code lock and council evidence with
  `git commit -m "docs: approve rate-aware implementation lock"`.
- [x] Push the locked branch after the independent audit is resolved and record
  the published lock boundary.

### Task 6: Execute, audit, interpret, and publish the frozen result

**Files:**
- Create: `results/mess3_rate_aware_clock_training.jsonl`
- Create: `results/mess3_rate_aware_clock_probes.jsonl`
- Create: `results/mess3_rate_aware_clock_audit.jsonl`
- Create: `results/mess3_rate_aware_clock_summary.jsonl`
- Create: `figures/mess3_rate_aware_clock_learning.png`
- Create: `figures/mess3_rate_aware_clock_forecasts.png`
- Create: `experiments/rate_aware_clock/RESULTS.md`
- Modify: `STATE.md`
- Modify: `README.md`
- Modify: `report.md`
- Modify: `experiments/rate_aware_clock/IMPLEMENTATION.md`

**Interfaces:**
- `make rate-aware-clock` produces the complete retained grid, audit, summary,
  and figures from the locked implementation.

- [x] Run one uninterrupted `/usr/bin/time -p make rate-aware-clock`, retaining
  a private execution log; resume only identical interrupted work.
- [x] Validate exact record counts 128/768/8/1, all raw provenance, forecast
  coefficients, checkpoint identity, token intersections, finite metrics,
  shuffled bounds, dissociation, support, step-zero exclusion, and seed-equal
  scoring before reading the verdict.
- [x] Write `RESULTS.md` with the frozen verdict, every seed error, all validity
  checks, wall/user/sys time, exploratory motivation, negative results, and
  limitations. Update `STATE.md`, README, and report without weakening the
  earlier registered claims.
- [x] Regenerate both figures only from committed raw JSONLs and verify hashes
  are stable.
- [x] Run `pytest -q` and `make smoke`; restore unrelated generated rewrites.
- [ ] Obtain independent raw-result and whole-branch reviews; fix code-quality
  defects only in clearly labeled post-result commits, preserving raw artifact
  hashes and decision logic.
- [x] Commit intended artifacts and result documentation with
  `git commit -m "experiment: report rate-aware clock validation"`.
- [ ] Push all intended artifacts, open and merge a PR to `main`, and
  verify local/remote `main` plus zero open PRs.
