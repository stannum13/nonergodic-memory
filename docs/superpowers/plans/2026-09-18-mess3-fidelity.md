# Mess3 Fidelity Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separately registered, CPU-executable reproduction of the paper's exact two-Mess3 nonergodic source and quantify recovery of its six-coordinate weighted joint belief geometry.

**Architecture:** Preserve the existing two-state experiment and select the data generator through an explicit configuration field. Represent each Mess3 edge-emitting process with the equivalent transition-then-emission factorization, which is exact for the published matrices, and use the stationary uniform prior. Extend held-out analysis with the joint target `P(C=c,S_t=s|x_<=t)`, geometric distortion metrics, and ground-truth-versus-reconstruction cone plots.

**Tech Stack:** Python 3.11, NumPy, PyTorch, scikit-learn, Matplotlib, PyYAML, pytest.

## Global Constraints

- Do not alter or overwrite the existing central experiment or its raw results.
- Use the exact published Mess3 parameters `(x=0.15, alpha=0.60)` and `(x=0.50, alpha=0.66)` with equal component weights.
- Treat this as a direct data/process reproduction and clearly disclose remaining architecture and compute differences from the paper.
- Use held-out sequences, trained/untrained controls, shuffled targets, and seeds 0, 1, and 2.
- Save raw outputs as JSONL and generate every figure only from those records.
- Run on CPU and make no GPU claim.

---

### Task 1: Exact Mess3 generator

**Files:**
- Modify: `src/nonergodic_memory/data/hmm.py`
- Modify: `src/nonergodic_memory/data/__init__.py`
- Test: `tests/test_hmm.py`

**Interfaces:**
- Produces: `make_mess3(alpha: float, x: float) -> HMM` and `make_mess3_mixture() -> HMMMixture`.
- Preserves: all existing `HMM`, `HMMMixture`, and source-overlap constructors.

- [ ] **Step 1: Write failing published-matrix tests**

```python
def test_mess3_factorization_matches_published_labeled_operators() -> None:
    hmm = make_mess3(alpha=0.6, x=0.15)
    labeled = np.stack([hmm.transition * hmm.emission[:, token][None, :] for token in range(3)])
    expected_a = np.array([[.42, .03, .03], [.09, .14, .03], [.09, .03, .14]])
    expected_b = np.array([[.14, .09, .03], [.03, .42, .03], [.03, .09, .14]])
    expected_c = np.array([[.14, .03, .09], [.03, .14, .09], [.03, .03, .42]])
    np.testing.assert_allclose(labeled, np.stack([expected_a, expected_b, expected_c]))
    np.testing.assert_allclose(hmm.initial, np.full(3, 1 / 3))


def test_published_mess3_mixture_has_two_three_state_components() -> None:
    mixture = make_mess3_mixture()
    assert mixture.vocab_size == 3
    assert [component.n_states for component in mixture.components] == [3, 3]
    np.testing.assert_allclose(mixture.weights, [0.5, 0.5])
```

- [ ] **Step 2: Run the tests and verify the constructors are missing**

Run: `pytest tests/test_hmm.py -q`

Expected: collection or test failure because `make_mess3` and `make_mess3_mixture` do not exist.

- [ ] **Step 3: Implement the exact factorization**

```python
def make_mess3(alpha: float, x: float) -> HMM:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    if not 0.0 <= x <= 0.5:
        raise ValueError("x must lie in [0, 0.5]")
    beta = (1.0 - alpha) / 2.0
    y = 1.0 - 2.0 * x
    transition = np.full((3, 3), x)
    np.fill_diagonal(transition, y)
    emission = np.full((3, 3), beta)
    np.fill_diagonal(emission, alpha)
    return HMM(transition, emission, np.full(3, 1.0 / 3.0))


def make_mess3_mixture() -> HMMMixture:
    return HMMMixture(
        [make_mess3(alpha=0.60, x=0.15), make_mess3(alpha=0.66, x=0.50)],
        [0.5, 0.5],
    )
```

- [ ] **Step 4: Run analytic tests**

Run: `pytest tests/test_hmm.py -q`

Expected: all HMM tests pass.

- [ ] **Step 5: Commit the generator**

```bash
git add src/nonergodic_memory/data/hmm.py src/nonergodic_memory/data/__init__.py tests/test_hmm.py
git commit -m "feat: add exact published Mess3 mixture"
```

### Task 2: Generator selection and provenance

**Files:**
- Modify: `src/nonergodic_memory/experiment.py`
- Modify: `src/train.py`
- Modify: `src/probe.py`
- Create: `configs/mess3_cpu.yaml`
- Test: `tests/test_training.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `make_mess3_mixture()` from Task 1.
- Produces: `mixture_from_config()` dispatch for `data.generator` values `simple` and `mess3`.
- Records: a `generator` field in training and probe JSONL rows.

- [ ] **Step 1: Write a failing config-dispatch test**

```python
def test_mixture_from_config_selects_published_mess3() -> None:
    config = minimal_config()
    config["data"]["generator"] = "mess3"
    config["data"].pop("overlap")
    mixture = mixture_from_config(config)
    assert mixture.vocab_size == 3
    assert [component.n_states for component in mixture.components] == [3, 3]
```

- [ ] **Step 2: Run the test and verify it fails on the required overlap lookup**

Run: `pytest tests/test_training.py::test_mixture_from_config_selects_published_mess3 -q`

Expected: FAIL because `mixture_from_config` only supports the simple overlap generator.

- [ ] **Step 3: Add explicit dispatch and a reusable provenance helper**

```python
def generator_name(config: dict) -> str:
    return str(config["data"].get("generator", "simple"))


def mixture_from_config(config: dict) -> HMMMixture:
    data = config["data"]
    generator = generator_name(config)
    if generator == "mess3":
        return make_mess3_mixture()
    if generator == "simple":
        return make_source_mixture(
            n_components=int(data.get("components", 2)), overlap=float(data["overlap"])
        )
    raise ValueError(f"unknown data generator: {generator}")
```

Use `generator_name(config)` in training and probing records. Record `overlap` only when present instead of fabricating an overlap value for Mess3.

- [ ] **Step 4: Add a CPU reproduction config**

```yaml
data:
  generator: mess3
  sequence_length: 64
  train_sequences: 2048
  test_sequences: 256
model:
  width: 32
  layers: 2
  heads: 4
  max_length: 128
train:
  epochs: 24
  batch_size: 64
  learning_rate: 0.003
probe:
  train_sequences: 1024
  test_sequences: 512
```

- [ ] **Step 5: Run dispatch and CLI tests**

Run: `pytest tests/test_training.py tests/test_cli.py -q`

Expected: all selected tests pass.

- [ ] **Step 6: Commit config support**

```bash
git add src/nonergodic_memory/experiment.py src/train.py src/probe.py configs/mess3_cpu.yaml tests/test_training.py tests/test_cli.py
git commit -m "feat: select Mess3 experiments by config"
```

### Task 3: Weighted joint-belief regression

**Files:**
- Modify: `src/nonergodic_memory/analysis.py`
- Modify: `src/probe.py`
- Test: `tests/test_analysis.py`

**Interfaces:**
- Produces: `ActivationTable.joint_belief`, `ProbeBundle.joint_regression`, and metrics `joint_belief_r2`, `joint_belief_mse`, `joint_distance_r2`.
- Definition: `joint_belief[..., c, s] = component_posterior[..., c] * state_posterior[..., c, s]`.

- [ ] **Step 1: Write a failing target-factorization test**

```python
def test_activation_table_joint_belief_is_weighted_conditional_state() -> None:
    mixture = make_mess3_mixture()
    batch = mixture.sample(4, 7, seed=10)
    model = GRUPredictor(vocab_size=3, width=8)
    table = collect_activations(model, batch, mixture)
    expected = (
        table.component_posterior[:, :, None]
        * table.state_posterior.reshape(-1, 2, 3)
    ).reshape(-1, 6)
    np.testing.assert_allclose(table.joint_belief, expected)
    np.testing.assert_allclose(table.joint_belief.sum(axis=1), 1.0)
```

- [ ] **Step 2: Run the test and verify the field is absent**

Run: `pytest tests/test_analysis.py::test_activation_table_joint_belief_is_weighted_conditional_state -q`

Expected: FAIL because `ActivationTable` has no `joint_belief`.

- [ ] **Step 3: Add joint targets and held-out regression**

Compute the flattened joint target when constructing `ActivationTable`. Fit a separate standardized ridge regression on training sequences, score it on test sequences, and add:

```python
joint_prediction = joint_regression.predict(test.hidden)
joint_mse = float(np.mean((test.joint_belief - joint_prediction) ** 2))
joint_r2 = float(r2_score(test.joint_belief, joint_prediction))
joint_distance_r2 = pairwise_distance_r2(test.joint_belief, joint_prediction, seed)
```

Define `pairwise_distance_r2` to sample at most 20,000 deterministic unordered pairs and score predicted pairwise Euclidean distances against exact pairwise distances. Shuffle joint targets with an independent permutation in the shuffled control.

- [ ] **Step 4: Add deterministic metric tests**

Test that identical coordinates produce distance `R²=1`, constant predictions produce a non-positive score, and two calls with the same seed return the same value.

- [ ] **Step 5: Run analysis tests**

Run: `pytest tests/test_analysis.py -q`

Expected: all analysis tests pass.

- [ ] **Step 6: Commit joint-belief analysis**

```bash
git add src/nonergodic_memory/analysis.py src/probe.py tests/test_analysis.py
git commit -m "feat: quantify weighted Mess3 belief geometry"
```

### Task 4: Reconstructible Mess3 geometry figure

**Files:**
- Create: `src/nonergodic_memory/mess3_figures.py`
- Modify: `src/probe.py`
- Create: `scripts/reproduce_mess3.sh`
- Modify: `scripts/figures.sh`
- Modify: `Makefile`
- Test: `tests/test_mess3_figures.py`

**Interfaces:**
- Produces: JSONL records of type `mess3_geometry` containing exact and reconstructed six-coordinate joint beliefs for a deterministic held-out subset.
- Produces: `figures/mess3_geometry.png` and `figures/mess3_metrics.png` solely from `results/mess3_reproduction.jsonl` and `results/mess3_training.jsonl`.
- Adds: `make reproduce-mess3`.

- [ ] **Step 1: Write failing figure tests using synthetic JSONL rows**

The test must create three seeds of probe rows plus geometry rows with `exact_b0` through `exact_b5` and `pred_b0` through `pred_b5`, call `generate_mess3_figures`, and assert both nonempty PNG files exist. A second test must remove one coordinate and assert a descriptive `ValueError`.

- [ ] **Step 2: Run the figure tests and verify the module is missing**

Run: `pytest tests/test_mess3_figures.py -q`

Expected: collection failure because `nonergodic_memory.mess3_figures` does not exist.

- [ ] **Step 3: Emit deterministic geometry rows**

For trained models only, use the fitted joint regression to reconstruct the held-out test targets. Select at most 2,000 evenly spaced rows and write all six exact and predicted coordinates with model, seed, component label, position, config digest, and runtime provenance.

- [ ] **Step 4: Implement the figures**

`mess3_geometry.png` must show exact and reconstructed component blocks as four 3D scatter panels, using the posterior mass of the plotted component as color. `mess3_metrics.png` must show mean ± population seed SD for joint-belief R², pairwise-distance R², and joint-belief MSE across trained, untrained, and shuffled-target controls.

- [ ] **Step 5: Add the experiment command**

```make
reproduce-mess3:
	bash scripts/reproduce_mess3.sh
```

The script trains Transformer seeds 0, 1, and 2 if compatible checkpoints are absent, runs held-out probing, and generates the two Mess3 figures.

- [ ] **Step 6: Run figure and CLI tests**

Run: `pytest tests/test_mess3_figures.py tests/test_cli.py -q`

Expected: all selected tests pass.

- [ ] **Step 7: Commit the reproducible command and figures**

```bash
git add src/nonergodic_memory/mess3_figures.py src/probe.py scripts/reproduce_mess3.sh scripts/figures.sh Makefile tests/test_mess3_figures.py tests/test_cli.py
git commit -m "feat: add reproducible Mess3 geometry experiment"
```

### Task 5: Register, run, interpret, and release the result

**Files:**
- Modify: `STATE.md`
- Modify: `README.md`
- Modify: `paper_notes.md`
- Modify: `report.md`
- Create: `results/mess3_training.jsonl`
- Create: `results/mess3_reproduction.jsonl`
- Create: `figures/mess3_geometry.png`
- Create: `figures/mess3_metrics.png`

**Interfaces:**
- Consumes: `make reproduce-mess3` from Task 4.
- Produces: a preregistered prediction, raw three-seed CPU evidence, an explicit fidelity table, and a reproduction-first interpretation.

- [ ] **Step 1: Register the prediction before running**

Append to `STATE.md`:

```markdown
## Registered Mess3 fidelity prediction

For the exact published two-Mess3 source, trained Transformer joint-belief R² will exceed its same-seed untrained control in all three seeds. Pairwise-distance R² is secondary. The experiment is a direct data/process reproduction but not an exact compute reproduction: width 32, two layers, absolute positions, LayerNorm, and CPU training differ from the published width-128 four-layer TransformerLens model with rotary positions, RMSNorm, gated GELU, and 45,000 optimization steps.
```

Commit this registration before producing result files.

- [ ] **Step 2: Run the complete Mess3 experiment**

Run: `make reproduce-mess3`

Expected: three trained Transformer checkpoints, training and reproduction JSONL files, and two figures. Preserve the result even if the registered prediction fails.

- [ ] **Step 3: Verify raw-result completeness**

Check that seeds `{0,1,2}` exist for trained/untrained and normal/shuffled controls, every normal row has finite joint metrics, all geometry rows contain 12 belief coordinates, and each six-coordinate exact target sums to one within numerical tolerance.

- [ ] **Step 4: Update the research narrative**

Add a report section headed `Direct Mess3 fidelity reproduction`. State the registered outcome seed by seed, disclose every architecture/compute difference, distinguish joint weighted beliefs from conditional-state probes, and avoid describing visual similarity as quantitative evidence. Update `paper_notes.md` with the exact matrix factorization and README with the new command.

- [ ] **Step 5: Run full verification**

Run: `pytest -q`

Expected: all tests pass.

Run: `make reproduce-mess3`

Expected: cached-checkpoint rerun reproduces the same numeric JSON values and regenerates both figures.

- [ ] **Step 6: Commit the registered result**

```bash
git add STATE.md README.md paper_notes.md report.md results/mess3_training.jsonl results/mess3_reproduction.jsonl figures/mess3_geometry.png figures/mess3_metrics.png
git commit -m "exp: reproduce weighted Mess3 belief geometry"
```

