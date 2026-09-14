# Paper notes

Source: Kyle J. Ray, Paul M. Riechers, and Adam S. Shai, “The Geometry of Nonergodic Composition,” *Belief Updates*, 9 September 2026, https://simplex.pub/nonergodic-geometry/ (accessed 14 September 2026).

## Claim being tested

The article treats a nonergodic source as a direct-sum composition: one component is selected for the entire sequence, so Bayesian prediction tracks a weight for each component and a normalized belief state within each component. The weighted component beliefs form telescoping geometries: evidence expands the likely component block and contracts unlikely blocks. If a network represents these beliefs linearly, a linear readout from activations should recover them.

The reported experiment uses two three-state Mess3 sources and a four-layer, width-128 decoder-only Transformer at context length 128. It reports held-out belief-vector regression around R² 0.985 after block three and around 0.99 after the final block, versus about 0.45 in an untrained network. This repository does not claim an exact numerical replication: it uses simpler two-state HMMs, much smaller models, shorter sequences, and three seeds so it can execute cheaply on CPU.

## Reproduction mapping

- Direct-sum component identity ↔ exact `p(c | x_0:t)` and its held-out linear regression/classification.
- Per-component belief geometry ↔ exact `p(s_t | c, x_0:t)` for the true component and held-out regression/classification.
- Residual-stream geometry ↔ final GRU hidden state or normalized final Transformer residual state.
- Visual geometry ↔ PCA coordinates saved in raw JSONL, supported by quantitative probes.
- Paper’s untrained comparison ↔ freshly initialized same-architecture controls for every seed.

## Deliberate extension

The article establishes correlational linear recoverability. This artifact asks a narrower causal question: does projecting out a probe-defined component or conditional-state subspace selectively damage an independently fitted held-out decoder while preserving the other one? Direction fitting, evaluator fitting, and intervention testing use three disjoint sequence sets. Because a low-rank projection can cause generic distribution shift, random subspaces and per-example norm-matched random perturbations are required controls. Shuffled-label directions test probe-fitting artifacts.

## Important non-equivalences

The local HMMs are conventional state-emission HMMs rather than the article’s edge-emitting Mess3 construction. PCA of raw activations is not the paper’s regression into weighted belief coordinates. The intervention acts only at the final representation immediately before the output head. Consequently, the result is a compact conceptual reproduction and falsifiable extension, not a reconstruction of the authors’ exact training run.
