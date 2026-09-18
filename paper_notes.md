# Paper notes

Source: Kyle J. Ray, Paul M. Riechers, and Adam S. Shai, [“The Geometry of Nonergodic Composition,” *Belief Updates*](https://simplex.pub/nonergodic-geometry/), 9 September 2026 (accessed 14 and 18 September 2026).

## Claim being tested

The article treats a nonergodic source as a direct-sum composition: one component is selected for the entire sequence, so Bayesian prediction tracks a weight for each component and a normalized belief state within each component. The weighted component beliefs form telescoping geometries: evidence expands the likely component block and contracts unlikely blocks. If a network represents these beliefs linearly, a linear readout from activations should recover them.

The published weighted-belief readout reaches approximately R² 0.985 after block three and 0.99 after block four, versus 0.45 without training. The original experiments here use simpler two-state HMMs as a conceptual reproduction and causal extension. The separately registered `make reproduce-mess3` experiment now reproduces the exact two-Mess3 observation process and weighted target on three CPU seeds. Its smaller network and training protocol do not reconstruct the published training run; the explicit fidelity table and measured result are in [the report](report.md#direct-mess3-fidelity-reproduction).

## Exact Mess3 matrix factorization

For each component, define `beta = (1 - alpha)/2`, `y = 1 - 2*x`, and

```text
A = [[y, x, x],       E = [[alpha, beta,  beta ],
     [x, y, x],            [beta,  alpha, beta ],
     [x, x, y]]            [beta,  beta,  alpha]]

T^(a) = A diag(alpha, beta, beta)
T^(b) = A diag(beta, alpha, beta)
T^(c) = A diag(beta, beta, alpha)
```

Thus `T^(k)[i,j] = A[i,j] E[j,k]`: transitioning from `i` to `j` and then emitting token `k` from `j` gives the published labeled operators exactly. This is an exact factorization, not an approximate replacement of an edge-emitting process. The implementation uses `(x,alpha) = (0.15,0.60)` and `(0.50,0.66)`, component weights `(1/2,1/2)`, and uniform initial state `pi = (1/3,1/3,1/3)`. Because `pi A = pi`, omitting a transition before the first emission in the existing sampler/filter leaves the first-token joint state distribution unchanged. Subsequent updates multiply by `A` and then the appropriate diagonal emission matrix. Matrix-identity and enumerated-short-sequence tests check this equivalence.

The six-coordinate target is

`q_t(c,s) = p(c | x_0:t) p(s_t=s | c,x_0:t)`.

Its full sum is one; each component's three-coordinate block sums to that component's posterior weight. The unweighted conditional target `p(s_t | c,x_0:t)` instead has two separately normalized blocks and sums to two. Recovering that target, or only the two component weights, does not establish recovery of the weighted six-coordinate geometry. The direct run fits a separate standardized ridge readout to `q_t` and scores held-out coordinate R²/MSE and sampled pairwise-distance R². It plots six exact and six unrepaired reconstructed coordinates; visual similarity is descriptive, not quantitative evidence.

## Reproduction mapping

- Direct-sum component identity ↔ exact `p(c | x_0:t)` and its held-out linear regression/classification.
- Conditional-state diagnostic ↔ regression of all `p(s_t | c, x_0:t)` blocks and classification within the true component; this is not the weighted geometry target.
- Direct Mess3 geometry ↔ six coordinates `p(c,s_t | x_0:t)`, held-out joint regression, and distance distortion.
- Residual-stream geometry ↔ final GRU hidden state or normalized final Transformer residual state.
- Visual geometry ↔ descriptive PCA for the earlier experiments; exact-versus-reconstructed weighted coordinates for Mess3, saved in raw JSONL.
- Paper’s untrained comparison ↔ freshly initialized same-architecture controls for every seed.

## Deliberate extension

The article establishes correlational linear recoverability. This artifact asks a narrower causal question: does projecting out a probe-defined component or conditional-state subspace selectively damage an independently fitted held-out decoder while preserving the other one? Direction fitting, evaluator fitting, and intervention testing use three disjoint sequence sets. Because a low-rank projection can cause generic distribution shift, random subspaces and per-example norm-matched random perturbations are required controls. Shuffled-label directions test probe-fitting artifacts.

## Important non-equivalences

The earlier two-state sources do not reproduce Mess3. The new Mess3 factorization does reproduce the emission process, but its 64-emission sequences omit BOS and provide 63 input/target positions; the published sequence protocol differs. The CPU run also changes width, depth, attention-head dimension, position encoding, normalization, MLP width/gating, initialization, probe site, optimizer settings, and optimization budget, as detailed in the report's fidelity table. PCA of raw activations remains distinct from regression into weighted coordinates. The causal intervention results belong to the earlier two-state setting and cannot be transferred to Mess3 from this correlational readout experiment.
