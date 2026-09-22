# Experimental state

## Hypothesis

For the exact published two-Mess3 mixture, the failed small-model geometry result may reflect limited predictive generalization from repeatedly training on 2,048 fixed sequences. At matched initialization, architecture, batch size, optimizer, sequence length, and update count, sampling fresh sequences per update should reduce held-out exact-predictive KL relative to reusing a fixed sequence pool.

## Last experiment

Exploratory Mess3 training-diversity diagnosis, registered at `7cac6cb` before any diagnosis output. For seeds 10/11, compared a fixed pool of 2,048 sequences with fresh batches at every update, holding initialization, architecture, batch size, sequence length, optimizer, and checkpoints fixed. Exact baselines quantify the available predictive signal; independent probes measure all three activation sites at steps 0/768/3,072.

## Result

- The registered primary prediction is supported in both exploratory seeds. At step 3,072, fresh/reused exact-predictive KL is 0.001649/0.203223 for seed 10 and 0.001301/0.179832 for seed 11.
- Uniform predictive KL is 0.008175; exact eight-token Bayes is 0.003313. Fresh training recovers 79.8%/84.4% of the uniform-to-full-Bayes gap, while reused training becomes much worse than uniform.
- At step 3,072, fresh block-2 joint-belief R² is 0.726/0.616 versus reused 0.251/0.250. Fresh final-norm R² is 0.677/0.628. Shuffled-target joint R² stays near zero.
- The uncached CPU command reported 3:02:13 wall time, almost consuming the four-hour exploratory cap.

## Interpretation

Fixed-pool sequence reuse explains a large part of the initial small-model failure: reused training generalizes poorly and eventually damages both prediction and geometry, whereas fresh data produces useful long-history prediction and a substantial deeper-layer weighted-belief representation. The result remains exploratory with two seeds and does not match the paper's architecture, training budget, or reported R². It supports a mechanism for the failed CPU run rather than an exact numerical reproduction.

## Next smallest experiment

Do not launch the planned five-seed confirmation with the current sampler: the exploratory run nearly exhausted its compute cap. First vectorize or batch the exact sampler, verify its distribution against the reference implementation, and benchmark one fresh-data run. Then preregister new confirmatory seeds if the projected cost is acceptable.

## Registered Mess3 fidelity prediction

For the exact published two-Mess3 source, trained Transformer joint-belief R² will exceed its same-seed untrained control in all three seeds. Pairwise-distance R² is secondary. The experiment is a direct data/process reproduction but not an exact compute reproduction: width 32, two layers, absolute positions, LayerNorm, and CPU training differ from the published width-128 four-layer TransformerLens model with rotary positions, RMSNorm, gated GELU, and 45,000 optimization steps.

## Registered exploratory training-diversity prediction

Registered before creating any diagnosis checkpoint, JSONL record, or figure. Configuration digest `aa2de784730aa352` compares `reused` and `fresh` sequence conditions for exploratory seeds 10 and 11 at steps 0, 768, and 3,072. Both conditions use the same initialized width-32 two-layer Transformer, sequence length 64, batch size 64, AdamW settings, and supervised tokens per update. The reused condition traverses a deterministic fixed pool of 2,048 sequences; the fresh condition samples a new batch of 64 sequences at every update.

The primary prediction is: at step 3,072, fresh-data training has lower held-out exact-predictive KL than reused-data training in both seeds 10 and 11. Layerwise component-posterior, conditional-state, and six-coordinate weighted joint-belief recovery are secondary outcomes with no directional success criterion. Seeds 10 and 11 are exploratory and cannot be reused for later confirmation.
