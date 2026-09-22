# Experimental state

## Hypothesis

For the exact published two-Mess3 mixture, the failed small-model geometry result may reflect limited predictive generalization from repeatedly training on 2,048 fixed sequences. At matched initialization, architecture, batch size, optimizer, sequence length, and update count, sampling fresh sequences per update should reduce held-out exact-predictive KL relative to reusing a fixed sequence pool.

## Last experiment

Direct Mess3 fidelity reproduction, registered at `74dcff2` before any experiment output. `make reproduce-mess3` trained width-32, two-layer Transformers on CPU for seeds 0/1/2 using the unchanged `configs/mess3_cpu.yaml`: 2,048 length-64 sequences, 24 epochs, 768 optimizer steps. Independent probe fits/tests used 1,024/512 sequences. Three training records, 12 normal/shuffled trained/untrained probe cells, 6,000 geometry points, and two figures are saved.

## Result

- The registered trained-over-untrained joint R² prediction failed in all three seeds. Trained R² is 0.327137/0.320975/0.315538 versus untrained 0.344238/0.333918/0.330558; paired differences are −0.017101/−0.012943/−0.015020.
- Pairwise-distance R² is negative: mean −1.315667 trained versus −1.358679 untrained. Joint MSE is 0.020913 trained versus 0.020439 untrained. All shuffled-target joint R² values are near zero.
- Conditional-state R² is 0.699709 trained versus 0.748687 untrained, while component-posterior R² is near zero. Conditional-state recovery is a different target from weighted joint geometry. Held-out exact-predictive KL is 0.015609 ± 0.001063 nats.

## Interpretation

The exact Mess3 source process is reproduced, but this smaller CPU training protocol does not reproduce high-fidelity weighted-belief recovery or improve it over initialization. Architecture, BOS/context protocol, initialization, optimization, and training exposure differ from the published run; the failure does not isolate its cause or refute the larger-model result. No configuration or seed was changed after inspecting outcomes. The earlier remote-history and budget-matching findings remain scoped to the two-state sources, and no Mess3 causal-erasure claim is made.

## Next smallest experiment

Run the registered exploratory Mess3 training-diversity diagnosis below. Interpret predictive competence against uniform, last-token, eight-token Bayes, and full-history Bayes before interpreting layerwise weighted-belief recovery.

## Registered Mess3 fidelity prediction

For the exact published two-Mess3 source, trained Transformer joint-belief R² will exceed its same-seed untrained control in all three seeds. Pairwise-distance R² is secondary. The experiment is a direct data/process reproduction but not an exact compute reproduction: width 32, two layers, absolute positions, LayerNorm, and CPU training differ from the published width-128 four-layer TransformerLens model with rotary positions, RMSNorm, gated GELU, and 45,000 optimization steps.

## Registered exploratory training-diversity prediction

Registered before creating any diagnosis checkpoint, JSONL record, or figure. Configuration digest `aa2de784730aa352` compares `reused` and `fresh` sequence conditions for exploratory seeds 10 and 11 at steps 0, 768, and 3,072. Both conditions use the same initialized width-32 two-layer Transformer, sequence length 64, batch size 64, AdamW settings, and supervised tokens per update. The reused condition traverses a deterministic fixed pool of 2,048 sequences; the fresh condition samples a new batch of 64 sequences at every update.

The primary prediction is: at step 3,072, fresh-data training has lower held-out exact-predictive KL than reused-data training in both seeds 10 and 11. Layerwise component-posterior, conditional-state, and six-coordinate weighted joint-belief recovery are secondary outcomes with no directional success criterion. Seeds 10 and 11 are exploratory and cannot be reused for later confirmation.
