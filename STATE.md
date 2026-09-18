# Experimental state

## Hypothesis

For the exact published two-Mess3 mixture, next-token training should improve linear recovery of the six weighted joint-belief coordinates over same-seed untrained controls. The earlier two-state experiments separately tested component/conditional-state readouts, selective erasure, and remote-history use.

## Last experiment

Direct Mess3 fidelity reproduction, registered at `74dcff2` before any experiment output. `make reproduce-mess3` trained width-32, two-layer Transformers on CPU for seeds 0/1/2 using the unchanged `configs/mess3_cpu.yaml`: 2,048 length-64 sequences, 24 epochs, 768 optimizer steps. Independent probe fits/tests used 1,024/512 sequences. Three training records, 12 normal/shuffled trained/untrained probe cells, 6,000 geometry points, and two figures are saved.

## Result

- The registered trained-over-untrained joint R² prediction failed in all three seeds. Trained R² is 0.327137/0.320975/0.315538 versus untrained 0.344238/0.333918/0.330558; paired differences are −0.017101/−0.012943/−0.015020.
- Pairwise-distance R² is negative: mean −1.315667 trained versus −1.358679 untrained. Joint MSE is 0.020913 trained versus 0.020439 untrained. All shuffled-target joint R² values are near zero.
- Conditional-state R² is 0.699709 trained versus 0.748687 untrained, while component-posterior R² is near zero. Conditional-state recovery is a different target from weighted joint geometry. Held-out exact-predictive KL is 0.015609 ± 0.001063 nats.

## Interpretation

The exact Mess3 source process is reproduced, but this smaller CPU training protocol does not reproduce high-fidelity weighted-belief recovery or improve it over initialization. Architecture, BOS/context protocol, initialization, optimization, and training exposure differ from the published run; the failure does not isolate its cause or refute the larger-model result. No configuration or seed was changed after inspecting outcomes. The earlier remote-history and budget-matching findings remain scoped to the two-state sources, and no Mess3 causal-erasure claim is made.

## Next smallest experiment

No further scientific experiment is registered. Any attempt to test a closer architecture or larger training budget should begin with a new preregistration and preserve this negative result. The present result and explicit fidelity differences are recorded in `report.md`.

## Registered Mess3 fidelity prediction

For the exact published two-Mess3 source, trained Transformer joint-belief R² will exceed its same-seed untrained control in all three seeds. Pairwise-distance R² is secondary. The experiment is a direct data/process reproduction but not an exact compute reproduction: width 32, two layers, absolute positions, LayerNorm, and CPU training differ from the published width-128 four-layer TransformerLens model with rotary positions, RMSNorm, gated GELU, and 45,000 optimization steps.
