# Experimental state

## Hypothesis

Small next-token predictors trained on a fixed-component mixture will expose linearly separable component identity and within-component predictive state. Probe-row-space erasure will cause selective damage: component erasure will reduce component accuracy more than conditional-state accuracy, and state erasure will do the converse, beyond rank- and norm-matched random controls.

## Last experiment

Source-overlap sweep at 0.00, 0.35, 0.70, and 0.90, seeds 0/1/2, on CPU. Each cell fixed sequence length 32, width/depth 32/2, 512 training sequences, 12 epochs, and three disjoint intervention datasets. This smaller sweep configuration is exploratory and separate from the larger central configuration.

## Result

- GRU trained-minus-untrained component-posterior R² gain across overlap 0.00/0.35/0.70/0.90: 0.074/0.153/0.193/−0.004. Transformer: 0.170/0.301/0.287/0.027.
- Full conditional-state R² gain stayed small: GRU −0.010/−0.009/0.004/0.012; Transformer 0.019/0.027/0.048/0.063.
- Intended component-erasure accuracy damage was nonmonotonic and seed-sensitive: GRU 0.013/0.053/0.017/0.042; Transformer 0.133/0.115/0.221/0.038.
- Intended state-erasure damage was GRU 0.019/0.040/0.097/0.033 and Transformer 0.192/0.174/0.177/0.151. Every value is a mean across three seeds; seed SDs are plotted in `figures/sweep_overlap.png`.

## Interpretation

The registered prediction of a monotonic decline in component-posterior training advantage with overlap is falsified. Gain peaks at intermediate overlap and collapses only when the sources are nearly identical. A plausible interpretation is a boundary effect: at zero overlap, recent tokens make component inference easy even for random features; at intermediate overlap, learned temporal integration adds value; at 0.90 there is little identifiable component signal to learn. This explanation is post hoc and should be tested by varying sequence length. The causal-erasure curves are also nonmonotonic and do not rescue a stable-subspace claim.

## Next smallest experiment

At fixed overlap 0.35, sweep sequence length through 8, 16, 32, and 64. The falsifiable post-hoc prediction is that trained-over-untrained component-posterior R² gain grows with length because learned integration should matter more when evidence is distributed over longer histories. Component count, width, and intervention depth follow only after this test.
