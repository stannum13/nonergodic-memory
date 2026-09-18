# Task 1 implementation report: exact Mess3 generator

## Scope

Implemented the exact published Mess3 HMM factorization and its two-component mixture.

- Added `make_mess3(alpha, x)` with the required parameter bounds, transition matrix, emission matrix, and uniform initial distribution.
- Added `make_mess3_mixture()` with components `(alpha=0.60, x=0.15)` and `(alpha=0.66, x=0.50)`, weighted equally.
- Exported both constructors from `nonergodic_memory.data`.
- Added published labeled-operator and mixture-shape tests.

## TDD evidence

The tests were written before the production constructors. Running `pytest tests/test_hmm.py -q` at that point failed during collection with:

```text
ImportError: cannot import name 'make_mess3' from 'nonergodic_memory.data.hmm'
```

After implementation, the focused suite passed:

```text
10 passed in 0.41s
```

The full test suite also passed:

```text
67 passed in 30.67s
```
