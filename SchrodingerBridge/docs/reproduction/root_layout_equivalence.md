# Root Layout Equivalence Check

The active Python modules were copied from `src/` to the `SchrodingerBridge/` root after the baseline reproduction passed. The legacy `src/` tree remains intact for comparison and compatibility during the transition.

## Fixed-Checkpoint Check

- Checkpoint: reproduced epoch 6.
- SHA-256: `67ca62f377f1606f2369904ebd9535f250d5b98caf254cb50ec834a788aec621`.
- Device and dtype: CPU, float32.
- Input: one deterministic `4 x 64 x 64` latent spanning `[-1, 1]`.
- Target style: style ID 2 and a deterministic horizontally flipped style latent.
- Time: `t=0.375`.
- Loading: strict state-dict loading in two isolated Python processes.

The old `src/` modules and promoted root modules produced exactly equal tensors:

| Head | Shape | Maximum absolute difference | Mean absolute difference |
|---|---|---:|---:|
| LL | `1 x 4 x 32 x 32` | 0.0 | 0.0 |
| LH | `1 x 4 x 32 x 32` | 0.0 | 0.0 |
| HL | `1 x 4 x 32 x 32` | 0.0 | 0.0 |

Path derivations that depended on the old extra `src/` level were then updated in the promoted copies only. No model computation was changed.

## Tests

```text
35 passed, 1 warning
```

The checked set covers the root import/entry-point contract, inference infrastructure, cleanup smoke tests, and spectral ODE behavior. The warning is an existing test-only tensor-to-float conversion warning.
