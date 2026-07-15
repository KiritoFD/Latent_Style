# Root Layout Equivalence Check

The active Python modules were copied from `src/` to the project root after the baseline reproduction passed. After the equivalence checks below passed, the legacy tree was moved to `archives/legacy-src-pre-root-20260715/` for provenance and removed from the active import surface.

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

## Full Evaluation Check

The promoted root evaluator was also run on the complete 750-image board. A second legacy-layout run was made to separate layout changes from ordinary cross-process GPU/VAE nondeterminism.

| Comparison | Mean absolute 8-bit channel difference | Changed channel fraction | Maximum difference |
|---|---:|---:|---:|
| Original legacy run vs. promoted root | 0.05731 | 0.05658 | 21 |
| Original legacy run vs. legacy repeat | 0.05737 | 0.05662 | 19 |
| Promoted root vs. legacy repeat | 0.05743 | 0.05667 | 20 |

The root-layout pixel drift is indistinguishable from a second run of the unchanged legacy evaluator. Aggregate metrics are similarly stable:

| Run | CLIP-S | LPIPS |
|---|---:|---:|
| Original legacy | 0.7073850 | 0.2507627 |
| Promoted root | 0.7074457 | 0.2507637 |
| Legacy repeat | 0.7074157 | 0.2507570 |

This full-board check passes. Exact PNG hashes are not a valid cross-process equivalence criterion for this GPU VAE path; fixed-model latent outputs remain exactly equal, and full-board drift must be compared against the legacy repeat envelope.

## Tests

```text
42 passed, 1 warning
```

The checked set covers the root import/entry-point contract, portable canonical configs, inference infrastructure, cleanup smoke tests, and spectral ODE behavior. The warning is an existing test-only tensor-to-float conversion warning.
