# 2026-06-18 TopoGate Multiblock Audit

This audit answers a very specific implementation question:

> Did `topogate_attention_gw` really use the TopoGate structure fingerprint we
> thought it used, or did it silently collapse to a narrower proxy?

## Finding

Before the 2026-06-18 fix, `src/losses.py::_structure_pairwise_cost(...)` called
`topogate_attention_gw` through a helper that effectively consumed only the
**last semantic body block**:

1. run the content-side probe through all `body_blocks`
2. read `model.last_semantic_topology_attn`
3. build one complexity descriptor from that last cached attention map

That was narrower than the phase-616 theory/docs contract, which described
TopoGate attention as a model-internal structure fingerprint rather than a
single-block summary.

The helper now aggregates **all semantic body blocks**:

1. clear each block's cached attention
2. run the neutral content-side OT probe through every body block
3. collect `last_topology_attn` per block, falling back to `last_attn`
4. convert each block attention map into a 4-value complexity descriptor
5. concatenate the per-block descriptors into the final TopoGate fingerprint

## Repro

```powershell
py -3.12 tools/probe_topogate_descriptor_coverage.py `
  --config docs/experiments/2026-06-18-stage1-lowrank-rerun-audit/baseline_h1_lowrank_config.json `
  --output-dir docs/experiments/2026-06-18-topogate-multiblock-audit `
  --device cpu
```

Artifacts:

- `summary.json`
- `per_block_topogate_descriptor.csv`

## Observed result

Current `summary.json` reports:

- `descriptor_blocks = 4`
- per-block cost means:
  - block 0 -> `0.0021682840306311846`
  - block 1 -> `0.0018197735771536827`
  - block 2 -> `0.0011960655683651567`
  - block 3 -> `0.0018775684293359518`
- last-block-only aggregate:
  - `last_block_cost_mean = 0.0018775684293359518`
  - `last_block_cost_var = 3.2766697586339433e-06`
- all-block aggregate:
  - `aggregate_cost_mean = 0.007061691954731941`
  - `aggregate_cost_var = 1.353352126898244e-05`
- aggregate vs last-block difference:
  - `aggregate_minus_last_mean_abs = 0.005184123292565346`

Interpretation:

1. all four body blocks expose nontrivial topology signal
2. the last block is not a sufficient proxy for the whole TopoGate structure path
3. the pre-fix h5/h6 interpretation was therefore incomplete

## Practical consequence

Any earlier h5/h6 result produced before this multiblock fix should be treated as
**stale** if it is being interpreted as evidence about the intended
`topogate_attention_gw` design.

That does not mean the old results were fake. It means they answered a narrower
question:

- "what happens if OT sees only the last TopoGate block?"

rather than:

- "what happens if OT sees the full semantic body topology fingerprint?"

## Guardrails

Two checks should now be standard for future `topogate_attention_gw` review:

1. `ot_topogate_probe_active == 1`
2. `ot_topogate_descriptor_blocks > 1` on multiblock models

If the second check regresses back to `1` unexpectedly, re-audit the descriptor
path before trusting any new h5/h6 conclusion.
