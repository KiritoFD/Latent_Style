# 713 WEAVE Style-Path Diagnosis

Date: 2026-07-13

Goal: build a probe-first theory of the current `aaai2027_v4` / T11 WEAVE architecture, then make only targeted model changes with DINO-S as the primary style metric.

## Current Working Assumptions

- Primary style metric: DINO-S. CLIP-S is secondary and mostly useful as a direction/style-affinity sanity check.
- The strongest current style path is endpoint high-frequency statistical alignment, not the learned cross-attention path.
- LL should remain protected unless a probe shows a measurable style bottleneck that cannot be solved in LH/HL/HH.
- Changes should be small, gated by config, and evaluated against DINO-S, LPIPS, and DINO-C together.

## Files

- `read_cache.md`: summary of documents and code paths already read.
- `theory_map.md`: current information-flow and bottleneck model.
- `probe_plan.md`: concrete probes and acceptance criteria.
- `runbook.md`: local/remote execution plan.
- `status.md`: live progress log.

