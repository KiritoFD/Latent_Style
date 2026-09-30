# 2026-06-18 Style-ID Eval Probe

## Purpose

When phase-618 variants were landing very close to each other, we needed to answer a more basic question first:

1. In plain no-reference eval, does changing only `style_id` actually change the model output?
2. If it changes, does the change become body-live, or does it stay trapped in decoder-only cosmetics?
3. What does `semantic_self_topology_blend` really do in the current implementation?

This directory contains the direct probe outputs from `tools/probe_styleid_eval_path.py`.

## Probe correction note

On 2026-06-18 we fixed a probe implementation bug for the repaired lowrank family.

The original trace path seeded the style branch with `encode_style_id(style_id)`.
That was wrong for `pure_latent_spatial`, because the real runtime path uses:

- `model._compute_style_code(...)`

and for this family the initial style code is:

- `0 + time_code`

rather than a style-id embedding.

After the fix, the probe now mirrors the real no-reference eval path much more closely.
The corrected outputs replace the earlier exaggerated lowrank internal-delta readings.

Later on the same day we also fixed a model/runtime ordering bug:

- the lowrank residual code-map had been decoded before the structured tokenizer resolved the style-specific code

That fix materially increases the live no-reference style separation for the repaired family.
See:

- `docs/experiments/2026-06-18-lowrank-code-map-order-audit/README.md`

## Main result

The repaired lowrank base is not dead. It is body-live.

- Old base (`legacy_factorized`, no repaired carrier):
  - `max_forward_pair_delta = 0.010786`
  - `max_body_pair_delta = 0.000000`
  - `max_decoder_pair_delta = 0.063219`
  - interpretation: style-id changes the final output a little, but the effect does not enter the body path

- Repaired lowrank base (`pure_latent_spatial` + matched-target residual code + lowrank code-map):
  - `max_forward_pair_delta = 0.010019`
  - `max_body_pair_delta = 0.059144`
  - `max_decoder_pair_delta = 0.078543`
  - `max_style_map_pair_delta = 0.014215`
  - interpretation: the no-reference style carrier is alive, and after the ordering fix it is substantially stronger than before

So the repaired family changed the implementation in a real way. The current bottleneck is no longer "style path totally dead"; it is "internal style actuation exists, but downstream anchoring still suppresses the final visible delta."

More precisely after the probe fix and the lowrank ordering fix:

- the repaired family is body-live
- the internal-vs-final gap is not nearly as dramatic as we first thought
- style first becomes live at the structured tokenizer map stage
- the lowrank residual branch now contributes real style separation instead of acting as a style-invariant residual

## TopoGate blend semantics: what the code actually does

The strongest hypothesis in `docs/618/why_style_weak.md` was:

- `blend=1.0` blocks style entirely

That is not what the current code does.

In `src/lancet_blocks.py`, `SemanticCrossAttn.forward(...)` mixes the attention logits between:

- style-driven `k_style`
- content-topology `k_content`

but `v` still comes from the style map.

So `blend=1.0` means:

- use content topology for the routing weights
- still read style values through `v`

It is a routing constraint, not a full style shutoff.

## Probe evidence for blend semantics

### Repaired lowrank base, gate on, blend 1.0

From `lowrank_base/summary.json`:

- `max_forward_pair_delta = 0.010019`
- `max_body_pair_delta = 0.059144`
- `first_live_stage_histogram = {"style_map_a_vs_b_mean_abs": 10}`

If `blend=1.0` truly blocked style, `max_body_pair_delta` should collapse toward zero. It does not.

### Repaired lowrank base, gate on, blend 0.0

From `lowrank_blend0p00/summary.json`:

- `max_forward_pair_delta = 0.010066`
- `max_body_pair_delta = 0.060304`
- `max_decoder_pair_delta = 0.079209`

Lowering blend helps, but only modestly. The style path was already alive at `blend=1.0`.

### Gate off proves when blend is a no-op

From:

- `lowrank_gatefalse_blend1p00/summary.json`
- `lowrank_gatefalse_blend0p00/summary.json`

these two summaries are numerically identical:

- `max_forward_pair_delta = 0.010066`
- `max_body_pair_delta = 0.060304`
- `max_decoder_pair_delta = 0.079209`
- `max_delta_pair_delta = 0.010066`

This matches the implementation:

- if `semantic_self_topology_gate = false`, the blend value is ignored
- `blend` only matters inside the gated topology path

## Corrected lowrank interpretation

The corrected probe adds an important implementation fact:

- `encoded_style_code_a_vs_b_mean_abs = 0.0` for the repaired lowrank base
- `resolved_code_a_vs_b_mean_abs ~= 0.0238`
- `first_live_stage = style_map`

So for the current `pure_latent_spatial` family:

1. style-id does not create an early style-specific global code before the structured tokenizer
2. the first style-specific difference appears when the structured tokenizer emits `global_code + spatial_map`
3. the main style carrier is the structured map plus lowrank residual map, not an early legacy-style lookup code

And after the lowrank ordering fix:

- `pre_structured_style_code_map_a_vs_b_mean_abs = 0.0`
- `post_resolved_style_code_map_a_vs_b_mean_abs ~= 0.0125`

So the lowrank branch had real style capacity, but it was previously decoded from the wrong code state.

This matters because it means:

- some earlier intuition imported from the legacy family does not apply to the repaired family
- "why are experiments close?" is now better framed as:
  - is the structured tokenizer producing too-small inter-style separation?
  - is the lowrank residual strong enough after being wired correctly?
  - are later stages preserving only a small portion of a small incoming style delta?

## What this means for phase-618 interpretation

1. A repaired lowrank run that stays near neighboring variants is not automatically a no-op.
   - the carrier may already be body-live
   - but the carrier amplitude itself can still be small

2. `blend=1.0` should not be described as "style cannot enter the network at all."
   - it is more accurate to say:
   - content topology dominates attention routing
   - style values still flow through the attention value path

3. The next real bottlenecks to audit are:
   - structured tokenizer inter-style separation
   - lowrank residual map contribution
   - later-stage retention of an already modest style delta

## Files

- `old_base/summary.json`
- `lowrank_base/summary.json`
- `lowrank_blend0p00/summary.json`
- `lowrank_blend0p40/summary.json`
- `lowrank_gatefalse_blend1p00/summary.json`
- `lowrank_gatefalse_blend0p00/summary.json`
