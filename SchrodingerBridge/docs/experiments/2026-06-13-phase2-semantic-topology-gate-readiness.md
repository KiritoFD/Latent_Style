# Phase 2: semantic_topology_gate Readiness

Date: 2026-06-13

## Role

- code-readiness note for a structure-control switch
- not a launched packet
- does not change the current phase-2 queue order
- only becomes relevant if `safe_rescan_r2` fails and the queue moves into training-side structure reentry

## Why This Exists

- `docs/plan/attn.md` already argued that the cleanest structure-preserving idea is:
  - keep content self-attention topology
  - inject style mainly through the value path
- round-1 `attn_sa_mod` proved that this idea is implementable, but not strong enough as a standalone promotion family
- phase-2 therefore needs a lighter version:
  - reusable inside `legacy_semantic_crossattn`
  - switchable by config
  - available as a structure-side control without reopening the whole attention-family sweep

## Implementation

- code paths:
  - [lancet_blocks.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/lancet_blocks.py)
  - [lancet_backbone.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/lancet_backbone.py)
- new effective runtime switches:
  - `semantic_self_topology_gate: bool`
  - `semantic_self_topology_blend: float`
- current behavior:
  - only affects `legacy_semantic_crossattn`
  - leaves `attn_sa_mod`, `attn_gw_ot`, `attn_gated_spade`, and `attn_pnp_selfinject` unchanged
  - keeps style-map values, but blends cross-attention logits toward content self-attention logits

## Intended Read

- this is a structure-preserving tool, not a style amplifier
- it is meant for future phase-2 structure-side packets if tokenizer-safe sweep is exhausted
- it should be evaluated against:
  - whether LPIPS stays in-band better than pure topology-anchor loss alone
  - whether style remains above the old safe shelf instead of collapsing

## Local Smoke

- synthetic local smoke on the current `safe_rescan_r2` config base with:
  - `backbone_attention_family = legacy_semantic_crossattn`
  - `semantic_self_topology_gate = true`
  - `semantic_self_topology_blend = 1.0`
- readout:
  - `forward -> [1, 4, 32, 32]`
  - `predict_transport_base -> [1, 4, 32, 32]`
  - `integrate_transport(num_steps=2) -> [1, 4, 32, 32]`
  - `body_blocks[0].last_attn -> [1, 256, 256]`
  - `body_blocks[0].last_topology_attn -> [1, 256, 256]`

## Queue Impact

- none yet
- current phase-2 order remains:
  - `safe_rescan_r2`
  - then structure-side reentry
  - then diagnostic-only I2SB
- if a structure-side packet is prepared later, this switch is now available without further model-plumbing work
