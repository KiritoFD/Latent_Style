# Faraday Split1 Paper-Safe Packet

Date: 2026-06-06

Scope:

- fixed-rule follow-up split:
  - `wikiart_stress1`
- methods currently closed on the exact same `5x5 / 750` test surface:
  - `IDT / no-op`
  - `LBM-F e1`

Split styles:

- `Color_Field_Painting`
- `High_Renaissance`
- `Mannerism_Late_Renaissance`
- `Pop_Art`
- `Realism`

## Closed packet roots

Retained model point:

- run root:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/wikiart_stress1_Color_Field_Painting__High_Renaissance__Mannerism_Late_Renaissance__Pop_Art__Realism_variant_f_b44_remote`
- retained checkpoint:
  - `epoch_0001.pt`
- full eval:
  - `full_eval/epoch_0001/summary.json`
- retained-point ArtFID:
  - `full_eval_artfid/epoch_0001/aggregate_targetwise_artfid.json`

Split-local unchanged reference:

- run root:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/wikiart_stress1_Color_Field_Painting__High_Renaissance__Mannerism_Late_Renaissance__Pop_Art__Realism_idt_5x5`
- full eval:
  - `summary.json`
- targetwise ArtFID:
  - `aggregate_targetwise_artfid.json`

## Split-local `IDT / no-op`

Closed metrics:

- transfer CLIP-S:
  - `0.685820`
- transfer LPIPS:
  - `0.000000`
- transfer targetwise ArtFID:
  - `369.0005`
- full CLIP-S:
  - `0.714689`
- full LPIPS:
  - `0.000000`
- full targetwise ArtFID:
  - `295.4004`
- eval wall:
  - `53.67s`

Immediate reading:

- even on this fixed-rule follow-up split, unchanged images already obtain high
  absolute target-style similarity
- this preserves the same evaluator pathology class that motivated Distinct5

## Retained `LBM-F e1`

Closed metrics:

- transfer CLIP-S:
  - `0.699701`
- transfer LPIPS:
  - `0.328629`
- transfer targetwise ArtFID:
  - `415.0147`
- full CLIP-S:
  - `0.722121`
- full LPIPS:
  - `0.324370`
- full targetwise ArtFID:
  - `365.8655`
- eval wall:
  - `99.14s`

## No-op-adjusted readout

`LBM-F e1 - IDT`:

- transfer `Δ clip_style`:
  - `+0.013881`
- full `Δ clip_style`:
  - `+0.007432`
- transfer `Δ targetwise ArtFID`:
  - `+46.0142`
  - read as worse than `IDT`, because lower ArtFID is better
- full `Δ targetwise ArtFID`:
  - `+70.4651`

Interpretation:

- `LBM-F e1` does exceed the unchanged baseline on target-style similarity
- the margin is positive but modest on this split
- at the same time, the unchanged image remains an extremely strong baseline by
  targetwise ArtFID because it pays zero displacement

## Safe claim boundary

What this packet now safely supports:

- the fixed-rule follow-up split reproduces the same broad evaluation regime:
  unchanged images still score strongly, and positive style movement should be
  read against an explicit `IDT` reference rather than raw absolute CLIP-S
- `LBM-F e1` produces positive split-local target-style gain over `IDT`
  (`+0.0139` transfer CLIP-S) on the first follow-up split

What this packet does **not** support:

- a broad multi-split generalization claim
- an ArtFID-based superiority claim on this split
- any claim about baseline comparison on this split, since `SaMAM/SaMST/LoRA`
  have not yet been closed here

## Current paper use

This is the first follow-up split packet that is safe to cite as:

- fixed-rule stress-family evidence for the persistence of the `IDT` control
  issue; and
- one additional split where `LBM-F` still clears the split-local unchanged
  reference

It should still be reported with narrow language:

- one follow-up split
- fixed rule
- same `5x5 / 750` test surface
- compact `LBM-F` retained point only
