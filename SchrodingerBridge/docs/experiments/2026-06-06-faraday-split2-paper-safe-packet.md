# Faraday Split2 Paper-Safe Packet

Date: 2026-06-06

Scope:

- fixed-rule follow-up split:
  - `wikiart_stress2`
- methods currently closed on the exact same `5x5 / 750` test surface:
  - `IDT / no-op`
  - `LBM-F e1`

Split styles:

- `Abstract_Expressionism`
- `Baroque`
- `Cubism`
- `Northern_Renaissance`
- `Post_Impressionism`

## Closed packet roots

Retained model point:

- run root:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/wikiart_stress2_Abstract_Expressionism__Baroque__Cubism__Northern_Renaissance__Post_Impressionism_variant_f_b44_remote`
- retained checkpoint:
  - `epoch_0001.pt`
- full eval:
  - `full_eval/epoch_0001/summary.json`
- retained-point ArtFID:
  - `full_eval_artfid/epoch_0001/aggregate_targetwise_artfid.json`

Split-local unchanged reference:

- run root:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/wikiart_stress2_Abstract_Expressionism__Baroque__Cubism__Northern_Renaissance__Post_Impressionism_idt_5x5`
- full eval:
  - `summary.json`
- targetwise ArtFID:
  - `aggregate_targetwise_artfid.json`

## Split-local `IDT / no-op`

Closed metrics:

- transfer CLIP-S:
  - `0.716762`
- transfer LPIPS:
  - `0.000000`
- transfer targetwise ArtFID:
  - `332.0953`
- full CLIP-S:
  - `0.739348`
- full LPIPS:
  - `0.000000`
- full targetwise ArtFID:
  - `265.8762`
- eval wall:
  - `53.97s`

Immediate reading:

- unchanged images again obtain high absolute target-style similarity on the
  follow-up split
- the same evaluator pathology class therefore persists beyond both Distinct5
  and split1

## Retained `LBM-F e1`

Closed metrics:

- transfer CLIP-S:
  - `0.724096`
- transfer LPIPS:
  - `0.319365`
- transfer targetwise ArtFID:
  - `402.1109`
- full CLIP-S:
  - `0.740629`
- full LPIPS:
  - `0.318825`
- full targetwise ArtFID:
  - `351.8889`
- eval wall:
  - `99.44s`

Closed full-eval curve:

| epoch | transfer CLIP-S | transfer LPIPS | full CLIP-S | full LPIPS | eval wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| `e1` | `0.724096` | `0.319365` | `0.740629` | `0.318825` | `99.44s` |
| `e2` | `0.720268` | `0.341043` | `0.735662` | `0.340144` | `94.10s` |
| `e3` | `0.717863` | `0.351921` | `0.732457` | `0.350668` | `93.68s` |

Retained point:

- `split2 F e1`

## No-op-adjusted readout

`LBM-F e1 - IDT`:

- transfer `Δ clip_style`:
  - `+0.007335`
- full `Δ clip_style`:
  - `+0.001281`
- transfer `Δ targetwise ArtFID`:
  - `+70.0155`
  - read as worse than `IDT`, because lower ArtFID is better
- full `Δ targetwise ArtFID`:
  - `+86.0127`

Interpretation:

- `LBM-F e1` still exceeds the split-local unchanged baseline on target-style
  similarity
- the margin is smaller than on split1, but remains positive
- the unchanged reference still wins on ArtFID because it pays zero
  displacement

## Safe claim boundary

What this packet now safely supports:

- a second fixed-rule follow-up split reproduces the same broad evaluation
  regime:
  unchanged images remain a strong target-style baseline, so raw absolute
  CLIP-style should be read with an explicit `IDT` control
- `LBM-F e1` again produces positive split-local target-style gain over `IDT`
  on this second split

What this packet does **not** support:

- broad multi-split generalization rhetoric beyond the currently closed splits
- an ArtFID superiority claim on this split
- any baseline comparison on this split, since `SaMAM/SaMST/LoRA` are not yet
  closed here

## Current paper use

This is now the second fixed-rule follow-up split that is safe to cite as:

- stress-family evidence for the persistence of the `IDT` control issue; and
- another split where compact `LBM-F` still clears the split-local unchanged
  reference
