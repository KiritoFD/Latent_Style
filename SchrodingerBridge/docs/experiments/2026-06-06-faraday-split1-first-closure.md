# Faraday Split1 First Closure

Date: 2026-06-06

Scope:

- follow-up split:
  - `wikiart_stress1`
- current closed method packet:
  - `LBM-F`

## What is already closed

Training is closed for:

- `wikiart_stress1_Color_Field_Painting__High_Renaissance__Mannerism_Late_Renaissance__Pop_Art__Realism_variant_f_b44_remote`

Retained artifacts:

- `epoch_0001.pt`
- `epoch_0002.pt`
- `epoch_0003.pt`
- `remote_train.log`

Training-side readout from the retained log:

- batch size:
  - `44`
- style count:
  - `5`
- source samples:
  - `1000` train latents per style
- runtime memory band:
  - peak allocated about `7.81 GiB`
  - peak reserved about `8.11 GiB`
- epoch wall:
  - about `33.3s` compute time per epoch in the landed packet

## Eval compatibility repair

The first deferred `full_eval` launch failed after training because evaluator
checkpoint loading hit a backward-compatibility gap:

- old checkpoint config objects could be deserialized without newer dataclass
  attributes such as `tokenizer_content_adaptive`

Repair applied:

- [config_schema.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/config_schema.py)

Current fix:

- `ModelConfig.to_dict()` and `validated()` now materialize any missing
  dataclass fields from defaults before reuse
- this keeps older checkpoint payloads evaluable without rewriting their saved
  `config.json`

After this repair, the rerun packet landed normally.

## Closed `full_eval` curve

Remote full-eval root:

- `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/wikiart_stress1_Color_Field_Painting__High_Renaissance__Mannerism_Late_Renaissance__Pop_Art__Realism_variant_f_b44_remote/full_eval`

Closed metrics:

| epoch | transfer CLIP-S | transfer LPIPS | full CLIP-S | full LPIPS | eval wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| `e1` | `0.699701` | `0.328629` | `0.722121` | `0.324370` | `99.14s` |
| `e2` | `0.700839` | `0.351362` | `0.721889` | `0.346829` | `94.20s` |
| `e3` | `0.699378` | `0.368591` | `0.718985` | `0.363227` | `94.25s` |

Immediate reading:

- `e1` is the best current retained point by `transfer CLIP-S + LPIPS`
- `e2` recovers a tiny amount of transfer style but clearly worsens LPIPS
- `e3` is worse on both style and LPIPS than `e1`

Current retained point:

- `split1 F e1`

## Still open

This packet is not yet paper-safe for split-level claim use because two pieces
are still pending:

1. standalone targetwise `ArtFID` for the retained `e1` point
2. split-local `IDT / no-op` packet on the same `wikiart_stress1` test surface

Current in-flight lane:

- `full_eval_artfid/epoch_0001` for retained `e1`

Current prep added for the next open item:

- [materialize_noop_eval_images.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/materialize_noop_eval_images.py)

Purpose:

- build a reusable split-local no-op packet under evaluator-compatible file
  names:
  - `{src_style}_{src_stem}_to_{tgt_style}.png`

## Safe conclusion now

What is safe to say already:

- the first follow-up stress split is no longer only selected/materialized; it
  now has a landed `LBM-F` train packet and a landed `e1..e3` full-eval curve
- within this split, the current retained point is again an early checkpoint,
  with later epochs mainly worsening LPIPS

What is not safe to say yet:

- positive-vs-IDT claim on `wikiart_stress1`
- targetwise-ArtFID claim on `wikiart_stress1`
- broader fixed-rule multi-split generalization claim
