# Payload Candidate Inventory

Date: 2026-06-03

Purpose:

- record the remote payload-recovery and candidate-inventory result for the
  tokenizer execution-alignment line;
- distinguish blocked same-family fallbacks from payload-backed new-family
  successors;
- keep the packet history inspectable before any explicit reselection.

## Search basis

Remote owner:

- `Linnaeus`

Primary searched roots:

- `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge`
- `I:\Github\Latent_Style_TokenizerClean`
- `I:\Github\Latent_Style`

Blocking family root confirmed to exist:

- `I:\Github\Latent_Style\SchrodingerBridge\exp\distinct5_512_ema_variant_h_hard_explore_queue_e3_b44_remote`

Surviving H-family evidence:

- `config.json`
- `logs\training_20260602_235921.csv`

Confirmed latent train root from the surviving H-family config:

- `/mnt/i/wikiart_distinct5_samam_512_latents_ema/train`

## Recovery result for the intended point

Original target:

- family:
  - `distinct5_512_ema_variant_h_hard_explore_queue_e3_b44_remote`
- checkpoint:
  - `epoch_0001`

Result:

- family root exists
- selected checkpoint payload does not exist
- same-family adjacent fallback `H e2` is also unavailable in the currently
  searched remote evidence surface
- no silent substitution was performed

## Candidate inventory within the searched paper-facing Distinct5 surface

| family | family root exists | checkpoint payloads | full_eval summaries / metrics | interpretation |
| --- | --- | --- | --- | --- |
| `H` = `distinct5_512_ema_variant_h_hard_explore_queue_e3_b44_remote` | yes | none found | none confirmed on remote | original packet family is blocked |
| `F` = `distinct5_512_ema_variant_f_annealed_prototype_ot_queue_e3_b44_remote` | yes | none found | none confirmed on remote | reviewed paper-facing point, but currently unusable as a payload-backed successor |
| `K` = `distinct5_512_ema_variant_k_content_adaptive_vq_queue_e3_b44_remote` | yes | `epoch_0001.pt`, `epoch_0002.pt`, `epoch_0003.pt` | yes for `epoch_0001/0002/0003` | payload-backed candidate; mechanism-family change required |
| `J` = `distinct5_512_ema_variant_j_aux_hard_swd_queue_e3_b44_remote` | yes | payload-backed | yes for `epoch_0001/0002/0003` | payload-backed candidate; mechanism-family change required |
| `L` = `distinct5_512_ema_variant_l_content_adaptive_annealed_queue_e3_b44_remote` | yes | payload-backed | yes for `epoch_0001/0002/0003` | payload-backed candidate; mechanism-family change required |
| `M` = `distinct5_512_ema_variant_m_style_gated_content_router_e3_b44_remote` | yes | payload-backed | yes for `epoch_0001/0002/0003` | payload-backed candidate; mechanism-family change required |

## Immediate conclusion

1. The original `H e1` packet cannot be launched as specified.
2. A same-family paper-safe fallback is not currently available on remote.
3. `F` does not rescue the old packet either, because it is also missing both
   payload and landed eval surface in the searched remote roots.
4. The next executable options are `K/J/L/M`, but any of them must be treated
   as a new mechanism-family packet rather than a fallback for the blocked
   `H e1` packet.
