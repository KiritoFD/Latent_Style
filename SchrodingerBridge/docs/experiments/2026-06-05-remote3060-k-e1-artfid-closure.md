# Remote 3060 `K_e1` ArtFID Closure

Date: 2026-06-05

Scope:

- close the retained remote owner-surface packet for
  `distinct5_512_ema_variant_k_content_adaptive_vq_queue_e3_b44_remote`
- reuse the existing `750` generated images
- emit a standalone `aggregate_targetwise_artfid.json` under the current
  evaluator contract

## Authoritative remote surface

- run root:
  - `I:\Github\Latent_Style\SchrodingerBridge\exp\distinct5_512_ema_variant_k_content_adaptive_vq_queue_e3_b44_remote`
- eval dir:
  - `I:\Github\Latent_Style\SchrodingerBridge\exp\distinct5_512_ema_variant_k_content_adaptive_vq_queue_e3_b44_remote\full_eval\epoch_0001`

## Why this closure was needed

Before the re-audit:

- `summary.json` existed but its `matrix_breakdown[*][*].art_fid` entries were
  all `null`
- no standalone `aggregate_targetwise_artfid.json` was present
- the retained packet was therefore not paper-safe under the current
  targetwise-ArtFID artifact rule

## Reuse-only evaluator closure

Local wrapper used for the remote WSL relaunch:

- [remote_k_e1_reuse_eval.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/_codex_tmp/remote_k_e1_reuse_eval.sh)

Remote log:

- `I:\Github\Latent_Style\SchrodingerBridge\_codex_tmp\remote_k_e1_reuse_eval.log`

Observed wall-clock window from the retained log:

- start:
  - `2026-06-05T21:14:29+08:00`
- end:
  - `2026-06-05T21:34:31+08:00`
- wall time:
  - about `20.03 min`

Key retained log lines:

- `Phase 1: Reuse generated images ... Reused 750 generated images`
- `Summary saved: ...\summary.json`
- `Targetwise ArtFID summary saved: ...\aggregate_targetwise_artfid.json`

## Closed artifact

New retained file:

- `I:\Github\Latent_Style\SchrodingerBridge\exp\distinct5_512_ema_variant_k_content_adaptive_vq_queue_e3_b44_remote\full_eval\epoch_0001\aggregate_targetwise_artfid.json`

Directory timestamp at check time:

- `2026/06/05 21:34`

The refreshed `summary.json` and standalone targetwise-ArtFID json were also
synced back to the local mirrored packet directory:

- `G:\GitHub\Latent_Style\SchrodingerBridge\exp\distinct5_512_ema_variant_k_content_adaptive_vq_queue_e3_b44_remote\full_eval\epoch_0001`

## Measured targetwise ArtFID from the closed packet

From `aggregate_targetwise_artfid.json`:

- all-pairs mean targetwise ArtFID:
  - `360.37285813760724`
- transfer-only mean targetwise ArtFID:
  - `406.15086907239305`
- identity-only mean targetwise ArtFID:
  - `177.26081439846382`

The transfer-only per-target means are:

- `Early_Renaissance`: `388.5066344133707`
- `Impressionism`: `389.85379302796184`
- `Minimalism`: `363.4558522977851`
- `Rococo`: `464.97216148460666`
- `Ukiyo_e`: `423.96590413824106`

## Audit consequence

This closure exposes a direct conflict with the current tracked same-cost
inventory row for `K_e1`, which still records:

- transfer targetwise ArtFID:
  - `161.9576574745`
- full targetwise ArtFID:
  - `157.1687499148`

Those older values should no longer be treated as authoritative for `K_e1`.
The packet is now closed under the current evaluator contract, but the
inventory/paper interpretation must be re-audited before `K_e1` ArtFID is used
as a paper-facing anchor again.
