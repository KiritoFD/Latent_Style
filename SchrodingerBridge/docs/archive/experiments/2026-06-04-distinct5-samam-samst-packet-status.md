# Distinct5 SaMST / SaMAM Packet Status (2026-06-04)

Scope: Distinct5 only. This note does not edit the paper. It checks whether a manuscript-usable packet can be closed from existing artifacts only.

## Conclusion

- `SaMST e5`: closed
- `SaMST e15`: closed
- `SaMAM step_003000` authoritative packet: closed
- `SaMAM step_003250` latest available packet: partial

Therefore the Distinct5 line now has closed packets from existing artifacts, but the absolute latest SaMAM point (`3250`) is still not fully closed.

## Closure rule used here

A packet is treated as closed when all of the following are present from existing artifacts:

1. full + transfer `CLIP-S` / `LPIPS`
2. targetwise `ArtFID`
3. same-scope timing when present in current artifacts
4. IDT-aligned per-image row proof, or an explicit missing-row report

## IDT reference

- IDT aligned rows:
  - `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\idt_eval_20260602\distinct5_512\idt_5x5\metrics.csv`

## SaMST

### SaMST e5

Paths:

- summary:
  - `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samst_distinct5_512_real_b1_e5_20260603\eval_bundle\eval_epoch5\epoch_0005\summary.json`
- per-image rows:
  - `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samst_distinct5_512_real_b1_e5_20260603\eval_bundle\eval_epoch5\epoch_0005\metrics.csv`
- targetwise ArtFID:
  - `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samst_distinct5_512_real_b1_e5_20260603\eval_bundle\eval_epoch5\epoch_0005\aggregate_targetwise_artfid.json`
- timing source:
  - `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samst_distinct5_512_real_b1_e5_20260603\eval_bundle\compare_e5_vs_e15\samst_distinct5_epoch_comparison.json`
  - `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samst_distinct5_512_real_b1_e5_20260603\run.log`

Metrics:

| point | scope | clip_style | content_lpips | targetwise_artfid |
| --- | --- | ---: | ---: | ---: |
| SaMST e5 | full | 0.7275811868 | 0.6270693954 | 432.0511083215 |
| SaMST e5 | transfer | 0.6989188100 | 0.6334999498 | 465.6860418255 |

Timing:

- train wall: `6958.502907 s`
- eval wall: `22.2205546 s`
- run.log start/finish:
  - `started=2026-06-03T17:57:31.111486`
  - `finished=2026-06-03T19:53:29.614393`

IDT row report:

- row count: `750`
- exact key match against IDT on `(src_style, tgt_style, src_image)`: `750/750`
- missing rows: `0`
- extra rows: `0`

Status: closed.

### SaMST e15

Paths:

- summary:
  - `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samst_distinct5_512_real_b2_e15_20260602\eval_epoch15\epoch_0015\summary.json`
- per-image rows:
  - `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samst_distinct5_512_real_b2_e15_20260602\eval_epoch15\epoch_0015\metrics.csv`
- targetwise ArtFID:
  - `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samst_distinct5_512_real_b2_e15_20260602\eval_epoch15\epoch_0015\aggregate_targetwise_artfid.json`
- timing source:
  - `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samst_distinct5_512_real_b1_e5_20260603\eval_bundle\compare_e5_vs_e15\samst_distinct5_epoch_comparison.json`
  - `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samst_distinct5_512_real_b2_e15_20260602\run.log`

Metrics:

| point | scope | clip_style | content_lpips | targetwise_artfid |
| --- | --- | ---: | ---: | ---: |
| SaMST e15 | full | 0.7247245136 | 0.6255497488 | 395.7117071285 |
| SaMST e15 | transfer | 0.6957412316 | 0.6319495817 | 444.4870406091 |

Timing:

- train wall: `20835.399642 s`
- eval wall: `157.2435011 s`
- run.log start/finish:
  - `started=2026-06-02T12:10:17.637052`
  - `finished=2026-06-02T17:57:33.036694`

IDT row report:

- row count: `750`
- exact key match against IDT on `(src_style, tgt_style, src_image)`: `750/750`
- missing rows: `0`
- extra rows: `0`

Status: closed.

## SaMAM

Important note: for late SaMAM points, the `curve_metrics.json/csv` artifacts are not the same packet as the later ArtFID reuse summaries. For packet use, the same-scope source is the ArtFID reuse packet because it carries `CLIP-S`, `LPIPS`, `ArtFID`, `metrics.csv`, and eval timing together.

### SaMAM step_003000 authoritative packet

Authoritative root:

- `I:\Github\Latent_Style\Related_Works\baseline_pipeline\results\samam_distinct5_512_mamba_b6_seg250_remote_wsl_20260601_2130_diag`

Packet paths:

- summary:
  - `I:\Github\Latent_Style\Related_Works\baseline_pipeline\results\samam_distinct5_512_mamba_b6_seg250_remote_wsl_20260601_2130_diag\eval_curve\step_003000_artfid_reuse\summary.json`
- per-image rows:
  - `I:\Github\Latent_Style\Related_Works\baseline_pipeline\results\samam_distinct5_512_mamba_b6_seg250_remote_wsl_20260601_2130_diag\eval_curve\step_003000_artfid_reuse\metrics.csv`
- timing/log source:
  - `I:\Github\Latent_Style\Related_Works\baseline_pipeline\results\samam_distinct5_512_mamba_b6_seg250_remote_wsl_20260601_2130_diag\segmented.log`

Metrics from the same-scope `summary.json`:

| point | scope | clip_style | content_lpips | targetwise_artfid |
| --- | --- | ---: | ---: | ---: |
| SaMAM 3000 | full | 0.6978010031 | 0.3220869704 | 345.6017453148 |
| SaMAM 3000 | transfer | 0.6646182162 | 0.3270938640 | 394.7662083479 |

Timing:

- successful segment train walls from `segmented.log`:
  - `2250 cumulative`: `27513.02 s` (from existing Distinct5 table/doc artifacts)
  - `2250 -> 2500`: `2941.42 s`
  - `2500 -> 2750`: `3144.38 s`
  - `2750 -> 3000`: `3156.25 s`
- cumulative train wall to 3000:
  - `36755.07 s` (`612.58 min`, about `10.21 h`)
- same-scope eval wall:
  - `289.31 s` from `EVAL_STEP_3000_WALL_SECONDS`

IDT row report:

- row count: `750`
- exact IDT slot coverage after normalizing only the IDT `src_image` prefix
  - IDT key: `(src_style, tgt_style, src_image without leading "src_style__")`
  - SaMAM key: `(src_style, tgt_style, src_image)`
- matched rows: `750/750`
- missing rows: `0`
- extra rows: `0`

Status: closed.

### SaMAM step_003250 latest available packet

Latest available paths:

- summary:
  - `/home/xy/samam_eval_local/step_003250_artfid_reuse2/summary.json`
- per-image rows:
  - `/home/xy/samam_eval_local/step_003250_artfid_reuse2/metrics.csv`
- local eval-only curve sidecar:
  - `/home/xy/samam_eval_local/step_003250_curve_named/curve_metrics.json`
- checkpoint path recorded by the local eval sidecar:
  - `/home/xy/samam_segments/step_003250_named/step_checkpoints/step-step=003250.ckpt`

Metrics from the same-scope `summary.json`:

| point | scope | clip_style | content_lpips | targetwise_artfid |
| --- | --- | ---: | ---: | ---: |
| SaMAM 3250 | full | 0.6970079633 | 0.3095833540 | 338.8172387965 |
| SaMAM 3250 | transfer | 0.6626959599 | 0.3145874888 | 389.6994417007 |

Timing:

- same-scope eval wall from `summary.json`:
  - `32.546313072 s`
- missing:
  - no current artifact proving cumulative train wall to `3250`
  - no authoritative synced packet under the main `I:` result root

IDT row report:

- row count: `750`
- exact key match against IDT on `(src_style, tgt_style, src_image)`: `750/750`
- missing rows: `0`
- extra rows: `0`

Why still partial:

1. the packet is local-only under `/home/xy/...`, not synced back to the authoritative `I:\...samam_distinct5_512_mamba_b6_seg250_remote_wsl_20260601_2130_diag` root
2. cumulative train timing to `3250` is not proven by current artifacts

Status: partial.

## Practical manuscript-safe reading

- Safe closed SaMST packets exist at `e5` and `e15`.
- A safe closed authoritative SaMAM packet exists at `step_003000`.
- The latest SaMAM point `step_003250` is usable only as a local-only partial packet unless its train-time provenance and authoritative sync are restored.
