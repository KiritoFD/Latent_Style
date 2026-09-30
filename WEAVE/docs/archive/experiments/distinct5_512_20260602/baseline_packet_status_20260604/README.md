# Distinct5-512 baseline packet status

Date: 2026-06-04

This packet audits existing artifacts only. No new long training run was started.

## Verdict

- `SaMST e5`: closed from existing artifacts
- `SaMST e15`: partial
- active-manuscript `SaMAM`: valid only through `step_002250`
- post-2250 `SaMAM` packets: retained only as audit history, not manuscript
  evidence

## Active manuscript boundary

For the active AAAI/LBM manuscript path, treat `SaMAM` Distinct5 evidence as
valid only through `step_002250`.

Use this manuscript row only:

| checkpoint | tr CLIP-S | tr LPIPS | tr ArtFID | delta vs IDT | train min |
|---|---:|---:|---:|---:|---:|
| SaMAM 2250 | 0.5523 | 0.3605 | 148.2 | -0.0877 | 458.6 |

Boundary rule:

- post-2250 SaMAM outputs are reproduction-chain failures for the active
  manuscript path unless a clean independent rerun closes a new aligned packet
- do not use `2500/3000/3250` as positive-IDT manuscript evidence
- do not use `394.8 / 10.2h` in the active manuscript path

## IDT reference

- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\idt_eval_20260602\distinct5_512\idt_5x5\metrics.csv`
- transfer-only IDT CLIP-S used in existing bootstrap artifacts: `0.6399224616587162`

## SaMST

| checkpoint | tr CLIP-S | tr LPIPS | tr ArtFID | full CLIP-S | full LPIPS | full ArtFID | train h | inference ms/img | aligned rows | missing rows | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| e5 | 0.698919 | 0.633500 | 465.6860 | 0.727581 | 0.627069 | 432.0511 | 1.93 | 430.761 | 750 | 0 | closed |
| e15 | 0.695741 | 0.631950 | 444.4870 | 0.724725 | 0.625550 | 395.7117 | 5.79 | missing | 750 | 0 | partial |

Key paths:

- e5 summary:
  - `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samst_distinct5_512_real_b1_e5_20260603\eval_bundle\eval_epoch5\epoch_0005\summary.json`
- e5 metrics:
  - `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samst_distinct5_512_real_b1_e5_20260603\eval_bundle\eval_epoch5\epoch_0005\metrics.csv`
- e5 ArtFID:
  - `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samst_distinct5_512_real_b1_e5_20260603\eval_bundle\eval_epoch5\epoch_0005\aggregate_targetwise_artfid.json`
- e5 inference timing source:
  - `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samst_distinct5_512_real_b1_e5_20260603\eval_bundle\eval_epoch5\generate.log`
- e15 summary:
  - `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samst_distinct5_512_real_b2_e15_20260602\eval_epoch15\epoch_0015\summary.json`
- e15 metrics:
  - `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samst_distinct5_512_real_b2_e15_20260602\eval_epoch15\epoch_0015\metrics.csv`
- e15 ArtFID:
  - `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samst_distinct5_512_real_b2_e15_20260602\eval_epoch15\epoch_0015\aggregate_targetwise_artfid.json`
- row-alignment reports:
  - `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\distinct5_512_20260602\baseline_packet_status_20260604\samst_e5_idt_aligned_rows.csv`
  - `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\distinct5_512_20260602\baseline_packet_status_20260604\samst_e15_idt_aligned_rows.csv`

Missing for SaMST e15:

- no same-run-root generation log or equivalent packet-bound inference timing artifact
- the older timing note
  - `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\2026-06-01-main-table-gap-analysis.md`
  gives `126.22s / 0.168 s/img`, but it is not bound to the current Distinct5 e15 packet root

## SaMAM

### Active manuscript checkpoint

Use only `step_002250` in the active AAAI/LBM manuscript path.

| checkpoint | tr CLIP-S | tr LPIPS | tr ArtFID | delta vs IDT | train min | status |
|---|---:|---:|---:|---:|---:|---|
| step_002250 | 0.5523 | 0.3605 | 148.2 | -0.0877 | 458.6 | manuscript-valid |

Interpretation relative to IDT:

- transfer CLIP-S remains below the IDT floor by `-0.0877`
- this supports the current manuscript wording: SaMAM 2250 improves
  art-domain diagnostics while failing target execution under CLIP-S/IDT
- do not upgrade the paper to a post-2250 SaMAM point without a clean
  independent rerun

### Post-2250 packets retained as audit history only

The following packets exist in the audit trail, but are excluded from the active
manuscript path by the boundary above.

- root:
  - `I:\Github\Latent_Style\Related_Works\baseline_pipeline\results\samam_distinct5_512_mamba_b6_seg250_remote_wsl_20260601_2130_diag`
- summary:
  - `I:\Github\Latent_Style\Related_Works\baseline_pipeline\results\samam_distinct5_512_mamba_b6_seg250_remote_wsl_20260601_2130_diag\eval_curve\step_003000_artfid_reuse\summary.json`
- metrics:
  - `I:\Github\Latent_Style\Related_Works\baseline_pipeline\results\samam_distinct5_512_mamba_b6_seg250_remote_wsl_20260601_2130_diag\eval_curve\step_003000_artfid_reuse\metrics.csv`
- timing source:
  - `I:\Github\Latent_Style\Related_Works\baseline_pipeline\results\samam_distinct5_512_mamba_b6_seg250_remote_wsl_20260601_2130_diag\segmented.log`

Metrics:

| checkpoint | tr CLIP-S | tr LPIPS | tr ArtFID | full CLIP-S | full LPIPS | full ArtFID | train h | inference ms/img | aligned rows | missing rows | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| step_003000 | 0.664618 | 0.327094 | 394.7662 | 0.697801 | 0.322087 | 345.6017 | 10.21 | missing | 750 | 0 | audit-only / excluded |

Interpretation:

- `step_003000` is not active manuscript evidence
- do not describe it as a positive-IDT SaMAM result in the paper
- do not use `394.8` targetwise ArtFID or `10.2h` training time in the active
  manuscript path

Missing for SaMAM 3000:

- same-scope inference `ms/img` is not recoverable from the current packet
- current same-scope timing is train wall plus full eval wall, not pure generation wall

### Later tuned candidate remains partial

Latest visible tuned candidate:

- summary:
  - `/home/xy/samam_eval_local/step_003250_artfid_reuse2/summary.json`
- metrics:
  - `/home/xy/samam_eval_local/step_003250_artfid_reuse2/metrics.csv`
- local eval sidecar:
  - `/home/xy/samam_eval_local/step_003250_curve_named/curve_metrics.json`

Metrics:

| checkpoint | tr CLIP-S | tr LPIPS | tr ArtFID | full CLIP-S | full LPIPS | full ArtFID | train h | inference ms/img | aligned rows | missing rows | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| step_003250 | 0.662696 | 0.314587 | 389.6994 | 0.697008 | 0.309583 | 338.8172 | missing | missing | 750 | 0 | audit-only / partial |

Missing for SaMAM 3250:

- no authoritative sync under the main `I:\...samam_distinct5_512_mamba_b6_seg250_remote_wsl_20260601_2130_diag` root
- no cumulative train wall proof to `3250`
- no same-scope inference `ms/img`

## Paper wording gate

Paper-safe upgrade:

- none for SaMAM beyond `2250`

Not yet safe to upgrade:

- do not claim `SaMAM closed above IDT` in the active manuscript
- do not claim a same-scope cost comparison for SaMAM because pure generation `ms/img` is still missing
- do not call the full SaMST line closed while `e15` lacks packet-bound inference timing
