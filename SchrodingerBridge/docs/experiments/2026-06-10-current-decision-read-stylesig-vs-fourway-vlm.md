# Current Decision Read: Stylesig vs Four-Way VLM

Date: 2026-06-10

This note compresses the current decision state into the shortest useful form.

## 1. Remote `stylesig` status

Current remote branch:

- `aaai2027_inmortal_knee_e13_spatial_carriergate_bodydecoder_qedgegated_pattn_stylesig_seed42_b8a2`

Current verified state:

- training is alive
- current retained checkpoints already include:
  - `epoch_0001.pt` through `epoch_0010.pt`
- log has progressed into:
  - `epoch 11`
- runtime memory remains machine-safe
- first eval outputs have now landed under:
  - `full_eval/epoch_0001`
  - `full_eval/epoch_0002`
  - `full_eval/epoch_0003`
- dedicated `full_eval_fast_snapshot` watcher is now also attached

First read:

- `epoch_0001 -> epoch_0005` shows a monotonic transfer-style increase
- but also a monotonic `LPIPS` worsening

Implication:

- `stylesig` is no longer train-only
- but its first read still looks like a familiar:
  - `style-up / structure-down`
  trajectory
- so it should still not be treated as a promoted frontier branch

## 2. Local four-way external-baseline VLM state

Current frozen board:

- [vlm_lbmpsv2_vs_seedream_vs_samst_vs_samam_20260610_snapshot5.method_summary.md](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_lbmpsv2_vs_seedream_vs_samst_vs_samam_20260610_snapshot5.method_summary.md)

Current `160`-case read:

- `Seedream`
  - still best overall
- `SaMAM_2250`
  - now very close in overall wins
  - clearly stronger on:
    - structure
    - artifact control
- `SaMST_e15`
  - style-active but not converting into wins
- `LBM-PS-v2_e13`
  - clearly below all three external baselines

Current local runtime caveat:

- the live `VLM` run is still usable
- but the current error file is dominated by transport/runtime failures, not by
  panel-format drift
- current observed error mix:
  - `ConnectionAbortedError 10053 = 3`
  - `HTTP 500 = 2`

Most important interpretation change:

- `SaMAM_2250` is no longer just a baseline to beat casually
- it is now a real paper-facing visual anchor
- especially if the claim space includes:
  - cleaner geometry
  - fewer artifacts
  - better usable image quality

## 3. Current theory-facing implication

The current evidence says the main unsolved gap is not just:

- `more style`

It is more specifically:

- how to increase target-style specificity enough to challenge `Seedream`
- without giving up the structure and artifact advantages that are already
  visible in `SaMAM`-like outputs

That means the strongest next mechanism should be judged against this concrete
failure mode:

- if it only raises internal cheap style metrics but still loses the visual board
  to `SaMAM`, it is not yet solving the real problem

## 4. Operational guidance

Do now:

1. keep `stylesig` training/eval closure moving beyond the first three points
2. continue frozen-snapshot local `VLM` reading, not live-CSV reading
3. treat `SaMAM` as a serious external comparator in any next-round promotion gate

Do not do now:

- do not claim the new mechanism opened the ceiling before the first eval lands
- do not keep using only `Seedream` as the visual comparison anchor
- do not read network-interrupted live `VLM` rows as authoritative without a snapshot freeze
