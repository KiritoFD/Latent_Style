# Remote / Local Evaluation Contract

Date: 2026-06-09

This note locks the current division of labor for the next round.

## Contract

Remote is the narrow execution surface.

Remote should do only:

- training
- retained checkpoint evaluation
- compact `CSV` production
- handoff packaging for the best few epochs or checkpoints

Remote should not be treated as the main interpretation surface.

Local is the broad review and direction surface.

Local should do:

- `VLM` review
- broader `IntroStyle + DINO` comparative audits
- multi-point plotting
- theory interpretation
- failure-mode diagnosis against `Seedream`
- next-step mechanism selection

## Practical rule

For each remote line:

1. remote closes `full_eval` or `fresh_localreview`
2. remote exports:
   - `summary.json`
   - `metrics.csv`
   - compact curve CSV
   - best-few handoff CSV
3. local takes only the shortlisted points forward for:
   - `VLM`
   - richer cross-method comparison
   - paper-facing explanation

Operational exception:

- if remote `IntroStyle` hits a path/filename robustness issue after the handoff CSV is already available,
  do not keep the remote GPU tied up on eval-debugging
- at that point:
  - remote should move on to the next training-side or packet-side task
  - local should absorb the shortlisted eval and complete the interpretation

## Current live example

Active remote training line:

- `aaai2027_inmortal_knee_e13_spatial_carriergate_bodydecoder_qedgegated_pattn_seed42_b8a2`

Most recently closed handoff example:

- `aaai2027_inmortal_knee_e13_spatial_carriergate_bodydecoder_edgegated_seed42_b8a2`

Current remote handoff file:

- [knee_spatial_carriergate_bodydecoder_edgegated_bestfew_handoff_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/knee_spatial_carriergate_bodydecoder_edgegated_bestfew_handoff_20260609.csv)

Current local review burden:

- compare the handoff points against:
  - `LBM-Knee`
  - `LBM-PS-v2`
  - `Seedream`
- decide whether the line is:
  - true positive
  - near-tie negative
  - or dead family

Prepared CPU-only VLM triplet manifests:

- [vlm_manifest_hold4twostage_e12_vs_knee_vs_seedream_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_manifest_hold4twostage_e12_vs_knee_vs_seedream_20260609.csv)
- [vlm_manifest_hold4twostage_e02_vs_knee_vs_seedream_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_manifest_hold4twostage_e02_vs_knee_vs_seedream_20260609.csv)
- [vlm_manifest_hold4twostage_e20_vs_knee_vs_seedream_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_manifest_hold4twostage_e20_vs_knee_vs_seedream_20260609.csv)
- [vlm_manifest_knee_spatial_e08_vs_knee_vs_seedream_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_manifest_knee_spatial_e08_vs_knee_vs_seedream_20260609.csv)
- [vlm_manifest_edgegated_e03_vs_knee_vs_seedream_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_manifest_edgegated_e03_vs_knee_vs_seedream_20260609.csv)
- [vlm_manifest_edgegated_e01_vs_knee_vs_seedream_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_manifest_edgegated_e01_vs_knee_vs_seedream_20260609.csv)
- [vlm_manifest_edgegated_e12_vs_knee_vs_seedream_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_manifest_edgegated_e12_vs_knee_vs_seedream_20260609.csv)

Current note on the edge-gated triplets:

- the CPU-only VLM entrypoints are prepared
- `EdgeGated_e03` is already well into a nontrivial evidence band
  - currently `247` completed cases
  - currently `11` candidate overall wins
- `EdgeGated_e01` and `EdgeGated_e12` are no longer just first-case probes
  - `EdgeGated_e01`: `140` completed, `7` candidate wins
  - `EdgeGated_e12`: `139` completed, `6` candidate wins

## Why this split

- the remote `3060` should stay focused on throughput and reproducible packet closure
- the local surface is better for expensive review, broader comparison, and theory-facing diagnosis
- this avoids spending the remote lane on interpretation work that does not change the training result itself
