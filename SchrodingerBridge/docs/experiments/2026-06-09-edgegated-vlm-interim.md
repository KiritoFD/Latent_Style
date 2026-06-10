# EdgeGated VLM Interim

Date: 2026-06-09

Scope:

- local CPU-only VLM triplet review:
  - `LBM-Knee_e13`
  - `Seedream_repaired750`
  - `EdgeGated_e03`
- companion CPU-only triplets are also live:
  - `EdgeGated_e01`
  - `EdgeGated_e12`

Manifest:

- [vlm_manifest_edgegated_e03_vs_knee_vs_seedream_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_manifest_edgegated_e03_vs_knee_vs_seedream_20260609.csv)

Raw outputs:

- [vlm_edgegated_e03_vs_knee_vs_seedream_20260609.jsonl](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_edgegated_e03_vs_knee_vs_seedream_20260609.jsonl)
- [vlm_edgegated_e03_vs_knee_vs_seedream_summary_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_edgegated_e03_vs_knee_vs_seedream_summary_20260609.csv)

Current completed cases:

- `340` for `EdgeGated_e03`
- `245` for `EdgeGated_e01`
- `244` for `EdgeGated_e12`

Current winner read:

- `best_overall`:
  - `Seedream_repaired750` wins `321 / 340`
  - `EdgeGated_e03` wins `19 / 340`
- secondary subwins:
  - `EdgeGated_e03` currently has:
    - `50` structure subwins
    - `64` artifact-control subwins

Current means:

- `LBM-Knee_e13`
  - style specificity: `2.00`
  - structure preservation: `3.15`
  - artifact control: `2.42`
- `Seedream_repaired750`
  - style specificity: `4.89`
  - structure preservation: `4.76`
  - artifact control: `4.68`
- `EdgeGated_e03`
  - style specificity: `2.69`
  - structure preservation: `3.86`
  - artifact control: `3.42`

Interpretation:

- this is not yet a promoted win
- but unlike the earlier `Hold4TwoStage` schedule family, it is at least showing the right *kind* of partial signal:
  - some structure-side wins
  - some artifact-side wins
- and now `19` sparse `best_overall` wins
- while still losing decisively on `best_overall` style/content judgement to `Seedream`

Companion triplet status:

- `EdgeGated_e01`
  - `Seedream 235 / 245`
  - candidate `10 / 245`
- `EdgeGated_e12`
  - `Seedream 235 / 244`
  - candidate `9 / 244`

Stage read:

- `still alive, but not yet sufficient`
