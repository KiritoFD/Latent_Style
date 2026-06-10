# Local VLM Method Summary

Date: 2026-06-09

Artifacts:

- summary table:
  - [vlm_distinct5_finalists_method_summary_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_distinct5_finalists_method_summary_20260609.csv)
- summary figure:
  - [vlm_distinct5_finalists_method_summary_20260609.png](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_distinct5_finalists_method_summary_20260609.png)
- source run:
  - [vlm_distinct5_finalists_full750_20260609.jsonl](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_distinct5_finalists_full750_20260609.jsonl)

Current completed cases:

- `713`

Compared methods:

- `LBM-Knee_e13`
- `LBM-PS-v2_e13`
- `Seedream_repaired750`

Current means:

- `LBM-Knee_e13`
  - style specificity: `2.07`
  - structure preservation: `3.29`
  - artifact control: `2.40`
  - best-overall wins: `0 / 713`
  - structure wins: `35 / 713`
  - artifact wins: `23 / 713`
- `LBM-PS-v2_e13`
  - style specificity: `1.50`
  - structure preservation: `2.12`
  - artifact control: `1.33`
  - best-overall wins: `3 / 713`
  - style wins: `4 / 713`
- `Seedream_repaired750`
  - style specificity: `4.96`
  - structure preservation: `4.93`
  - artifact control: `4.96`
  - best-overall wins: `709 / 713`
  - style wins: `708 / 713`
  - structure wins: `673 / 713`
  - artifact wins: `685 / 713`

Interpretation:

- the VLM is not merely calling `Seedream` more stylistic
- it is also consistently rating it higher on:
  - structure preservation
  - artifact control
- this is especially important because it pushes against the naive expectation that stronger style should necessarily cost more geometry
- `LBM-Knee` does keep a limited secondary advantage on some structure-only and artifact-only subvotes over `LBM-PS-v2`

Current implication:

- `LBM-PS-v2` is now strongly disfavored by every serious review axis we have:
  - full750 `IntroStyle`
  - full750 `DINO`
  - local VLM
- `LBM-Knee` remains the best internal point, but the current VLM still prefers `Seedream` overwhelmingly
- the next internal mechanism burden remains:
  - improve `IntroStyle` over `LBM-Knee`
  - keep DINO in the same regime
  - and close part of the large VLM gap to `Seedream`
