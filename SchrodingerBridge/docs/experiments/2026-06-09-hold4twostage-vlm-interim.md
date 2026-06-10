# Hold4TwoStage VLM Interim

Date: 2026-06-09

Scope:

- local CPU-only VLM triplet review:
  - `LBM-Knee_e13`
  - `Seedream_repaired750`
  - `Hold4TwoStage_e12`
- and a second local CPU-only triplet:
  - `LBM-Knee_e13`
  - `Seedream_repaired750`
  - `Hold4TwoStage_e02`

Manifest:

- [vlm_manifest_hold4twostage_e12_vs_knee_vs_seedream_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_manifest_hold4twostage_e12_vs_knee_vs_seedream_20260609.csv)

Raw outputs:

- [vlm_hold4twostage_e12_vs_knee_vs_seedream_20260609.jsonl](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_hold4twostage_e12_vs_knee_vs_seedream_20260609.jsonl)
- [vlm_hold4twostage_e02_vs_knee_vs_seedream_20260609.jsonl](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_hold4twostage_e02_vs_knee_vs_seedream_20260609.jsonl)
- [vlm_hold4twostage_e20_vs_knee_vs_seedream_20260609.jsonl](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_hold4twostage_e20_vs_knee_vs_seedream_20260609.jsonl)

Current completed cases:

- `616` for `Hold4TwoStage_e12`
- `566` for `Hold4TwoStage_e02`
- `549` for `Hold4TwoStage_e20`

Current winner read:

- `Hold4TwoStage_e12`:
  - `Seedream_repaired750` currently wins `596 / 616`
  - `Hold4TwoStage_e12` currently wins `20 / 616`
- `Hold4TwoStage_e02`:
  - `Seedream_repaired750` currently wins `550 / 566`
  - `Hold4TwoStage_e02` currently wins `15 / 566`
- `Hold4TwoStage_e20`:
  - `Seedream_repaired750` currently wins `530 / 549`
  - `Hold4TwoStage_e20` currently wins `18 / 549`

Qualitative interim pattern:

- `Seedream` is still dominating on:
  - style specificity
  - structure preservation
  - artifact control
- `Hold4TwoStage_e12` does not look like a hidden win over `LBM-Knee`
- `Hold4TwoStage_e02` looks even weaker
- `Hold4TwoStage_e20` is also currently weak
- even with the larger completed set, the family still stays below `LBM-Knee` on the more meaningful internal read:
  - `LBM-Knee` keeps noticeably better mean structure and artifact scores
- if anything, the early VLM read is consistent with the current local `IntroStyle + DINO` judgment:
  - `Hold4TwoStage` is not emerging as the next promoted family

Current implication:

- the schedule-only continuation story remains weak
- even before the VLM batch finishes, the existing evidence stack is already converging toward:
  - `do not promote Hold4TwoStage`
