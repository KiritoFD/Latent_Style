# attn_sa_mod VLM Stageclose Snapshot

- Snapshot date: `2026-06-10`
- Purpose: freeze the external-baseline `VLM` evidence used for the round-1 `attn_sa_mod` reject decision

Frozen artifacts:

- board:
  - [round1_attn_sa_mod_vlm_stageclose_20260610_board.md](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_sa_mod_vlm_stageclose_20260610_board.md)
  - [round1_attn_sa_mod_vlm_stageclose_20260610_board.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_sa_mod_vlm_stageclose_20260610_board.csv)
- per-candidate summaries:
  - [round1_attn_sa_mod_vlm_stageclose_20260610_e08.method_summary.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_sa_mod_vlm_stageclose_20260610_e08.method_summary.csv)
  - [round1_attn_sa_mod_vlm_stageclose_20260610_e24.method_summary.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_sa_mod_vlm_stageclose_20260610_e24.method_summary.csv)

Frozen read:

- `AttnSA_e08 vs Seedream vs SaMAM`
  - valid cases: `200`
  - wins:
    - `AttnSA_e08 = 2`
    - `SaMAM_2250 = 104`
    - `Seedream_repaired750 = 94`
- `AttnSA_e24 vs Seedream vs SaMAM`
  - valid cases: `169`
  - wins:
    - `AttnSA_e24 = 0`
    - `SaMAM_2250 = 97`
    - `Seedream_repaired750 = 72`

Interpretation:

- this snapshot is already decisive for round-1 promotion:
  - both shortlisted `attn_sa_mod` checkpoints are far below `SaMAM / Seedream`
  - the family does not convert its stable fast-curve closure into paper-facing visual competitiveness
