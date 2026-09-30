# tok_d_vlm_prompt Remote Run Log

- Run dir: `./exp/inmortal-exp/aaai2027_round1_tok_d_vlm_prompt_seed42_b8a2`

- new-data retry on `2026-06-12`:
  - tokenizer-family infra blockers were already cleared before this retry:
    - matching `wikiarts_5_full_notest` DINO cache
    - corrected DINO patch-grid to latent-grid reshape logic
  - first strict retry:
    - `batch=8`
    - entered training
    - later hit runtime guard at about `11896MiB`
    - above the hard `11.3GiB` cap
  - second strict retry:
    - `batch=7`
    - entered training
    - 180-second health read about `8532MiB`
    - below the requested floor
  - current conclusion:
    - `tok_d_vlm_prompt` now has a clean strict `7/8` memory bracket
    - keep it in `recalibration_needed`

<!-- ROUND1_AUTO_STATUS:START -->
## Auto Status

- Family id: `tok_d_vlm_prompt`
- Run name: `aaai2027_round1_tok_d_vlm_prompt_seed42_b8a2`
- Remote run dir: `./exp/inmortal-exp/aaai2027_round1_tok_d_vlm_prompt_seed42_b8a2`
- Config: [aaai2027_round1_tok_d_vlm_prompt_seed42_b8a2.json](G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/round1_full_sweep/aaai2027_round1_tok_d_vlm_prompt_seed42_b8a2.json)
- Manifest status: `recalibration_needed`
- Local fast root: [round1_tok_d_vlm_prompt_fast_local](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_tok_d_vlm_prompt_fast_local)
- Local review root: [round1_tok_d_vlm_prompt_localreview](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_tok_d_vlm_prompt_localreview)
- Prelaunch switch smoke: `ok`
- Switch smoke artifact: [round1_tok_d_vlm_prompt_switch_smoke_latest.json](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_tok_d_vlm_prompt_switch_smoke_latest.json)
- Switch smoke row count: `1`
- Tokenizer warmstart config: [aaai2027_round1_tok_d_vlm_prompt_warmstart_seed42_b8a2.json](G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/round1_full_sweep/warmstart/aaai2027_round1_tok_d_vlm_prompt_warmstart_seed42_b8a2.json)
- Tokenizer reconstruction-pretrain config: [aaai2027_round1_tok_d_vlm_prompt_reconpretrain_seed42_b8a2.json](G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/round1_full_sweep/pretrain/aaai2027_round1_tok_d_vlm_prompt_reconpretrain_seed42_b8a2.json)
<!-- ROUND1_AUTO_STATUS:END -->
