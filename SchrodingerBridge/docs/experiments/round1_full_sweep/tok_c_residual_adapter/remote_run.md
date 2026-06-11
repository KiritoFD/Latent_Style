# tok_c_residual_adapter Remote Run Log

- Run dir: `./exp/inmortal-exp/aaai2027_round1_tok_c_residual_adapter_seed42_b8a2`

- new-data retry on `2026-06-12`:
  - the family entered a real formal lane on the first strict retry:
    - `batch=8`
    - representative live read about `10125MiB`
  - but the same run later drifted down and was killed by the under-band guard:
    - about `8313MiB`
    - before the first retained checkpoint landed
  - second strict retry:
    - `batch=9`
    - launched successfully
    - but later hit the hard cap at about `11898MiB`
    - again before the first retained checkpoint landed
  - current conclusion:
    - `tok_c_residual_adapter` now has a clean strict `8/9` bracket
    - keep it in `recalibration_needed`

<!-- ROUND1_AUTO_STATUS:START -->
## Auto Status

- Family id: `tok_c_residual_adapter`
- Run name: `aaai2027_round1_tok_c_residual_adapter_seed42_b8a2`
- Remote run dir: `./exp/inmortal-exp/aaai2027_round1_tok_c_residual_adapter_seed42_b8a2`
- Config: [aaai2027_round1_tok_c_residual_adapter_seed42_b8a2.json](G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/round1_full_sweep/aaai2027_round1_tok_c_residual_adapter_seed42_b8a2.json)
- Manifest status: `recalibration_needed`
- Local fast root: [round1_tok_c_residual_adapter_fast_local](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_tok_c_residual_adapter_fast_local)
- Local review root: [round1_tok_c_residual_adapter_localreview](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_tok_c_residual_adapter_localreview)
- Prelaunch switch smoke: `ok`
- Switch smoke artifact: [round1_tok_c_residual_adapter_switch_smoke_latest.json](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_tok_c_residual_adapter_switch_smoke_latest.json)
- Switch smoke row count: `1`
- Tokenizer warmstart config: [aaai2027_round1_tok_c_residual_adapter_warmstart_seed42_b8a2.json](G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/round1_full_sweep/warmstart/aaai2027_round1_tok_c_residual_adapter_warmstart_seed42_b8a2.json)
- Tokenizer reconstruction-pretrain config: [aaai2027_round1_tok_c_residual_adapter_reconpretrain_seed42_b8a2.json](G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/round1_full_sweep/pretrain/aaai2027_round1_tok_c_residual_adapter_reconpretrain_seed42_b8a2.json)
<!-- ROUND1_AUTO_STATUS:END -->
