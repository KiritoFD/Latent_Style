# tok_b_cross_image Remote Run Log

- Run dir: `./exp/inmortal-exp/aaai2027_round1_tok_b_cross_image_seed42_b8a2`

- new-data retry on `2026-06-12`:
  - matching `wikiarts_5_full_notest` DINO cache is now available
  - first strict tokenizer-tail retry:
    - `batch=8`
    - entered a real formal lane
    - representative early live read about `9793MiB`
  - later read:
    - the same `epoch_0001` run drifted down to about `8279MiB`
    - and was killed by the strict under-band guard before the first retained checkpoint landed
  - current conclusion:
    - keep `tok_b_cross_image` in `recalibration_needed`
    - next useful retry is `batch=9` under the same strict contract

<!-- ROUND1_AUTO_STATUS:START -->
## Auto Status

- Family id: `tok_b_cross_image`
- Run name: `aaai2027_round1_tok_b_cross_image_seed42_b8a2`
- Remote run dir: `./exp/inmortal-exp/aaai2027_round1_tok_b_cross_image_seed42_b8a2`
- Config: [aaai2027_round1_tok_b_cross_image_seed42_b8a2.json](G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/round1_full_sweep/aaai2027_round1_tok_b_cross_image_seed42_b8a2.json)
- Manifest status: `recalibration_needed`
- Local fast root: [round1_tok_b_cross_image_fast_local](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_tok_b_cross_image_fast_local)
- Local review root: [round1_tok_b_cross_image_localreview](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_tok_b_cross_image_localreview)
- Prelaunch switch smoke: `ok`
- Switch smoke artifact: [round1_tok_b_cross_image_switch_smoke_latest.json](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_tok_b_cross_image_switch_smoke_latest.json)
- Switch smoke row count: `1`
- Tokenizer warmstart config: [aaai2027_round1_tok_b_cross_image_warmstart_seed42_b8a2.json](G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/round1_full_sweep/warmstart/aaai2027_round1_tok_b_cross_image_warmstart_seed42_b8a2.json)
- Tokenizer reconstruction-pretrain config: [aaai2027_round1_tok_b_cross_image_reconpretrain_seed42_b8a2.json](G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/round1_full_sweep/pretrain/aaai2027_round1_tok_b_cross_image_reconpretrain_seed42_b8a2.json)
<!-- ROUND1_AUTO_STATUS:END -->
