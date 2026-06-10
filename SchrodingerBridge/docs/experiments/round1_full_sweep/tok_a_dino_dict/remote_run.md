# tok_a_dino_dict Remote Run Log

- Run dir: `./exp/inmortal-exp/aaai2027_round1_tok_a_dino_dict_seed42_b8a2`
- Launch attempt on `2026-06-10`:
  - queue was allowed to advance because `attn_sa_mod` had already moved from `running` to `reviewing`
  - the remote launcher refused the formal start before execution
  - refusal reason:
    - remote prelaunch GPU memory was `8968 MiB`
    - allowed single-lane prelaunch ceiling was `7000 MiB`
  - decision:
    - keep `tok_a_dino_dict` in `planned`
    - retry only after the remote `3060` returns to the formal single-lane idle band
  - note:
    - config-side DINO cache wiring is already present in:
      - [aaai2027_round1_tok_a_dino_dict_seed42_b8a2.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/round1_full_sweep/aaai2027_round1_tok_a_dino_dict_seed42_b8a2.json)
- Remote occupancy follow-up:
  - repeated samples after the refusal oscillated between about `5356 MiB` and `8968 MiB`
  - because round-1 train launches also require runtime memory in the `9000-11300 MiB` band, the launcher was tightened to derive an effective prelaunch ceiling of:
    - `min(requested_prelaunch, max_runtime - min_runtime)`
    - for current round-1 train settings this is `min(7000, 11300 - 9000) = 2300 MiB`
  - decision update:
    - do not retry `tok_a_dino_dict` until remote baseline occupancy is well below `2300 MiB`

<!-- ROUND1_AUTO_STATUS:START -->
## Auto Status

- Family id: `tok_a_dino_dict`
- Run name: `aaai2027_round1_tok_a_dino_dict_seed42_b8a2`
- Remote run dir: `./exp/inmortal-exp/aaai2027_round1_tok_a_dino_dict_seed42_b8a2`
- Config: [aaai2027_round1_tok_a_dino_dict_seed42_b8a2.json](G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/round1_full_sweep/aaai2027_round1_tok_a_dino_dict_seed42_b8a2.json)
- Manifest status: `planned`
- Local fast root: [round1_tok_a_dino_dict_fast_local](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_tok_a_dino_dict_fast_local)
- Local review root: [round1_tok_a_dino_dict_localreview](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_tok_a_dino_dict_localreview)
- Prelaunch switch smoke: `ok`
- Switch smoke artifact: [round1_tok_a_dino_dict_switch_smoke_latest.json](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_tok_a_dino_dict_switch_smoke_latest.json)
- Switch smoke row count: `1`
<!-- ROUND1_AUTO_STATUS:END -->




