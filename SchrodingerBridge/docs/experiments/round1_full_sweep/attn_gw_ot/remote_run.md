# attn_gw_ot Remote Run Log

- Run dir: `./exp/inmortal-exp/aaai2027_round1_attn_gw_ot_seed42_b8a2`
- Config:
  - [aaai2027_round1_attn_gw_ot_seed42_b8a2.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/round1_full_sweep/aaai2027_round1_attn_gw_ot_seed42_b8a2.json)
- Remote train log:
  - `/mnt/i/Github/Latent_Style/exp/inmortal-exp/aaai2027_round1_attn_gw_ot_seed42_b8a2_train.log`
- Dataset root:
  - `/mnt/i/wikiarts_5_full_notest_latents_ema/train`
- Cache roots:
  - `/mnt/i/wikiarts_5_full_notest_latents_ema/train/.latent_cache/manifest.json`
  - `/mnt/i/wikiarts_5_full_notest_latents_ema/train/.latent_cache/prototype_pairing_top8.pt`

Launch sequence on `2026-06-10`:

- attempt 1:
  - batch size `13`
  - health sample `7921 MiB`
  - decision: rejected as under-band; kept `planned`
- attempt 2:
  - batch size raised to `15`
  - health sample `8992 MiB`
  - decision: still rejected because the old launcher used a zero-slack lower-bound check
- launcher correction:
  - `launch_remote_wsl_command.py` now applies `128 MiB` slack only to the minimum-band check
  - the `11.0 GiB` upper cap remains strict
- attempt 3:
  - batch size `15`
  - authoritative health sample `8970 MiB / 12288 MiB`
  - decision at launch: accepted
  - later runtime sample:
    - `11979 MiB / 12288 MiB`
  - decision after runtime audit:
    - invalid for formal use
    - stopped immediately because it crossed the paper-facing cap
- runtime-guard fix:
  - [launch_remote_wsl_command.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_wsl_command.py) now supports a continuous runtime VRAM guard inside the generated remote WSL launcher
  - [launch_remote_round1_family_train.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_round1_family_train.py) now passes the round-1 guard cap through
- attempt 4:
  - batch size reduced to `12`
  - launch-time minimum band temporarily relaxed to `7000 MiB` for this recalibration, while the runtime guard stays at `11000 MiB`
  - authoritative relaunch health sample:
    - `10711 MiB / 12288 MiB`
  - current state:
    - this is the active formal lane
    - fast-eval watcher is still intentionally skipped because local GPU is reserved for WSL `SaMAM` repro
    - a deferred local fast-eval launcher is now armed and waiting for the `SaMAM` repro to truly finish before it starts

<!-- ROUND1_AUTO_STATUS:START -->
## Auto Status

- Family id: `attn_gw_ot`
- Run name: `aaai2027_round1_attn_gw_ot_seed42_b8a2`
- Remote run dir: `./exp/inmortal-exp/aaai2027_round1_attn_gw_ot_seed42_b8a2`
- Config: [aaai2027_round1_attn_gw_ot_seed42_b8a2.json](G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/round1_full_sweep/aaai2027_round1_attn_gw_ot_seed42_b8a2.json)
- Manifest status: `running`
- Local fast root: [round1_attn_gw_ot_fast_local](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_gw_ot_fast_local)
- Local review root: [round1_attn_gw_ot_localreview](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_gw_ot_localreview)
<!-- ROUND1_AUTO_STATUS:END -->

