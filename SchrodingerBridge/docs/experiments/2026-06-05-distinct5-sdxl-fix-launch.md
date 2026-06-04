# Distinct5-512 SDXL-fix Local Launch

Date: 2026-06-05

Scope: launch record for the local `Distinct5-512` SDXL-latent experiment using
the current `LBM-K` family on the RTX 4070 Laptop GPU.

## Inputs

Latent root:

- `F:/wikiart_distinct5_512_latents_sdxl_fix/train`

Test image root:

- `F:/wikiart_distinct5_samam_512_classview/test`

Styles:

- `Early_Renaissance`
- `Impressionism`
- `Minimalism`
- `Rococo`
- `Ukiyo_e`

## Prepared Artifacts

Packed latent cache:

- `F:/wikiart_distinct5_512_latents_sdxl_fix/train/.latent_cache/packed`

Pairing cache:

- `F:/wikiart_distinct5_512_latents_sdxl_fix/train/.latent_cache/prototype_pairing_top8.pt`

Pairing cache sidecar:

- `F:/wikiart_distinct5_512_latents_sdxl_fix/train/.latent_cache/prototype_pairing_top8.json`

Config:

- [local_distinct5_512_sdxl_fix_k_b32_e8.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/archive/20260605_local_distinct5_sdxl_fix/local_distinct5_512_sdxl_fix_k_b32_e8.json)

## Launch

Training command:

```powershell
python src/run.py --config configs/archive/20260605_local_distinct5_sdxl_fix/local_distinct5_512_sdxl_fix_k_b32_e8.json
```

Checkpoint root:

- `G:/GitHub/Latent_Style/SchrodingerBridge/exp/local_distinct5_512_sdxl_fix_k_b32_e8`

Live stderr log:

- [local_distinct5_sdxl_fix_train.err.log](/G:/GitHub/Latent_Style/SchrodingerBridge/_codex_tmp/local_distinct5_sdxl_fix_train.err.log)

Live stdout log:

- [local_distinct5_sdxl_fix_train.out.log](/G:/GitHub/Latent_Style/SchrodingerBridge/_codex_tmp/local_distinct5_sdxl_fix_train.out.log)

Training CSV log root:

- `G:/GitHub/Latent_Style/SchrodingerBridge/exp/local_distinct5_512_sdxl_fix_k_b32_e8/logs`

## Current Status At Launch Audit

Verified before writing this note:

- all `5` styles were fully encoded with `1000` latent files each
- packed cache build completed
- prototype-aware pairing cache build completed with `20000` routes
- training entered `Epoch 1/8`
- run stayed numerically healthy past the early-step region that previously
  killed the raw `stabilityai/sdxl-vae` fp16 path

This note is only a launch/status record. It does not claim any metric outcome
yet.
