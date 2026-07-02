# Round2 Follow-on: wave2_sde_noise

- Winner family: `tok_pure_latent_spatial`
- Winner checkpoint: `/mnt/i/Github/Latent_Style/exp/inmortal-exp/aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2_c11/epoch_0002.pt`
- Source manifest: `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\round2_pure_sde\round2_family_manifest.csv`

## Generated Configs
- sde_i2sb_sigma_0p25:
  - Config: `G:\GitHub\Latent_Style\SchrodingerBridge\configs\aaai2027\round2_pure_sde\followon\tok_pure_latent_spatial\aaai2027_round2_sde_i2sb_sigma_0p25_seed42_b8a2_from_tok_pure_latent_spatial_epoch_0002.json`
  - Run name: `aaai2027_round2_sde_i2sb_sigma_0p25_seed42_b8a2_from_tok_pure_latent_spatial_epoch_0002`

## Runtime-Fix Rerun

- Relaunch config:
  - `G:\GitHub\Latent_Style\SchrodingerBridge\configs\aaai2027\round2_pure_sde\followon\tok_pure_latent_spatial\aaai2027_round2_sde_i2sb_sigma_0p25_seed42_b8a2_from_tok_pure_latent_spatial_epoch_0002_rtfix.launch.json`
- Relaunch run name:
  - `aaai2027_round2_sde_i2sb_sigma_0p25_seed42_b8a2_from_tok_pure_latent_spatial_epoch_0002_rtfix`
- Why:
  - the old `c25clean` lane was launched before the bridge-runtime sigma fix
  - `rtfix` is the first rerun that uses the corrected true-I2SB runtime contract from the start
- Current runtime read:
  - 20s health check was under-band at `6821 MiB`
  - later warmup samples reached `8617 MiB` and then `9285 MiB`
  - no further batch bump is currently needed before the first retained checkpoint
