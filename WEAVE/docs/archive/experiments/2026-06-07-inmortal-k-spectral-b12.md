# `K_spectral` Safety Rerun

Date: 2026-06-07

Reason for rerun:

- the first formal `K_spectral b16` launch crossed the current remote `11.5 GiB` paper-facing cap
- that makes the `b16` launch invalid as a formal evidence surface, even if it is numerically healthy

Corrective action:

- keep the mechanism unchanged
- reduce only:
  - `training.batch_size: 16 -> 12`
- rerun as:
  - [inmortal_k_spectral_seed42_b12.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_k_spectral_seed42_b12.json)

Interpretation rule:

- this is a machine-contract correction, not a mechanism verdict change
- any paper-facing `K_spectral` read should come from the `b12` packet, not the over-cap `b16` packet

## Closed readout

Remote packet:

- run dir:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/inmortal-exp/aaai2027_inmortal_k_spectral_seed42_b12`

Training surface:

- batch: `12`
- evaluated checkpoints: `e1` to `e6`
- trainer-recorded peak memory on the retained curve:
  - allocated about `6.09 GB`
  - reserved about `6.46 GB`

Transfer curve:

| epoch | transfer CLIP-style | transfer LPIPS |
| --- | ---: | ---: |
| `e1` | `0.6776` | `0.3373` |
| `e2` | `0.6788` | `0.3637` |
| `e3` | `0.6776` | `0.3583` |
| `e4` | `0.6740` | `0.3417` |
| `e5` | `0.6783` | `0.3308` |
| `e6` | `0.6773` | `0.3229` |

Best retained point by the current promotion rule:

- `e2`
  - transfer `clip_style = 0.6788`
  - transfer `content_lpips = 0.3637`
  - full `clip_style = 0.7101`
  - full `content_lpips = 0.3522`

Additional note:

- the final evaluated point `e6` has the lowest LPIPS on this packet:
  - transfer `clip_style = 0.6773`
  - transfer `content_lpips = 0.3229`

Interpretation:

- `K_spectral` is stable and machine-safe after the `b12` correction
- it does not improve the current ceiling frontier
- relative to the other pure-kinetic families:
  - raw style is nearly tied with `K_manifold`
  - LPIPS is worse than `K_manifold`
  - LPIPS is also slightly worse than the best `K_spatial` point

Mechanism conclusion:

- the FFT-orthogonal split is not enough, by itself, to break the trivial-solution ceiling
- this packet supports the same high-level conclusion as `K_spatial` and `K_manifold`:
  - changing kinetic geometry alone is insufficient
  - the stronger signal still comes from target redesign and proximal structure

Status:

- `K_spectral b12` is a closed negative-to-neutral control packet
- keep it as the completed single-family control for the spectral split idea
