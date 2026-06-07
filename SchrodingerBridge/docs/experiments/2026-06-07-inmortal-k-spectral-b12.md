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
