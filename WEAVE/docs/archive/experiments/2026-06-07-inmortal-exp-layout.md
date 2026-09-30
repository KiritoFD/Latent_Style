# inmortal-exp Layout

Date: 2026-06-07

Purpose:

- give the `inmortal` round one authoritative experiment root
- make missing fast-eval checkpoints easy to audit
- keep old run directories usable without moving them

Remote authoritative root:

- `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/inmortal-exp`

Rules from this point forward:

- new `inmortal` training runs should save to:
  - `./exp/inmortal-exp/<run_name>`
- the bundle root may contain symlinks to older legacy run directories under:
  - `./exp/aaai2027_inmortal_*`
- stage summaries and missing-eval audits should scan `exp/inmortal-exp` first

Operational helpers:

- prepare bundle root:
  - `tools/experiments/prepare_inmortal_exp_root.py`
- build stage summary and missing-eval list:
  - `tools/experiments/build_inmortal_stage_summary.py`
