# SA-SWD Axis Ablation Bundle

Date: 2026-06-03

This directory tracks the fixed-base `semantic` vs `random` projection-axis
ablation for the terminal SWD term on Distinct5-512.

## Purpose

This packet is the next paper-blocking experiment after the repaired
endpoint-metric trio. Its job is narrow:

- test whether semantic projection-axis selection contributes beyond ordinary
  random-axis terminal SWD;
- keep the rest of the `H` mainline fixed;
- determine whether the paper may continue to center SA-SWD as a novelty claim.

## Base family

- model family:
  - `distinct5_512_ema_variant_h_hard_explore_queue_e3`
- comparison scope:
  - Distinct5-512 strict `5x5 / 750` full eval
- hardware:
  - remote `RTX 3060`
- formal batch:
  - `44`
- seed:
  - `42`

## Config packet

Base remote config:

- `configs/aaai2027/saswd_axis_h_base_seed42_b44.json`

Generated matched configs:

- `configs/aaai2027/saswd_axis_h_base_seed42_b44_saswd_semantic.json`
- `configs/aaai2027/saswd_axis_h_base_seed42_b44_saswd_random.json`

Launch manifest:

- `docs/experiments/2026-06-03-saswd-axis-ablation/launch_manifest_20260603.md`

## Variable under test

Only the terminal projection-axis source should differ:

- `terminal_swd_axis_source = semantic`
- `terminal_swd_axis_source = random`

Everything else should remain matched:

- same queue family
- same seed
- same batch size
- same terminal SWD weight
- same kinetic term
- same eval contract

## Launch status

Current status:

- semantic arm finished training through `epoch_0003.pt` on remote `3060`
- semantic arm auto full-eval is now running from `full_eval/epoch_0001`
- random arm remains queued behind semantic on remote `3060`
- live semantic log:
  - `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp\saswd_axis_h_base_seed42_b44_saswd_semantic\remote_train.log`
- current remote path:
  - `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp\saswd_axis_h_base_seed42_b44_saswd_semantic`
- latest GPU report during auto eval:
  - `100% util, 2322/12288 MiB, 122.92 W`
- latest execution snapshot:
  - training reached `Epoch 3/3, 113/113`; remote task remained healthy and
    advanced into deferred full-eval with no crash markers

## Acceptance gate

Paper-safe positive closure requires at least one of:

1. semantic axes improve the style/content frontier over random axes; or
2. semantic axes hold comparable CLIP-style and LPIPS while improving broader
   artifact diagnostics.

If random axes match or beat semantic axes, the paper must stop centering
semantic projection-axis selection as a proven novelty and instead present it as
one tested design choice.
