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

- semantic arm is fully completed on remote `3060`
- random arm is now actively running on remote `3060`
- live semantic log:
  - `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp\saswd_axis_h_base_seed42_b44_saswd_semantic\remote_train.log`
- current semantic run path:
  - `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp\saswd_axis_h_base_seed42_b44_saswd_semantic`
- semantic task state after completion:
  - `SB_SASWD_H_SEM_S42`: ready, last result `0`
- random task state:
  - `SB_SASWD_H_RAND_S42`: running
- live random log:
  - `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp\saswd_axis_h_base_seed42_b44_saswd_random\remote_train.log`
- first random-arm heartbeat:
  - entered `Epoch 1/3`
  - first recorded step window: `0/113`
  - GPU: `100% util, 8930/12288 MiB, 56.71 W`
- later runtime-risk heartbeat:
  - latest seen progress: `Epoch 1/3`, `36/113`
  - GPU: `100% util, 12080/12288 MiB, 66.30 W`
  - observed step times stretched to roughly `9-14s/it`
  - provisional interpretation: memory-pressure / degraded-throughput risk on
    this exact packet, not yet a hard crash
- blocker-level runtime heartbeat:
  - latest seen progress: `Epoch 1/3` completed, `Epoch 2/3` entered at `2/113`
  - epoch-1 compute time alone reached `971.7s`
  - GPU: `100% util, 11217/12288 MiB, 71.42 W`
  - current interpretation: this exact random-arm run is no longer a credible
    normal-speed formal remote-3060 run; if it completes, its quality summaries
    may still be diagnostically useful, but its wall-clock behavior should not
    be treated as normal formal evidence

## Completed semantic-arm results

Remote summary roots:

- `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp\saswd_axis_h_base_seed42_b44_saswd_semantic\full_eval\epoch_0001\summary.json`
- `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp\saswd_axis_h_base_seed42_b44_saswd_semantic\full_eval\epoch_0002\summary.json`
- `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp\saswd_axis_h_base_seed42_b44_saswd_semantic\full_eval\epoch_0003\summary.json`

Key semantic-arm metrics:

| epoch | full clip_style | full lpips | transfer clip_style | transfer lpips |
|---|---:|---:|---:|---:|
| `e1` | `0.6963431751` | `0.3313274828` | `0.6650688401` | `0.3391037583` |
| `e2` | `0.6987337865` | `0.3608344315` | `0.6684234888` | `0.3702646600` |
| `e3` | `0.6961391042` | `0.3415375931` | `0.6645665071` | `0.3506202180` |

Provisional reading before the random arm lands:

- best full-view style is `e2`, but it pays a clear LPIPS penalty;
- best full-view LPIPS is `e1`;
- the semantic arm alone does **not** yet justify a positive SA-SWD novelty
  claim because the matched random-axis control has not been run.

## Current blocker interpretation

The packet is currently split into two evidence classes:

1. `semantic` is a valid completed arm with usable full-eval summaries;
2. `random` is still an open matched control, but the current remote runtime
   state has crossed into a throughput-blocker regime.

Until the random arm either:

- finishes with summaries,
- crashes,
- or is relaunched in a healthier runtime state,

Gate B remains open. If the current run completes, its quality-only comparison
may still be usable, but its runtime behavior must be logged as abnormal rather
than treated as representative formal-speed evidence.

## Best-effort blocker diagnosis

Current evidence ranks the likely causes as:

1. `C` - the random-axis path itself is materially heavier or less efficient;
2. `A` - generic host/runtime interference is a distant second;
3. `B` - an accidental resolved-config mismatch is unlikely.

Remote evidence supporting that ranking:

- the semantic and random configs are effectively identical except for
  `bridge.terminal_swd_axis_source`;
- the semantic arm completed normally on the same remote host and queue family;
- the random arm shows pathological throughput on the same machine and branch,
  with VRAM pinned near the limit and very low effective progress.

Current handling decision:

- do **not** relaunch the same packet yet;
- allow the current random run to finish if it keeps making progress;
- treat any resulting summaries as quality-only evidence;
- do **not** treat the current random-arm wall clock as representative formal
  speed evidence.

Reviewer-side policy link:

- `docs/reviews/aaai2027_gate_b_runtime_anomaly_policy_20260603.md`

## Acceptance gate

Paper-safe positive closure requires at least one of:

1. semantic axes improve the style/content frontier over random axes; or
2. semantic axes hold comparable CLIP-style and LPIPS while improving broader
   artifact diagnostics.

If random axes match or beat semantic axes, the paper must stop centering
semantic projection-axis selection as a proven novelty and instead present it as
one tested design choice.
