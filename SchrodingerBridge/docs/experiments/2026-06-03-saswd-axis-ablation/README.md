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
- random arm is also completed on remote `3060`
- live semantic log:
  - `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp\saswd_axis_h_base_seed42_b44_saswd_semantic\remote_train.log`
- current semantic run path:
  - `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp\saswd_axis_h_base_seed42_b44_saswd_semantic`
- semantic task state after completion:
  - `SB_SASWD_H_SEM_S42`: ready, last result `0`
- random task state:
  - `SB_SASWD_H_RAND_S42`: completed, last result `0`
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
- hard-stall runtime heartbeat:
  - latest seen progress: `Epoch 2/3`, `96/113`
  - GPU: `100% util, 11226/12288 MiB, 72.21 W`
  - step times remain near `~8s/it` with no recovery
  - current interpretation: the run has crossed from degraded throughput into a
    stop-worthy blocker as a formal execution, but it may still be allowed to
    continue only for quality-only evidence while it keeps making real progress
- completion heartbeat:
  - training finished at `2026-06-03 05:33:28`
  - full eval landed for `epoch_0001`, `epoch_0002`, and `epoch_0003`
  - retained evidence class:
    - `quality_only`

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

Semantic-arm reading in isolation:

- best full-view style is `e2`, but it pays a clear LPIPS penalty;
- best full-view LPIPS is `e1`;
- the semantic arm alone does **not** yet justify a positive SA-SWD novelty
  claim because the matched random-axis control has not been run.

## Completed random-arm results

Remote summary roots:

- `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp\saswd_axis_h_base_seed42_b44_saswd_random\full_eval\epoch_0001\summary.json`
- `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp\saswd_axis_h_base_seed42_b44_saswd_random\full_eval\epoch_0002\summary.json`
- `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp\saswd_axis_h_base_seed42_b44_saswd_random\full_eval\epoch_0003\summary.json`

Key random-arm metrics:

| epoch | full clip_style | full lpips | transfer clip_style | transfer lpips |
|---|---:|---:|---:|---:|
| `e1` | `0.6860751852` | `0.2691072983` | `0.6506762915` | `0.2692665908` |
| `e2` | `0.6863345393` | `0.2706820921` | `0.6507716310` | `0.2710198916` |
| `e3` | `0.6863872107` | `0.2706962664` | `0.6508313735` | `0.2713293984` |

Random-arm reading:

- all three epochs cluster tightly;
- best full-view LPIPS is `e1`;
- the arm is admissible for `quality_only` evidence, not for formal runtime
  evidence.

## Pair interpretation after both arms landed

Gate B is no longer missing a control arm. The packet now closes as a
completed-but-runtime-anomalous pair:

1. `semantic` is a normal completed arm with usable full-eval summaries;
2. `random` is a completed arm whose quality summaries are usable, but whose
   runtime is not admissible as representative formal-speed evidence.

Quality-side comparison:

- semantic retains a raw style advantage of roughly `+0.010` to `+0.012`
  CLIP-style over random across matched epochs;
- random retains a large LPIPS advantage of roughly `-0.060` to `-0.090`
  against semantic across matched epochs.

Immediate implication:

- this packet does **not** support a clean positive novelty claim that semantic
  projection-axis selection dominates the matched random-axis control on the
  reviewed style/content trade-off;
- at minimum, the paper must demote semantic projection-axis selection from a
  proven win to a tested design choice unless reviewer re-audit finds a safer
  narrower interpretation.

## Best-effort runtime diagnosis

Current evidence ranks the likely causes as:

1. `C` - the random-axis path itself is materially heavier or less efficient;
2. `A` - generic host/runtime interference is a distant second;
3. `B` - an accidental resolved-config mismatch is unlikely.

Remote evidence supporting that ranking:

- the semantic and random configs are effectively identical except for
  `bridge.terminal_swd_axis_source`;
- the semantic arm completed normally on the same remote host and queue family;
- the random arm completed on the same machine and branch but sustained
  pathological throughput, with VRAM pinned near the limit and very low
  effective progress for most of training.

Current handling decision:

- keep the current completed random arm as `quality_only` evidence;
- do **not** use its wall clock in any fair timing or efficiency claim;
- do **not** relaunch immediately just to seek a prettier runtime trace;
- if a future paper argument still requires a fair runtime comparison for the
  random axis, relaunch only under an explicitly healthier runtime condition and
  log it as a new packet.

Reviewer-side policy link:

- `docs/reviews/aaai2027_gate_b_runtime_anomaly_policy_20260603.md`

## Acceptance gate

Paper-safe positive closure requires at least one of:

1. semantic axes improve the style/content frontier over random axes; or
2. semantic axes hold comparable CLIP-style and LPIPS while improving broader
   artifact diagnostics.

Based on the landed pair, that positive closure is not yet supported. The next
safe step is reviewer re-audit, not manuscript escalation.
