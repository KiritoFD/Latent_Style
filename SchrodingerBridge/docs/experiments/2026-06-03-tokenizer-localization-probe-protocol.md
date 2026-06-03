# Tokenizer Localization Probe Protocol

Date: 2026-06-03

Purpose:

- define the next mechanism-side experiment after the landed `L`-family
  execution-alignment successor packet;
- localize whether the remaining tokenizer ceiling is primarily
  representation-side or executor-side;
- keep the next formal run aligned with the current review boundary rather than
  opening another loosely scoped tokenizer sweep.

## 1. Question

The landed `L`-family successor packet showed that tokenizer code geometry maps
only moderately into executed geometry, while executed movement tracks
`delta_idt` more closely than raw code geometry.

That still leaves the key identification question open:

> when only one side is allowed to adapt, does most of the recoverable gain
> come from a fresher tokenizer or from a fresher LANCET executor?

This is the cleanest next probe because it attacks the current weak-reject
driver directly:

- tokenizer-side weakness is not ruled out;
- executor-side attenuation is not proven dominant;
- current evidence is still only partial mechanism closure.

## 2. Shared base and required arms

Run the two freeze-direction arms as a matched pair around the same reviewed
Distinct5 checkpoint:

- shared family:
  - `distinct5_512_ema_variant_l_content_adaptive_annealed_queue_e3`
- shared checkpoint:
  - `epoch_0001`
- paper-facing reason:
  - this is the same `L` family already used by the landed execution-alignment
    successor packet, so the localization probe stays on the current
    manuscript-facing mechanism surface rather than dropping back to an older
    legacy256 tokenizer route

Shared config base:

- `SchrodingerBridge/configs/aaai2027/tokenizer_localization_l_e1_seed42_b44_base.json`

### Arm A: fresh style branch, frozen executor

Meaning:

- keep the current LANCET consumer fixed;
- reinitialize the style-side control branch;
- train only the style-side control branch to see how much gain is recoverable
  inside the current execution landscape.

Current config path:

- `SchrodingerBridge/configs/aaai2027/tokenizer_localization_l_e1_stylebranch_seed42_b44.json`

Key freeze mode:

- `training.freeze_mode = style_branch`

Resume source:

- `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/distinct5_512_ema_variant_l_content_adaptive_annealed_queue_e3_b44_remote/epoch_0001.pt`

Load/reset rule:

- load the reviewed `L e1` executor;
- ignore and reinitialize:
  - `style_tokenizer.*`
  - `style_spatial_id_16`

### Arm B: frozen style branch, fresh executor

Meaning:

- lock the style-side control branch from the reviewed `L e1` checkpoint;
- leave the executor random;
- train only the executor side to see how much gain is recoverable from better
  execution alone.

Current config path:

- `SchrodingerBridge/configs/aaai2027/tokenizer_localization_l_e1_executoronly_seed42_b44.json`

Key freeze mode:

- `training.freeze_mode = executor_only`

Resume source:

- `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/distinct5_512_ema_variant_l_content_adaptive_annealed_queue_e3_b44_remote/epoch_0001.pt`

Load/freeze rule:

- load only:
  - `style_tokenizer.*`
  - `style_spatial_id_16`
- leave the rest of the executor randomly initialized and trainable

## 3. Scope and runtime policy

Dataset:

- Distinct5-512 only

Environment:

- formal runs on the remote RTX 3060 only

Logging:

- every arm must produce:
  - resolved config snapshot
  - per-epoch training log
  - per-epoch full-eval summaries
  - final note-ready metrics for both `full` and `transfer-only`

Output rule:

- do not bury the runs inside ad hoc local-only surfaces;
- pair the final run roots with a dated note under `docs/experiments/`;
- append landed status to `aaai2027_master_experiment_log.csv`.

## 4. Readout contract

The probe is not judged by raw CLIP-style alone.

Minimum readout:

1. quality metrics:
   - `clip_style`
   - `content_lpips`
   - `clip_dir`
2. no-op-aware readout:
   - `delta_idt` or equivalent no-op-adjusted style gain
3. representation/execution diagnostics:
   - tokenizer code separation
   - executed delta separation
   - code-to-output alignment

Why:

- the paper's current tokenizer claim is about executed representation, not
  code geometry alone.

## 5. Interpretation rules

### If Arm A wins clearly

Safe reading:

- the current control object is still a stronger bottleneck candidate than the
  executor.

Unsafe leap:

- `the next correct tokenizer factorization is already proven`

### If Arm B wins clearly

Safe reading:

- current tokenizer codes were more usable than the current executor allowed.

Unsafe leap:

- `tokenizer-side weakness is closed`

### If both improve

Safe reading:

- the bottleneck is joint and cannot yet be localized to one side only.

### If neither improves materially

Safe reading:

- the current direct-atom-residual family may be near its useful ceiling under
  this training protocol, and the next change should target the representation
  form itself rather than the freeze direction.

## 6. Promotion gate

This probe becomes paper-facing only if at least one of the following is true:

1. one arm materially improves `delta_idt` with non-catastrophic LPIPS change;
2. one arm materially strengthens the code-to-output localization story beyond
   the current landed `L`-family packet;
3. the paired result cleanly rules out one previously live tokenizer claim.

Otherwise:

- keep it as internal mechanism evidence and do not promote it into manuscript
  headline text.

## 7. Immediate next step

Before launch, verify on the remote `3060` that all of the following are true:

1. the reviewed `L`-family `epoch_0001` checkpoint is present at the exact
   remote path recorded above;
2. the new `aaai2027/tokenizer_localization_l_e1_*` configs resolve cleanly on
   the remote clean worktree;
3. the new `executor_only` freeze mode is present in the remote code copy.

If any of these fail, write a path-truth note first and do not launch a
substitute silently.
