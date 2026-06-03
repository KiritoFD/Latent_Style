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

## 2. Required arms

Run the two freeze-direction arms as a matched pair.

### Arm A: fresh tokenizer, frozen executor

Meaning:

- keep the current LANCET consumer fixed;
- train only the tokenizer/style branch;
- measure how much better style control can become inside the existing
  execution landscape.

Current config path:

- `SchrodingerBridge/configs/tokenizer_t01_direct_atom_residual_tokonly_from_backbone_e16.json`

Key freeze mode:

- `training.freeze_mode = tokenizer_only`

Resume source:

- `exp/tokenizer_t01_factorized_backbone_e16/epoch_0016.pt`

### Arm B: frozen tokenizer, fresh executor

Meaning:

- lock a trained tokenizer;
- reinitialize and train the LANCET consumer/backbone;
- measure how much gain is recoverable from better execution alone.

Current config path:

- `SchrodingerBridge/configs/tokenizer_t01_direct_atom_residual_frozen_tok_fresh_lancet_e16.json`

Key freeze mode:

- `training.freeze_mode = backbone_only`

Resume chain:

1. tokenizer warmup checkpoint from:
   - `exp/tokenizer_t01_direct_atom_residual_warmup_e2_from_backbone_e16/epoch_0002.pt`
2. then train fresh LANCET consumer with tokenizer frozen

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

Before launch, verify on the remote `3060` that the required resume chain is
actually available:

1. `tokenizer_t01_factorized_backbone_e16/epoch_0016.pt`
2. `tokenizer_t01_direct_atom_residual_warmup_e2_from_backbone_e16/epoch_0002.pt`

If either payload is missing, write a path-truth note first and do not launch a
substitute silently.
