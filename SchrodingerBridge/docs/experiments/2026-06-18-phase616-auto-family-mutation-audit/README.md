# 2026-06-18 phase616_auto Family Mutation Audit

## Finding

`tools/experiments/phase616_auto.py` had a silent base-mutation bug:

- every generated run config passed through `_shared_model_data_defaults(...)`
- that helper unconditionally rewrote:
  - `model.tokenizer_family = "legacy_factorized"`
  - `model.style_tokenizer = "factorized"`

This means a phase-618 run could start from a repaired lowrank base on disk, but the emitted per-run `config.json` would still be downgraded back to the old legacy family.

## Why this matters

Phase-618 conclusions depend on a repaired no-reference carrier base:

- `tokenizer_family = pure_latent_spatial`
- `matched_target_conditioning_mode = both`
- `matched_target_style_encoder_mode = residual`
- `style_code_spatial_mode = lowrank`

If the auto launcher rewrites the family back to `legacy_factorized`, then the experiment is no longer testing the intended model family.

That invalidates "close result" interpretation, because the tested model may never have contained the repaired carrier in the first place.

## Direct reproduction

We reproduced the issue by calling `_prepare_run_config(...)` on:

- `docs/experiments/2026-06-18-stage1-lowrank-rerun-audit/baseline_h1_lowrank_config.json`

Observed before the fix:

- base:
  - `tokenizer_family = pure_latent_spatial`
  - `matched_target_conditioning_mode = both`
  - `style_code_spatial_mode = lowrank`
- generated run config:
  - `tokenizer_family = legacy_factorized`
  - `matched_target_conditioning_mode = both`
  - `style_code_spatial_mode = lowrank`

So the repaired family was being partially preserved at the field level, while the tokenizer family itself was silently reverted. That is exactly the kind of "the model did not actually change the way we thought it did" failure mode we were trying to audit.

## Fix applied

`_shared_model_data_defaults(...)` now uses non-destructive defaults instead of overwriting explicit base settings:

- `model.setdefault("tokenizer_family", "legacy_factorized")`
- `model.setdefault("style_tokenizer", "factorized")`
- `model.setdefault("semantic_self_topology_gate", True)`
- `model.setdefault("semantic_self_topology_blend", 1.0)`

This keeps old phase-616 behavior for configs that omit those fields, but preserves repaired phase-618 bases.

## Added guardrail

`_validate_phase618_repaired_lowrank_base(...)` now also requires:

- `model.tokenizer_family == "pure_latent_spatial"`

So style-sweep can no longer accept an old legacy family and call it a repaired lowrank base.

## Post-fix verification

After the patch, the same `_prepare_run_config(...)` reproduction now preserves:

- `tokenizer_family = pure_latent_spatial`
- `matched_target_conditioning_mode = both`
- `style_code_spatial_mode = lowrank`
- `style_code_spatial_scale = 0.35`

The old remote base from:

- `docs/experiments/2026-06-18-remote-h1-e18-diagnosis/remote_config.json`

is now correctly rejected by `_validate_phase618_repaired_lowrank_base(...)`.

## Practical consequence

Any phase-618 auto runs launched through the old `phase616_auto.py` helper should be treated as suspect if they were supposed to use the repaired lowrank family.

In particular:

- phase-618 old-OT reruns launched via `run_phase618_ot_rerun.sh`
- phase-618 style-sweep launches via `run_phase618_style_sweep.sh`

must be rerun from scratch after this patch if we want the results to mean what their run names claim.

## Related generator fix

The same family-downgrade class of mistake also remained in:

- `tools/experiments/gen_lite_batch.py`

That legacy helper still hard-coded:

- `model.tokenizer_family = "legacy_factorized"`
- `model.style_tokenizer = "factorized"`

even when the caller intended to regenerate a repaired phase-618 OT batch.

It is now patched to:

1. preserve the explicit base family
2. validate the repaired lowrank carrier by default
3. refuse to silently regenerate old legacy-family phase-618 runs unless legacy mode is explicitly opted in

So any old manual batch regenerated through the pre-fix `gen_lite_batch.py` should also be treated as stale for phase-618 interpretation.
