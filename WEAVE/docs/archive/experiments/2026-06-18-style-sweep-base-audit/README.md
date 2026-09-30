# Style-Sweep Base Audit

This audit answers a very specific phase-618 wiring question:

> When `stage3_style` results are close or misleading, is the sweep itself weak, or did
> the runner accidentally execute on the pre-repair base instead of the repaired
> low-rank no-reference carrier base?

The answer is now clear: the base matters enough to change the meaning of the sweep.

## 1. Repro

Style-sweep variant spec:

- `style_sweep_variant_spec.json`

Old style-sweep base probe:

```powershell
py -3.12 tools/probe_config_effectiveness.py `
  --config docs/experiments/2026-06-18-remote-h1-e18-diagnosis/remote_config.json `
  --variant-spec docs/experiments/2026-06-18-style-sweep-base-audit/style_sweep_variant_spec.json `
  --output-dir docs/experiments/2026-06-18-style-sweep-base-audit/old_base_probe `
  --device cpu
```

Repaired low-rank base probe:

```powershell
py -3.12 tools/probe_config_effectiveness.py `
  --config docs/experiments/2026-06-18-stage1-lowrank-rerun-audit/baseline_h1_lowrank_config.json `
  --variant-spec docs/experiments/2026-06-18-style-sweep-base-audit/style_sweep_variant_spec.json `
  --output-dir docs/experiments/2026-06-18-style-sweep-base-audit/lowrank_base_probe `
  --device cpu
```

## 2. Old-base result

On the old `remote_config.json` base:

- `r1-r6` classify as `train_graph_only`
  - `plain_eval_delta -> 0.0`
  - `configured_delta -> 0.016-0.026`
  - `anatomy_code_body_dead_spatial_body_live -> true`
- `r7/r8/r9/r10` classify as `plain_eval_change`
  - because they inject the low-rank repair into a base that did not have it

Meaning:

1. the old base still has the dead no-reference code carrier
2. low-rank variants look strong there partly because they are repairing the base
3. any style-sweep run launched on that base is confounded:
   - it mixes "repair the carrier" with
   - "test the bold direction"

## 3. Repaired-base result

On the repaired low-rank base:

- `r1-r6` classify as `plain_eval_change`
  - `plain_eval_delta -> 0.00044-0.00083`
  - `configured_delta -> 0.0088-0.0141`
  - `anatomy_code_body_dead_spatial_body_live -> false`
- `r7`, `r8`, and `r10` classify as `no_effect`
  - because the repaired base already contains the low-rank carrier they were adding
- `r9` still classifies as `plain_eval_change`
  - because it combines the repaired carrier with an actual blend change

Meaning:

1. once the low-rank carrier is already in the base, the real stage3-style lever is the
   blend change itself
2. `r7/r8/r10` are not meaningful "new experiments" on the repaired base
3. they should not be allowed to dominate stage3-style selection as if they were bold
   theory wins

## 4. Implementation conclusion

This was not just a theory problem.

The original `run_phase618_style_sweep.sh` path could launch `style-sweep` without
forcing the repaired low-rank base. That means a nominally phase-618 sweep could still
run on the old phase-2-era base contract.

That is an implementation-level experiment wiring bug because it changes the scientific
meaning of the sweep:

- on the old base:
  - low-rank variants look like strong candidates
- on the repaired base:
  - those same variants collapse to `no_effect`

## 5. Fix applied

The runner is now repaired in two layers:

1. `tools/experiments/run_phase618_style_sweep.sh`
   - force-cleans the stage root
   - generates a repaired low-rank base config
   - passes it through `--base-cfg`
2. `tools/experiments/phase616_auto.py`
   - `style-sweep` now rejects bases that do not already have:
     - `matched_target_conditioning_mode=both`
     - `matched_target_style_encoder_mode=residual`
     - `style_code_spatial_mode=lowrank`

## 6. Bottom line

If future `stage3_style` runs are close:

1. first check whether the run started from the repaired low-rank base
2. if yes, then closeness is much more likely to reflect real weakness of the blend
   hypothesis or objective mismatch
3. if no, the sweep is scientifically confounded and should not be trusted
