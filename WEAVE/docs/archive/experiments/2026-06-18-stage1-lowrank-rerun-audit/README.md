# Stage1 Low-Rank Rerun Audit

This audit answers the next phase-618 question after the original `h1` diagnosis:

> Once the low-rank no-reference style carrier is enabled, are the old `h0`-`h6`
> OT variants still collapsing because the evaluated graph never changes, or are
> they now real training differences that survive the previous implementation bug?

It should be read together with:

- `docs/experiments/2026-06-18-stage1-config-effect-probe`
- `docs/experiments/2026-06-18-stage1-training-effect-probe`
- `docs/experiments/2026-06-18-remote-h1-e18-diagnosis`
- `docs/experiments/2026-06-18-stage1-lowrank-rerun-audit/conditioning_sensitivity_probe`
- `docs/experiments/2026-06-18-topogate-multiblock-audit`

## 1. Baseline

Authoritative rerun base:

- remote file pulled from the live rerun root:
  - `remote_base_phase618_ot_rerun_lowrank.json`

Probe baseline:

- `baseline_h1_lowrank_config.json`

How it was built:

1. start from the live rerun base with:
   - `matched_target_conditioning_mode="both"`
   - `matched_target_style_encoder_mode="residual"`
   - `style_code_spatial_mode="lowrank"`
   - `style_code_spatial_scale=0.35`
2. apply the `h1_linear_fm` bridge settings:
   - `bridge_path_mode="linear"`
   - `coupling_cost_composition="structure_only"`
   - `coupling_structure_cost_mode="self_affinity_gw"`
   - `bridge_sigma=0.0`

This gives the correct comparison point for the old stage1 family under the repaired
no-reference style carrier.

## 2. Repro

Config-effect differential probe:

```powershell
py -3.12 tools/probe_config_effectiveness.py `
  --config docs/experiments/2026-06-18-stage1-lowrank-rerun-audit/baseline_h1_lowrank_config.json `
  --variant-spec docs/experiments/2026-06-18-stage1-training-effect-probe/stage1_variant_spec.json `
  --output-dir docs/experiments/2026-06-18-stage1-lowrank-rerun-audit/config_effect_probe `
  --device cpu
```

Training-path differential probe:

```powershell
py -3.12 tools/probe_training_variant_effect.py `
  --config docs/experiments/2026-06-18-stage1-lowrank-rerun-audit/baseline_h1_lowrank_config.json `
  --variant-spec docs/experiments/2026-06-18-stage1-training-effect-probe/stage1_variant_spec.json `
  --output-dir docs/experiments/2026-06-18-stage1-lowrank-rerun-audit/training_effect_probe `
  --device cpu
```

## 3. Main finding

The low-rank repair changes the baseline anatomy exactly the way we wanted:

- `anatomy_code_body_dead_spatial_body_live -> false`
- baseline code-only path:
  - `first_hires_block_gate1_a_vs_b_mean_abs -> 0.0018754458287730813`
  - `h_body_a_vs_b_mean_abs -> 0.062431029975414276`
  - `delta_a_vs_b_mean_abs -> 0.012378660961985588`

Meaning:

1. the no-reference path is no longer body-dead
2. the evaluated graph now has genuine body-level style actuation
3. the earlier "all stage1 results are tied because plain eval never changed" story is no
   longer sufficient once this repaired base is used

## 4. What did not change

Even on this repaired base, the old OT variants still do **not** change the executed
no-reference forward graph relative to `h1_lowrank`:

```text
h0_vertical_fm        -> plain vs_base_forward_mean_abs = 0.0
h2_euclidean_ot       -> plain vs_base_forward_mean_abs = 0.0
h3_sde_noise          -> plain vs_base_forward_mean_abs = 0.0
h4_unbalanced_ot      -> plain vs_base_forward_mean_abs = 0.0
h5_topogate_attention -> plain vs_base_forward_mean_abs = 0.0
h6_combined_topogate  -> plain vs_base_forward_mean_abs = 0.0
```

This is expected:

- these stage1 variants modify OT construction, bridge sampling, or training targets
- they do not directly rewire the runtime forward graph
- the low-rank carrier fixes style actuation for the family as a whole, not pairwise
  forward differences between `h0` and `h6`

So after the repair, "config-effect on plain forward" and "training-effect on OT target"
must be treated as two different questions.

## 5. What does change

The training-effect probe shows the old OT family is still training-real:

- `h0_vertical_fm`
  - classification: `bridge_only_change`
  - `x_t_vs_base_mean_abs -> 0.08638186007738113`
  - `target_velocity_vs_base_mean_abs -> 0.30444979667663574`

- `h2_euclidean_ot`
  - classification: `ot_or_target_change`
  - `matched_target_vs_base_mean_abs -> 0.45671725273132324`
  - `objective_target_vs_base_mean_abs -> 0.45671725273132324`

- `h3_sde_noise`
  - classification: `bridge_only_change`
  - `x_t_vs_base_mean_abs -> 0.08667436987161636`
  - `target_velocity_vs_base_mean_abs -> 0.30537015199661255`

- `h4_unbalanced_ot`
  - classification: `ot_or_target_change`
  - `matched_target_vs_base_mean_abs -> 0.06527001410722733`
  - `ot_target_gini -> 0.039943695068359375`

- `h5_topogate_attention`
  - classification: `ot_or_target_change`
  - `matched_target_vs_base_mean_abs -> 0.3728516399860382`
  - `ot_structure_cost_mean -> 1.0`
  - `ot_topogate_probe_active -> 1.0`
  - `ot_topogate_descriptor_blocks -> 4.0`

- `h6_combined_topogate`
  - classification: `ot_or_target_change`
  - `matched_target_vs_base_mean_abs -> 0.3767605721950531`
  - `bridge_sigma -> 0.02`
  - `ot_topogate_probe_active -> 1.0`
  - `ot_topogate_descriptor_blocks -> 4.0`

Meaning:

1. the old OT variants are not silent no-ops after the low-rank repair
2. `h5/h6` really do switch into the new TopoGate complexity descriptor path
3. if the rerun curves still stay tightly clustered, that is much stronger evidence of
   theory weakness or objective weakness, not the old no-reference-style-path bug

### Important implementation correction for `h5/h6`

The current `h5/h6` numbers above are **post-fix** numbers after
`topogate_attention_gw` was repaired to aggregate all semantic body blocks.

Before the 2026-06-18 multiblock fix, the same mode effectively looked only at the
last body block attention cache. See:

- `docs/experiments/2026-06-18-topogate-multiblock-audit/README.md`

So any older h5/h6 result or probe artifact captured before that fix should be
treated as stale if it is being used to support conclusions about the intended
full-body TopoGate descriptor.

### Important metric-contract note for `h5/h6`

`ot_structure_cost_mean = 1.0` under `topogate_attention_gw` does **not** mean the
structure cost collapsed.

It means the returned TopoGate blended structure term is already mean-normalized
inside `_structure_pairwise_cost(...)`.

On the repaired low-rank rerun base:

- `h5_topogate_attention`
  - `ot_structure_transport_cost_var -> 0.059765417128801346`
  - `ot_topogate_complexity_term_var -> 0.22340138256549835`
  - `ot_latent_affinity_term_var -> 0.05051263049244881`
  - `ot_total_cost_matrix_var -> 0.01826178841292858`
  - `ot_topogate_structure_blend_weight -> 0.5`
  - `ot_topogate_descriptor_blocks -> 4.0`

Meaning:

1. the TopoGate complexity branch is not a constant matrix
2. its normalized contrast is stronger than the latent-affinity branch
3. the current TopoGate descriptor covers all 4 semantic body blocks, not only the
   last one
3. earlier reads that treated `ot_structure_cost_mean = 1.0` as a failure signal were
   using a misleading metric contract

## 6. Current remote rerun sanity check

After the phase-618 relaunch was cleaned up and re-pinned to `batch_size=16`
(because `B20` crossed the `11.3 GB` safety line on the repaired base), the live
remote `h0_vertical_fm` rerun again shows the repaired style path is actually active
during real training:

- `matched_target_style_code_abs -> 0.04795800894498825`
- `style_spatial_code_map_abs -> 0.07173637300729752`
- `style_spatial_map_abs -> 1.199442744255066`
- `matched_target_style_code_active -> 1.0`
- `style_spatial_source_target_latent -> 1.0`
- `style_spatial_code_map_residual -> 1.0`
- `gpu_vram_used_gb_peak -> 10.53515625`
- `gpu_power_w_peak -> 150.73`

The first post-relaunch full eval (`epoch_0001`) also landed successfully:

- `transfer_clip_style -> 0.6708698003987472`
- `transfer_content_lpips -> 0.3777350581333333`
- `converged -> false`
- `eval_wall_total_sec -> 208.55800735400408`

And the next corrected live points already show the more important trajectory:

- `epoch_0002 -> 0.6620107294122377 / 0.3341379833333334`
- `epoch_0003 ->` training continued; best epoch still remained `epoch_0001`
- `epoch_0004 -> 0.6605690497159957 / 0.3502975537666667`
- `epoch_0005 -> 0.6617938352624575 / 0.3437728258266667`
- `epoch_0006 -> 0.6610736021399499 / 0.36138213324999996`
- `since_best -> 3`, `since_last_pareto -> 2`, `converged -> false`

So the repaired rerun is not sitting on an exact implementation no-op. It is producing
an initially style-strong point and then drifting back toward the old close-result band.
That is much closer to an objective / training-dynamics weakness than a dead family bug.

The next corrected family member (`h1_linear_fm`) is already separating in the
same rerun root:

- `epoch_0001 -> 0.6526460257669291 / 0.33696081576666664`
- `epoch_0002 -> 0.6493085829416911 / 0.40545237836666664`
- `epoch_0003 -> 0.6470608141521612 / 0.4265378871166666`

This matters because it is no longer a “same numbers with a different run name”
pattern. On the corrected base, at least `h0` vs `h1` are already measurably
separated in the real remote rerun.

There was also a convergence-contract mismatch in the surrounding automation:

- `src/run.py` only honored `round2_convergence.json["converged"]`
- `phase616_auto.py` outer monitoring separately stopped a run once the
  objective-gap best had survived `patience` epochs
- so corrected `h0` could legally stop and advance while `round2_convergence.json`
  still showed `converged=false`

That mismatch is now repaired by emitting a shared stop packet in
`round2_convergence.json`:

- `objective_best_epoch`
- `objective_epochs_since_best`
- `objective_patience_converged`
- `stop_ready`
- `stop_reason`

and both `src/run.py` and `phase616_auto.py` now honor the same signal.

Meaning:

1. the low-rank style carrier is not only a random-init probe artifact
2. it is present in the live rerun training job under the corrected remote launcher
3. per-epoch eval and convergence tracking are live on the rerun root
4. the corrected live trajectory already argues against "the model never changed at all"
5. the current rerun no longer has a stop-criterion split between the trainer and the outer auto-runner
6. the current rerun is now a valid test of whether old OT hypotheses help once the
   no-reference style path is no longer crippled

## 6.5 Conditioning-source observability fix

The repaired low-rank base also exposed a probe-contract gap:

- runtime debug already emitted `style_spatial_source_structured_map`
- `tools/probe_conditioning_sensitivity.py` originally did not export it
- this made the `mode=none` and `mode=code` rows look internally inconsistent

After fixing the probe export and rerunning it, the no-reference source semantics are:

- `mode=none -> structured_map + residual code_map`
- `mode=code -> structured_map + residual code_map`
- `mode=spatial -> target_style_latent + residual code_map`

So the repaired base is not hiding another "legacy zero path" bug. See:

- `docs/experiments/2026-06-18-stage1-lowrank-rerun-audit/conditioning_sensitivity_probe/README.md`

## 7. Bottom line

The state after this audit is:

1. the earlier implementation bug was real and the low-rank repair addresses it
2. the repaired base now has body-level no-reference style actuation
3. the old OT variants still differ mainly through training-time target construction
4. therefore any continued near-tie in the rerun should be interpreted as:
   - old OT theory not moving the objective enough
   - or the current objective not rewarding those changes enough
   - not as proof that the model family is still implementation-dead

That is the threshold we needed before moving harder into the bold directions from
`docs/618/bold_directions.md`.
