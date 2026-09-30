# Remote h1 Diagnosis

This folder captures a remote artifact pull and local code-path diagnosis for:

- remote run: `exp/20250618_lite_ot_vertical_auto/h1_linear_fm`
- pulled checkpoint: `epoch_0018.pt`
- pulled config: `remote_config.json`
- pulled eval curve: `clip_lpips_curve.csv`

Primary purpose: determine whether the architecture/config levers in this run are live at all, independent of final metric quality.

## 1. Remote facts

The remote `clip_lpips_curve.csv` shows:

- best LPIPS in this run appears at `epoch_0018` with:
  - `transfer_clip_style = 0.6679457206030687`
  - `transfer_content_lpips = 0.2765800356716666`

The pulled config has three code-relevant properties:

1. `semantic_self_topology_gate = true`
2. `semantic_self_topology_blend = 1.0`
3. `tokenizer_content_adaptive = false`

That third point matters: the content-router override-erasure bug is not active in this config. So if the style-code path is weak here, it is not because the content router is overwriting it.

## 2. Random-init diagnosis

The authoritative implementation-level diagnosis is the random-init probe:

```powershell
py -3.12 tools/probe_conditioning_sensitivity.py `
  --config docs/experiments/2026-06-18-remote-h1-e18-diagnosis/remote_config.json `
  --output-dir docs/experiments/2026-06-18-remote-h1-e18-diagnosis/probe_random_init
```

Key results from `probe_random_init/summary.json`:

- conditioning:
  - `spatial -> 0.029726596549153328`
  - `code -> 0.0007738805725239217`
  - `both -> 0.02729835733771324`
- topology:
  - `gate0_blend0 vs gate0_blend1 -> 0.0`
  - `gate1_blend0 vs gate1_blend1 -> 0.031597238034009933`

Interpretation:

1. `semantic_self_topology_blend` is an exact no-op when `semantic_self_topology_gate=false`.
2. The topology lever becomes live when the gate is enabled.
3. In this h1 config, the pure global code path is already much weaker than the spatial matched-target path at random init:
   - `0.0007739 / 0.0297266 ~= 2.6%`
4. So even before training, this architecture is heavily biased toward spatial matched-target conditioning and weak global style-code actuation.

## 2.1 Path anatomy: where does code-only style first become live?

Before the live-init repair, a one-off local trace on the same random-init config showed:

- `encoded_code_delta -> 0.008771349675953388`
- `first_hires_block_gate1_delta -> 0.0`
- `h_fused_code_only_delta -> 0.0`
- `h_dec_code_only_delta -> 0.0`
- `h_mod_code_only_delta -> 0.0033200972247868776`
- `delta_code_only_delta -> 0.0005722185014747083`
- `style_map_spatial_delta -> 1.179891586303711`
- `h_body_spatial_delta -> 0.24209201335906982`

Meaning:

- the matched-target style code existed
- but the no-reference executed path did not change until `dec_mod`
- the semantic body still only reacted to the spatial matched-target path

After the repair, run:

```powershell
py -3.12 tools/probe_conditioning_sensitivity.py `
  --config docs/experiments/2026-06-18-remote-h1-e18-diagnosis/remote_config.json `
  --output-dir docs/experiments/2026-06-18-remote-h1-e18-diagnosis/probe_random_init_live_init
```

Key results from `probe_random_init_live_init/path_anatomy.csv`:

- `code_only_no_reference`
  - `first_hires_block_gate1_a_vs_b_mean_abs -> 0.003666731994599104`
  - `h_fused_a_vs_b_mean_abs -> 0.0013707999605685472`
  - `h_dec_pre_mod_a_vs_b_mean_abs -> 0.006331180687993765`
  - `delta_a_vs_b_mean_abs -> 0.0009615437593311071`
- `spatial_matched_target`
  - `style_map_a_vs_b_mean_abs -> 1.175713062286377`
  - `h_body_a_vs_b_mean_abs -> 0.25299525260925293`
  - `delta_a_vs_b_mean_abs -> 0.02723843604326248`

Meaning:

1. the code-only path is no longer hard-dead before `dec_mod`
2. skip fusion and decoder features now respond to style-code differences
3. but `h_body` is still unchanged in the code-only path
4. so the core mismatch remains: training gets a strong matched-target spatial body path, while no-reference inference still has no body-level style actuation

## 2.2 Minimal repair: low-rank code-map override

A new optional path was added:

- `style_code_spatial_mode = lowrank`

This synthesizes a `style_code -> map_16` spatial carrier from the existing global style code:

- if matched-target spatial conditioning is present, the code-map is added as a residual
- if no reference spatial source exists, the code-map becomes the primary body style source

Saved probe artifact:

- `probe_lowrank_code_map_override.json`
- `probe_random_init_lowrank_cli/summary.json`

Key results:

- `conditioning_code_forward_delta -> 0.004973988048732281`
- `conditioning_spatial_forward_delta -> 0.026586372405290604`

And anatomy:

- `code_only_no_reference`
  - `h_body_a_vs_b_mean_abs -> 0.05634588375687599`
  - `delta_a_vs_b_mean_abs -> 0.009910744614899158`
- `spatial_matched_target`
  - `h_body_a_vs_b_mean_abs -> 0.2625107765197754`
  - `delta_a_vs_b_mean_abs -> 0.026605140417814255`

Meaning:

1. the no-reference path is no longer body-dead
2. the code-only path is still weaker than matched-target spatial conditioning
3. but this is the first repair that makes no-reference style reach the semantic body at all

Repro command:

```powershell
py -3.12 tools/probe_conditioning_sensitivity.py `
  --config docs/experiments/2026-06-18-remote-h1-e18-diagnosis/remote_config.json `
  --output-dir docs/experiments/2026-06-18-remote-h1-e18-diagnosis/probe_random_init_lowrank_cli `
  --override model.style_code_spatial_mode='"lowrank"' `
  --override model.style_code_spatial_hidden_dim=64 `
  --override model.style_code_spatial_rank=8 `
  --override model.style_code_spatial_base_hw=16 `
  --override model.style_code_spatial_scale=0.35
```

## 2.3 Cache-order fix verification

The output-appearance cache is now written after spatial-source resolution.

Verified by:

- `probe_random_init_post_cache_fix_v2/summary.json`

Key rows:

- `spatial`
  - `style_spatial_map_abs -> 0.9807956218719482`
  - `cached_output_style_map_abs -> 0.9807956218719482`
- `both`
  - `style_spatial_map_abs -> 0.9874134063720703`
  - `cached_output_style_map_abs -> 0.9874134063720703`

Meaning:

- downstream output-appearance consumers now see the resolved matched-target spatial map instead of the pre-resolution placeholder

## 2.4 No-reference endpoint fallback now carries style maps

With `style_code_spatial_mode=lowrank`, direct local checks now show:

- `appearance_map_present -> true`
- `appearance_map_abs -> 0.06679472327232361`
- `proximal_map_abs -> 0.7689206600189209`

Meaning:

- no-reference `output_appearance` fallback no longer returns an empty `StyleMaps.map_16`
- the legacy proximal refinement branch also receives a non-empty spatial style carrier

## 2.5 Config-effect audit: which levers change training only, and which change no-reference eval?

The new differential probe is:

```powershell
py -3.12 tools/probe_config_effectiveness.py `
  --config docs/experiments/2026-06-18-remote-h1-e18-diagnosis/remote_config.json `
  --variant-spec docs/experiments/2026-06-18-remote-h1-e18-diagnosis/config_effect_variants.json `
  --output-dir docs/experiments/2026-06-18-remote-h1-e18-diagnosis/config_effect_probe `
  --device cpu
```

This probe compares a baseline config against named override variants under four contexts:

- `plain`: no matched target, no explicit code override
- `configured`: whatever the variant says the real run would use
- `spatial`: matched-target spatial path only
- `code`: explicit style-code override only

It also copies shared baseline weights into each variant before probing, so differences are caused by the config delta instead of random init drift.

Key baseline fact from `config_effect_probe/summary.json`:

- `anatomy_code_body_dead_spatial_body_live -> true`

That is the current `h1` contract mismatch in one line:

- training-style spatial conditioning changes `h_body`
- no-reference code-only style still does not

### Blend sweep result

For `blend_0p20`, `blend_0p40`, `blend_0p60`, and `blend_0p00`:

- `plain`
  - `vs_base_forward_mean_abs -> 0.0`
  - `style_response_forward_mean_abs -> 0.0`
- `configured` / `spatial`
  - `blend_0p20 -> 0.02586447075009346`
  - `blend_0p40 -> 0.021669115871191025`
  - `blend_0p60 -> 0.016345953568816185`
  - `blend_0p00 -> 0.029262322932481766`

Meaning:

1. these blend sweeps are not broken no-ops in the real `h1` training graph because `semantic_self_topology_gate=true`
2. but they are exact no-ops in the plain no-reference path
3. so a run can have a real train-time blend effect and still look nearly unchanged at no-reference eval

This is the most important correction to the earlier simplistic story "blend=1.0 locks style, therefore all blend sweeps are fake." The stronger and more precise statement is:

- in `h1`, blend changes are real for the matched-target spatial branch
- but the benchmarked no-reference path has no spatial source to modulate, so plain inference stays identical

### Low-rank code-map result

The first variants that materially change the no-reference plain path are the low-rank code-map repairs:

- `code_map_lowrank`
  - `plain vs_base_forward_mean_abs -> 0.06238806992769241`
  - `configured vs_base_forward_mean_abs -> 0.004142487421631813`
  - `anatomy_code_body_dead_spatial_body_live -> false`
- `code_map_lowrank_both`
  - `plain vs_base_forward_mean_abs -> 0.10839318484067917`
  - `configured vs_base_forward_mean_abs -> 0.007987639866769314`
  - `anatomy_code_body_dead_spatial_body_live -> false`
- `code_map_lowrank_both_blend_0p40`
  - `plain vs_base_forward_mean_abs -> 0.10840653628110886`
  - `configured vs_base_forward_mean_abs -> 0.023664431646466255`
  - `anatomy_code_body_dead_spatial_body_live -> false`

Meaning:

1. low-rank `style_code -> map_16` is the first tested repair that changes the benchmarked no-reference eval graph itself
2. once that carrier exists, the old "all groups look the same" failure mode is no longer explained by eval-path identity
3. after this point, remaining similarity between experiments is much more likely to reflect real theory weakness or optimization weakness rather than a silent path no-op

This probe is now the best single answer to the phase-618 question:

> Did this experiment actually change the model we evaluate, or only the model we train?

## 3. Trained checkpoint diagnosis

For completeness, the pulled checkpoint was also probed:

```powershell
py -3.12 tools/probe_conditioning_sensitivity.py `
  --config docs/experiments/2026-06-18-remote-h1-e18-diagnosis/remote_config.json `
  --checkpoint docs/experiments/2026-06-18-remote-h1-e18-diagnosis/epoch_0018.pt `
  --output-dir docs/experiments/2026-06-18-remote-h1-e18-diagnosis/probe
```

Key results:

- `spatial -> 0.0007440653862431645`
- `code -> 0.0005612247623503208`
- `both -> 0.0009249740978702903`

We should not over-index on these training-state numbers for root-cause diagnosis, but they do show one useful thing:

- the trained weights respond only weakly to matched-target changes overall
- this is consistent with a content-preserving training dynamic that learns to suppress style-conditioned transport

## 4. Zero-init reality check

The codebase contains many zero-initialized style-side or residual-side heads, including:

- `src/lancet_backbone.py`
  - `output_appearance_head[-1]`
- `src/model.py`
  - style injectors
  - carrier/gate injectors
  - spatial carrier-gate injectors
  - several style-delta / style-section / proximal output layers
- `src/lancet_blocks.py`
  - multiple `gamma` parameters start at zero
  - several routing/projection layers start with zero weights or zero bias

This does not by itself prove a bug. But it does explain why many style-side branches can stay effectively dormant unless they receive strong early gradients.

## 5. Diagnosis

The main code-level conclusions are:

1. Historical blend-only sweeps are invalid if the gate was off.
2. The override-erasure bug was real, but it is not the explanation for this specific h1 config because `tokenizer_content_adaptive=false`.
3. Even with `semantic_self_topology_gate=true`, blend sweeps still leave the plain no-reference path unchanged unless a no-reference spatial carrier exists.
4. The h1 architecture itself gives the global style-code path very little actuation compared with the spatial matched-target path.
5. The low-rank code-map repair is the first tested lever that changes the no-reference eval graph and flips `anatomy_code_body_dead_spatial_body_live` to `false`.
6. Zero-initialized style-side modules likely amplify the "everything looks the same" failure mode by making weak style branches hard to wake up.
7. There was also a real cache-order bug: output-appearance context was being cached before the resolved spatial style map existed.
8. Even after waking part of the code path, the semantic body still only reacts strongly to the spatial matched-target path unless we add an explicit no-reference spatial carrier. That train/eval mismatch is the next bottleneck.

## 6. Action

For future reruns and ablations:

1. trust random-init liveness probes before trusting full training curves
2. run `probe_config_effectiveness.py` before trusting any sweep that changes only config levers such as blend, conditioning mode, or low-rank spatial carrier
3. rerun old OT comparisons only after the conditioning-path fixes
4. treat style-code-only conclusions with caution unless the probe shows a meaningful nonzero executed delta
5. explicitly test whether zero-initialized style heads should be warmed, rescaled, or partially de-zeroed
6. next repair should target no-reference body-level style actuation, not just stronger decoder-only code paths
7. the new low-rank code-map path is the first candidate that actually satisfies that condition in random-init anatomy probes
