# 616 Clean OT Probe Round 8: Cost Composition x Target Geometry

## Why this round exists

The 2026-06-17 implementation audit established three reasons the earlier OT
results were not yet a clean read of the 616 theory:

1. the retained line still used mixed `appearance_plus_structure` OT
2. the target geometry still defaulted to stochastic `sample`
3. `pure_vertical_flow` had been applied to the target but not the bridge noise

Round 8 addresses the first two points directly with a matched 2x2 table before
we interpret later OT outcomes as evidence for or against the theory itself.

## Matrix

All four rows keep fixed:

- same parent family: clean `self_affinity_gw` fast probe lane
- same `sinkhorn_unbalanced`
- same `pure_vertical_flow` target projection
- same one-epoch / 60-step / transfer-only eval contract

Changed axes:

1. OT cost composition
   - `appearance_plus_structure`
   - `structure_only`
2. target geometry
   - `sample`
   - `barycentric_topk = 4`

## Configs

- control:
  - [phase616_clean_ot_probe_selfaffgw_mix_sample_faststep60_e1_authoritative.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_ot_probe_selfaffgw_mix_sample_faststep60_e1_authoritative.json)
- row B:
  - [phase616_clean_ot_probe_selfaffgw_structureonly_sample_faststep60_e1_authoritative.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_ot_probe_selfaffgw_structureonly_sample_faststep60_e1_authoritative.json)
- row C:
  - [phase616_clean_ot_probe_selfaffgw_mix_barytopk4_faststep60_e1_authoritative.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_ot_probe_selfaffgw_mix_barytopk4_faststep60_e1_authoritative.json)
- row D:
  - [phase616_clean_ot_probe_selfaffgw_structureonly_barytopk4_faststep60_e1_authoritative.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_ot_probe_selfaffgw_structureonly_barytopk4_faststep60_e1_authoritative.json)

## Expected readout

Questions this round should answer:

1. Does `structure_only` improve transfer once mixed appearance cost is removed?
2. Does deterministic barycentric target geometry reduce noisy or unstable OT behavior?
3. Does the best row come from cleaner theory alignment, or does the retained
   mixed sampled line remain strongest even after this cleanup?

## Launchers

- local/WSL batch:
  - [run_phase616_clean_ot_probe_round8_targetgeometry2x2.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_phase616_clean_ot_probe_round8_targetgeometry2x2.sh)
- remote launcher:
  - [launch_phase616_clean_ot_probe_round8_targetgeometry2x2_remote.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_phase616_clean_ot_probe_round8_targetgeometry2x2_remote.sh)

## Closure rule

If no row beats the retained mixed-sampled control on transfer while also
improving OT observability, then OT cleanup should be considered closer to
exhausted and priority should shift to the remaining 616 mechanisms:

- bridge-noise geometry
- stats / photometry track
- later tokenizer-side geometry work

## Live status

### 2026-06-17 16:35 CST partial read

Round 8 has started on the remote lane.

Observed status:

- `mix_sample` finished train + eval
- `structureonly_sample` reached late training steps and had created its run
  directory, but had not yet produced final eval artifacts at this read
- the barycentric rows had not started yet at this read

### Row A: `appearance_plus_structure` + `sample`

Eval summary from:

- `/home/xy/Latent_Style/SchrodingerBridge_phase616/SchrodingerBridge/exp/aaai2027_phase616_clean_ot_probe_selfaffgw_mix_sample_faststep60_e1_authoritative/full_eval_transfer_probe60/curve_summary.json`

Key metrics:

- `transfer_clip_style = 0.69798`
- `transfer_content_lpips = 0.64951`
- `eval_wall_total_sec = 221.93`
- `generation_sec = 120.80`
- `vae_decode_sec = 59.25`

GPU summary:

- mean VRAM `2.61 GiB`
- peak VRAM `9.63 GiB`
- mean util `39.5%`
- peak util `96%`
- mean power `71.1 W`

Mid-train observability snapshot around step 50:

- `ot_cost = 2.5690`
- `ot_target_gini = 0.0456`
- `ot_target_max_mass = 0.3438`
- `base_structural_drift = 0.0831`
- `fiber_energy_ratio = 0.7220`
- `low_freq_leak = 2.3328`
- `training_bridge_noise_projection_active = 0.0`

Interpretation:

- this reproduces the implementation-audit baseline cleanly
- the row is strong enough on style to remain relevant
- but it still confirms that the bridge-noise half of the vertical geometry is
  off in this line

### Row B: `structure_only` + `sample`

Eval summary from:

- `/home/xy/Latent_Style/SchrodingerBridge_phase616/SchrodingerBridge/exp/aaai2027_phase616_clean_ot_probe_selfaffgw_structureonly_sample_faststep60_e1_authoritative/full_eval_transfer_probe60/curve_summary.json`

Key metrics:

- `transfer_clip_style = 0.69363`
- `transfer_content_lpips = 0.72141`
- `eval_wall_total_sec = 215.98`
- `generation_sec = 118.14`
- `vae_decode_sec = 57.05`

GPU summary:

- mean VRAM `2.67 GiB`
- peak VRAM `8.79 GiB`
- mean util `36.8%`
- peak util `95%`
- mean power `72.0 W`

Matched delta vs Row A (`mix_sample`):

- `CLIP-S`: `-0.00435`
- `LPIPS`: `+0.07190`
- eval wall: `-5.96s`

Current interpretation:

- removing appearance cost **did not help** under the sampled target geometry
- style changed only marginally downward
- LPIPS degraded materially
- this is negative evidence against the claim that the retained mixed OT line
  only looked strong because appearance cost was hiding a purer structure win

So the first half of the 2x2 already says:

- `structure_only` is not automatically better
- if a cleaner OT gain exists, it is more likely to come from target-geometry
  cleanup than from deleting appearance cost alone

### Row C: `appearance_plus_structure` + `barycentric_topk = 4`

Eval summary from:

- `/home/xy/Latent_Style/SchrodingerBridge_phase616/SchrodingerBridge/exp/aaai2027_phase616_clean_ot_probe_selfaffgw_mix_barytopk4_faststep60_e1_authoritative/full_eval_transfer_probe60/curve_summary.json`

Key metrics:

- `transfer_clip_style = 0.66990`
- `transfer_content_lpips = 0.77725`
- `eval_wall_total_sec = 213.76`
- `generation_sec = 116.87`
- `vae_decode_sec = 56.84`

GPU summary:

- mean VRAM `2.65 GiB`
- peak VRAM `6.51 GiB`
- mean util `40.9%`
- peak util `97%`
- mean power `73.3 W`

Mid-train observability snapshot around step 50:

- `ot_cost = 2.5690`
- `ot_target_gini = 0.0456`
- `ot_target_max_mass = 0.3438`
- `base_structural_drift = 0.0500`
- `fiber_energy_ratio = 0.3896`
- `low_freq_leak = 1.9764`
- `training_bridge_noise_projection_active = 0.0`

Matched delta vs Row A (`mix_sample`):

- `CLIP-S`: `-0.02808`
- `LPIPS`: `+0.12774`
- eval wall: `-8.18s`

Interpretation:

- switching the target geometry from `sample` to deterministic
  `barycentric_topk=4` did **not** stabilize the retained mixed OT line
- this row lost both style and structure materially against the sampled control
- the read is especially important because it weakens the claim that earlier OT
  failures were mainly due to stochastic target selection noise

### Row D: `structure_only` + `barycentric_topk = 4`

Eval summary from:

- `/home/xy/Latent_Style/SchrodingerBridge_phase616/SchrodingerBridge/exp/aaai2027_phase616_clean_ot_probe_selfaffgw_structureonly_barytopk4_faststep60_e1_authoritative/full_eval_transfer_probe60/curve_summary.json`

Key metrics:

- `transfer_clip_style = 0.63161`
- `transfer_content_lpips = 0.74008`
- `eval_wall_total_sec = 216.24`
- `generation_sec = 115.99`
- `vae_decode_sec = 57.11`

GPU summary:

- mean VRAM `2.64 GiB`
- peak VRAM `6.51 GiB`
- mean util `37.9%`
- peak util `97%`
- mean power `70.7 W`

Mid-train observability snapshot around step 50:

- `ot_cost = 1.8940`
- `ot_target_gini = 0.1278`
- `ot_target_max_mass = 0.3891`
- `base_structural_drift = 0.0638`
- `fiber_energy_ratio = 0.3886`
- `low_freq_leak = 2.1944`
- `training_bridge_noise_projection_active = 0.0`

Matched delta vs Row A (`mix_sample`):

- `CLIP-S`: `-0.06637`
- `LPIPS`: `+0.09057`
- eval wall: `-5.70s`

Interpretation:

- `structure_only` remained negative after the target geometry was cleaned up
- the more concentrated barycentric transport did not uncover a hidden
  structure-only win; it pushed style down even further
- the higher `ot_target_gini` / `ot_target_max_mass` also suggests this row is
  not reducing hubness in the direction we wanted

## Round-8 verdict

All three alternatives lost to the retained Row A control on `transfer`:

- Row B (`structure_only + sample`) lost modest style and much worse LPIPS
- Row C (`mix + barycentric_topk4`) lost both style and LPIPS decisively
- Row D (`structure_only + barycentric_topk4`) was the weakest style row in the matrix

Operational conclusion:

- the 2026-06-17 implementation audit was still necessary because it cleaned up
  false ambiguity in the earlier 616 reads
- but after that cleanup, this authoritative 2x2 does **not** support OT cost
  composition or deterministic barycentric target geometry as the main missing
  breakthrough
- OT is therefore downgraded from primary rescue hypothesis to
  `diagnostic / infrastructure-retained`

Recommended next queue for 616:

1. bridge-noise geometry with explicit vertical noise projection
2. stats / photometry track (`transport_stats_mode`)
3. only later, tokenizer-side geometry once the bridge itself is clean
