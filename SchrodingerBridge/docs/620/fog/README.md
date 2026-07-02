# 620 Fog / Whitening Diagnosis

Date: 2026-06-20

## Goal

Do not patch the 620 architecture speculatively. First locate where the whitening / fogging enters the real inference path, then make targeted fixes.

## Current Symptom

- Generated transfers from the current 620 no-text local run look washed out, low-contrast, and close to an identity-like pale reconstruction.
- User confirmed this was already present before the newer text branch, so text conditioning is not the primary suspect.

## Quantitative Benchmark

We now need a benchmark stricter than CLIP-S / LPIPS alone.

The gating requirement for this phase is:

- reduce 620 whitening / fogging until it is on the same order as the Seedream reference / IDT control level,
- using an explicit image-space fog metric in addition to the latent-space probes.

### WFI: Whitening / Fog Index

We use:

- `SchrodingerBridge/tools/probe_620_fog_whiteness_index.py`

The current WFI packet includes:

1. `contrast_ratio = luminance_std / luminance_mean`
2. `dynamic_range = (p95 - p5) / (p95 + p5)`
3. `saturation_mean`
4. `edge_energy`
5. `luminance_std`
6. `wfi_score`

where:

- lower `wfi_score` is healthier,
- higher contrast / range / saturation generally means less fog,
- and we should compare not only pooled means, but also:
  - `all_pairs`
  - `identity`
  - `style_transfer`

because some target styles are naturally flatter or paler than others.

### Seedream repaired750 reference

Reference packet:

- `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\seedream45_api\distinct5_512_seedream45_windhub_20260607_repaired750`

Reference experiment note:

- `SchrodingerBridge/docs/experiments/2026-06-07-distinct5-seedream45-repaired750.md`

On the current local WFI sample (`30` images from the repaired750 packet), we measured:

- `avg_contrast_ratio ~= 0.4244`
- `avg_dynamic_range ~= 0.6218`
- `avg_saturation_mean ~= 0.3642`
- `avg_luminance_std ~= 0.1950`
- `avg_wfi_score ~= 0.1581`

Interpretation:

- this is our first practical "healthy image-space" baseline,
- but it is still only a pooled sample baseline,
- so the next stricter benchmark should compare:
  - Seedream `identity`
  - Seedream `style_transfer`
  - 620 `identity`
  - 620 `style_transfer`

using the same WFI script over `metrics.csv`.

### White/fog acceptance gate for this phase

Before we resume broader 620 roadmap work such as:

- adding text,
- changing cross-attention forms,
- or continuing broader DINO variants,

we should first satisfy:

1. 620 `identity` fog level is not materially worse than the Seedream / IDT reference bucket
2. 620 `style_transfer` fog level is not materially worse than the Seedream healthy band for the same test distribution
3. this is supported by both:
   - image-space WFI
   - latent-space endpoint probes

## Known Evidence

### Config facts

Reference run:

- `G:\GitHub\Latent_Style\SchrodingerBridge\src\exp\620_spatial_bridge\620_swd16_notext_vlen004_b8_local\config.json`

Relevant fields:

- `model.contract_family = "620_spatial_bridge"`
- `model.style_text_enabled = false`
- `model.output_appearance_alignment_mode = "none"`
- `model.transport_prediction_mode = "velocity"`
- `bridge.bridge_sigma = 0.02`
- `full_eval.num_steps = 16`

So the current whitening is not explained by the text path or the output appearance alignment head.

### Training/eval evidence already observed

Training log:

- `G:\GitHub\Latent_Style\SchrodingerBridge\src\exp\620_spatial_bridge\620_swd16_notext_vlen004_b8_local\logs\training_20260620_060936.csv`

Observed:

- `velocity_abs` is around `0.09-0.10`
- `target_velocity_abs` is around `0.51-0.54`

This suggests the learned velocity magnitude is much smaller than the training target magnitude.

Eval summary:

- `G:\GitHub\Latent_Style\SchrodingerBridge\src\exp\620_spatial_bridge\620_swd16_notext_vlen004_b8_local\full_eval\epoch_0008\summary.json`

Observed:

- `transfer clip_style ~= 0.665`
- `clip_s_delta_idt ~= -0.175`
- runtime observability shows:
  - `model_style_gate_value ~= 0.047`
  - `model_cross_attn_entropy ~= 5.54`
  - `model_velocity_abs ~= 0.0946`

Representative bad image:

- `G:\GitHub\Latent_Style\SchrodingerBridge\src\exp\620_spatial_bridge\620_swd16_notext_vlen004_b8_local\full_eval\epoch_0008\images\Early_Renaissance_andrea-mantegna_adoration-of-the-magi-central-panel-from-the-altarpiece_to_Impressionism.png`

Visual read:

- low contrast
- pale / misty look
- weak style movement

## Diagnosis Questions

We need probes that separate these cases:

1. `predict_endpoint` is already low-energy and washed out in latent space.
2. `predict_endpoint` is fine, but `integrate_transport` collapses contrast over repeated I2SB steps.
3. Latent transport is acceptable, but `VAE decode` or latent scaling makes the final image look foggy.
4. The model mostly preserves low frequency while failing to move high frequency / contrast energy.

## Probe Protocol

Primary probe:

- `SchrodingerBridge/tools/probe_620_fog_path.py`

It should measure, per sample and per stage:

- latent mean / std / abs mean
- latent lowpass std
- latent highpass RMS
- delta RMS to source and target
- decoded luminance mean / std
- decoded gradient energy
- decoded distance to source decode and target decode

Stages to compare:

- source latent
- target latent
- `predict_endpoint(t=0)`
- `integrate(num_steps=1)`
- `integrate(num_steps=4)`
- `integrate(num_steps=8)`
- `integrate(num_steps=16)` when affordable

The first read we want:

- does contrast collapse already exist at endpoint?
- does it worsen as NFE increases?
- does decode amplify the collapse?

## Execution Order

1. Implement probe without changing model behavior.
2. Run `py_compile`.
3. Run local probe on the known bad local checkpoint.
4. Summarize local evidence here.
5. Run the same probe on the remote 3060 WSL lane.
6. Only after that, propose targeted fixes.

## Remote Notes

Target host:

- `ssh -p 2222 administrator@100.115.18.62`

Target WSL repo:

- `/mnt/i/Github/Latent_Style/SchrodingerBridge`

Use the existing remote launcher contract from:

- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\616\tools.md`
- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\620\Infra.md`

Keep this phase eval-only and diagnostic first.

## Round 1 Remote Findings

### 2026-06-20 remote probe launch

First remote probe packet:

- launcher task: `620_fog_probe_smoke`
- result: failed before inference due to missing remote DINO cache path

Observed error:

- `FileNotFoundError: DINO cache not found: /mnt/i/Github/Latent_Style/eval_cache/offline_pairing/dinov2_small_wikiart_distinct5_train_cache.pt`

Interpretation:

- the remote eval chain was live
- the first blocker was infra/path alignment, not model execution

Second remote probe packet:

- launcher task: `620_fog_probe_smoke_v2`
- result: reached dataset sampling, then failed on DINO sidecar alignment

Observed error:

- `KeyError: "Missing DINO sidecar for ('Ukiyo_e', 'Ukiyo_e__utagawa-kuniyoshi_tametomo-rescued-from-the-sea-monster-by-tengu')"`

Interpretation:

- remote dataset root and cache roots were close enough to start real loading
- however the remote DINO cache does not fully align with the current latent dataset stems
- we still do not yet have evidence that whitening originates in:
  - endpoint prediction
  - integration
  - or VAE decode
- because the real input contract is still broken before a full probe sample completes

### Immediate next move

Do not patch the model yet.

Instead:

1. make the fog probe resilient to partial DINO sidecar coverage
2. skip or record missing-sidecar samples instead of aborting the whole packet
3. collect the first successful fog statistics on the aligned subset
4. then decide whether the next problem is model-path whitening or cache-generation drift

### 2026-06-20 remote probe v3

Third remote probe packet:

- launcher task: `620_fog_probe_smoke_v3`
- settings:
  - `sample_count=1`
  - `max_scan_multiplier=50`
  - skip missing-sidecar samples instead of aborting

Observed artifacts:

- `fog_stage_metrics.csv`: empty
- `fog_sample_summary.csv`: empty
- `fog_skipped_samples.csv`: non-empty
- `summary.json`: written successfully

Interpretation:

- the probe now completes as a diagnostic job
- but it still collects `0` valid aligned samples within the scanned window
- so the current remote DINO cache mismatch is not a one-off bad sample; it is broad enough to block even a tiny fog packet

This is now the strongest current evidence:

- before blaming the 620 architecture for whitening on remote, we must first restore dataset-to-DINO sidecar alignment

## Current Best Read

At this moment, the primary verified blocker is:

- remote DINO cache / latent dataset alignment is broken

Most likely root cause from code and artifact audit:

- current remote cache usage was pointing at `dinov2_small_train_cache.pt`
- repository launch remnants show that file was historically built from an older generic root:
  - image root: `style_data/train`
  - latent root: `latent-256`
- current 620 run and current latent dataset use:
  - latent root: `/mnt/i/wikiart_distinct5_samam_512_latents_ema/train`
  - image root should be aligned to `/mnt/i/wikiart_distinct5_samam_512_classview/train`

So the strongest explanation is:

- remote DINO cache is stale / built from the wrong dataset lineage, not merely misnamed

Supporting code evidence:

- `tools/experiments/build_offline_dino_pairing_cache.py` writes sidecars only for image/latent stem pairs found in the chosen dataset roots
- `tools/experiments/dino_cache_utils.py` supports both `style__stem` and `stem` aliases
- after upgrading dataset alias handling, the previously failing stems still do not exist in the currently referenced cache
- that means the stems are absent from the cache itself, not just inaccessible by naming convention

Not yet verified:

- whether whitening starts at `predict_endpoint`
- whether it worsens during `integrate_transport`
- whether VAE decode amplifies it

## Next Required Action

1. audit the remote DINO cache build contract against the exact latent dataset stems now in use
2. rebuild or repair the remote DINO cache so selected target stems resolve correctly
3. rerun `probe_620_fog_path.py`
4. only then propose model-side fixes for whitening

Remote rebuild entry prepared:

- `G:\GitHub\Latent_Style\SchrodingerBridge\tools\experiments\run_620_rebuild_remote_dino_cache.sh`

## Round 2 Remote Findings

### 2026-06-20 remote DINO rebuild verified

Remote rebuild artifacts now exist and match the intended balanced distinct5 training set:

- cache:
  - `/mnt/i/Github/Latent_Style/eval_cache/offline_pairing/dinov2_small_wikiart_distinct5_train_cache.pt`
- pairing plan:
  - `/mnt/i/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/prototype_pairing_top8.pt`
- logs:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/docs/620/logs/620_rebuild_dino_cache_stdout.log`
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/docs/620/logs/620_rebuild_dino_plan_stdout.log`

Verified cache build summary:

- `n_rows = 5000`
- per-style counts:
  - `Early_Renaissance = 1000`
  - `Impressionism = 1000`
  - `Minimalism = 1000`
  - `Rococo = 1000`
  - `Ukiyo_e = 1000`

Interpretation:

- the remote cache / pairing infra is now aligned to the intended dataset lineage
- the previous remote fog blocker was real infra drift, and it is now cleared

### 2026-06-20 remote probe after rebuild

Probe packet:

- launcher task: `620_fog_probe_after_rebuild_v1`
- output dir:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/docs/620/fog/remote_probe_after_rebuild_v1`

Observed artifacts:

- `fog_stage_metrics.csv`: non-empty
- `fog_sample_summary.csv`: non-empty
- `fog_skipped_samples.csv`: empty
- `summary.json`: written successfully

So this is the first successful real-path fog measurement on valid 620 samples.

### Headline probe result

From `summary.json` stage averages:

- source image std: `0.2176`
- endpoint image std: `0.1878`
- endpoint / source image std ratio: `0.8630`
- source latent std: `0.8245`
- endpoint latent std: `0.6980`
- source latent highpass RMS: `0.5900`
- endpoint latent highpass RMS: `0.5676`
- endpoint / source latent highpass ratio: `0.9622`
- endpoint / source image gradient ratio: `1.0640`

Interpretation:

- the whitening / fogging is **already present at `predict_endpoint(t=0)`**
- this is **not primarily a high-frequency collapse**
- edge / gradient energy is largely preserved
- the stronger collapse is in overall latent/image dynamic range, especially low-frequency / contrast structure

### Integration is not the main culprit

Comparing `predict_endpoint_t0` to `integrate_nfe_1/4/8/16`:

- `img_std` stays nearly flat:
  - endpoint: `0.1878`
  - nfe4: `0.1893`
  - nfe8: `0.1892`
  - nfe16: `0.1895`
- `latent_high_rms` also stays nearly flat:
  - endpoint: `0.5676`
  - nfe16: `0.5676`
- `to_source_img_delta_rms` stays small:
  - endpoint: `0.0597`
  - nfe16: `0.0551`

Interpretation:

- repeated I2SB integration is **not progressively washing the image out**
- once the endpoint is predicted, the solver mostly preserves that already-compressed state
- the dominant whitening source is upstream of multi-step transport, not downstream in the solver loop

### Training objective evidence now lines up with the probe

The local training log for the same run already showed:

- `endpoint_low_to_source ~= 0.103`
- `endpoint_low_to_target ~= 0.518`
- `endpoint_low_target_ratio ~= 5.03`
- `target_base_shift ~= 0.573`
- `velocity_abs ~= 0.151`
- `target_velocity_abs ~= 0.617`

Interpretation:

- under the current 620 vertical objective, the predicted endpoint low-frequency component stays much closer to the source than to the target
- this matches the fog probe: high-frequency movement exists, but low-frequency contrast / tone migration is underpowered

## Updated Best Read

What is now verified:

1. remote DINO cache misalignment was a real blocker and is fixed
2. whitening begins at `predict_endpoint(t=0)`
3. multi-step `integrate_transport` is not the primary washout source
4. the current 620 objective / endpoint behavior preserves too much source-side low-frequency structure while only partially moving style-relevant contrast statistics

What is still not yet isolated:

- whether the main defect is:
  - the vertical target definition itself
  - insufficient low-frequency supervision in the 620 loss
  - under-scaled velocity magnitude from the 620 head / block stack
  - or a combination of those three

## Next Required Action

Do not change the solver first.

Instead, inspect and probe the 620 endpoint-learning path:

1. compare current vertical target construction against the actual desired low-frequency transfer behavior
2. probe whether `pred_velocity` is systematically under-scaled relative to `target_velocity`
3. add a focused endpoint-path probe for low-frequency migration if the current fog probe is not enough
4. only then make a targeted 620 model/loss change and rerun the same remote fog packet for before/after comparison

## First Targeted Fix Attempt

### 2026-06-20 low-frequency endpoint supervision

Based on the successful remote fog probe, the first targeted fix is:

- keep the 620 solver unchanged
- keep the DINO patch conditioning unchanged
- add an explicit low-frequency endpoint loss in `losses620.py`
- wire it through the existing bridge config field:
  - `bridge.w_content_lowpass_anchor`

Implementation intent:

- the current vertical FM target already supervises high-frequency migration
- the fog probe showed the real deficit is low-frequency / contrast migration
- so the new term directly penalizes:
  - `L1(lowpass(z_hat1), lowpass(target_style))`

This is intentionally narrow:

- it does not speculate about the solver
- it does not rewrite the 620 block stack
- it only adds supervision exactly where the probe showed the endpoint is too source-anchored

### Early smoke signal

New config:

- `G:\GitHub\Latent_Style\SchrodingerBridge\configs\620_spatial_bridge_lowfreqfix.json`

Remote launcher task:

- `620_lowfreqfix_smoke1`

Before the full training epoch finished, the built-in CPU endpoint decomposition probe already reported:

- `endpoint_low_to_source = 0.1100`
- `endpoint_low_to_target = 0.3530`
- `endpoint_low_target_ratio = 3.21`

Compared with the earlier training/eval evidence from the unfixed run:

- prior `endpoint_low_target_ratio ~= 5.03`

Interpretation:

- the first targeted fix moves the endpoint in the correct direction
- low-frequency target attraction is measurably stronger than before
- this is not yet final proof that whitening is solved
- but it is the first architecture/loss change that is directly consistent with the fog diagnosis

### Infra note

This smoke launch also exposed a launcher-only issue:

- the remote 620 training process started normally
- but the host-side health check treated `~12 GB` runtime VRAM as a ceiling failure

That is a launcher guard issue, not a model failure.
The 620 remote launcher was updated to use a looser runtime ceiling for follow-up runs.

### 2026-06-20 debug training result: first fix is directionally wrong in real training

To separate launcher issues from model behavior, a minimal remote debug lane was added:

- `G:\GitHub\Latent_Style\SchrodingerBridge\configs\620_spatial_bridge_lowfreqfix_debug.json`

Settings:

- `batch_size = 16`
- `num_workers = 0`
- `stop_after_global_steps = 2`
- full eval disabled

This lane completed successfully on remote and produced:

- checkpoint:
  - `/mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_lowfreqfix_debug_b16_gs2_smoke/epoch_0001.pt`
- training log:
  - `/mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_lowfreqfix_debug_b16_gs2_smoke/logs/training_20260620_170341.csv`

Observed training metrics after 2 optimizer steps:

- `endpoint_low_to_source = 0.0499`
- `endpoint_low_to_target = 0.5898`
- `endpoint_low_target_ratio = 12.42`
- `low_freq_leak = 0.0143`
- `velocity_abs = 0.0162`
- `target_velocity_abs = 0.6275`

Compare with the earlier unfixed local baseline:

- baseline `endpoint_low_to_source = 0.0958`
- baseline `endpoint_low_to_target = 0.5114`
- baseline `endpoint_low_target_ratio = 5.35`
- baseline `low_freq_leak = 0.1330`
- baseline `velocity_abs = 0.1458`
- baseline `target_velocity_abs = 0.6069`

Interpretation:

- the added low-frequency endpoint penalty did **not** improve the desired low-frequency migration
- instead, it collapsed low-frequency motion magnitude:
  - `velocity_abs` fell sharply
  - `low_freq_leak` fell sharply
  - endpoint low-frequency became even **closer to source**
  - and **farther from target** in relative terms

So this first targeted fix is informative, but wrong:

- it suppresses low-frequency dynamics
- it does not solve whitening
- it should not be promoted as the main repair path

## Updated Direction

The evidence now points away from “add more endpoint low-frequency penalty”.

The better next move is:

1. change the **vertical target definition itself** so the model is asked to transport the right low-frequency component,
2. rather than penalizing the endpoint after the fact while keeping the same source-anchored target path.

In other words:

- the problem is more fundamental than missing endpoint regularization
- the current vertical construction is likely asking the model to keep too much source-side low-frequency structure
- we should next probe / revise the bridge target path, not just stack another auxiliary loss

## Second Targeted Fix: Change the Vertical Low-Frequency Path

### 2026-06-20 target-linear vertical path

Instead of adding another endpoint penalty, the vertical state construction was changed so low frequency can move from source to target during training:

- `bridge.training_target_projection_low_mode = "target_linear"`
- `bridge.w_content_lowpass_anchor = 0.0`

Implementation behavior:

- old `"all"` mode:
  - low frequency in `x_t` stays source-anchored
  - target velocity only supervises the high-frequency residual
- new `"target_linear"` mode:
  - `x_low = (1-t) * c_low + t * t_low`
  - target velocity includes both:
    - `t_high - c_high`
    - `t_low - c_low`

This is the first fix that changes the actual training target instead of punishing the endpoint after the fact.

### Debug lane result

Config:

- `G:\GitHub\Latent_Style\SchrodingerBridge\configs\620_spatial_bridge_targetlinear_debug.json`

Remote checkpoint:

- `/mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_targetlinear_debug_b16_gs2_smoke/epoch_0001.pt`

Training log:

- `/mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_targetlinear_debug_b16_gs2_smoke/logs/training_20260620_171317.csv`

Key comparison:

| lane | endpoint_low_to_source | endpoint_low_to_target | endpoint_low_target_ratio | low_freq_leak | velocity_abs | target_velocity_abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.0958 | 0.5114 | 5.3451 | 0.1330 | 0.1458 | 0.6069 |
| lowfreqfix_debug | 0.0499 | 0.5898 | 12.4233 | 0.0143 | 0.0162 | 0.6275 |
| targetlinear_debug | 0.3188 | 0.2932 | 1.0225 | 0.0141 | 0.0162 | 0.9253 |

Interpretation:

- `lowfreqfix_debug` made the endpoint even more source-anchored
- `targetlinear_debug` is the first lane where endpoint low frequency is no longer clearly stuck at the source
- the endpoint low-frequency distance to target is now slightly smaller than the distance to source

This is still only a tiny 2-step debug lane, so it is not enough by itself. But directionally it matches the actual fog diagnosis much better than the auxiliary low-frequency loss.

## Fog Probe Comparison

### Baseline bad lane

Probe:

- `/mnt/i/Github/Latent_Style/SchrodingerBridge/docs/620/fog/remote_probe_after_rebuild_v1/summary.json`

Headline:

- endpoint latent highpass vs source: `0.9622`
- endpoint image gradient vs source: `1.0640`
- endpoint image std vs source: `0.8630`

Read:

- edges are mostly preserved
- image dynamic range is compressed
- whitening is already present at `predict_endpoint(t=0)`

### Target-linear debug lane

Probe:

- `/mnt/i/Github/Latent_Style/SchrodingerBridge/docs/620/fog/remote_probe_targetlinear_debug_epoch1/summary.json`

Headline:

- endpoint latent highpass vs source: `1.0029`
- endpoint image gradient vs source: `1.0187`
- endpoint image std vs source: `1.0241`

Selected stage averages at `predict_endpoint_t0`:

- `latent_std = 0.8514`
- `img_std = 0.2380`
- `img_grad_rms = 0.1106`
- `to_source_img_delta_rms = 0.0186`
- `to_target_img_delta_rms = 0.3657`

Comparison to the baseline remote probe:

- baseline endpoint image std ratio: `0.8630`
- target-linear endpoint image std ratio: `1.0241`

Interpretation:

- the earlier endpoint dynamic-range collapse is no longer visible in this debug probe
- the solver was never the main whitening source
- revising the vertical low-frequency target is the first change that appears to remove the fog signature itself

## Current Conclusion

Current evidence supports the following causal chain:

1. remote DINO cache drift was a real infra blocker and was fixed first
2. whitening originates at the predicted endpoint, not from repeated I2SB stepping
3. high-frequency structure is mostly intact even in the bad lane
4. the main defect is source-anchored low-frequency transport in the vertical target path
5. adding an auxiliary endpoint low-frequency penalty suppresses motion and is not the fix
6. letting the low-frequency path move linearly toward target is the first evidence-backed repair

## Next Action

Run a non-debug remote smoke lane with the target-linear path, then rerun:

1. training-metric comparison
2. `probe_620_fog_path.py`
3. if needed, only then tune magnitude / stability from the new target path

## Non-Debug Smoke Verification

### 2026-06-20 first real smoke launch: observability path OOM, not model-path failure

After the target-linear debug lane looked promising, a real remote smoke lane was launched:

- run: `620_targetlinear_swd8_sigma002_nfe8_b80_smoke`
- launcher task: `620_targetlinear_smoke1`

That first launch failed before finishing epoch 1 with:

- `torch.cuda.OutOfMemoryError` inside `src/blocks620.py`
- the failing line was the explicit `attn = softmax(q @ k^T)` path used only to compute cross-attention entropy and pixel entropy diagnostics

Interpretation:

- this was **not** evidence against the target-linear path itself
- the real culprit was observability code reconstructing a full attention matrix in the hot training path
- that defeated the intended memory savings from SDPA

Targeted fix:

- keep the real cross-attention forward path on `scaled_dot_product_attention`
- compute entropy / pixel-entropy from a sampled subset of queries instead of materializing the full attention matrix at training batch size

So there are now two distinct fixes in play:

1. architectural fix for whitening:
   - `training_target_projection_low_mode = "target_linear"`
2. infra fix for liveness at realistic batch size:
   - sampled attention statistics instead of full attention reconstruction

### 2026-06-20 second real smoke launch: completed successfully

Relaunched run:

- launcher task: `620_targetlinear_smoke2`
- checkpoint dir:
  - `/mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_targetlinear_swd8_sigma002_nfe8_b80_smoke`
- checkpoint:
  - `/mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_targetlinear_swd8_sigma002_nfe8_b80_smoke/epoch_0001.pt`
- training log:
  - `/mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_targetlinear_swd8_sigma002_nfe8_b80_smoke/logs/training_20260620_172329.csv`

End-of-epoch training metrics:

- `loss = 2.5552`
- `loss_fm = 1.4677`
- `loss_swd_ss = 0.1309`
- `loss_edge_ss = 0.3184`
- `endpoint_low_to_source = 0.3320`
- `endpoint_low_to_target = 0.3430`
- `endpoint_low_target_ratio = 1.0416`
- `fiber_energy_ratio = 1.8290`
- `low_freq_leak = 0.1472`
- `target_base_shift = 0.6328`
- `velocity_abs = 0.1610`
- `target_velocity_abs = 0.9338`
- `style_gate_value = 0.0497`
- `cross_attn_entropy = 5.5313`

Runtime evidence:

- epoch completed successfully in about `53.1s`
- observed mean VRAM usage was about `5.70 / 6.11 GB`
- no OOM after the attention-statistics fix

Interpretation:

- the target-linear path is now live in a realistic smoke lane, not only in a 2-step debug toy run
- the cross-attention observability path no longer destabilizes training
- endpoint low-frequency is still not strongly target-biased yet, but it is much closer to parity than the old source-anchored baseline

## Smoke Fog Probe Verification

Probe:

- `/mnt/i/Github/Latent_Style/SchrodingerBridge/docs/620/fog/remote_probe_targetlinear_smoke_epoch1/summary.json`

Headline:

- endpoint latent highpass vs source: `1.1229`
- endpoint image gradient vs source: `0.9689`
- endpoint image std vs source: `1.0914`

Selected stage averages:

| stage | latent_std | latent_high_rms | img_std | img_grad_rms | to_source_img_delta_rms | to_target_img_delta_rms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| source | 0.7916 | 0.5802 | 0.2022 | 0.0828 | 0.0000 | 0.3196 |
| endpoint | 0.9763 | 0.6515 | 0.2207 | 0.0803 | 0.0583 | 0.3348 |
| nfe16 | 0.9819 | 0.6491 | 0.2202 | 0.0791 | 0.0609 | 0.3343 |

Comparison to the bad baseline probe:

| probe lane | endpoint img std vs source | endpoint latent high vs source | endpoint img grad vs source |
| --- | ---: | ---: | ---: |
| baseline bad lane | 0.8630 | 0.9622 | 1.0640 |
| targetlinear debug | 1.0241 | 1.0029 | 1.0187 |
| targetlinear smoke | 1.0914 | 1.1229 | 0.9689 |

Interpretation:

- the smoke lane continues the same direction already seen in the debug lane
- the earlier endpoint dynamic-range compression is no longer present
- integration still does not materially worsen or rescue the endpoint; the endpoint remains the deciding stage
- the main whitening signature has been removed in this smoke checkpoint

## Updated Current Read

What is now evidence-backed:

1. the original whitening came from source-anchored low-frequency endpoint behavior
2. the solver was never the main washout source
3. the target-linear vertical path removes the endpoint dynamic-range collapse in both debug and real smoke runs
4. the realistic-batch remote lane needed a separate infra fix in `blocks620.py` because full attention reconstruction for entropy metrics caused OOM
5. after that fix, the real smoke lane is stable and probe-clean enough to justify a longer formal run

## Next Action

Run a longer remote formal target-linear lane and then repeat:

1. training-log comparison against baseline
2. `probe_620_fog_path.py` on the formal checkpoint
3. if the formal lane stays de-fogged, move on to broader eval and metric expansion

## Formal Lane Follow-Through

### 2026-06-20 formal target-linear run is live and stays de-fogged through epoch 3

Formal lane:

- launcher task: `620_targetlinear_formal_b64`
- run dir:
  - `/mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_targetlinear_swd8_sigma002_nfe8_b64`

The formal run was launched at `batch_size = 64` to keep a margin below the realistic 3060 memory ceiling after the attention-statistics fix.

Observed checkpoints:

- `epoch_0001.pt`
- `epoch_0002.pt`
- `epoch_0003.pt`

Training log:

- `/mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_targetlinear_swd8_sigma002_nfe8_b64/logs/training_20260620_172903.csv`

Selected epoch summaries:

| epoch | loss | loss_fm | endpoint_low_to_source | endpoint_low_to_target | endpoint_low_target_ratio | low_freq_leak | velocity_abs | style_gate_value | cross_attn_entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.6199 | 1.4483 | 0.3310 | 0.3355 | 1.0248 | 0.1393 | 0.1524 | 0.0491 | 5.5313 |
| 2 | 2.5820 | 1.4508 | 0.3341 | 0.3292 | 0.9995 | 0.1398 | 0.1694 | 0.0504 | 5.5313 |
| 3 | 2.5590 | 1.4280 | 0.3347 | 0.3262 | 0.9839 | 0.1376 | 0.1747 | 0.0657 | 5.5156 |

Interpretation:

- the target-linear path does not regress back toward the old source-anchored low-frequency endpoint behavior
- by epoch 3, the endpoint low-frequency distance to target is now consistently smaller than the distance to source
- the observed `style_gate_value` also rises above the earlier smoke level
- the cross-attention entropy remains in the intended healthy band

### Formal fog probe at epoch 1

Probe:

- `/mnt/i/Github/Latent_Style/SchrodingerBridge/docs/620/fog/remote_probe_targetlinear_formal_epoch1/summary.json`

Headline:

- endpoint latent highpass vs source: `1.1275`
- endpoint image gradient vs source: `0.9510`
- endpoint image std vs source: `1.1161`

Interpretation:

- the formal lane reproduces the smoke result
- the original endpoint dynamic-range collapse is still absent

### Formal fog probe at epoch 3

Probe:

- `/mnt/i/Github/Latent_Style/SchrodingerBridge/docs/620/fog/remote_probe_targetlinear_formal_epoch3/summary.json`

Headline:

- endpoint latent highpass vs source: `1.2036`
- endpoint image gradient vs source: `1.1549`
- endpoint image std vs source: `1.0852`

Selected stage averages:

| stage | latent_std | latent_high_rms | img_std | img_grad_rms | to_source_img_delta_rms | to_target_img_delta_rms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| source | 0.7916 | 0.5802 | 0.2022 | 0.0828 | 0.0000 | 0.4212 |
| endpoint | 0.9249 | 0.6983 | 0.2194 | 0.0957 | 0.1123 | 0.3856 |
| nfe16 | 0.9275 | 0.7011 | 0.2198 | 0.0961 | 0.1134 | 0.3850 |

Interpretation:

- the formal lane stays de-fogged through at least epoch 3
- endpoint dynamic range remains above the source baseline instead of collapsing below it
- integration still only perturbs the endpoint slightly; it is not the whitening source
- compared with epoch 1, the endpoint now shows stronger style movement in image-space gradient and a larger departure from the source image

## Updated Current Conclusion

Current evidence now supports a stronger version of the earlier claim:

1. the old whitening behavior was caused by the source-anchored low-frequency vertical target path
2. switching to `training_target_projection_low_mode = "target_linear"` removes the endpoint contrast collapse
3. that improvement survives not only a tiny debug lane and a 1-epoch smoke lane, but also a continuing formal remote run through at least epoch 3
4. the separate attention-statistics OOM was an infra-side observability bug and is fixed

What still remains before claiming the problem is fully solved:

- finish the full formal run
- probe a later checkpoint near the end of training
- inspect final transfer outputs / full eval to confirm the visual fix persists in the actual generation set, not only in fog metrics

## New Late-Training Finding

### 2026-06-20 formal epoch 6: endpoint behavior drifts again, but the solver now compensates

Later in the same formal run, the training metrics continued improving:

| epoch | loss | loss_fm | endpoint_low_to_source | endpoint_low_to_target | endpoint_low_target_ratio | low_freq_leak | velocity_abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 2.4732 | 1.3566 | 0.3756 | 0.3069 | 0.8246 | 0.2103 | 0.2485 |
| 5 | 2.3875 | 1.2854 | 0.4072 | 0.2880 | 0.7125 | 0.2756 | 0.3118 |
| 6 | 2.3222 | 1.2427 | 0.4304 | 0.2790 | 0.6514 | 0.3193 | 0.3603 |

So by the training objective's own low-frequency metrics, the model keeps moving toward the target.

However, the fog probe on epoch 6 shows a **different** inference-path regime from earlier checkpoints:

Probe:

- `/mnt/i/Github/Latent_Style/SchrodingerBridge/docs/620/fog/remote_probe_targetlinear_formal_epoch6/summary.json`

Headline:

- endpoint latent highpass vs source: `1.1149`
- endpoint image gradient vs source: `1.0461`
- endpoint image std vs source: `0.7276`

Selected stage averages:

| stage | img_std | img_grad_rms | to_source_img_delta_rms | to_target_img_delta_rms |
| --- | ---: | ---: | ---: | ---: |
| source | 0.2022 | 0.0828 | 0.0000 | 0.4212 |
| endpoint | 0.1471 | 0.0866 | 0.3218 | 0.2870 |
| nfe4 | 0.1971 | 0.0951 | 0.2306 | 0.3107 |
| nfe8 | 0.2052 | 0.0952 | 0.2203 | 0.3173 |
| nfe16 | 0.2093 | 0.0951 | 0.2155 | 0.3209 |

Interpretation:

- by epoch 6, `predict_endpoint(t=0)` is again low-contrast in image-space
- **but unlike the original baseline**, multi-step integration now materially changes the result and restores much of the lost dynamic range
- this means the earlier diagnosis still holds for the original bad lane, but later training introduces a second regime:
  - endpoint becomes too aggressive / noisy in latent-space displacement to source
  - multi-step I2SB averaging now partially corrects it instead of merely preserving it

So the whitening story is now more precise:

1. original baseline whitening:
   - endpoint low-frequency was too source-anchored
   - solver was mostly irrelevant
2. target-linear early/mid training:
   - endpoint dynamic range becomes healthy
   - solver still mostly preserves the endpoint
3. target-linear later training (at least by epoch 6):
   - endpoint image dynamic range regresses
   - solver begins to rescue part of that regression

This means the architecture fix is directionally correct, but the full problem is **not yet fully settled**.

The remaining question is now:

- how to preserve the good early target-linear endpoint behavior through late training,
- instead of letting later optimization drift into an endpoint-bad / solver-compensated regime.

### 2026-06-20 formal epoch 8: endpoint drift persists and becomes slightly stronger

The formal run completed through epoch 8.

Final training rows:

| epoch | loss | loss_fm | loss_swd_ss | endpoint_low_to_source | endpoint_low_to_target | endpoint_low_target_ratio | low_freq_leak | velocity_abs | style_gate_value | cross_attn_entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 2.3222 | 1.2427 | 0.1299 | 0.4304 | 0.2790 | 0.6514 | 0.3193 | 0.3603 | 0.0803 | 5.5199 |
| 7 | 2.2883 | 1.2168 | 0.1289 | 0.4393 | 0.2744 | 0.6268 | 0.3336 | 0.3789 | 0.0807 | 5.5217 |
| 8 | 2.2738 | 1.2099 | 0.1270 | 0.4470 | 0.2694 | 0.6050 | 0.3492 | 0.3958 | 0.0808 | 5.5221 |

So by the training objective, the model keeps moving in the same direction:

- lower `endpoint_low_to_target`
- lower `endpoint_low_target_ratio`
- higher `velocity_abs`
- stable style gate and cross-attention entropy

However, the fog probe on the final checkpoint shows that the late endpoint drift is still present:

Probe:

- `/mnt/i/Github/Latent_Style/SchrodingerBridge/docs/620/fog/remote_probe_targetlinear_formal_epoch8/summary.json`

Headline:

- endpoint latent highpass vs source: `1.1115`
- endpoint image gradient vs source: `1.0070`
- endpoint image std vs source: `0.6541`

Selected stage averages:

| stage | img_std | img_grad_rms | latent_high_rms | to_source_img_delta_rms | to_target_img_delta_rms |
| --- | ---: | ---: | ---: | ---: | ---: |
| source | 0.2022 | 0.0828 | 0.5802 | 0.0000 | 0.2950 |
| target | 0.1611 | 0.0777 | 0.5370 | 0.2950 | 0.0000 |
| endpoint | 0.1323 | 0.0834 | 0.6449 | 0.2497 | 0.2359 |
| nfe4 | 0.1831 | 0.0889 | 0.7240 | 0.1685 | 0.2540 |
| nfe8 | 0.1914 | 0.0891 | 0.7338 | 0.1598 | 0.2590 |
| nfe16 | 0.1956 | 0.0892 | 0.7384 | 0.1559 | 0.2617 |

Interpretation:

- compared with epoch 6, the endpoint image-space dynamic range degrades a bit further:
  - epoch 6 endpoint `img_std`: `0.1471`
  - epoch 8 endpoint `img_std`: `0.1323`
- but the solver compensation remains real and strong:
  - endpoint `img_std`: `0.1323`
  - `nfe8` `img_std`: `0.1914`
  - `nfe16` `img_std`: `0.1956`
- that compensation is not a free win:
  - endpoint `to_source_img_delta_rms`: `0.2497`
  - `nfe16` `to_source_img_delta_rms`: `0.1559`
  - endpoint `to_target_img_delta_rms`: `0.2359`
  - `nfe16` `to_target_img_delta_rms`: `0.2617`
  - so the solver restores contrast partly by pulling the image back toward the source decode, not by moving it closer to the target decode
- the endpoint is not simply "blurry everywhere":
  - latent high-frequency energy stays above source
  - image gradient energy stays roughly at or above source
  - what collapses most clearly is low-frequency image dynamic range / contrast

This sharpens the current diagnosis:

1. the original baseline failure mode was fixed:
   - whitening no longer originates from a source-anchored endpoint that survives unchanged through the solver
2. a second late-training failure mode remains:
   - the direct endpoint prediction drifts toward low-contrast image-space statistics
   - multi-step I2SB integration compensates for much of that drift, but partly by re-anchoring toward the source image
3. therefore the current 620 target-linear path is usable only with the solver in the loop;
   `predict_endpoint(t=0)` by itself is still not trustworthy late in training

## Current Status

What is now firmly established:

- we do have a probe-backed root-cause story for the original whitening lane
- `training_target_projection_low_mode = "target_linear"` was a real architectural improvement
- late-training whitening is now a narrower endpoint-consistency problem, not a whole-path collapse

What is still not solved:

- why late optimization keeps improving the training low-frequency target metric while worsening the decoded endpoint contrast
- whether that mismatch should be fixed by:
  - endpoint-specific regularization,
  - training schedule / checkpoint selection,
  - or tighter consistency pressure between `predict_endpoint` and the multi-step solver trajectory

So the next step should remain evidence-first:

- add a probe aimed specifically at the late-training endpoint/solver mismatch
- then make a targeted fix instead of guessing

## New Endpoint-Focused Probe Round

### 2026-06-20 solver trace: the solver is not the primary whitening source

We added:

- `SchrodingerBridge/tools/probe_620_solver_trace.py`

and ran it on the late problematic checkpoint:

- `/mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_targetlinear_swd8_sigma002_nfe8_b64/epoch_0008.pt`

with both:

- `sigma = 0.02`
- `sigma = 0.0`

Key result:

- the traces are nearly identical with and without sigma
- so the solver's behavior here is driven by deterministic mean updates, not noise injection

More importantly, the per-step low-frequency direction is not source-seeking:

- average `step_low_to_source_cos ~= -0.867`
- average `step_low_to_target_cos ~= +0.637`

Interpretation:

- the solver is not the main cause of whitening
- it is actually pushing low-frequency image structure away from the source and toward the target
- the remaining problem is better described as:
  - the direct endpoint prediction is bad at early times
  - the solver partially repairs that bad endpoint with small, more stable updates

So this probe sharpens the blame assignment:

- whitening is now mainly an endpoint pathology
- the solver is a compensator, not the original offender in the current regime

### 2026-06-20 endpoint time sweep: the worst endpoint behavior is concentrated near t=0

We added:

- `SchrodingerBridge/tools/probe_620_endpoint_time_sweep.py`

and ran it on the same late checkpoint.

Observed trend:

- `t = 0.0`:
  - `img_mean ~= 0.9410`
  - `img_std ~= 0.0691`
- `t = 0.5`:
  - `img_mean ~= 0.7926`
  - `img_std ~= 0.1991`
- `t = 0.875`:
  - `img_mean ~= 0.6370`
  - `img_std ~= 0.2746`

Interpretation:

- the endpoint gets much less washed out as `t` increases
- the strongest pathology is the direct source-side jump at `t ~= 0`
- this strongly suggests that late training does not preserve the early-time endpoint regime well enough

This is the most useful new design signal so far:

- if we want to fix whitening, we should target the early-time endpoint regime first
- not blindly change the whole solver path

## Failed Fix Branch

### 2026-06-20 direct source-endpoint auxiliary loss over-corrects and collapses back toward source

We tried a new loss branch:

- `source_endpoint_aux_weight`

with config:

- `SchrodingerBridge/configs/620_spatial_bridge_targetlinear_endpointaux.json`

This branch adds explicit `t=0` source-endpoint supervision during training.

Smoke run:

- checkpoint:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/620_spatial_bridge/620_targetlinear_endpointaux_swd8_sigma002_nfe8_b80/epoch_0001.pt`

Fog probe:

- `/mnt/i/Github/Latent_Style/SchrodingerBridge/docs/620/fog/remote_probe_targetlinear_endpointaux_epoch1/summary.json`

Headline:

- endpoint latent highpass vs source: `1.1117`
- endpoint image gradient vs source: `0.9747`
- endpoint image std vs source: `1.0823`

At first glance this looks acceptable, because the endpoint is no longer washed out.

However the stage details show why this is a bad fix:

- endpoint `to_source_img_delta_rms ~= 0.0546`
- endpoint `to_target_img_delta_rms ~= 0.3073`

Interpretation:

- the endpoint is now extremely close to the source image
- it is farther from the target image than even the earlier healthy smoke lane
- so the whitening did not disappear because the model learned a better target-facing endpoint
- it disappeared because the endpoint largely collapsed back toward source appearance

The endpoint time sweep confirms the same failure mode:

- `t = 0.0`:
  - `img_std ~= 0.3008`
  - `img_to_source_rms ~= 0.0634`
  - `img_to_target_rms ~= 0.3805`

So this branch should be treated as a rejected fix:

- it improves contrast
- but by destroying the intended transfer movement

That is not the repair we want.

## Current Next Move

The stronger current hypothesis is:

- we should increase training pressure on low-`t` samples
- without adding a separate hard source-endpoint loss that can re-anchor the whole endpoint back to source

To test that, the next branch is:

- bias the training-time `t` sampling distribution toward low `t`
- keep the original target-linear objective otherwise unchanged

### 2026-06-20 low-t biased sampling also over-anchors the endpoint toward source

We tested a gentler branch:

- `SchrodingerBridge/configs/620_spatial_bridge_targetlinear_tlow.json`

This branch does not add a separate source-endpoint loss. It only biases the training `t` sampling distribution toward low `t`.

Remote smoke checkpoint:

- `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/620_spatial_bridge/620_targetlinear_tlow_swd8_sigma002_nfe8_b80/epoch_0001.pt`

Fog probe:

- `/mnt/i/Github/Latent_Style/SchrodingerBridge/docs/620/fog/remote_probe_targetlinear_tlow_epoch1/summary.json`

Headline:

- endpoint latent highpass vs source: `1.0327`
- endpoint image gradient vs source: `1.0107`
- endpoint image std vs source: `0.9898`

At first glance this looks balanced:

- not washed out
- not obviously over-sharpened

But the endpoint deltas show the real issue:

- endpoint `to_source_img_delta_rms ~= 0.0498`
- endpoint `to_target_img_delta_rms ~= 0.2899`

Compared with the earlier healthy target-linear smoke lane:

- earlier endpoint `to_source_img_delta_rms ~= 0.0583`
- earlier endpoint `to_target_img_delta_rms ~= 0.3348`

So the low-`t` biased branch still moves in the wrong direction:

- it reduces whitening
- but again does so mainly by shrinking the endpoint back toward source appearance

The endpoint time sweep confirms this:

- `t = 0.0`:
  - `img_to_source_rms ~= 0.0449`
  - `img_to_target_rms ~= 0.3641`
- `t = 0.875`:
  - `img_to_source_rms ~= 0.0061`
  - `img_to_target_rms ~= 0.3787`

Interpretation:

- the entire time sweep stays too close to source
- so merely emphasizing low-`t` training is not enough
- it encourages the model to avoid whitening by reducing endpoint movement

This branch should also be treated as rejected.

## Updated Design Signal

The stronger current read is now:

1. the main pathology is still in the endpoint path
2. naive fixes that emphasize early-time endpoint behavior tend to collapse back toward source
3. the next fix should constrain **endpoint energy / decoded contrast failure directly**
   without rewarding source similarity as an escape route

## Mathematical Hypothesis Loop

We are now moving to a stricter loop:

1. state a mathematical hypothesis
2. derive the corresponding architecture change
3. run a quantitative probe or training experiment
4. update or reject the hypothesis from the result

### H1: whitening is a shrinkage solution in endpoint space

For a source latent `x`, target latent `y`, and predicted endpoint `y_hat`, define:

- `delta = y - x`
- `move = y_hat - x`
- `alpha = <move, delta> / ||delta||^2`

Interpretation:

- `alpha ~= 1`: endpoint reaches the target direction correctly
- `0 < alpha < 1`: endpoint falls inside the source-target interpolation segment
- `alpha ~= 0`: the endpoint barely moves

If whitening is mainly a shrinkage pathology, then late bad checkpoints should show:

- `alpha(t=0)` clearly below `1`
- small or moderate orthogonal residual
- decoded endpoint closer to source than target

This is the mathematically clean version of:

- "the model learned too little movement in the target direction"

### H2: the style path is too weak near early-time endpoint prediction

For a fixed state `x_t`, compare two target style conditions `s1` and `s2`:

- `S_latent = ||y_hat(x_t, t, s1) - y_hat(x_t, t, s2)|| / ||phi(s1) - phi(s2)||`

where `phi` is the DINO style representation used at runtime.

If the style path is too weak, then:

- `S_latent(t=0)` will be small
- endpoint prediction will behave more like a conditional mean over targets than a style-specific endpoint

This is the mathematically clean version of:

- "the model is not letting style meaningfully control the endpoint"

### H3: shrinkage may be band-specific or global

Split each latent into:

- `x = x_low + x_high`
- `y = y_low + y_high`
- `y_hat = y_hat_low + y_hat_high`

and define separate projection coefficients:

- `alpha_low`
- `alpha_high`

If:

- `alpha_high << alpha_low`, then whitening is mostly a high-frequency collapse
- both are small, then the pathology is a global endpoint shrinkage

This is the mathematically clean version of:

- "is the model only losing texture, or is the whole endpoint underpowered?"

## New Quantitative Probe Contract

Added:

- `SchrodingerBridge/tools/probe_620_hypothesis_metrics.py`

This probe writes:

- `hypothesis_metrics.csv`
- `hypothesis_skipped_samples.csv`
- `summary.json`

Per sample and per time `t`, it measures:

- `latent_alpha_mean`
- `latent_shrink_gap = max(0, 1 - alpha)`
- `latent_orth_over_delta`
- `low_alpha_mean`
- `high_alpha_mean`
- `style_sensitivity_latent`
- `style_sensitivity_img`
- endpoint distance to source and target in latent/image space

Primary read order:

1. inspect `t=0` alpha
2. compare `low_alpha` vs `high_alpha`
3. inspect style sensitivity at the same `t`
4. decide whether the redesign should prioritize:
   - endpoint-first parameterization
   - stronger style-controlled trunk routing
   - low/high split heads
   - or a combination

## Immediate Next Experiment

Run the new hypothesis probe on the current late bad checkpoint:

- checkpoint:
  - `/mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_targetlinear_swd8_sigma002_nfe8_b64/epoch_0008.pt`
- launcher:
  - `SchrodingerBridge/tools/experiments/launch_remote_620_hypothesis_probe.py`

We should treat that result as the new decision anchor before touching the next architecture redesign.

## 2026-06-20 Hypothesis Probe Result

We ran:

- `SchrodingerBridge/tools/probe_620_hypothesis_metrics.py`

on:

- `/mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_targetlinear_swd8_sigma002_nfe8_b64/epoch_0008.pt`

Output:

- `/mnt/i/Github/Latent_Style/SchrodingerBridge/docs/620/fog/remote_probe_targetlinear_hypothesis_epoch8/summary.json`

This is the first quantitative result that directly tests the shrinkage hypothesis.

### Main result: H1 is strongly supported

At `t = 0.0`:

- `latent_alpha_mean ~= 0.1633`
- `latent_shrink_gap ~= 0.8367`
- `latent_orth_over_delta ~= 0.4318`

Interpretation:

- the endpoint is moving only about `16%` of the way toward the target direction in latent space
- the dominant pathology is under-travel along the correct target direction
- the error is not mainly random orthogonal drift

This is the strongest quantitative confirmation so far that whitening is an endpoint shrinkage solution.

### Band read: the early-time collapse is especially severe in the high band

At `t = 0.0`:

- `low_alpha_mean ~= 0.4263`
- `high_alpha_mean ~= -0.0501`

Interpretation:

- low-frequency endpoint motion is weak but still target-facing
- high-frequency endpoint motion is effectively absent and slightly wrong-signed on average

This matters because it rules out a pure low-frequency-only explanation.

The current bad endpoint is better described as:

- some low-frequency movement survives
- but the high-frequency endpoint path is almost completely collapsed at early time

That is a more precise mathematical version of the visual "fog / whitening" read.

### Style path read: style is not absent, but it is not converting into correct endpoint travel

At `t = 0.0`:

- `style_sensitivity_latent ~= 8.7509`
- `style_sensitivity_img ~= 4.2075`

Interpretation:

- swapping style DINO conditions does cause substantial endpoint change
- so the failure is not "style has no effect"
- instead, style-conditioned changes are not being converted into enough target-facing endpoint displacement

This is an important correction to a weaker earlier story.

The better read is:

- style enters the network
- but the endpoint parameterization and optimization still prefer a shrinkage solution

### Time trend

As `t` increases:

- `latent_alpha_mean` rises from `0.1633` at `t=0.0`
- to `0.5607` at `t=0.5`
- to `0.9036` at `t=0.875`

and:

- `high_alpha_mean` rises from `-0.0501`
- to `0.4985`
- to `0.8927`

Interpretation:

- the model can represent the target-facing endpoint much better once the state is already closer to target
- the hardest source-side endpoint regime remains the main failure zone

This quantitatively matches the earlier endpoint time sweep.

## Updated Design Consequence

The new result changes the redesign target in a useful way.

We should no longer summarize the problem as:

- "style conditioning is too weak"

That is incomplete.

The sharper version is:

1. style conditioning is live
2. early-time endpoint transport is strongly shrunk
3. the shrinkage is especially catastrophic in the high band
4. therefore the next architecture change should not merely amplify style sensitivity
5. it should explicitly make the network predict a target-facing endpoint, especially high-band endpoint structure, instead of allowing it to hide behind a low-amplitude velocity solution

So the next architecture hypothesis should be:

- endpoint-first parameterization
- explicit low/high endpoint heads
- velocity derived from endpoint, not learned as the primary object

That is now the best mathematically supported next move.

## 2026-06-20 Endpoint-First Low/High Smoke Result

We implemented a minimal endpoint-first branch:

- config:
  - `SchrodingerBridge/configs/620_spatial_bridge_targetlinear_endpointlowhigh.json`
- checkpoint:
  - `/mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_targetlinear_endpointlowhigh_swd8_sigma002_nfe8_b80_smoke/epoch_0001.pt`

and probed it with:

- `/mnt/i/Github/Latent_Style/SchrodingerBridge/docs/620/fog/remote_probe_targetlinear_endpointlowhigh_epoch1/summary.json`

This branch did **not** improve the pathology in the way we wanted.

### Headline comparison against the old bad checkpoint

Old bad target-linear checkpoint at `t = 0.0`:

- `latent_alpha_mean ~= 0.1633`
- `high_alpha_mean ~= -0.0501`
- `style_sensitivity_latent ~= 8.7509`
- `endpoint_img_to_source_rms ~= 0.2446`
- `endpoint_img_to_target_rms ~= 0.2468`
- `endpoint_img_std ~= 0.1345`

New endpoint-lowhigh smoke at `t = 0.0`:

- `latent_alpha_mean ~= -0.0404`
- `high_alpha_mean ~= -0.0472`
- `style_sensitivity_latent ~= 0.00285`
- `endpoint_img_to_source_rms ~= 0.0269`
- `endpoint_img_to_target_rms ~= 0.3272`
- `endpoint_img_std ~= 0.2129`

### Interpretation

This branch changed the failure mode, but not in the desired direction.

What improved:

- decoded contrast is higher
- `endpoint_img_std` increased
- orthogonal residual became very small

What got worse:

- `latent_alpha_mean` moved from weakly positive to slightly negative at `t=0`
- `high_alpha_mean` stayed negative
- style sensitivity collapsed by several orders of magnitude
- endpoint image became much closer to source and much farther from target

So this is **not** a real fix.

It removes fog mostly by collapsing the endpoint back toward source appearance, while also nearly disabling style-conditioned movement.

### Updated design consequence

This is a useful rejection.

It tells us the first endpoint-first implementation was too unconstrained:

- explicit endpoint heads alone are not enough
- low/high decomposition alone is not enough
- if the style path is not coupled tightly enough into those heads, the model takes the easiest available solution:
  - predict a source-like endpoint with slightly sharper statistics
  - keep style sensitivity near zero

So the next redesign should **not** be "endpoint-first low/high" in this bare form.

The next mathematically motivated revision should require both:

1. endpoint-first parameterization
2. explicit target-facing style actuation inside the endpoint heads

In practical terms, the next branch should likely add one of:

- style-conditioned FiLM or AdaLN directly inside the low/high endpoint heads
- a style-actuated delta basis for endpoint high-band prediction
- or a residual endpoint parameterization that predicts target-facing movement relative to the known source decomposition, instead of free unconstrained endpoint offsets

This branch should be treated as rejected in its current form.

## 2026-06-20 Style-Actuated Endpoint-Head Smoke Result

We then tested a stronger but still minimal branch:

- config:
  - `SchrodingerBridge/configs/620_spatial_bridge_targetlinear_endpointstylehead.json`
- checkpoint:
  - `/mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_targetlinear_endpointstylehead_swd8_sigma002_nfe8_b80_smoke/epoch_0001.pt`
- probe:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/docs/620/fog/remote_probe_targetlinear_endpointstylehead_epoch1/summary.json`

This version injects the style global token directly into the low/high endpoint heads.

### Headline comparison at `t = 0.0`

Bare endpoint-lowhigh smoke:

- `latent_alpha_mean ~= -0.0404`
- `high_alpha_mean ~= -0.0472`
- `style_sensitivity_latent ~= 0.00285`
- `endpoint_img_to_source_rms ~= 0.0269`
- `endpoint_img_to_target_rms ~= 0.3272`

Style-actuated endpoint-head smoke:

- `latent_alpha_mean ~= -0.0406`
- `high_alpha_mean ~= -0.0481`
- `style_sensitivity_latent ~= 0.2285`
- `endpoint_img_to_source_rms ~= 0.0284`
- `endpoint_img_to_target_rms ~= 0.3267`

### Interpretation

This branch proves one useful point:

- direct style injection into the endpoint heads does revive style sensitivity somewhat

But it is still far from enough:

- early-time `alpha` remains negative
- early-time `high_alpha` remains negative
- endpoint remains extremely close to source
- target-facing movement still does not recover

So this branch is also rejected as a real fix.

The strongest current read is now:

1. style must act inside the endpoint head pathway
2. simple additive style offsets at the endpoint output are too weak
3. the next branch should let style modulate the endpoint-head feature maps themselves

That points to a stronger head-local conditioning mechanism such as:

- style-conditioned FiLM / AdaLN on the endpoint head trunk
- or a style-actuated endpoint delta basis

## 2026-06-20 FiLM Formal Result

We ran the stronger FiLM endpoint-head branch formally on remote 3060 WSL:

- run:
  - `/mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_film_formal`
- eval checkpoint used for diagnosis:
  - `epoch_0005.pt`
- remote probes:
  - `remote_probe_film_formal_hypothesis_epoch5`
  - `remote_probe_film_formal_fog_epoch5`

### Formal eval summary

From the remote `full_eval/epoch_0005/summary.json`:

- `style_transfer clip_style ~= 0.6723`
- `style_transfer clip_t ~= 0.2056`
- `style_transfer content_lpips ~= 0.2915`
- `style_transfer clip_s_delta_idt ~= 0.0324`

So this branch is not strong enough to justify moving on to broader roadmap work yet.

### Probe interpretation

At `t = 0.0`:

- `latent_alpha_mean ~= 0.1232`
- `low_alpha_mean ~= 0.2222`
- `high_alpha_mean ~= -0.0326`
- `style_sensitivity_latent ~= 10.1141`
- `endpoint_img_to_source_rms ~= 0.1978`
- `endpoint_img_to_target_rms ~= 0.3019`

Fog-path headline:

- `endpoint_latent_high_vs_source_ratio ~= 1.0774`
- `endpoint_img_grad_vs_source_ratio ~= 1.0415`
- `endpoint_img_std_vs_source_ratio ~= 0.9320`

Interpretation:

- FiLM does **not** look like a dead style path
- and it does **not** look like the endpoint is simply losing all contrast
- instead, it is a clearer example of the deeper failure:
  - style sensitivity is real
  - endpoint statistics are not especially fog-starved
  - but the endpoint is still not traveling far enough in the correct target-facing direction

So this branch is rejected as a whitening fix.

It is better summarized as:

- "not foggy enough to explain the whole failure"
- but still "not target-facing enough to solve transfer"

## 2026-06-20 Direction-Loss Smoke Result

We then tested a probe-aligned directional loss branch:

- config:
  - `SchrodingerBridge/configs/620_spatial_bridge_targetlinear_direction.json`
- smoke checkpoint:
  - `/mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_targetlinear_direction_swd8_sigma002_nfe8_b16_smoke/epoch_0001.pt`
- probe:
  - `remote_probe_targetlinear_direction_epoch1`

Headline at `t = 0.0`:

- `latent_alpha_mean ~= -0.0066`
- `low_alpha_mean ~= -0.0125`
- `high_alpha_mean ~= 0.0002`
- `style_sensitivity_latent ~= 0.0020`
- `endpoint_img_to_source_rms ~= 0.0209`
- `endpoint_img_to_target_rms ~= 0.3676`

Interpretation:

- this smoke is too short to act as a final verdict on the loss idea,
- but its current state is clearly unusable:
  - endpoint is almost snapped back to source
  - style sensitivity has nearly vanished

So the current directional-loss smoke should **not** be treated as evidence of success.

The correct operational read is:

- the 4-step smoke is inconclusive as a long-run training verdict,
- but the present implementation can already collapse into source-like motion with near-zero style actuation,
- so it should not be promoted without a fuller rerun and stronger representation constraints.

## DINO Status For This Phase

Current evidence does **not** justify deleting DINO yet.

Why:

- the baseline bad branch still shows large `style_sensitivity_latent`
- the FiLM branch also shows strong style sensitivity
- therefore the style-conditioned DINO path is alive

What current evidence does justify:

- DINO does not automatically solve whitening
- endpoint parameterization can waste the style signal and still land in a shrinkage solution

So the present stance is:

1. do **not** assume DINO is the root cause
2. do **not** keep DINO by default forever
3. after the whitening-focused endpoint redesign is tested, run an explicit keep-vs-cut comparison:
   - `target_dino_patches`
   - stronger latent-only / intrinsic alternative
4. if DINO does not provide a clear improvement in:
   - WFI
   - `clip_t`
   - `clip_s_delta_idt`
   - or probe-side target-facing endpoint motion

then it should be cut decisively

## 2026-06-21: 基于数学理论的重设计

### 理论诊断

经过完整的数学分析循环（数学思考→实验→修正理论），达成以下结论：

**SWD梯度平坦性**（已被推翻为主要原因）：
- 梯度探针验证：$\nabla \mathrm{SWD}(v=0) = 0.044$，非零
- 但 $\cos(\nabla \mathrm{SWD}, v_{\mathrm{target}}) = -0.024$（基本正交）
- Loss landscape 极小值在 $s=1.0$（正确解），不在 $s=0.16$

**修正后的根因：条件期望坍缩**：
- gate=0.05 → style 信号太弱 → 模型学到 $E[v_{\mathrm{target}} | x, t]$ 而非 $v_{\mathrm{target}}(x, t, \mathrm{style})$
- 不同 style 的 velocity 方向互相抵消，边际 velocity 幅度远小于 style-specific velocity
- 形成"平凡解"吸引子：gate 小 → style 梯度小 → gate 不增长 → 正反馈锁死

### 理论文档

| 文档 | 路径 | 内容 |
|------|------|------|
| SWD梯度平坦性定理 | `theory/swd_flatness.md` | SWD梯度的分段常数性质，排序稳定性定理 |
| 平凡解收敛性分析 | `theory/trivial_solution.md` | 为什么模型收敛到shrinkage解 |
| SB动力学分析 | `theory/sb_dynamics.md` | 条件期望坍缩、style信号强度、相图分析 |
| NSWD理论 | `theory/noisy_swd.md` | 噪声打破排序稳定性的数学原理 |

### 修复方案（2026-06-21实施）

| 优先级 | 修复 | 原理 | 实现 |
|--------|------|------|------|
| P0 | gate=0.3 | 增强style信号，打破条件期望坍缩 | model620.py: gate_init=0.3 |
| P0 | 大容量endpoint head (3层, 无GroupNorm) | 表达能力+无动态范围压缩 | model620.py: 3层Conv+SiLU |
| P1 | NSWD (σ=0.02) | 打破SWD梯度正交性 | losses620.py: noise_sigma |

### Smoke Test 结果 (gate=0.3, epoch=1)

**修复前后对比**：

| 指标 | gate=0.05 (旧) | gate=0.3 (新) | 变化 |
|------|---------------|---------------|------|
| gate_value | 0.064 | **0.297** | **+365%** |
| velocity_abs | 0.186 | **0.216** | **+16%** |
| endpoint_pred_abs | 0.686 | 0.629 | -8% |
| all_pairs clip_style | 0.700 | 0.696 | -1% |
| clip_s_delta_idt | 0.060 | 0.056 | -7% |
| content_lpips | 0.272 | 0.293 | +8% |
| identity clip_style | 0.841 | 0.836 | -1% |

**结论**：gate修复成功（gate值确认在0.3），velocity magnitude提升（shrinkage减少）。但1 epoch内style transfer质量未改善——符合理论预测：模型从phase 1（边际velocity）过渡到phase 2（style-specific velocity）需要更多训练。5-epoch训练已启动。

**关键发现**：
- `tswd=0.0000` 是占位符（losses620.py中terminal_swd恒为zero），不是bug
- Base config中的 `style_cross_attn_gate_init: 0.05` 覆盖了代码默认值0.3——这是之前修复无效的根本原因
- 已修复 `create_whitening_fix_configs.py` 在生成配置时显式设置gate=0.3

## 2026-06-21: 探针揭示深层问题 + StyleFiLM实现

### Velocity Direction 探针结果

在gate=0.3修复后，alpha从0.16提升到0.605（+278%），但风格迁移质量仍未改善。深入探针发现：

- **cos_sim(v(style_1), v(style_2)) = 0.9995** — 模型输出几乎相同的velocity方向，无论style输入是什么
- 问题不是velocity magnitude太小，而是velocity DIRECTION不随style变化

### 根因：Cross-Attention平均化瓶颈

Cross-attention的数学形式：
$$\text{CA}(x, S) = \text{softmax}(Q(x) \cdot K(S)^T / \sqrt{d}) \cdot V(S)$$

$Q(x)$ 由content决定，对256个style tokens产生相似的softmax分布。$V(S)$ 的加权和在不同style之间几乎不变，导致模型学到的是对style的边缘期望而非条件于特定style的velocity。

### StyleFiLM：绕过Cross-Attention的Style注入

**数学定义**：
$$\text{FiLM}(x; s) = (1 + \gamma(s)) \odot x + \beta(s)$$

其中 $\gamma(s), \beta(s) \in \mathbb{R}^C$ 由MLP从style_global预测，提供通道级特征调制。

**关键设计**：
- 在每个SpatialBridgeBlock620中，cross-attention shortcut之后、FFN之前插入FiLM
- Zero-init确保训练开始时FiLM=identity
- Per-block独立FiLM，不同深度学习不同层次的style调制
- 直接梯度路径：$\partial\mathcal{L}/\partial\theta_{\text{style}} \propto \partial\mathcal{L}/\partial v \cdot \partial\gamma/\partial\theta_{\text{style}}$（绕过softmax）

**理论文档**：`theory/style_film.md`

### 代码改动

| 文件 | 改动 |
|------|------|
| `blocks620.py` | 添加 `film_enabled` 参数、FiLM投影层、forward接受 `style_global` |
| `model620.py` | 读取 `style_film_enabled` 配置，传递 `style_global` 到blocks |
| `config_schema.py` | 添加 `style_film_enabled: bool = False` |
| `create_whitening_fix_configs.py` | 添加 `620_film_gate03_smoke` 变体 |

### 预期验证指标

- `film_gamma_abs` / `film_beta_abs`：从0开始增长（FiLM被激活）
- cos_sim(v1, v2)：目标从0.9995降到<0.95
- WFI：保持不恶化
- 监控 `film_enabled` debug字段
