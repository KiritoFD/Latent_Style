# Distinct5-512 LANCET Representation and Speed Plan

## Scope

Dataset source is fixed to `F:\wikiart_distinct5_samam_512`.

Only these folders are used:

- `train_flat/style`
- `test_flat/style`

Classes are fixed:

- `Early_Renaissance`
- `Impressionism`
- `Minimalism`
- `Rococo`
- `Ukiyo_e`

No training-time external semantic supervision is allowed. DINO, CLIP loss, classifier loss, and external semantic feature matching are excluded. CLIP and LPIPS remain evaluation-only metrics.

## Prepared Paths

Class-folder image view:

- Windows: `F:\wikiart_distinct5_samam_512_classview\train\<style>\*.jpg`
- Windows: `F:\wikiart_distinct5_samam_512_classview\test\<style>\*.jpg`
- WSL: `/mnt/f/wikiart_distinct5_samam_512_classview/train/<style>/*.jpg`
- WSL: `/mnt/f/wikiart_distinct5_samam_512_classview/test/<style>/*.jpg`

EMA latent output:

- Windows: `F:\wikiart_distinct5_samam_512_latents_ema\train`
- Windows: `F:\wikiart_distinct5_samam_512_latents_ema\test`
- WSL: `/mnt/f/wikiart_distinct5_samam_512_latents_ema/train`
- WSL: `/mnt/f/wikiart_distinct5_samam_512_latents_ema/test`

Packed train cache:

- `/mnt/f/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/manifest.json`
- `/mnt/f/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/packed/*.pt`

Prototype-aware pairing cache for Variant E:

- `/mnt/f/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/prototype_pairing_top8.pt`

Environment note:

- Current WSL distro name: `f`
- WSL repo path: `/mnt/g/GitHub/Latent_Style`
- Current local WSL check on 2026-06-02: system `python3` imports a broken `torch` package without `torch.utils.data`; do not use it for LANCET training until the local WSL Python environment is repaired or a known-good venv is selected.
- Windows Python `C:\Users\xy\AppData\Local\Programs\Python\Python312\python.exe` has `torch 2.11.0+cu128` and CUDA on the local RTX 4070 Laptop GPU, and was used for the local Variant M smoke below.
- Windows Python should not be used for reading the original flat symlink/reparse entries under `F:\wikiart_distinct5_samam_512\train_flat\style`. Use the generated classview/latent paths above.

## Code Entry Points

Data tools:

- `tools/prepare_distinct5_classview.py`
- `tools/encode_image_folder_latents.py`
- `tools/build_latent_packed_cache.py`
- `tools/build_latent_prototype_pairing_cache.py`

Training:

- `src/run.py`

Evaluation:

- `src/utils/run_evaluation.py`

## Configs

Baseline:

- `configs/distinct5_512_ema_baseline_direct_atom_residual.json`

Representation variants:

- `configs/distinct5_512_ema_variant_a_class_prototypes.json`
- `configs/distinct5_512_ema_variant_b_global_vq.json`
- `configs/distinct5_512_ema_variant_c_content_guided_spatial.json`
- `configs/distinct5_512_ema_variant_d_vq_content_guided.json`
- `configs/distinct5_512_ema_variant_e_latent_prototype_ot_queue.json`
- `configs/distinct5_512_ema_variant_f_annealed_prototype_ot_queue_e3.json`
- `configs/distinct5_512_ema_variant_g_stratified_prototype_ot_queue_e3.json`
- `configs/distinct5_512_ema_variant_h_hard_explore_queue_e3.json`
- `configs/distinct5_512_ema_variant_i_dual_target_mix_queue_e3.json`
- `configs/distinct5_512_ema_variant_j_aux_hard_swd_queue_e3.json`
- `configs/distinct5_512_ema_variant_k_content_adaptive_vq_queue_e3.json`
- `configs/distinct5_512_ema_variant_l_content_adaptive_annealed_queue_e3.json`
- `configs/distinct5_512_ema_variant_m_style_gated_content_router_e3.json`

All formal configs use:

- EMA VAE
- `virtual_length_multiplier=1.0`
- packed latent cache
- checkpoint every epoch
- full eval every epoch
- deferred full eval after training process releases GPU memory
- `only_lpips_clip_style=true`

## Model Variants

Baseline keeps the current `direct_atom_residual` tokenizer: per-style direct code plus shared atom residual.

Variant A uses per-class prototypes. Each style learns K class-local style vectors and a mixture over them. Spatial prior can follow the same mixture.

Variant B uses a global VQ-style codebook. All styles share atoms; each class learns a sparse distribution over shared atoms.

Variant C keeps class prototypes but makes the 16x16 spatial prior content-guided. Content latent features adjust prototype-map weights per image.

Variant D combines global VQ atoms and content-guided spatial routing. This is the main representation candidate.

Variant E adds prototype-aware latent target pairing. It builds target queues from VAE latent statistics, gradients, and frequency amplitudes only; the normal LANCET/SWD training objective then consumes the selected target latents.

Variant F keeps Variant E's tokenizer and prototype-aware latent target queue, but changes target sampling into a short curriculum. It uses rank-biased sampling over the pairing cache and anneals the active target top-k from easy targets to the full top-8 set over three epochs. This is still an internal VAE-latent queue policy; it does not add any external semantic supervision.

Variant G keeps Variant F's queue curriculum but replaces stochastic target-rank sampling with deterministic rank-stratified sampling. The goal is to reduce per-batch target-hardness jitter while keeping the same internal VAE-latent queue and active top-k schedule.

Variant H keeps the same global VQ plus content-guided tokenizer and the same prototype-aware latent target queue, but freezes the clean active target set at top-2 and injects 15% hard-rank exploration from top-8. The goal is to preserve Variant F's clean-target LPIPS behavior while adding a small stochastic hard-target pressure for style.

Variant I tests whether hard-target pressure can be made smoother than H. It keeps the clean top-2 target as the anchor and mixes in a top-8 hard target directly in VAE latent space with alpha 0.20. This is a target-shaping ablation, not a new external supervision signal.

Variant J keeps the clean top-2 main target and adds a separate auxiliary hard-target terminal SWD term. It tests whether hard-target pressure should be expressed as a second objective rather than as sampler replacement.

Variant K keeps Variant H's fixed clean top-2 plus sparse hard-explore sampler, but makes the global VQ style-code atom logits content-adaptive. A small zero-initialized router reads only internal VAE-latent content statistics and adds a per-image residual to the class atom logits before the atom mixture is converted to `style_code`.

Variant L keeps Variant F's easy-to-hard queue curriculum and adds the same content-adaptive global VQ atom-logit router as K. It tests whether K's style gain can be combined with F's LPIPS-preserving target schedule.

Variant M keeps Variant K's fixed clean top-2 plus sparse hard-explore sampler, and adds a target-style scalar gate on the content-adaptive VQ atom-logit router. This is motivated by the K/L matrix diagnosis: target `Minimalism` pays a large LPIPS cost from content-adaptive routing while gaining little style. The gate is initialized to 1.0, so the initial behavior matches K, and each target style can learn to reduce or preserve router strength.

Variants A/B/C/D/E/F/G/H/I/J/K/L/M now use tokenizer latent-stat initialization. At trainer construction time, the tokenizer reads the packed EMA latent cache, extracts only VAE-latent internal statistics, and initializes:

- class-local prototype tables for A/C,
- shared VQ atom tables and style atom logits for B/D/E/F/G/H/I/J/K/L/M.

This is not a new supervision signal. It is an initialization prior from the same training latents already consumed by LANCET, and it is disabled for the baseline so the baseline remains the current `direct_atom_residual` path.

## Profiling

Training logs include existing timing:

- `data_time_sec`
- `forward_time_sec`
- `backward_time_sec`
- `optimizer_time_sec`
- `compute_time_sec`
- `epoch_time_sec`

When `training.profile_modules=true`, loss metrics also include:

- `profile_tokenizer_sec`
- `profile_backbone_forward_sec`
- `profile_execution_budget_sec`
- `profile_diffeomorphic_stroke_sec` when the stroke path is enabled
- `profile_model_forward_sec`
- `profile_ot_match_sec`
- `profile_aux_loss_sec`
- `profile_terminal_swd_sec`

Evaluation uses `--profile_timing` for generation, VAE encode/decode, PNG save, LPIPS, and CLIP timing breakdown.

## Acceptance

Any variant is worth retaining if either:

- `clip_style` improves over the Distinct5 baseline, or
- `content_lpips` decreases over the Distinct5 baseline.

Both do not need to improve simultaneously.

The final report should include:

- 8-epoch curve for baseline and retained variants
- all 5x5 `clip_style`
- all 5x5 `content_lpips`
- train epoch time and sec/step
- generation sec/img
- full eval wall time
- module-level profile table
- failed-variant reason

## Current Status

As of 2026-06-02:

- Class-folder view has been generated in WSL.
- EMA train/test encoding is complete.
- Counts are verified:
  - Train image view: 1000 images per class, 5000 total.
  - Test image view: 30 images per class, 150 total.
  - Train latents: 1000 latents per class, 5000 total.
  - Test latents: 30 latents per class, 150 total.
- Packed latent cache is complete for train and test.
- Variant E prototype-aware latent pairing cache is complete:
  - 8 prototypes per class.
  - top-8 target routes.
  - 20000 routes total.
- Variant F annealed pairing cache sampling is implemented:
  - sample mode: `rank_biased`.
  - rank schedule: `easy_to_hard`.
  - active top-k schedule observed on remote: 2 -> 5 -> 8.
  - curriculum length: 3 epochs.
- Variant G rank-stratified pairing cache sampling is implemented:
  - sample mode: `rank_biased_stratified`.
  - rank schedule: `easy_to_hard`.
  - active top-k schedule observed on remote: 2 -> 5 -> 8.
  - curriculum length: 3 epochs.
- Variant H hard-explore pairing cache sampling is implemented:
  - sample mode: `rank_biased`.
  - rank schedule: `fixed`.
  - active top-k: 2.
  - hard exploration: 15% probability from top-8.
  - curriculum length: 0 epochs.
- Variant I dual-target latent mix is implemented:
  - sample mode: `rank_biased`.
  - rank schedule: `fixed`.
  - active top-k: 2.
  - hard replacement: disabled.
  - dual target mix: alpha 0.20 from top-8 hard target.
  - curriculum length: 0 epochs.
- Model construction and CPU finite shape smoke passed for baseline and variants A-E.
- WSL CUDA train smoke passed for the baseline:
  - Config: `_codex_tmp/distinct5_smoke_baseline_vlen004.json`
  - Batch: 16
  - Steps: 12
  - Checkpoint: `exp/_smoke_distinct5_512_ema_baseline_vlen004/epoch_0001.pt`
  - Peak reserved VRAM: 3.30 GB
  - Epoch wall: 10.5 s
- 25-image eval smoke passed:
  - Output: `exp/_smoke_distinct5_512_ema_baseline_vlen004/full_eval/epoch_0001_smoke25_timing`
  - PNG count: 25
  - CLIP backend: HF local cache
  - LPIPS loaded and executed
  - Smoke-only all-pairs `clip_style=0.6661`, `content_lpips=0.3836`
  - Timing fields now persist in `summary.json`: `load_lancet`, `load_vae`, `encode_inversion`, `lancet_generation`, `vae_decode`, `uint8_cpu_copy`, `image_save_submit`, `eval_metrics_loop`, `eval_total`, `wall_total`.
  - Smoke wall time: 12.15 s for 25 generated images plus metrics; this is not a formal throughput number.
- Variant train smoke passed for A-E at batch 8 and `virtual_length_multiplier=0.01`.
- Tokenizer latent-stat initialization smoke passed:
  - Variant A: `_codex_tmp/distinct5_smoke_variant_a_latent_init.json`, batch 4, `virtual_length_multiplier=0.002`, checkpoint `exp/_smoke_distinct5_variant_a_latent_init/epoch_0001.pt`, peak reserved 0.88 GB.
  - Variant B: `_codex_tmp/distinct5_smoke_variant_b_latent_init.json`, batch 4, `virtual_length_multiplier=0.002`, checkpoint `exp/_smoke_distinct5_variant_b_latent_init/epoch_0001.pt`, peak reserved 0.90 GB.
- Module profiler smoke verified these keys in training metrics: `profile_tokenizer_sec`, `profile_backbone_forward_sec`, `profile_execution_budget_sec`, `profile_model_forward_sec`, `profile_aux_loss_sec`, `profile_terminal_swd_sec`.
- Fixed a tokenizer compatibility bug in `src/trainer.py`: `channels_last` conversion now touches only rank-4 tensors, so rank-3 prototype/codebook parameters remain valid under PyTorch 2.11.
- Remote 3060 status:
  - Distinct5 data and VAE eval cache have been copied to `/mnt/i`.
  - Remote py_compile passed after code sync.
  - Batch ladder result:
    - b36 was valid as a smoke/calibration run only; total `nvidia-smi` memory was about 7.9 GB, below the formal 9 GB floor.
    - b40 was also below the formal floor at about 8.7 GB total memory.
    - b44 is the current formal remote batch; total `nvidia-smi` memory is about 9.55-9.60 GB and peak reserved memory is about 9.10 GB.
  - First wait script incorrectly used `query-compute-apps`, which returns 0 under this WSL setup; it launched b48 while SaMAM still occupied about 7.5 GB and OOMed. The wait script now gates on total `memory.used < 1500 MiB`.
  - Full eval is deferred until training ends, so the evaluator does not load a second LANCET/VAE while the training model and optimizer still occupy GPU memory.
  - The active SaMAM segmented process was found to have `MAX_STEPS=5000`; it was stopped before launching the formal LANCET run because it would otherwise block the LANCET queue indefinitely.

## Remote Formal Baseline: b44 Direct Atom Residual

Run:

- Config: `_codex_tmp/remote_distinct5_baseline_b44_full.json`
- Output: `exp/distinct5_512_ema_baseline_direct_atom_residual_b44_remote`
- Batch: 44
- Epochs: 8
- Checkpoints: `epoch_0001.pt` through `epoch_0008.pt`
- Eval: deferred full eval for every checkpoint, 750 all-pairs images per checkpoint.

Training speed:

- Epoch 1 wall: 70.77 s, including first-step/DataLoader warmup.
- Epoch 2-8 wall: about 62.2 s/epoch.
- Effective training throughput: about 79.9 samples/s after warmup.
- Total train-only wall: about 8.4 min.
- Peak training memory: 8.77 GB allocated, 9.08-9.10 GB reserved; `nvidia-smi` total memory was about 9.55-9.60 GB.

Eval speed:

- First full eval wall: 150.6 s, including global reference feature cache miss.
- Later full eval metric loop: about 24 s after reference cache hit, plus generation/PNG save/load overhead.
- Full deferred eval for 8 checkpoints ran from 04:45:18 to 05:04:47 remote time, about 19.5 min.
- Train plus all 8 full evals took about 28 min wall.

750-image all-pairs results:

| Epoch | clip_style | content_lpips |
|---|---:|---:|
| 1 | 0.686958 | 0.446756 |
| 2 | 0.686081 | 0.457773 |
| 3 | 0.686083 | 0.457186 |
| 4 | 0.686925 | 0.452478 |
| 5 | 0.685235 | 0.456191 |
| 6 | 0.684219 | 0.461261 |
| 7 | 0.685796 | 0.459917 |
| 8 | 0.687649 | 0.452743 |

Conclusion:

- Best `clip_style` is epoch 8 at `0.687649`.
- Best `content_lpips` is epoch 1 at `0.446756`.
- This direct-atom-residual baseline is a weak negative baseline on Distinct5. It does not approach the historical strong 512 result and should not be treated as a successful representation design.
- Next active run is Variant A, per-class prototype tokenizer with VAE-latent-stat initialization, using the same b44 remote setup.

## Remote Variant A: Per-Class Prototypes

Run:

- Config: `_codex_tmp/remote_distinct5_variant_a_b44_full.json`
- Output: `exp/distinct5_512_ema_variant_a_class_prototypes_b44_remote`
- Batch: 44
- Epochs: 8
- Tokenizer: class-local 4-prototype mixture with VAE-latent-stat initialization.

Training:

- `Initialized class_prototypes tokenizer from VAE latent statistics.`
- Model params: 4,567,517.
- Peak training memory: 8.80 GB allocated, 9.08-9.13 GB reserved; `nvidia-smi` total memory about 9.58-9.64 GB.
- Epoch speed is effectively unchanged from the baseline, about 62 s/epoch after warmup.

750-image all-pairs results:

| Epoch | clip_style | content_lpips |
|---|---:|---:|
| 1 | 0.681630 | 0.446381 |
| 2 | 0.680428 | 0.476418 |
| 3 | 0.681846 | 0.473576 |
| 4 | 0.682672 | 0.466868 |
| 5 | 0.681248 | 0.473303 |
| 6 | 0.680320 | 0.473405 |
| 7 | 0.683380 | 0.472045 |
| 8 | 0.684946 | 0.462296 |

Conclusion:

- Best `clip_style` is epoch 8 at `0.684946`, below the direct-atom baseline.
- Best `content_lpips` is epoch 1 at `0.446381`, only `0.000375` lower than the baseline best. This is too small to count as a meaningful representation gain.
- Failure reason: class-local prototypes add capacity but do not create a better shared style geometry. They likely partition each class into weak local means instead of learning transferable style atoms.
- Next active run is Variant B, global VQ codebook with VAE-latent-stat initialization.

## Remote Variant B: Global VQ Codebook

Run:

- Config: `_codex_tmp/remote_distinct5_variant_b_b44_full.json`
- Output: `exp/distinct5_512_ema_variant_b_global_vq_b44_remote`
- Batch: 44
- Epochs: 8
- Tokenizer: shared 64-atom global VQ-style codebook with top-8 mixture and VAE-latent-stat initialization.

Training:

- `Initialized global_vq tokenizer from VAE latent statistics.`
- Model params: 6,016,949.
- Peak training memory: 8.80 GB allocated, 9.11-9.15 GB reserved; `nvidia-smi` total memory about 9.62-9.66 GB.
- The first Variant B attempt was invalid because a SaMAM scheduled task restarted and occupied about 7.5 GB. SaMAM Distinct5 scheduled tasks were then disabled and Variant B was rerun from scratch.

750-image all-pairs results:

| Epoch | clip_style | content_lpips |
|---|---:|---:|
| 1 | 0.677212 | 0.468214 |
| 2 | 0.679746 | 0.460498 |
| 3 | 0.678800 | 0.457317 |
| 4 | 0.682498 | 0.446238 |
| 5 | 0.683059 | 0.447180 |
| 6 | 0.682828 | 0.452562 |
| 7 | 0.685666 | 0.447441 |
| 8 | 0.687321 | 0.444600 |

Conclusion:

- Best `clip_style` is epoch 8 at `0.687321`, slightly below the direct-atom baseline `0.687649`.
- Best `content_lpips` is epoch 8 at `0.444600`, about `0.00216` lower than the baseline best.
- Retention decision: weak retain by the OR gate for LPIPS only. It does not solve the style representation problem, but the shared VQ geometry is less damaging to content than class-local prototypes.
- Next active run is Variant C, class prototypes plus content-guided spatial routing.

## Remote Variant C: Content-Guided Spatial Routing

Run:

- Config: `_codex_tmp/remote_distinct5_variant_c_b44_full.json`
- Output: `exp/distinct5_512_ema_variant_c_content_guided_spatial_b44_remote`
- Batch: 44
- Epochs: 8
- Tokenizer: class-local prototypes plus content-guided spatial routing.

Training:

- `Initialized class_prototypes tokenizer from VAE latent statistics.`
- Model params: 4,618,275.
- Peak training memory: 8.87 GB allocated, 9.15-9.17 GB reserved; `nvidia-smi` total memory about 9.67-9.69 GB.
- Epoch 1 wall was about 70 s with warmup. Epoch 2-8 compute time was about 59.7-59.8 s/epoch.
- Training completed cleanly and saved `epoch_0001.pt` through `epoch_0008.pt`.

Eval note:

- A separate SaMAM scheduled task restarted during the original deferred eval after `epoch_0004`, occupying about 7.6 GB and interrupting the LANCET eval path.
- `epoch_0001` through `epoch_0004` were completed before this contamination. The partial `epoch_0005` eval directory was removed, SaMAM tasks were disabled again, and `epoch_0005` through `epoch_0008` were rerun with `_codex_tmp/remote_distinct5_variant_c_eval_5_8.sh`.
- The final table below uses clean summaries for all eight checkpoints.

750-image all-pairs results:

| Epoch | clip_style | content_lpips |
|---|---:|---:|
| 1 | 0.690336 | 0.432616 |
| 2 | 0.690659 | 0.422593 |
| 3 | 0.687908 | 0.428443 |
| 4 | 0.686210 | 0.429728 |
| 5 | 0.685986 | 0.441034 |
| 6 | 0.688118 | 0.447314 |
| 7 | 0.689421 | 0.449902 |
| 8 | 0.689030 | 0.445063 |

Conclusion:

- Best `clip_style` is epoch 2 at `0.690659`, higher than direct-atom baseline `0.687649`.
- Best `content_lpips` is epoch 2 at `0.422593`, substantially lower than direct-atom baseline `0.446756`.
- Retention decision: strong retain. This is the first Distinct5 tokenizer variant that improves both style and LPIPS under the OR gate and also under the stricter AND comparison.
- Design conclusion: content-conditioned spatial routing is useful; class-local prototypes alone were not. The useful signal appears to come from letting content features route spatial style priors, not from increasing class token capacity.
- Training longer is not useful for this variant: the best point is early (`epoch_0002`), and later epochs mostly trade LPIPS away without a style breakthrough.
- Next active run is Variant D, global VQ atoms plus content-guided spatial routing.

## Remote Variant D: Global VQ + Content-Guided Spatial Routing

Run:

- Config: `_codex_tmp/remote_distinct5_variant_d_b44_full.json`
- Output: `exp/distinct5_512_ema_variant_d_vq_content_guided_b44_remote`
- Batch: 44
- Epochs: 8
- Tokenizer: shared global VQ atoms plus content-guided spatial routing.

Training:

- `Initialized global_vq tokenizer from VAE latent statistics.`
- Model params: 6,092,023.
- Peak training memory: 8.86 GB allocated, 9.19 GB reserved; `nvidia-smi` total memory about 9.65-9.71 GB.
- Epoch 1 wall was about 69 s with warmup. Epoch 2-8 compute time was about 59.6-59.8 s/epoch.
- Training completed cleanly and saved `epoch_0001.pt` through `epoch_0008.pt`.

750-image all-pairs results:

| Epoch | clip_style | content_lpips |
|---|---:|---:|
| 1 | 0.689761 | 0.415599 |
| 2 | 0.686349 | 0.440081 |
| 3 | 0.686785 | 0.443708 |
| 4 | 0.687284 | 0.447290 |
| 5 | 0.686562 | 0.453334 |
| 6 | 0.686255 | 0.456307 |
| 7 | 0.687917 | 0.451140 |
| 8 | 0.689149 | 0.443262 |

Conclusion:

- Best `clip_style` is epoch 1 at `0.689761`, higher than direct-atom baseline but lower than Variant C best `0.690659`.
- Best `content_lpips` is epoch 1 at `0.415599`, the best Distinct5 LANCET result so far.
- Retention decision: strong retain by the OR gate. It is the best content-preserving representation variant so far, but it is not a style breakthrough over Variant C.
- Design conclusion: global VQ atoms help content preservation when paired with content-guided routing. However, the shared VQ route does not add style strength over class-prototype content-guided routing.
- Training longer is actively harmful for this variant. The best checkpoint is `epoch_0001`; later checkpoints quickly lose LPIPS while style remains flat.
- Variant E has completed. See the next section for the prototype-aware latent OT queue results.

## Remote Variant E: Prototype-Aware Latent OT Queue

Run:

- Config: `_codex_tmp/remote_distinct5_variant_e_b44_full.json`
- Output: `exp/distinct5_512_ema_variant_e_latent_prototype_ot_queue_b44_remote`
- Batch: 44
- Epochs: 8
- Tokenizer: shared global VQ atoms plus content-guided spatial routing, with prototype-aware target pairing from VAE latent statistics.
- Pairing cache: `/mnt/i/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/prototype_pairing_top8.pt`
- Pairing mode: cross-style uniform top-8 routes; 20000 source-target routes.

Training:

- `Initialized global_vq tokenizer from VAE latent statistics.`
- `Loaded pairing cache ... with 20000 source-target routes.`
- Model params: 6,092,023.
- Peak training memory: 8.86 GB allocated, 9.18-9.19 GB reserved; `nvidia-smi` total memory during the clean training segment was about 9.2 GB.
- Epoch 1-7 completed first and saved `epoch_0001.pt` through `epoch_0007.pt`.
- A SaMAM scheduled task restarted and occupied the GPU during epoch 8, interrupting the original run at about 31% of epoch 8.
- SaMAM scheduled tasks were disabled, SaMAM was killed, and training resumed from `epoch_0007.pt` to complete `epoch_0008.pt`.
- Resume epoch 8 summary: `loss=7.6024`, `terminal_swd=7.6562`, `|v|=0.341`, `data=0.3s`, `compute=73.9s`, peak `8.86/9.18GB`.

Eval:

- Full eval was completed for all checkpoints `epoch_0001.pt` through `epoch_0008.pt` using `_codex_tmp/remote_distinct5_eval_variant_e_1_8.sh`.
- The eval scheduled task `Distinct5VariantEEval18` exited with result `0`.
- Eval metric loop time was about 24-26 s per checkpoint after reference cache hit. Full per-checkpoint wall is still dominated by 750-image generation and PNG save/load, as in previous variants.
- CLIP and LPIPS were evaluation only; no CLIP/DINO/classifier/external semantic supervision entered training.

750-image all-pairs results:

| Epoch | clip_style | content_lpips |
|---|---:|---:|
| 1 | 0.697347 | 0.358167 |
| 2 | 0.695356 | 0.339921 |
| 3 | 0.693462 | 0.333086 |
| 4 | 0.691414 | 0.350552 |
| 5 | 0.691366 | 0.362751 |
| 6 | 0.693713 | 0.369907 |
| 7 | 0.696153 | 0.371873 |
| 8 | 0.697092 | 0.359256 |

Conclusion:

- Best `clip_style` is epoch 1 at `0.697347`, higher than Variant C best `0.690659` and Variant D best `0.689761`.
- Best `content_lpips` is epoch 3 at `0.333086`, much lower than Variant D best `0.415599`.
- Retention decision: strong retain. Variant E is the current best Distinct5 LANCET representation variant under the OR gate and also improves both headline metrics against all earlier Distinct5 LANCET runs.
- Design conclusion: the useful step was not adding another external semantic model. The useful step was making the target queue representation-aware using only internal VAE latent statistics, then letting the same LANCET/SWD objective train against a less noisy target distribution.
- Training longer is still not uniformly useful. Style peaks at epoch 1, LPIPS at epoch 3, and epochs 4-8 do not improve the Pareto point. Future E-family runs should use early stopping or shorter schedules, then spend saved time on representation ablations.

## Remote Variant F: Annealed Prototype-Aware Latent OT Queue

Run:

- Config: `_codex_tmp/remote_distinct5_variant_f_e3_b44.json`
- Source config: `configs/distinct5_512_ema_variant_f_annealed_prototype_ot_queue_e3.json`
- Output: `exp/distinct5_512_ema_variant_f_annealed_prototype_ot_queue_e3_b44_remote`
- Batch: 44
- Epochs: 3
- Tokenizer: shared global VQ atoms plus content-guided spatial routing, with prototype-aware target pairing from VAE latent statistics.
- Pairing cache: `/mnt/i/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/prototype_pairing_top8.pt`
- Pairing mode: `rank_biased`.
- Pairing schedule: `easy_to_hard`, active top-k 2 -> 5 -> 8, rank power 1.5.

Training:

- `Initialized global_vq tokenizer from VAE latent statistics.`
- `Loaded pairing cache ... with 20000 source-target routes.`
- Model params: 6,092,023.
- The Windows scheduled task `Distinct5VariantFE3B44` completed with result `0`.
- No SaMAM process was active during this run.
- Peak training memory: 8.86 GB allocated, 9.19 GB reserved; `nvidia-smi` total memory during the run was about 9.6-9.7 GB.
- Epoch summaries:
  - Epoch 1: compute 69.9 s, peak 8.86/9.19 GB.
  - Epoch 2: compute 59.6 s, peak 8.86/9.19 GB.
  - Epoch 3: compute 55.4 s, peak 8.86/9.19 GB.

Eval:

- Deferred full eval completed for checkpoints `epoch_0001.pt` through `epoch_0003.pt`.
- Per-checkpoint full eval wall:
  - epoch 1: 140.1 s.
  - epoch 2: 149.0 s.
  - epoch 3: 145.2 s.
- Metric loop time after reference cache hit was about 22.7-24.4 s per checkpoint.
- CLIP and LPIPS were evaluation only; no CLIP/DINO/classifier/external semantic supervision entered training.

750-image all-pairs results:

| Epoch | clip_style | content_lpips |
|---|---:|---:|
| 1 | 0.696915 | 0.318645 |
| 2 | 0.695708 | 0.319789 |
| 3 | 0.695381 | 0.329646 |

Conclusion:

- Best `clip_style` is epoch 1 at `0.696915`, effectively tied with Variant E's best `0.697347` and below it by `0.000432`.
- Best `content_lpips` is epoch 1 at `0.318645`, improving Variant E's best `0.333086` by `0.014441`.
- Retention decision: strong retain by the OR gate. It is the current best Distinct5 LANCET point for content preservation while keeping the Variant E-level style score.
- Design conclusion: the active-top-k curriculum is useful. Rank-biased easy-to-hard target sampling keeps the cleaner target distribution benefit of Variant E but reduces the later drift seen in the 8-epoch uniform queue run.
- Training protocol conclusion: for this E-family, a 3-epoch short schedule is currently more rational than 8 epochs. The best Pareto point appears immediately at epoch 1; longer schedules are only useful if a new representation change shifts the early curve, not as default compute spending.

## Remote Variant G: Stratified Prototype-Aware Latent OT Queue

Run:

- Config: `_codex_tmp/remote_distinct5_variant_g_e3_b44.json`
- Source config: `configs/distinct5_512_ema_variant_g_stratified_prototype_ot_queue_e3.json`
- Output: `exp/distinct5_512_ema_variant_g_stratified_prototype_ot_queue_e3_b44_remote`
- Batch: 44
- Epochs: 3
- Tokenizer: shared global VQ atoms plus content-guided spatial routing, with prototype-aware target pairing from VAE latent statistics.
- Pairing cache: `/mnt/i/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/prototype_pairing_top8.pt`
- Pairing mode: `rank_biased_stratified`.
- Pairing schedule: `easy_to_hard`, active top-k 2 -> 5 -> 8, rank power 1.5.

Training:

- Model params: 6,092,023.
- The Windows scheduled task `Distinct5VariantGE3B44` completed with result `0`.
- SaMAM scheduled tasks were disabled before launch; no SaMAM process was active during the run.
- Peak training memory: 8.86 GB allocated, 9.19 GB reserved; `nvidia-smi` total memory during training was about 9.5-9.7 GB.
- Epoch summaries:
  - Epoch 1: compute 69.4 s, peak 8.86/9.19 GB.
  - Epoch 2: compute 59.7 s, peak 8.86/9.19 GB.
  - Epoch 3: compute 59.7 s, peak 8.86/9.19 GB.

Eval:

- Deferred full eval completed for checkpoints `epoch_0001.pt` through `epoch_0003.pt`.
- Per-checkpoint full eval wall:
  - epoch 1: 146.7 s.
  - epoch 2: 144.9 s.
  - epoch 3: 150.8 s.
- Metric loop time after reference cache hit was about 24.2-26.8 s per checkpoint.

750-image all-pairs results:

| Epoch | clip_style | content_lpips |
|---|---:|---:|
| 1 | 0.696931 | 0.345240 |
| 2 | 0.697271 | 0.341884 |
| 3 | 0.696132 | 0.332391 |

Conclusion:

- Best `clip_style` is epoch 2 at `0.697271`, slightly above Variant F's best `0.696915` but still below Variant E's best `0.697347`.
- Best `content_lpips` is epoch 3 at `0.332391`, better than Variant E's best `0.333086` by only `0.000695`, but worse than Variant F's best `0.318645` by `0.013746`.
- Retention decision: reject as a current best candidate. It does not improve the E/F Pareto front.
- Design conclusion: deterministic rank stratification is not the right queue regularizer here. Removing stochastic rank jitter makes the curve style-stable but loses the strong content-preservation gain that F obtained. The queue needs controlled target diversity, not fully deterministic hardness coverage.

## Remote Variant H: Fixed Clean Top-2 + Hard Exploration Queue

Run:

- Config: `_codex_tmp/remote_distinct5_variant_h_e3_b44.json`
- Source config: `configs/distinct5_512_ema_variant_h_hard_explore_queue_e3.json`
- Output: `exp/distinct5_512_ema_variant_h_hard_explore_queue_e3_b44_remote`
- Batch: 44
- Epochs: 3
- Tokenizer: shared global VQ atoms plus content-guided spatial routing, with prototype-aware target pairing from VAE latent statistics.
- Pairing cache: `/mnt/i/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/prototype_pairing_top8.pt`
- Pairing mode: `rank_biased`.
- Pairing schedule: `fixed`, active top-k 2, hard exploration probability 0.15 from top-8, rank power 1.5.

Training:

- `Initialized global_vq tokenizer from VAE latent statistics.`
- `Loaded pairing cache ... with 20000 source-target routes.`
- Model params: 6,092,023.
- SaMAM scheduled tasks were disabled before launch; no SaMAM process was active during the run.
- Remote py_compile passed for `src/config_schema.py`, `src/run.py`, and `src/utils/dataset.py` before launch.
- Peak training memory: 8.86 GB allocated, 9.19 GB reserved; `nvidia-smi` total memory during training was about 9.7 GB.
- Epoch summaries:
  - Epoch 1: compute 70.2 s, data 0.3 s, peak 8.86/9.19 GB.
  - Epoch 2: compute 59.6 s, data 0.2 s, peak 8.86/9.19 GB.
  - Epoch 3: compute 59.7 s, data 0.2 s, peak 8.86/9.19 GB.

Eval:

- Deferred full eval completed for checkpoints `epoch_0001.pt` through `epoch_0003.pt`.
- Per-checkpoint full eval wall from trainer log:
  - epoch 1: 149.9 s.
  - epoch 2: 144.5 s.
  - epoch 3: 144.8 s.
- `summary.json` timing profile reports wall totals of 97.6 s, 96.0 s, and 95.5 s respectively. The trainer-level wall includes subprocess launch, model/cache setup, grid summary generation, and cleanup overhead.
- Stable profile components per checkpoint:
  - LANCET latent generation: about 5.1-5.4 s for 750 images.
  - VAE decode: about 52.6-52.8 s for 750 images.
  - eval metric loop: about 24.2-24.5 s for 750 images.
  - VAE decode remains the dominant eval bottleneck; LANCET generation itself is not the bottleneck.

750-image all-pairs results:

| Epoch | clip_style | content_lpips |
|---|---:|---:|
| 1 | 0.697363 | 0.321333 |
| 2 | 0.699383 | 0.348407 |
| 3 | 0.696762 | 0.339958 |

Conclusion:

- Best `clip_style` is epoch 2 at `0.699383`, improving Variant E's previous best `0.697347` by `0.002036`.
- Best `content_lpips` is epoch 1 at `0.321333`, worse than Variant F's best `0.318645` by `0.002688`, but better than Variant E and G.
- Retention decision: strong retain by the OR gate. Variant H is the current best Distinct5 LANCET point for `clip_style`; Variant F remains the best LPIPS point.
- Design conclusion: fixed clean top-2 sampling plus sparse hard exploration is better than deterministic rank stratification and better for style than the pure easy-to-hard curriculum. The useful queue regularizer is controlled stochastic hard-target exposure, not deterministic hardness coverage.
- Next direction should compare F and H directly with a small grid over hard exploration probability and active top-k, rather than adding external semantic supervision or larger tokenizers.

## Remote Variant I: Clean Top-2 + Soft Hard-Target Latent Mix

Run:

- Config: `_codex_tmp/remote_distinct5_variant_i_e3_b44.json`
- Source config: `configs/distinct5_512_ema_variant_i_dual_target_mix_queue_e3.json`
- Output: `exp/distinct5_512_ema_variant_i_dual_target_mix_queue_e3_b44_remote`
- Batch: 44
- Epochs: 3
- Tokenizer: shared global VQ atoms plus content-guided spatial routing, with prototype-aware target pairing from VAE latent statistics.
- Pairing cache: `/mnt/i/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/prototype_pairing_top8.pt`
- Pairing mode: `rank_biased`.
- Pairing schedule: `fixed`, active top-k 2.
- Hard replacement: disabled.
- Dual target mix: clean top-2 target plus top-8 hard target mixed in VAE latent space with alpha 0.20.

Training:

- `Initialized global_vq tokenizer from VAE latent statistics.`
- `Loaded pairing cache ... with 20000 source-target routes.`
- Model params: 6,092,023.
- SaMAM scheduled tasks were disabled before launch; no SaMAM process was active during the run.
- Local smoke passed first: batch 4, 2 steps, finite loss, checkpoint saved.
- Remote py_compile passed for `src/config_schema.py`, `src/run.py`, and `src/utils/dataset.py` before launch.
- Peak training memory: 8.86 GB allocated, 9.19 GB reserved; `nvidia-smi` total memory during training was about 9.7 GB.
- Epoch summaries:
  - Epoch 1: compute 70.2 s, data 0.3 s, peak 8.86/9.19 GB.
  - Epoch 2: compute 59.8 s, data 0.2 s, peak 8.86/9.19 GB.
  - Epoch 3: compute 59.9 s, data 0.2 s, peak 8.86/9.19 GB.

Eval:

- Deferred full eval completed for checkpoints `epoch_0001.pt` through `epoch_0003.pt`.
- Per-checkpoint full eval wall from trainer log:
  - epoch 1: 147.7 s.
  - epoch 2: 146.3 s.
  - epoch 3: 146.2 s.
- `summary.json` timing profile reports wall totals of 97.4 s, 95.6 s, and 96.4 s respectively.
- Stable profile components per checkpoint:
  - LANCET latent generation: about 5.2-5.5 s for 750 images.
  - VAE decode: about 52.6-52.7 s for 750 images.
  - eval metric loop: about 24.2-24.5 s for 750 images.
- Dual target mixing adds no meaningful training or eval runtime overhead versus H.

750-image all-pairs results:

| Epoch | clip_style | content_lpips |
|---|---:|---:|
| 1 | 0.696485 | 0.347966 |
| 2 | 0.696633 | 0.384613 |
| 3 | 0.694144 | 0.366362 |

Conclusion:

- Best `clip_style` is epoch 2 at `0.696633`, below Variant H's best `0.699383` and below Variant E/F/G style-level results.
- Best `content_lpips` is epoch 1 at `0.347966`, worse than Variant F's best `0.318645`, Variant H's best `0.321333`, and Variant E/G.
- Retention decision: reject. It does not improve the E/F/H Pareto front.
- Design conclusion: convex interpolation between clean and hard target latents is not a good representation move. It lowers terminal SWD loss numerically but appears to create an ambiguous target manifold point that weakens both style and LPIPS. Hard-target pressure should remain stochastic/discrete or be expressed as a separate SWD term, not by mixing target latents before the bridge.

## Remote Variant J: Clean Top-2 + Auxiliary Hard-Target SWD

Run:

- Config: `_codex_tmp/remote_distinct5_variant_j_e3_b44.json`
- Source config: `configs/distinct5_512_ema_variant_j_aux_hard_swd_queue_e3.json`
- Output: `exp/distinct5_512_ema_variant_j_aux_hard_swd_queue_e3_b44_remote`
- Batch: 44
- Epochs: 3
- Tokenizer: shared global VQ atoms plus content-guided spatial routing, with prototype-aware target pairing from VAE latent statistics.
- Pairing cache: `/mnt/i/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/prototype_pairing_top8.pt`
- Main target: clean fixed top-2 sampled target.
- Auxiliary target: hard top-8 target, not mixed into latent; used only as a separate terminal SWD term.
- Auxiliary weight: `terminal_swd_aux_weight=3.0`.

Training:

- Local smoke passed first: batch 4, finite loss, checkpoint saved, `terminal_swd_aux=2.25`, `aux_target_ratio=0.875`.
- Remote config check passed: `batch_size=44`, `epochs=3`, `aux_weight=3.0`, `aux_topk=8`, `dual_mix=0.0`.
- SaMAM Windows scheduled tasks were disabled; a lingering WSL `tmux` SaMAM segmented run was stopped before launch.
- `nohup` did not persist under this WSL entry path, so the formal run was launched in a dedicated `tmux` session.
- Peak training memory: 8.99 GB allocated, 9.19 GB reserved; `nvidia-smi` total memory during training was about 9.7 GB.
- Aux target was active in checkpoints with `aux_target_ratio=0.798472`.
- Epoch summaries:
  - Epoch 1: compute 70.2 s, data 0.5 s, `terminal_swd_aux=1.382812`, peak 8.99/9.19 GB.
  - Epoch 2: compute 60.3 s, data 0.3 s, `terminal_swd_aux=1.257812`, peak 8.99/9.19 GB.
  - Epoch 3: compute 60.3 s, data 0.3 s, `terminal_swd_aux=1.187500`, peak 8.99/9.19 GB.

Eval:

- Deferred full eval completed for checkpoints `epoch_0001.pt` through `epoch_0003.pt`.
- Per-checkpoint full eval wall from trainer log:
  - epoch 1: 153.2 s.
  - epoch 2: 145.5 s.
  - epoch 3: 145.1 s.
- Stable profile components visible in `summary.json`:
  - VAE decode: about 52.7-52.8 s for 750 images.
  - eval metric loop: about 24.3-27.0 s for 750 images.

750-image all-pairs results:

| Epoch | clip_style | content_lpips |
|---|---:|---:|
| 1 | 0.697653 | 0.332274 |
| 2 | 0.696313 | 0.349714 |
| 3 | 0.694811 | 0.340217 |

Conclusion:

- Best `clip_style` is epoch 1 at `0.697653`, below Variant H's `0.699383`.
- Best `content_lpips` is epoch 1 at `0.332274`, below Variant F's `0.318645` and worse than Variant H's `0.321333`.
- Retention decision: reject for the current Pareto front.
- Design conclusion: separating the hard target as an auxiliary SWD term is cleaner than Variant I's latent premix, but it still behaves as extra terminal pull rather than a better representation. It lowers train terminal losses but does not improve eval style or LPIPS. The hard-target signal is useful when it occasionally replaces/samples the real target distribution, as in Variant H, not when added as a second simultaneous target constraint.
- Implementation note: trainer now accepts `aux_target_style` / `aux_target_valid`, `OMFLoss` supports `terminal_swd_aux_weight`, dataset supports `pairing_cache_aux_target_topk`, and future training CSV logs include `terminal_swd_aux` plus `aux_target_ratio`.

## Remote Variant K: Content-Adaptive Global VQ Atom Routing

Run:

- Config: `_codex_tmp/remote_distinct5_variant_k_e3_b44.json`
- Source config: `configs/distinct5_512_ema_variant_k_content_adaptive_vq_queue_e3.json`
- Output: `exp/distinct5_512_ema_variant_k_content_adaptive_vq_queue_e3_b44_remote`
- Batch: 44
- Epochs: 3
- Tokenizer: shared global VQ atoms plus content-guided spatial routing, with an additional content-adaptive residual over VQ atom logits.
- Pairing cache: `/mnt/i/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/prototype_pairing_top8.pt`
- Pairing mode: `rank_biased`.
- Pairing schedule: `fixed`, active top-k 2, hard exploration probability 0.15 from top-8, rank power 1.5.

Design:

- The router reads only internal VAE-latent content statistics from the LANCET feature path: mean, std, absolute mean, local highpass absolute mean, and total energy.
- The router output is added to class atom logits before top-k atom softmax. It does not see target latents, CLIP, LPIPS, DINO, classifiers, or any external semantic feature.
- The final router linear layer is zero-initialized, so the initial behavior matches Variant H and any effect must be learned during training.

Training:

- Local smoke passed first: batch 4, finite loss, checkpoint saved, and `content_atom_*` debug fields appeared.
- Remote finite shape check passed on WSL before formal launch.
- The formal run was launched in tmux because `nohup` had been unreliable in this WSL entry path.
- Peak training memory: 8.86 GB allocated, 9.19-9.20 GB reserved; `nvidia-smi` total memory during training was about 9.6-9.7 GB.
- Epoch summaries:
  - Epoch 1: compute 69.4 s, data 0.3 s, peak 8.86/9.19 GB.
  - Epoch 2: compute 60.1 s, data 0.2 s, peak 8.86/9.20 GB.
  - Epoch 3: compute 60.0 s, data 0.4 s, peak 8.86/9.20 GB.

Eval:

- Deferred full eval completed for checkpoints `epoch_0001.pt` through `epoch_0003.pt`.
- Per-checkpoint full eval wall from trainer log:
  - epoch 1: 153.2 s.
  - epoch 2: 145.4 s.
  - epoch 3: 145.6 s.
- Stable profile components visible in `summary.json`:
  - LANCET latent generation: 5.33-5.65 s for 750 images.
  - VAE decode: 52.62-52.77 s for 750 images.
  - eval metric loop: 24.22-24.60 s for 750 images.
- Runtime is effectively unchanged from H/I/J; K adds tokenizer compute but not a meaningful eval bottleneck.

750-image all-pairs results:

| Epoch | clip_style | content_lpips |
|---|---:|---:|
| 1 | 0.700995 | 0.362294 |
| 2 | 0.698433 | 0.370882 |
| 3 | 0.698514 | 0.368187 |

Tokenizer learning diagnostic:

- `numeric_debug.jsonl` confirms that the zero-initialized content router moved during training:
  - `tokenizer_content_atom_delta_abs` reached about 0.034-0.037.
  - `tokenizer_content_atom_effective_count` stayed around 7.8 active atoms under top-8 routing.
  - `tokenizer_grad_atom_logits_weight` was nonzero, with observed values around 0.03-0.066 in sampled debug rows.

Conclusion:

- Best `clip_style` is epoch 1 at `0.700995`, improving Variant H's previous best `0.699383` by `0.001612`.
- Best `content_lpips` is epoch 1 at `0.362294`, worse than Variant F/H/J and far worse than the F LPIPS best `0.318645`.
- Retention decision: retain for style only. K is the first Distinct5 LANCET variant to cross `clip_style=0.700`, so the representation idea is not empty. However, it shifts the Pareto point toward style at a clear LPIPS cost.
- Design conclusion: content-adaptive atom-logit routing is useful for style discrimination, but without an explicit content-preserving constraint or queue policy adjustment it increases endpoint movement in a way LPIPS penalizes. The next K-family test should not simply raise the router gain; it should combine the K router with the cleaner F curriculum or reduce the router gain while preserving H's sparse hard exploration.

## Remote Variant L: Content-Adaptive Atom Routing + Annealed Queue

Run:

- Config: `_codex_tmp/remote_distinct5_variant_l_e3_b44.json`
- Source config: `configs/distinct5_512_ema_variant_l_content_adaptive_annealed_queue_e3.json`
- Output: `exp/distinct5_512_ema_variant_l_content_adaptive_annealed_queue_e3_b44_remote`
- Batch: 44
- Epochs: 3
- Tokenizer: shared global VQ atoms plus content-guided spatial routing, with the same content-adaptive atom-logit router as K.
- Pairing cache: `/mnt/i/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/prototype_pairing_top8.pt`
- Pairing mode: `rank_biased`.
- Pairing schedule: `easy_to_hard`, active top-k 2 -> 5 -> 8, rank power 1.5.
- Hard exploration: disabled. This intentionally follows F, not H.

Design:

- This is the direct mechanism-composition test after K: keep K's learned content-adaptive style-code router, but replace H's fixed top-2 plus hard-explore sampler with F's LPIPS-preserving easy-to-hard curriculum.
- Training still uses only VAE latent targets, SWD/OT, and internal prototype queue statistics. CLIP, LPIPS, DINO, classifiers, and external semantic features are evaluation-only or absent.

Training:

- Local config expansion verified: `tokenizer_content_adaptive=true`, `pairing_cache_rank_schedule=easy_to_hard`, `pairing_cache_explore_prob=0.0`.
- Remote py_compile passed before launch.
- SaMAM scheduled tasks were disabled before launch; no SaMAM process was active.
- Peak training memory: 8.86 GB allocated, 9.19-9.20 GB reserved; `nvidia-smi` total memory during training was about 9.7 GB.
- Epoch summaries:
  - Epoch 1: compute 70.0 s, data 0.3 s, peak 8.86/9.19 GB.
  - Epoch 2: compute 59.8 s, data 0.2 s, peak 8.86/9.20 GB.
  - Epoch 3: compute 59.9 s, data 0.2 s, peak 8.86/9.20 GB.

Eval:

- Deferred full eval completed for checkpoints `epoch_0001.pt` through `epoch_0003.pt`.
- Per-checkpoint full eval wall from trainer log:
  - epoch 1: 150.7 s.
  - epoch 2: 145.1 s.
  - epoch 3: 145.3 s.
- Stable profile components visible in `summary.json`:
  - LANCET latent generation: 5.33-5.46 s for 750 images.
  - VAE decode: 52.62-52.77 s for 750 images.
  - eval metric loop: 24.09-24.34 s for 750 images.
- Runtime remains unchanged from F/H/K; the content-adaptive router is not an infra bottleneck.

750-image all-pairs results:

| Epoch | clip_style | content_lpips |
|---|---:|---:|
| 1 | 0.697777 | 0.339710 |
| 2 | 0.696230 | 0.365100 |
| 3 | 0.696625 | 0.362950 |

Tokenizer learning diagnostic:

- `numeric_debug.jsonl` confirms the content router learned:
  - `tokenizer_content_atom_delta_abs` moved from 0 at the first step to a max observed value of about 0.0395, mean about 0.031 across logged rows.
  - `tokenizer_content_atom_effective_count` stayed around 7.8 active atoms under top-8 routing.
  - `tokenizer_grad_atom_logits_weight` was nonzero, with observed values up to about 0.0506 in sampled rows.

Conclusion:

- Best `clip_style` is epoch 1 at `0.697777`, below K's `0.700995` and H's `0.699383`.
- Best `content_lpips` is epoch 1 at `0.339710`, below K but worse than F's `0.318645`, H's `0.321333`, and J's `0.332274`.
- Retention decision: reject for the current Pareto front.
- Design conclusion: K's style boost does not transfer to F's easy-to-hard curriculum. The evidence suggests that content-adaptive atom routing needs sparse hard-target exposure from H to lift style; the F curriculum suppresses the style gain but still does not recover F's LPIPS. This rules out the simple "K router + F curriculum" composition as the next basis.

## Variant M: Style-Gated Content-Adaptive Router

Run:

- Config: `_codex_tmp/remote_distinct5_variant_m_e3_b44.json`
- Source config: `configs/distinct5_512_ema_variant_m_style_gated_content_router_e3.json`
- Output: `exp/distinct5_512_ema_variant_m_style_gated_content_router_e3_b44_remote`
- Dataset: `F:\wikiart_distinct5_samam_512` classview/EMA-latent Distinct5 split.
- Batch: 44
- Epochs: 3 configured.
- Tokenizer: Variant K global VQ plus content-guided spatial routing, with a target-style scalar gate on the content-adaptive atom-logit residual.
- Pairing cache: prototype-aware top-8 routes built only from VAE latent statistics.
- Pairing mode: fixed clean top-2 plus 15% hard exploration from top-8.

Operational boundary:

- This run was launched during the remote phase and then stopped after the user explicitly redirected work back to local-only experimentation.
- Only the already-read epoch 1/2 summaries are recorded here. No further remote collection is required for this document.
- Treat this result as a rejected partial ablation, not as a new active branch.

Available 750-image all-pairs results:

| Epoch | clip_style | content_lpips |
|---|---:|---:|
| 1 | 0.698726 | 0.346543 |
| 2 | 0.696810 | 0.345800 |

Training/eval notes:

- Formal training memory during the run was in the same b44 band as H/K/L: about 9.6-9.7 GB by `nvidia-smi`.
- Epoch 1 compute was about 70.2 s; epoch 3 compute log was about 59.9 s before eval.
- Available eval metric loop time was about 24.5 s/checkpoint, consistent with H/K/L.

Local smoke after switching back to local-only:

- Config used: `_codex_tmp/distinct5_smoke_variant_m_local_windows.json`
- This was a transient smoke override under `_codex_tmp`; it is not a formal committed config.
- Dataset: `F:\wikiart_distinct5_samam_512_latents_ema\train`
- Runtime: Windows Python 3.12, `torch 2.11.0+cu128`, local RTX 4070 Laptop GPU.
- Batch: 8
- `virtual_length_multiplier`: 0.02
- Steps: 12
- Result: finite forward/backward/optimizer path, checkpoint saved at `exp/_smoke_distinct5_512_ema_variant_m_style_gated_local_windows/epoch_0001.pt`.
- Final smoke loss: `11.6485`; terminal SWD: `11.5000`; kinetic: `0.1277`.
- Peak smoke memory: 1.66 GB allocated, 1.83 GB reserved.
- `numeric_debug.jsonl` contains `tokenizer_content_atom_gate_mean/min/max=1.0` at the first logged step, as expected from the gate initialization. The short smoke only logged step 1, so it validates wiring, not gate learning dynamics.

Conclusion:

- Best available M style is epoch 1 at `0.698726`, below K's `0.700995` and H's `0.699383`.
- Best available M LPIPS is epoch 2 at `0.345800`, worse than F's `0.318645`, H's `0.321333`, and J's `0.332274`.
- Retention decision: reject for the current Pareto front.
- Design conclusion: a free style gate initialized at 1.0 does not immediately solve K's style-vs-LPIPS tradeoff. If this idea is revisited, the gate needs either a style-specific prior, stronger regularization, or an explicit target-style diagnostic objective inside the VAE-latent-only regime. It should not be the next default branch.

## Current Best Distinct5 LANCET Points

| Model | Best epoch | clip_style | content_lpips | Decision |
|---|---:|---:|---:|---|
| Baseline direct atom residual | 8 / 1 | 0.687649 | 0.446756 | weak baseline |
| Variant A class prototypes | 8 / 1 | 0.684946 | 0.446381 | reject |
| Variant B global VQ | 8 | 0.687321 | 0.444600 | weak retain for LPIPS |
| Variant C content-guided spatial | 2 | 0.690659 | 0.422593 | retain |
| Variant D VQ + content-guided | 1 | 0.689761 | 0.415599 | retain |
| Variant E latent prototype OT queue | 1 / 3 | 0.697347 | 0.333086 | strong retain |
| Variant F annealed prototype OT queue | 1 | 0.696915 | 0.318645 | current best LPIPS |
| Variant G stratified prototype OT queue | 2 / 3 | 0.697271 | 0.332391 | reject |
| Variant H hard-explore prototype OT queue | 2 / 1 | 0.699383 | 0.321333 | current best style |
| Variant I dual-target latent mix queue | 2 / 1 | 0.696633 | 0.347966 | reject |
| Variant J auxiliary hard-target SWD queue | 1 | 0.697653 | 0.332274 | reject |
| Variant K content-adaptive VQ atom routing | 1 | 0.700995 | 0.362294 | current best style, retain style-only |
| Variant L content-adaptive annealed queue | 1 | 0.697777 | 0.339710 | reject |
| Variant M style-gated content router | 1 / 2 | 0.698726 | 0.345800 | reject, partial available result |

Immediate next design direction:

- Keep Variant F/H/K as the current Distinct5 basis: F for LPIPS pressure, H for balanced style with tolerable LPIPS, and K for the strongest style-only point.
- Do not add DINO/CLIP/classifier training supervision.
- Explore cheaper E/K-family ablations: hard-explore probability, fixed active top-k, route temperature, prototype count, stochastic queue hardness, content-guided routing strength, and content-adaptive atom-router gain.
- Do not pursue convex target-latent mixing as the default hard-target mechanism. If dual targets are revisited, implement them as separate weighted SWD terms rather than a pre-mixed latent target.
- Do not pursue simultaneous auxiliary hard-target SWD as the default either; the better evidence is sparse hard-target exposure in the target sampler.
- Do not treat the K result as proof that more tokenizer adaptivity is always better. The useful signal is style-only so far; LPIPS degraded sharply.
- Do not treat "content-adaptive router plus F curriculum" as the obvious fix; L shows it loses K's style boost and still misses F's LPIPS.
- Do not treat the M style gate as a solved LPIPS repair. The available M data is below H/K for style and below F/H/J for LPIPS.
- Use a short-schedule protocol first because best style and best LPIPS both appear in the first three epochs. Only expand to 8 epochs if an ablation improves the early curve.
- Do not pursue deterministic rank stratification further unless it is paired with an explicit stochastic diversity source.

Smoke matrix:

| Variant | Loss | Terminal SWD | Epoch sec | Peak reserved GB | Checkpoint |
|---|---:|---:|---:|---:|---|
| A class prototypes | 14.9155 | 14.6875 | 8.61 | 1.74 | `exp/_smoke_distinct5_512_ema_variant_a_class_prototypes_b8_vlen001/epoch_0001.pt` |
| B global VQ | 14.4136 | 14.1875 | 0.79 | 1.75 | `exp/_smoke_distinct5_512_ema_variant_b_global_vq_b8_vlen001/epoch_0001.pt` |
| C content-guided spatial | 14.8407 | 14.5000 | 0.81 | 1.78 | `exp/_smoke_distinct5_512_ema_variant_c_content_guided_spatial_b8_vlen001/epoch_0001.pt` |
| D VQ + content-guided | 14.7427 | 14.4375 | 0.80 | 1.75 | `exp/_smoke_distinct5_512_ema_variant_d_vq_content_guided_b8_vlen001/epoch_0001.pt` |
| E latent prototype OT queue | 11.1700 | 11.0625 | 0.76 | 1.75 | `exp/_smoke_distinct5_512_ema_variant_e_latent_prototype_ot_queue_b8_vlen001/epoch_0001.pt` |
| M style-gated content router | 11.6485 | 11.5000 | 8.0 | 1.83 | `exp/_smoke_distinct5_512_ema_variant_m_style_gated_local_windows/epoch_0001.pt` |

The smoke losses are not benchmark results. They only validate that each representation path can load packed latents, run forward/backward, and save checkpoints.
