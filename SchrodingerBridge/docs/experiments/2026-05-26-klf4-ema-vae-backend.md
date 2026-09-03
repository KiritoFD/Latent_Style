# 2026-05-26 KL-f4 / SD15-EMA VAE Backend Notes

## Objective

Replace the previous SD15-MSE/SDXL VAE backends at 256x256 and test whether a more suitable first-stage autoencoder can improve LANCET style/content performance:

- `CompVis KL-f4`: legacy LDM first-stage autoencoder, 256 image -> 3x64x64 latent.
- `sd-vae-ft-ema`: SD 1.5 EMA VAE, 256 image -> 4x32x32 latent, avoiding MSE-smoothed VAE.

Primary target remains `clip_style > 0.72`; secondary target is `content_lpips < 0.53`.

## Why SDXL Was Paused

SDXL was stable after rebuilding fp32 latents, but multiple architecture/loss routes stayed below the target style:

- minimal SDXL: about `clip_style=0.667`, very good LPIPS but weak style.
- stronger SWD / plain 4ch: saturated near `0.674`.
- OMF/t01-style strong 4ch: did not recover SD15 behavior and often worsened LPIPS.
- high W/K t01max: reached only about `0.679`, with LPIPS around `0.75`.

The working conclusion is that SDXL VAE has high capacity but poor 256x256 operator compatibility for the current LANCET assumptions. It is not deleted, but it is no longer the first priority.

## KL-f4 Loader Fix

KL-f4 is a legacy CompVis LDM checkpoint, not a normal Diffusers repo. The loader now:

- downloads `https://ommer-lab.com/files/latent-diffusion/kl-f4.zip`;
- extracts `model.ckpt` and the autoencoder YAML;
- loads with `torch.load(..., weights_only=False)` because PyTorch 2.6 otherwise rejects Lightning objects in the trusted CompVis checkpoint;
- converts to Diffusers `AutoencoderKL.from_single_file` using a synthesized original config with `first_stage_config`;
- forces `vae.config.scaling_factor = 1.0`, because this standalone first-stage autoencoder is not an SD latent scaled by `0.18215`.

Smoke result:

- latent shape: `1 x 3 x 64 x 64`
- finite ratio: `1.0`
- decode returns `1 x 3 x 256 x 256`

## KL-f4 Memory Calibration

KL-f4 has 4x the spatial tokens of f8 VAEs, so old batch sizes are unsafe.

| variant | batch | status | peak VRAM |
|---|---:|---|---:|
| `klf4_t01_w20` | 48 | OOM/killed | 12086 MB |
| `klf4_t01_w20_b40` | 40 | train_ok | 12049 MB |
| `klf4_t01_w20_b32` | 32 | train_ok | 11135 MB |
| `klf4_t01_w20_b28` | 28 | train_ok | 9676 MB |
| `klf4_t01_w20_b24` | 24 | train_ok | 8443 MB |

Decision: formal KL-f4 runs use `batch=28`, which lands near the 10G engineering target.

## Active KL-f4 Run

Remote path:

`I:\Github\Latent_Style\SchrodingerBridge\exp\vae_backend_256_klf4_ema_full\klf4_t01_w20_b28`

Config:

- VAE: `kl-f4`
- latent root: `I:\Github\Latent_Style\latent-256-kl-f4`
- latent shape: `3x64x64`
- batch: `28`
- training: `8` epochs
- eval epochs: `6,7,8`
- actuator: t01-style diffeomorphic stroke, W20/K1, patches `[3,5,7,15]`

Pending: read eval summaries once epoch 6/7/8 finish.

### First KL-f4 Readout

The first formal KL-f4 run finished successfully at the target memory level.

| epoch | clip_style | content_lpips | EC | peak VRAM |
|---:|---:|---:|---:|---:|
| 6 | 0.6601 | 0.4586 | 0.3573 | 9796 MB |
| 7 | 0.6584 | 0.4595 | 0.3559 | 9796 MB |
| 8 | 0.6595 | 0.4593 | 0.3566 | 9796 MB |

Interpretation:

- KL-f4's 64x64 latent support did not automatically recover high CLIP style.
- LPIPS is acceptable, but style is much worse than the original t01 anchor (`~0.726/0.517`).
- This means the bottleneck is not only rank/spatial support. The f4 latent coordinate system likely needs a different actuator or encode/decode convention.

Immediate follow-up checks:

- Compare KL-f4 posterior sample vs mode/reconstruction behavior. If stochastic posterior sampling is too noisy for this VAE, re-encode with mode latents.
- Try KL-f4-specific patch/strength settings rather than copying t01: larger macro patches and lower diffeomorphic warp/color pressure.
- Run `sd-vae-ft-ema` as the non-MSE f8 control before investing too much into KL-f4.

## EMA Plan

After KL-f4 first readout, run `sd-vae-ft-ema` as a direct f8 control:

- VAE alias: `ema`
- latent root: `latent-256-sd15-ema`
- same t01-style actuator and eval protocol.

This tells us whether the old VAE issue was specifically MSE smoothing, or whether the decisive gain comes from KL-f4's f4 spatial support.

### EMA Smoke

`ema_t01_w20` 1 epoch / 30 batch smoke finished:

- batch: `128`
- peak VRAM: `9490 MB`
- status: `train_ok`

Decision: use `batch=128` for the formal EMA run. It satisfies the 10G engineering target and is directly comparable to the original f8 setup.

Active formal EMA run:

`I:\Github\Latent_Style\SchrodingerBridge\exp\vae_backend_256_klf4_ema_full\ema_t01_w20`

First EMA readout:

| epoch | clip_style | content_lpips | EC | peak VRAM |
|---:|---:|---:|---:|---:|
| 6 | 0.7237 | 0.5755 | 0.3072 | 9495 MB |
| 7 | 0.7231 | 0.5797 | 0.3039 | 9495 MB |
| 8 | 0.7224 | 0.5764 | 0.3060 | 9495 MB |

Interpretation:

- `sd-vae-ft-ema` restores the desired style range (`>0.72`), unlike KL-f4 and SDXL.
- Content LPIPS is too high (`~0.576`), so the next axis is content guarding rather than style chasing.
- This already suggests that the MSE-finetuned VAE was not the only viable f8 backend; EMA is compatible with the original LANCET t01 style actuator, but needs lower visible deformation.

Active EMA content-guard queue:

- `ema_guard_w16`: lower terminal SWD and weaker warp/color.
- `ema_guard_w18_patch357`: remove macro patch 15, keep local texton patches.
- `ema_guard_w20_lowwarp`: keep W20 but strongly reduce warp/color actuator.

Goal: preserve `clip_style >= 0.72` while moving LPIPS below `0.53`.

### EMA Content Guard Readout: `ema_guard_w16`

`ema_guard_w16` finished training and eval on epochs `6/7/8`.

| epoch | clip_style | content_lpips | EC | peak VRAM |
|---:|---:|---:|---:|---:|
| 6 | 0.7183 | 0.5230 | 0.3426 | 9490 MB |
| 7 | 0.7207 | 0.5256 | 0.3419 | 9490 MB |
| 8 | 0.7192 | 0.5235 | 0.3427 | 9490 MB |

Interpretation:

- The content guard is effective: LPIPS drops from the plain EMA t01 baseline (`~0.576`) to `~0.523-0.526`.
- Style stays near the target boundary. Epoch 7 is the current best balanced EMA point: `clip_style=0.7207`, `content_lpips=0.5256`.
- Compared with original t01 (`~0.726 / ~0.517`), EMA is now close but still not clearly superior: it has slightly weaker style and slightly weaker content. It is, however, much better than KL-f4 and SDXL in style compatibility.

Current decision:

- Continue the active queue. `ema_guard_w18_patch357` tests whether removing the macro `15` patch and raising terminal SWD to `18` can recover style while keeping the LPIPS gain.
- `ema_guard_w20_lowwarp` tests whether the original style pressure can be kept while visible deformation is reduced through lower warp/color actuation.

## Architecture-Level VAE Interpretation

The updated target is stricter: `clip_style > 0.72` and `content_lpips < 0.45`. Under this target, parameter-only tuning is not enough. The two non-MSE VAEs imply different operator designs.

### SD15 EMA

`sd-vae-ft-ema` is still an f8, 4-channel latent backend. Its spatial token count and channel count match the original SD15 family, so the old 6-channel standard diffeomorphic head (`4 residual + 2 warp`) is structurally legal. The readouts show that it is also style-compatible: plain EMA t01 reaches `clip_style~0.723`.

The failure mode is not style capacity; it is content motion. LPIPS rises to `~0.576`, and content-guarded `ema_guard_w16` only lowers it to `~0.523`. Therefore the EMA architecture search should ask:

- Is standard 6ch (`4 residual + 2 warp`) too geometry-active for the `<0.45` LPIPS target?
- Can a 7ch factorized head (`4 residual + 1 amplitude + 2 warp`) move style into high-frequency amplitude while keeping warp near zero?
- Does a pure 4ch residual/color route preserve content enough, and if so, can stronger SWD recover style?

Decision: EMA should keep 6ch as a controlled branch, but the preferred next architecture is factorized amplitude with near-zero warp, not stronger standard warp.

### CompVis KL-f4

KL-f4 is not just "a sharper VAE". It changes the latent topology:

- channel count: `3`, not `4`;
- spatial support: `64x64`, not f8 `32x32`;
- scaling convention: standalone `scaling_factor=1.0`;
- current t01-copy result: good content (`LPIPS~0.459`) but very weak style (`clip_style~0.660`).

This means the old 6ch idea cannot be copied literally. For KL-f4:

- standard diffeo is `3 residual + 2 warp = 5ch`;
- factorized amplitude is `3 residual + 1 amplitude + 2 warp = 6ch`;
- patch sizes should shift upward because f4 has twice the latent spatial resolution per image axis.

Decision: KL-f4 should not be judged only by the t01-copy standard head. The next fair test is posterior-mode latents plus f4-scale larger patches and a 6ch factorized-amplitude head. If this still cannot lift style above `0.72`, KL-f4 is likely content-preserving but style-incompatible with the current LANCET loss surface.

### KL-f4 Fair-Mode Follow-Up

Audit:

- The existing full KL-f4 encoded dataset was `latent-256-kl-f4`.
- It used posterior `sample` latents.
- `latent-256-kl-f4-mode` did not exist yet.
- The old `vae_backend_256_klf4_ema_full` ledger was mixed with EMA rows, so KL-f4 sample metrics were re-read from the actual `summary.json` files.

Existing KL-f4 sample-latent result:

| epoch | clip_style | content_lpips |
|---:|---:|---:|
| 6 | 0.6601 | 0.4586 |
| 7 | 0.6584 | 0.4595 |
| 8 | 0.6595 | 0.4593 |

Fair follow-up launched:

`I:\Github\Latent_Style\SchrodingerBridge\exp\vae_backend\klf4_mode_fair`

Variant:

`klf4_mode_stylepush_w40_p5917`

Changes relative to the sample-latent t01-copy:

- Encode full training set with posterior `mode`, not posterior `sample`.
- Use KL-f4-aware patch sizes `[5, 9, 17]` for the 64x64 f4 latent grid.
- Raise terminal SWD to `40`.
- Keep batch at `28`, the previously selected safe 10G-class batch.
- Eval epochs `6/7/8`.

Question:

If this still stays near `clip_style~0.66`, KL-f4 is likely not a useful drop-in backend for the current LANCET objective at 256. If it jumps substantially, then the earlier failure was an unfair latent convention / patch-scale mismatch rather than a backend limitation.

### Immediate Architecture Matrix

The next VAE-aware experiments should be small but structurally distinct:

| backend | branch | output head | expected behavior |
|---|---|---|---|
| EMA | 4ch residual | no diffeo | content floor, likely weak style |
| EMA | standard 6ch | residual + warp | known high style, risky LPIPS |
| EMA | factorized 7ch | residual + amplitude + near-zero warp | best chance for style with LPIPS below 0.45 |
| KL-f4 | standard 5ch | residual + warp | t01-copy baseline, currently weak style |
| KL-f4 | factorized 6ch | residual + amplitude + warp | correct KL-f4 analogue of the old "6ch" idea |

The phrase "6 channels" is therefore VAE-relative. For f8 4ch VAEs it means standard residual+warp. For KL-f4 3ch, the meaningful 6ch design is factorized residual+amplitude+warp.

## EMA Fragmentation Attribution

Visual review folder:

`G:\GitHub\Latent_Style\SchrodingerBridge\exp\vae_backend\vae_backend_256_status\ema_guard_grids_review`

The compared grids show that `ema_guard_w16` is the most visually stable recent EMA point, while `ema_guard_w18_patch357` and `ema_guard_w20_lowwarp` recover more style at the cost of local broken strokes and edge bleeding. The key observation is that `ema_guard_w20_lowwarp` already reduces `diffeomorphic_warp_strength` to `0.01`, yet the fragmentation remains visible. Therefore the current failure should not be attributed to warp alone.

Working diagnosis:

- Warp is likely an amplifier, especially near content edges, but not the root cause.
- High terminal SWD/style pressure can push local texture statistics through the residual/color branch even when warp is weak.
- EMA's non-MSE latent has stronger high-frequency curvature than the MSE VAE, so the old color/residual path can reveal cross-edge texture leakage after decoding.
- Macro patch `15` and W18/W20 style pressure correlate with stronger visual fragmentation. Removing macro patch improves content somewhat but does not fully solve the issue.
- If fragmentation survives with `use_diffeomorphic_stroke=false`, the culprit is mostly SWD/residual/decoder style injection; if it disappears, the diffeomorphic branch is the dominant cause.

Attribution queue launched on remote:

`I:\Github\Latent_Style\SchrodingerBridge\exp\vae_backend\ema_arch_attribution`

| variant | purpose |
|---|---|
| `ema_hardcontent_w18_anchor` | low warp + Gaussian lowpass + active gradient mask + structure barrier |
| `ema_hardcontent_w24_anchor` | same as above, stronger style pressure |
| `ema_amp_only_w24_anchor` | factorized amplitude with `warp_strength=0`, tests whether style can move into amplitude without geometry motion |
| `ema_identity_w24_anchor` | hard-content variant plus identity pairs, tests unnecessary content motion |
| `ema_plain4_w20_anchor` | direct no-diffeo counterpart to `ema_guard_w20_lowwarp`, isolates whether W20 fragmentation survives without warp channels |
| `ema_plain4_spectral_iso_w32` | pure residual high-SWD content floor with no diffeomorphic branch |

Current readout rule:

- If `ema_plain4_w20_anchor` is still broken, do not keep blaming warp. Move to SWD patch/style-pressure and residual/color isolation.
- If `ema_amp_only_w24_anchor` keeps style while reducing fragmentation, the next main EMA route should be factorized amplitude with near-zero warp.
- If only the hard-content/barrier variants improve LPIPS but lose style, the barrier is useful but not sufficient; tune style pressure and patch bands around it.

### First Attribution Readout

`ema_hardcontent_w18_anchor` finished epochs `6/7/8`:

| epoch | clip_style | content_lpips | EC |
|---:|---:|---:|---:|
| 6 | 0.6671 | 0.4945 | 0.3372 |
| 7 | 0.6677 | 0.4944 | 0.3376 |
| 8 | 0.6672 | 0.4944 | 0.3373 |

Interpretation:

- The strong barrier/anchor setup does reduce content damage relative to plain EMA t01 and the W18/W20 content-guard variants.
- It also collapses style far below the target. This is not a viable main route.
- This result supports the idea that edge barriers can suppress fragmentation, but a global delta barrier is too blunt: it blocks the same high-frequency style transport that makes EMA useful.
- The queue was reprioritized to run `ema_amp_only_w24_anchor` and `ema_plain4_w20_anchor` before the remaining barrier-like variants, because those two answer the warp attribution question more directly.

`ema_amp_only_w24_anchor` finished epochs `6/7/8`:

| epoch | clip_style | content_lpips | EC |
|---:|---:|---:|---:|
| 6 | 0.6682 | 0.5092 | 0.3280 |
| 7 | 0.6708 | 0.5113 | 0.3278 |
| 8 | 0.6695 | 0.5123 | 0.3265 |

Interpretation:

- Moving style through a factorized amplitude route with `warp_strength=0` is not enough under the current anchor/barrier setup.
- LPIPS improves relative to plain EMA t01, but style collapses almost as badly as the hard-content barrier branch.
- This weakens the hypothesis that "amplitude-only can replace warp" by itself. If amplitude is useful later, it probably needs less global structure barrier and a more permissive style path.
- The decisive attribution check is now `ema_plain4_w20_anchor`: if no-diffeo W20 keeps the same visual damage, SWD/residual pressure is sufficient to cause fragmentation; if it is visually clean but low-style, diffeomorphic routing was the missing style actuator.

`ema_plain4_w20_anchor` finished epochs `6/7/8`:

| epoch | clip_style | content_lpips | EC |
|---:|---:|---:|---:|
| 6 | 0.7007 | 0.4215 | 0.4053 |
| 7 | 0.7001 | 0.4240 | 0.4033 |
| 8 | 0.7007 | 0.4248 | 0.4030 |

Visual read:

- This is the cleanest recent EMA attribution grid. Edges and object geometry are much more coherent than `ema_guard_w18_patch357`, `ema_guard_w20_lowwarp`, and `ema_amp_only_w24_anchor`.
- It crosses the content target (`content_lpips < 0.45`) but misses the style target by about `0.02`.
- Therefore W20 SWD itself is not enough to cause the severe fragmentation. The damaging ingredient is the stronger geometry/style actuator, especially diffeomorphic routing near content boundaries.
- This no-diffeo result is the new clean base. The next architecture should add the smallest possible style actuator on top of this base, not return to the original strong 6ch warp path.

Current next test: `ema_plain4_spectral_iso_w32`, which keeps no diffeomorphic branch but raises style pressure and uses smaller style patches. If it reaches `clip_style>0.72` while preserving LPIPS, the EMA path can stay 4ch. If it does not, add a constrained style-only actuator to the plain4 base.

`ema_plain4_spectral_iso_w32` finished epochs `6/7/8`:

| epoch | clip_style | content_lpips | EC |
|---:|---:|---:|---:|
| 6 | 0.7025 | 0.4342 | 0.3974 |
| 7 | 0.7024 | 0.4371 | 0.3954 |
| 8 | 0.7026 | 0.4379 | 0.3949 |

Interpretation:

- Increasing pure 4ch SWD/style pressure from W20 to W32 only improves style by about `0.002`.
- Content remains comfortably below the LPIPS target, so the clean 4ch route is not capacity-limited for content.
- The style ceiling of pure residual 4ch appears to be around `0.70` under this loss family.
- Remaining gap to `clip_style>0.72` likely needs a small style actuator, but it must be much more constrained than the original 6ch warp branch.

Next queue:

`I:\Github\Latent_Style\SchrodingerBridge\exp\vae_backend\ema_cleanbase_push`

| variant | purpose |
|---|---|
| `ema_plain4_styleloss_w24` | clean 4ch base plus style energy / contrastive / residual-direction / spectral-amplitude losses |
| `ema_plain4_canvas_w24` | clean 4ch base plus edge-masked latent highpass canvas |
| `ema_microdiffeo_w20` | clean 4ch base plus microscopic factorized actuator (`warp_strength=0.0015`) |
| `ema_microdiffeo_styleloss_w20` | microscopic actuator plus light style losses |

The desired result is a small style lift from `~0.70` to `>0.72` while preserving the clean-base LPIPS band (`~0.42-0.44`).

### Clean-Base Push Readout

`ema_plain4_styleloss_w24`:

| epoch | clip_style | content_lpips | EC |
|---:|---:|---:|---:|
| 6 | 0.7044 | 0.4473 | 0.3893 |
| 7 | 0.7043 | 0.4508 | 0.3868 |
| 8 | 0.7045 | 0.4517 | 0.3863 |

`ema_plain4_canvas_w24`:

| epoch | clip_style | content_lpips | EC |
|---:|---:|---:|---:|
| 6 | 0.7045 | 0.4459 | 0.3903 |
| 7 | 0.7043 | 0.4493 | 0.3878 |
| 8 | 0.7041 | 0.4500 | 0.3873 |

`ema_microdiffeo_w20`:

| epoch | clip_style | content_lpips | EC |
|---:|---:|---:|---:|
| 6 | 0.6609 | 0.4714 | 0.3493 |
| 7 | 0.6607 | 0.4666 | 0.3524 |
| 8 | 0.6606 | 0.4644 | 0.3538 |

Interpretation:

- Style-side losses and latent canvas add only about `+0.004` style over the clean W20 base and push LPIPS to the target boundary.
- The microscopic diffeomorphic actuator is actively bad: it loses style and content at the same time.
- This reinforces the current diagnosis: clean 4ch is the correct content-preserving base, but this training loss family has a style ceiling near `0.70-0.705`.
- The next cheap test is inference strength on the clean W20 checkpoint, because the best clean checkpoint has LPIPS margin (`0.4215`) that might absorb a small style-strength increase.

Inference strength sweep on `ema_plain4_w20_anchor` epoch 6:

| style_strength | clip_style | content_lpips |
|---:|---:|---:|
| 1.05 | 0.7006 | 0.4216 |
| 1.10 | 0.7005 | 0.4215 |
| 1.15 | 0.7005 | 0.4216 |
| 1.20 | 0.7005 | 0.4215 |

Interpretation:

- Inference strength is effectively saturated for this clean 4ch checkpoint. It does not unlock the missing style.
- The clean base's style ceiling is therefore a learned-vector-field limitation, not merely an inference scaling issue.
- The next real route should be architectural: a style-only, non-geometric actuator that is less restrictive than the failed microdiffeo branch but still avoids the original strong warp path.

## EMA Non-Geometric Style Actuator Queue

The next queue is:

`I:\Github\Latent_Style\SchrodingerBridge\exp\vae_backend\ema_moment_dynamic`

| variant | purpose |
|---|---|
| `ema_plain4_moment_w20` | endpoint channel mean/std matching to style latent; no geometry channel |
| `ema_plain4_premoment_w20` | small pre-integration mean/std blend; gentle global style statistics |
| `ema_plain4_dynamic_w24` | dynamic style-conditioned output head; style capacity without warp |
| `ema_plain4_dynamic_moment_w20` | dynamic head plus endpoint moment matching |

These are deliberately non-geometric. They test whether the missing `~0.02` style can be recovered by style statistics and output-head capacity rather than by spatial warp.

`ema_plain4_moment_w20` epoch 6 eval:

| epoch | clip_style | content_lpips | EC |
|---:|---:|---:|---:|
| 6 | 0.7006 | 0.4219 | 0.4050 |

Visual read:

- The grid remains close to the clean plain4 base, with no obvious new geometry tearing.
- It does not improve style over `ema_plain4_w20_anchor`.
- Endpoint channel moment matching is therefore not the missing style actuator; it mostly preserves the existing vector-field behavior.

Full non-geometric queue readout:

| variant | best useful epoch | clip_style | content_lpips | EC | read |
|---|---:|---:|---:|---:|---|
| `ema_plain4_moment_w20` | 6 | 0.7006 | 0.4219 | 0.4050 | no gain over clean plain4 |
| `ema_plain4_premoment_w20` | 6 | 0.7008 | 0.4219 | 0.4051 | no gain |
| `ema_plain4_dynamic_w24` | 6 | 0.7127 | 0.4711 | 0.3770 | style capacity improves, content fails |
| `ema_plain4_dynamic_moment_w20` | 6 | 0.7070 | 0.4366 | 0.3984 | content-compatible but style still short |

Conclusion:

- Global moment matching is essentially neutral.
- The dynamic style-conditioned output head is a real style-capacity knob, but by itself it moves too much latent content.
- The next targeted path is not more moment matching; it is edge-safe geometry/style actuation.

## Edge-Safe Warp Attribution

Code-level diagnosis after inspecting `src/utils/diffeomorphic.py`:

- `diffeomorphic_metric_mask_gamma` was applied to `color_delta`.
- The actual `spatial_warp` still went into `grid_sample` without that metric mask.
- `_texture_tangent_warp` deliberately strengthens motion on high-gradient texture/edge support via `1 - exp(-grad * gate_strength)`.
- At 256x256, this means the branch is most likely to resample exactly where trees, windows, limbs, and thin contours are fragile.

Patch applied:

- Reuse the existing metric mask and multiply it into `spatial_warp` before `grid_sample`.
- Apply the same fix in both standard and `factorized_amp` diffeomorphic paths.

Next targeted test:

`ema_edgesafe_diffeo_w20`

- Weak factorized diffeo branch on the clean EMA base.
- `diffeomorphic_metric_mask_gamma=1.25`, `diffeomorphic_metric_mask_use_z0=true`.
- Purpose: test whether a small geometry/style actuator can recover style above the plain4 ceiling without reintroducing boundary fragmentation.
- Remote run started at `I:\Github\Latent_Style\SchrodingerBridge\exp\vae_backend\ema_edgesafe_diffeo`.

Result:

| epoch | clip_style | content_lpips | EC |
|---:|---:|---:|---:|
| 6 | 0.6609 | 0.4768 | 0.3458 |
| 7 | 0.6619 | 0.4734 | 0.3485 |
| 8 | 0.6610 | 0.4707 | 0.3499 |

Visual read:

- The edge-safe warp grid is less sharply fragmented than the earlier high-style diffeomorphic grids.
- But it becomes washed out and weakly stylized, with worse content than the clean plain4 base.
- This strongly suggests the old diffeomorphic branch's style lift came from exactly the risky edge/texture resampling path. Once that path is made boundary-safe, the style actuator loses its usefulness.

Attribution update:

- Severe local breakage is primarily a warp/edge-resampling artifact amplified by the diffeomorphic branch.
- Terminal SWD alone is not sufficient to produce the severe fragmentation, because `ema_plain4_w20_anchor` is visually clean.
- Safer warp is not enough to close the style gap; it prevents the artifact but also removes the style mechanism.
- The best current EMA branches are therefore:
  - clean/content: `ema_plain4_w20_anchor`, `clip_style~0.7007`, `LPIPS~0.4215`;
  - style-capacity probe: `ema_plain4_dynamic_w24`, `clip_style~0.7127`, but LPIPS too high;
  - balanced non-geometric: `ema_plain4_dynamic_moment_w20`, `clip_style~0.7070`, `LPIPS~0.4366`.

### Fragmentation Diagnosis Update

The most likely cause of the broken visual grids is not patch pressure by itself. The evidence separates the failure modes:

| branch | geometry path | best useful row | visual implication |
|---|---|---:|---|
| `ema_plain4_w20_anchor` | no warp | `0.7007 / 0.4215` | clean, content target passes |
| `ema_plain4_dynamic_w24` | no warp | `0.7127 / 0.4711` | style capacity rises, but content drifts globally |
| `ema_edgesafe_diffeo_w20` | edge-masked weak warp | `0.6619 / 0.4734` | less sharp tearing, but weak/wash-out style |
| older EMA diffeo guards | active tangent warp | `~0.72 / >0.52` | high style, visible broken edges |

Mechanistically, `_texture_tangent_warp` intentionally amplifies motion on high-gradient support, then `grid_sample` resamples the latent tensor there. At 256x256, those are exactly the fragile regions: tree branches, window grids, body contours, and thin foreground structures. The previous metric-mask bug made this worse because color was masked but spatial warp was not. After masking spatial warp as well, the artifact is reduced, but the style lift disappears.

So the current working conclusion is:

- Local fragmentation = high-gradient spatial warp / resampling artifact.
- Global content drift = dynamic/non-geometric style capacity without enough structure guard.
- SWD/patch pressure = necessary style driver, but not sufficient to explain the severe broken-contour pattern.

Two targeted probes are prepared:

| variant | purpose |
|---|---|
| `ema_warptv_diffeo_w20` | same weak diffeo path, but adds raw warp energy/TV regularization and divergence-free projection to test whether smooth low-energy warp can survive |
| `ema_dynamic_guard_w28` | no warp; stronger dynamic style head with stronger content/edge guard to test the non-geometric route |

The remote GPU is currently occupied by the KL-f4 fair-mode run, so these probes should be launched after that finishes.

## KL-f4 Fair-Mode Result

`klf4_mode_stylepush_w40_p5917` completed on the remote 3060.

| epoch | clip_style | content_lpips | EC |
|---:|---:|---:|---:|
| 6 | 0.6522 | 0.4887 | 0.3335 |
| 7 | 0.6540 | 0.4852 | 0.3367 |
| 8 | 0.6537 | 0.4852 | 0.3365 |

This was the fair KL-f4 test:

- posterior `mode` latents instead of noisy samples;
- larger f4-scale patch set `[5, 9, 17]`;
- strong style pressure (`terminal_swd=40`);
- 10GB-class training memory (`peak_gpu_memory_mb=10330`);
- eval on epochs `6/7/8`.

Result:

- It is worse than the earlier KL-f4 sample-latent result (`clip_style~0.66`, `content_lpips~0.459`).
- It is far below SD15 EMA clean/dynamic routes (`clip_style~0.70-0.713`) and below the target (`clip_style>0.72`, `content_lpips<0.45`).
- The extra f4 spatial support does not translate into useful CLIP-style transfer under the current LANCET/SWD objective.

Working conclusion:

KL-f4 should not remain a main backend candidate for the 256x256 paper route. Its latent geometry appears mismatched to the current style objective. If we revisit f4 later, it should be through a fundamentally different f4/wavelet-specific architecture, not by more patch/loss sweeps.

The follow-up EMA queue started immediately afterward:

| variant | purpose |
|---|---|
| `ema_warptv_diffeo_w20` | test smooth low-energy warp after fragmentation diagnosis |
| `ema_dynamic_guard_w28` | test stronger no-warp dynamic style capacity with tighter content guards |

## EMA Fragmentation Follow-Up

`ema_warptv_diffeo_w20`:

| epoch | clip_style | content_lpips | EC |
|---:|---:|---:|---:|
| 6 | 0.6610 | 0.4764 | 0.3461 |
| 7 | 0.6616 | 0.4727 | 0.3489 |
| 8 | 0.6608 | 0.4707 | 0.3498 |

`ema_dynamic_guard_w28`:

| epoch | clip_style | content_lpips | EC |
|---:|---:|---:|---:|
| 6 | 0.7078 | 0.4477 | 0.3909 |
| 7 | 0.7072 | 0.4650 | 0.3783 |
| 8 | 0.7075 | 0.4597 | 0.3822 |

Readout:

- Smooth, low-energy, divergence-free warp still fails. It is less useful than the clean no-warp route and does not recover style.
- This supports the stronger attribution: the old high-style diffeo result was exploiting unsafe edge resampling, not a stable geometric style mechanism.
- The dynamic no-warp route is the only currently useful style-capacity direction, but guarding it enough to hit `content_lpips<0.45` leaves style at only `~0.708`.
- Visual grids confirm the metric split: `warptv` is washed out; `dynamic_guard` is clearer and more painterly but still short of the `0.72` style target.

Current EMA frontier:

| role | variant | clip_style | content_lpips |
|---|---|---:|---:|
| clean/content | `ema_plain4_w20_anchor` e6 | 0.7007 | 0.4215 |
| guarded dynamic | `ema_dynamic_guard_w28` e6 | 0.7078 | 0.4477 |
| style probe | `ema_plain4_dynamic_w24` e6 | 0.7127 | 0.4711 |

Next step:

Before more training, run inference-strength sweeps on `ema_dynamic_guard_w28` e6 and `ema_plain4_dynamic_w24` e6. This checks whether an existing checkpoint can move along the frontier without changing weights.

### Dynamic Inference-Strength Sweep

`ema_dynamic_guard_w28` epoch 6:

| style_strength | clip_style | content_lpips |
|---:|---:|---:|
| 1.05 | 0.7076 | 0.4476 |
| 1.10 | 0.7076 | 0.4477 |
| 1.15 | 0.7075 | 0.4477 |

`ema_plain4_dynamic_w24` epoch 6:

| style_strength | clip_style | content_lpips |
|---:|---:|---:|
| 0.85 | 0.7056 | 0.4300 |
| 0.90 | 0.7083 | 0.4439 |
| 0.95 | 0.7107 | 0.4576 |

Conclusion:

There is no hidden inference-scale point satisfying `clip_style>0.72` and `content_lpips<0.45`. The dynamic checkpoints move along the expected tradeoff curve: style rises only when LPIPS leaves the acceptable band. Further progress needs a better no-warp architecture/loss balance, not inference scaling.

### Dynamic Frontier Training

Two stronger no-warp dynamic variants were tested after inference scaling failed.

`ema_dynamic_frontier_w32`:

| epoch | clip_style | content_lpips | EC |
|---:|---:|---:|---:|
| 6 | 0.7093 | 0.4690 | 0.3767 |
| 7 | 0.7082 | 0.4841 | 0.3653 |
| 8 | 0.7086 | 0.4794 | 0.3689 |

`ema_dynamic_frontier_guard_w36`:

| epoch | clip_style | content_lpips | EC |
|---:|---:|---:|---:|
| 6 | 0.7089 | 0.4637 | 0.3802 |
| 7 | 0.7077 | 0.4768 | 0.3703 |
| 8 | 0.7078 | 0.4730 | 0.3730 |

Readout:

- The extra dynamic capacity and higher SWD pressure do not break the `~0.71` style ceiling.
- The new points are strictly worse than `ema_dynamic_guard_w28` e6 for the target region because LPIPS moves above `0.45` while style remains below `0.72`.
- This strongly suggests the no-warp EMA dynamic family has reached its current engineering ceiling.

Current conclusion for `sd-vae-ft-ema`:

- It is the strongest of the newly tested VAE backends.
- It is not yet superior to the original paper route under the target criterion because it has not achieved `clip_style>0.72` and `content_lpips<0.45`.
- Its useful operating region is clean and content-preserving (`0.700-0.708` style, `0.42-0.448` LPIPS).
- Its high-style region comes from unsafe spatial warp, which causes visible fragmentation and fails content.

## SaMST/SaMam-Inspired Highpass SConv Probe

Reading SaMST and SaMam changed the next EMA probe. Both papers get useful style capacity from style-conditioned local operators:

- SaMST uses a compact style representation to generate depthwise convolution kernels (`SConv`), AdaIN parameters, and channel gates.
- SaMam repeats the same pattern inside a Mamba decoder: SConv for local style geometry, SAIN/SCM for controlled modulation, and zero-init so the block begins close to identity.

This matches our current diagnosis:

- Free spatial warp is the wrong style mechanism at 256 because it resamples exactly on fragile high-gradient boundaries.
- Pure no-warp dynamic heads preserve geometry but saturate around `clip_style~0.708-0.713`.
- Therefore the missing actuator should be local and style-conditioned, but not coordinate-moving.

Code change:

- Added optional `style_highpass_depthwise_head` to `LatentAdaCUT`.
- It is disabled by default.
- It takes the current input latent `x`, extracts a highpass band, generates per-channel depthwise kernels from the style code, and adds the filtered highpass response as an extra residual delta.
- Kernels are zero-initialized by default and bounded with `tanh`, following the SaMam zero-init/identity principle.
- It is deliberately applied only to highpass content, not to lowpass color/structure, so it should create local textons without a free warp field.

Local smoke:

- `py -3 -m py_compile` passes for `config_schema.py`, `lancet_backbone.py`, `lancet_runtime.py`, and `run_vae_backend_256_probe.py`.
- A tiny CPU forward pass with `dynamic_style_operator_head=True` and `style_highpass_depthwise_head=True` returned finite output.
- Probe dry-run generated valid configs.

Remote queue prepared:

| variant | intent |
|---|---|
| `ema_sconv_hp_w28_guard` | guarded highpass depthwise operator; target `LPIPS<0.45` while lifting style above dynamic guard |
| `ema_sconv_hp_w36_style` | stronger highpass depthwise operator; target crossing `clip_style>0.72` with acceptable LPIPS drift |

Expected diagnostic signature:

- If this works, `clip_style` should rise beyond the current no-warp ceiling while `content_lpips` remains closer to `ema_dynamic_guard_w28` than to old diffeo guards.
- If style does not rise, then style-conditioned local highpass filtering is insufficient in the current latent/SWD objective.
- If LPIPS rises sharply, the issue is not spatial warp alone; the generated highpass texton branch itself is disturbing content phase.

### Highpass SConv Result

Remote task:

- Machine: 3060 host, scheduled task `LANCET_EMA_SConv_Highpass`.
- Output: `exp/vae_backend/ema_sconv_highpass`.
- Light local mirror: `exp/vae_backend/ema_sconv_highpass_light`.
- Training memory target was met: guarded probe peaked at `9832 MB`; style probe peaked at `10130 MB`.

Metrics:

| variant | epoch | clip_style | content_lpips | EC | peak MB |
|---|---:|---:|---:|---:|---:|
| `ema_sconv_hp_w28_guard` | 6 | 0.7060 | 0.4402 | 0.3952 | 9832 |
| `ema_sconv_hp_w28_guard` | 7 | 0.7054 | 0.4529 | 0.3859 | 9832 |
| `ema_sconv_hp_w28_guard` | 8 | 0.7057 | 0.4499 | 0.3882 | 9832 |
| `ema_sconv_hp_w36_style` | 6 | 0.7112 | 0.4999 | 0.3556 | 10130 |
| `ema_sconv_hp_w36_style` | 7 | 0.7069 | 0.5152 | 0.3427 | 10130 |
| `ema_sconv_hp_w36_style` | 8 | 0.7090 | 0.5102 | 0.3473 | 10130 |

Visual read:

- `ema_sconv_hp_w28_guard` is stable and often visually clean, but it is conservative. It does not lift style beyond `ema_dynamic_guard_w28`.
- `ema_sconv_hp_w36_style` produces stronger paint texture, but the gain is bought mostly by larger texture/color perturbation. The best epoch reaches only `clip_style=0.7112`, while LPIPS is already essentially `0.50`.
- The grid shows the same failure mode as the metrics: highpass style filtering can create more local activity, but without semantic/edge locking it does not know where the activity should go. It adds painterly energy rather than object-aware textons.

Conclusion:

- The SaMST/SaMam-inspired highpass SConv head is numerically stable and fits the intended 10 GB runtime envelope.
- It does not break the EMA no-warp style ceiling. The strongest row is not better than the previous `ema_plain4_dynamic_w24` style point (`0.7127 / 0.4711`) and is much worse on LPIPS.
- The guarded row preserves the target LPIPS range but loses style.
- Therefore a bare style-conditioned highpass operator is insufficient. The missing component is not merely local filtering; it is edge/semantic/support gating or phase locking for where the local style residual is allowed to act.

Next model implication:

- Do not spend another broad sweep on ungated highpass SConv strength.
- The next meaningful probe should keep the operator-bank idea but add a content support gate:
  - derive an active support mask from content latent highpass / edge magnitude;
  - let style kernels act mainly inside active texture/support regions;
  - suppress large highpass residuals in flat semantic regions;
  - optionally add a highpass phase-consistency penalty so the generated residual does not anti-align with content structure.

### Support-Gated Highpass SConv Result

Follow-up remote task:

- Machine: 3060 host, scheduled task `LANCET_EMA_SConv_Support`.
- Output: `exp/vae_backend/ema_sconv_support`.
- Light local mirror: `exp/vae_backend/ema_sconv_support_light`.
- Both runs stayed inside the intended 10 GB class: peak memory `10307 MB` and `10265 MB`.

Metrics:

| variant | epoch | clip_style | content_lpips | EC | peak MB |
|---|---:|---:|---:|---:|---:|
| `ema_sconv_support_w30_guard` | 6 | 0.7110 | 0.4680 | 0.3782 | 10307 |
| `ema_sconv_support_w30_guard` | 7 | 0.7084 | 0.4812 | 0.3675 | 10307 |
| `ema_sconv_support_w30_guard` | 8 | 0.7098 | 0.4783 | 0.3704 | 10307 |
| `ema_sconv_support_w40_style` | 6 | 0.7168 | 0.5261 | 0.3397 | 10265 |
| `ema_sconv_support_w40_style` | 7 | 0.7141 | 0.5395 | 0.3289 | 10265 |
| `ema_sconv_support_w40_style` | 8 | 0.7145 | 0.5355 | 0.3319 | 10265 |

Visual read:

- `w30_guard` is visually cleaner than the style-push branch, but the output still reads as latent-local texture/color pressure rather than object-aware style placement. It raises style slightly over `ema_dynamic_guard_w28` (`0.7078 / 0.4477`) but pays too much LPIPS (`0.4680`).
- `w40_style` is the first no-warp/operator-bank branch to clearly move above the previous `~0.713` no-warp ceiling (`0.7168`), but the price is unacceptable: LPIPS jumps to `0.5261`, EC collapses to `0.3397`, and grids show large texture/color fields overwhelming content structure.
- Compared with bare highpass SConv (`0.7112 / 0.4999`), support gating improves the style ceiling but does not solve content preservation. The gate is acting as a high-frequency activity mask, not as semantic placement or phase locking.

Conclusion:

- Content highpass support gating is directionally useful for style capacity, but insufficient for the target `clip_style > 0.72` and `content_lpips < 0.45`.
- The current support mask is too primitive: it says "where the image has local energy", not "where this style texton belongs" or "which phase/orientation preserves the content edge".
- Further increasing SConv strength is not the right next move. The next architecture step should add semantic/edge/phase routing:
  - semantic support from the existing SWD/attention assignment, so style kernels act on matched semantic regions rather than all textured regions;
  - edge-side suppression or bilateral gating to prevent kernels from crossing object boundaries;
  - highpass phase-consistency regularization so generated textons align with content gradients instead of washing across them.

Updated frontier after this probe:

| role | variant | clip_style | content_lpips | interpretation |
|---|---|---:|---:|---|
| best target-region EMA | `ema_dynamic_guard_w28` e6 | 0.7078 | 0.4477 | closest to LPIPS target |
| best no-warp style probe | `ema_sconv_support_w40_style` e6 | 0.7168 | 0.5261 | style improves, content fails |
| best SConv guarded probe | `ema_sconv_support_w30_guard` e6 | 0.7110 | 0.4680 | support gate helps style but not enough |
| unsafe high-style EMA | `ema_t01_w20` e6 | 0.7237 | 0.5755 | reaches style target through visible deformation |

Working decision:

EMA remains the only replacement VAE worth continuing. KL-f4 and SDXL are no longer mainline for the 256x256 objective. The EMA path is now bottlenecked by operator routing, not by raw style strength. To beat the target, LANCET needs a semantic/phase-aware local style operator; more unguided SWD, warp, or SConv strength only moves along the same bad style-content tradeoff.

### Next Probe: Semantic-Confidence Gated SConv

Implementation change prepared after the support-gate result:

- `SemanticCrossAttn` already stores `last_attn`; runtime already retrieves the final semantic attention tensor while building the body representation.
- A new optional `style_highpass_depthwise_semantic_gate` passes that attention confidence into the highpass SConv branch.
- The gate uses per-content-token max assignment confidence, reshaped back to the latent grid and upsampled to the output latent resolution.
- This is deliberately still cheap and local: it does not add a new segmentation model or external semantic encoder.

Hypothesis:

- The failed support gate only knew "where content has high-frequency energy".
- The semantic gate should instead bias highpass texton kernels toward locations where the model has a confident content-to-style assignment.
- If it works, the best result should keep the style gain of `w40_style` (`~0.717`) while pulling LPIPS materially below `0.526`.
- If it fails, the current learned semantic attention is not a reliable routing signal for visible style operators, and the next step must use a stronger object/edge/phase diagnostic rather than internal attention confidence.

Prepared variants:

| variant | purpose |
|---|---|
| `ema_sconv_semantic_w34_guard` | moderate style pressure, support gate + semantic confidence gate, target better than `0.7110 / 0.4680` |
| `ema_sconv_semantic_w44_style` | style-push branch, tests whether semantic routing can approach/cross `0.72` with less LPIPS damage than `0.7168 / 0.5261` |

Local verification:

- AST parse passed for `config_schema.py`, `lancet_backbone.py`, `lancet_runtime.py`, and `run_vae_backend_256_probe.py`.
- Tiny CPU forward with support + semantic highpass gates returned finite output: `torch.Size([2, 4, 16, 16])`.

### Semantic-Confidence Gated SConv Result

Implementation note:

- The first eval attempt for `ema_sconv_semantic_w34_guard` failed because `semantic_self_topology_gate` hit a bf16 dtype mismatch in `torch.lerp`.
- Fix: cast `topology_painted` back to `painted_tokens.dtype` inside `SemanticCrossAttn`.
- Training checkpoints were valid, so only eval was rerun for `w34_guard`.
- Corrected local ledger: `exp/vae_backend/ema_sconv_semantic_light/vae_backend_256_results_repaired.csv`.

Metrics:

| variant | epoch | clip_style | content_lpips | EC | peak MB |
|---|---:|---:|---:|---:|---:|
| `ema_sconv_semantic_w34_guard` | 6 | 0.7118 | 0.4823 | 0.3685 | 10980 |
| `ema_sconv_semantic_w34_guard` | 7 | 0.7091 | 0.4960 | 0.3574 | 10980 |
| `ema_sconv_semantic_w34_guard` | 8 | 0.7104 | 0.4915 | 0.3612 | 10980 |
| `ema_sconv_semantic_w44_style` | 6 | 0.7144 | 0.5286 | 0.3367 | 10978 |
| `ema_sconv_semantic_w44_style` | 7 | 0.7093 | 0.5437 | 0.3237 | 10978 |
| `ema_sconv_semantic_w44_style` | 8 | 0.7111 | 0.5384 | 0.3282 | 10978 |

Readout:

- Semantic confidence gating does not improve the frontier. It is worse than support-only gating in both guarded and style-push regimes.
- `w34_guard` is weaker than `ema_sconv_support_w30_guard` on LPIPS (`0.4823` vs `0.4680`) while only barely improving style (`0.7118` vs `0.7110`).
- `w44_style` is weaker than `ema_sconv_support_w40_style` in both style and LPIPS (`0.7144 / 0.5286` vs `0.7168 / 0.5261`).

Interpretation:

- The existing internal `SemanticCrossAttn` confidence is not a good enough visible-style routing signal.
- It likely measures assignment sharpness in the learned latent body, not correctness of object-level style placement after decoding.
- Combining it with high-frequency support also over-constrains useful texture placement: it suppresses some style energy while still failing to protect content from broad color/texture drift.

Updated decision:

- Stop sweeping internal-attention confidence gates for this EMA branch.
- The next useful diagnostic should compare model deltas against the Seedream 4.5 golden outputs at the image/patch level:
  - where Seedream changes color but preserves object boundaries;
  - where LANCET introduces high-frequency residuals across boundaries;
  - whether our CLIP style gains come from broad palette shifts rather than target-style local textons.
- If we continue architecture work, the gate must come from explicit image-space/object-edge/patch-phase diagnostics, not raw internal attention confidence.

### Seedream 4.5 Gap Diagnostic

Added script:

- `tools/experiments/diagnose_seedream_gap.py`

Inputs:

- Seedream 4.5 golden images: `Related_Works/baseline_pipeline/results/seedream45_api/protocol_a_800/images`
- original VAE t01 output: `exp/diffeomorphic_tangent_sweep/t01_ws0p03_g6_nl0p05/full_eval/epoch_0008/images`
- EMA support-gated SConv mirrors copied from remote:
  - `exp/diagnostics/seedream_gap/inputs/ema_sconv_support_w30_guard_e6`
  - `exp/diagnostics/seedream_gap/inputs/ema_sconv_support_w40_style_e6`

Outputs:

- detail CSV: `exp/diagnostics/seedream_gap/seedream_gap_image_metrics.csv`
- summary CSV: `exp/diagnostics/seedream_gap/seedream_gap_summary.csv`
- worst-case contact sheet: `exp/diagnostics/seedream_gap/seedream_gap_worst_cases.png`

The diagnostic uses only image-space statistics, not CLIP/LPIPS:

- `flat_color_flood`: residual magnitude in source-flat regions where the output is also low-gradient;
- `highpass_delta_energy`: high-frequency residual energy;
- `highpass_phase_cos`: cosine between source Laplacian and output-source Laplacian;
- RGB mean/std shift;
- source-edge residual and edge/flat ratio.

All-pair summary over the 721 Seedream-comparable images:

| method | mean abs delta | RGB shift | flat flood | highpass energy | phase cos | flat flood gap vs Seedream | highpass gap vs Seedream |
|---|---:|---:|---:|---:|---:|---:|---:|
| Seedream 4.5 | 0.1508 | 0.2020 | 0.1401 | 0.1011 | -0.3524 | 0.0000 | 0.0000 |
| t01 original VAE e8 | 0.1573 | 0.2115 | 0.1449 | 0.1346 | -0.8097 | +0.0048 | +0.0335 |
| EMA SConv support w30 guard e6 | 0.1604 | 0.2125 | 0.1539 | 0.1433 | -0.7516 | +0.0138 | +0.0422 |
| EMA SConv support w40 style e6 | 0.1746 | 0.2164 | 0.1656 | 0.1548 | -0.7165 | +0.0255 | +0.0537 |

Per-target failure concentration:

- Worst gaps are on Monet and Vangogh.
- `ema_support_w40_style_e6` has the largest excess flat-region flood:
  - Monet: `flat_flood_gap=+0.0565`, `highpass_gap=+0.0848`;
  - Vangogh: `flat_flood_gap=+0.0529`, `highpass_gap=+0.0695`.
- The guarded branch is cleaner but still inherits the same failure:
  - Monet: `flat_flood_gap=+0.0519`, `highpass_gap=+0.0686`.

Visual read from `seedream_gap_worst_cases.png`:

- Seedream preserves object silhouettes and often repaints inside existing semantic regions. It changes palette strongly, but the change is organized along the source layout.
- Our t01/EMA outputs often add gray-blue or green high-frequency residuals over the whole region. The stronger SConv branch increases local texture, but the texture is not phase-locked to the source boundary.
- The face/anime cases are especially diagnostic: Seedream keeps eyes, glasses, and hair boundaries sharp while changing color/stroke. Our outputs wash texture across those boundaries.
- For Monet/Vangogh, the current model tends to create a global mist of highpass energy rather than target-style local textons.

Interpretation:

- The bottleneck is not insufficient style energy. The `ema_t01_w20` branch already reaches `clip_style=0.7237`, and support SConv reaches `0.7168`; both fail because style energy is spatially misrouted.
- The useful missing object is an explicit image/latent-space edge-phase router. Internal attention confidence did not work because assignment sharpness is not the same as visible boundary correctness.
- The next model change should not be another unguided style-weight or SConv-strength sweep. It should force style residuals to be local and phase-compatible:
  - derive a boundary/flat mask from the source latent or decoded source;
  - suppress highpass residual in flat low-confidence regions;
  - align generated highpass residual with source Laplacian/gradient phase near object boundaries;
  - allow stronger texton injection only inside semantic regions, not across boundary rings.

Next concrete probe:

- Start from the best target-region EMA branch, `ema_dynamic_guard_w28` e6 (`0.7078 / 0.4477`), or the saved frontier point `ema_dynamic_frontier_w32` e6 (`0.7093 / 0.4690`).
- Add an edge/phase loss rather than a new free operator:
  - penalty on highpass residual in source-flat regions;
  - penalty on highpass residual anti-alignment around source-edge rings;
  - optional relaxed LPIPS target up to `0.49-0.50` if `clip_style` crosses `0.72` with locally organized textons.
- Success should be judged by both CLIP/LPIPS and the Seedream-gap diagnostic: `highpass_delta_energy_vs_seedream` and `flat_color_flood_vs_seedream` must not grow like the `w40_style` branch.

### Edge/Phase Loss Probe

Implementation:

- Added `w_flat_highpass_suppression` to penalize high-frequency residuals in source-flat regions.
- Added `w_edge_phase_alignment` to penalize anti-phase Laplacian residuals around source edge support.
- Both are latent-space losses and default to zero, so prior configs remain reproducible.
- Training logs now include:
  - `flat_highpass_suppression`
  - `edge_phase_alignment`

Smoke:

- Remote task: `LANCET_EMA_EdgePhase_Smoke`
- Variants: `ema_edgephase_w32_flatguard`, `ema_edgephase_w40_styleguard`
- Setting: 1 epoch, 30 batches, no eval.
- Result: both `train_ok`; peak VRAM around `9.94 GB`; no OOM/non-finite.

Full run:

- Remote task: `LANCET_EMA_EdgePhase_Full`
- Output: `exp/vae_backend/ema_edgephase`
- Full eval epochs: 6/7/8.

Metrics:

| variant | epoch | clip_style | content_lpips | EC | peak MB |
|---|---:|---:|---:|---:|---:|
| `ema_edgephase_w32_flatguard` | 6 | 0.7091 | 0.4672 | 0.3778 | 9945 |
| `ema_edgephase_w32_flatguard` | 7 | 0.7080 | 0.4822 | 0.3666 | 9945 |
| `ema_edgephase_w32_flatguard` | 8 | 0.7085 | 0.4772 | 0.3704 | 9945 |
| `ema_edgephase_w40_styleguard` | 6 | 0.7104 | 0.5414 | 0.3258 | 9944 |
| `ema_edgephase_w40_styleguard` | 7 | 0.7064 | 0.5488 | 0.3187 | 9944 |
| `ema_edgephase_w40_styleguard` | 8 | 0.7073 | 0.5442 | 0.3223 | 9944 |

Seedream-gap diagnostic after pulling epoch-6 images locally:

| method | flat flood gap vs Seedream | highpass gap vs Seedream | phase cos | readout |
|---|---:|---:|---:|---|
| `t01_original_vae_e8` | +0.0048 | +0.0335 | -0.8097 | old VAE still lowest artifact gap, despite lower-quality style placement |
| `ema_support_w30_guard_e6` | +0.0138 | +0.0422 | -0.7516 | guarded SConv: style local energy, still too much flat-region activity |
| `ema_edgephase_w32_flatguard_e6` | +0.0119 | +0.0453 | -0.7325 | flat flood improves slightly, but highpass mist increases; no style gain |
| `ema_support_w40_style_e6` | +0.0255 | +0.0537 | -0.7165 | style-push support SConv: more flat-region pollution |
| `ema_edgephase_w40_styleguard_e6` | +0.0289 | +0.0491 | -0.7381 | highpass gap improves slightly vs w40 SConv, but flat flood and LPIPS worsen |

Readout:

- The flat loss does what it says in the mild branch: flat flood gap drops from `+0.0138` to `+0.0119`.
- But the energy is not routed into correct textons. It leaks into other high-frequency residuals (`highpass gap +0.0453`, worse than `+0.0422`) and CLIP style stays at `~0.709`.
- Stronger style pressure plus weak edge/phase losses fails badly: `0.7104 / 0.5414`, so this is not a hidden path to `0.72`.
- This proves the current bottleneck is architectural routing, not a missing scalar penalty. A loss can mildly discourage one artifact signature, but the model then finds a different highpass residual unless the operator itself is constrained.

Updated decision:

- Do not continue scalar edge/phase-loss sweeps as the main route.
- The next useful design must apply the constraint inside the style actuator:
  - edge-aware residual decomposition before the final output head;
  - separate flat-region lowpass color path from edge/texton highpass path;
  - explicit mask/gate on the generated highpass residual, not only a penalty after the residual is produced.
- The safe base remains `ema_dynamic_guard_w28` (`0.7078 / 0.4477`) and `ema_dynamic_frontier_w32` (`0.7093 / 0.4690`). The style target is reachable only if the new actuator can add organized textons without increasing flat flood/highpass mist.

### Routed Residual Actuator Probe

Implementation:

- Added an optional output residual router in `LatentAdaCUTRuntimeMixin._compute_delta`.
- Config switches:
  - `output_residual_router`
  - `output_router_kernel`
  - `output_router_edge_gamma`
  - `output_router_highpass_floor`
  - `output_router_lowpass_strength`
  - `output_router_edge_lowpass_suppression`
- The router decomposes the predicted residual:
  - `delta_low = avg_pool(delta)`;
  - `delta_high = delta - delta_low`;
  - `delta_high` is multiplied by a source-derived edge/high-frequency support gate;
  - `delta_low` remains the smooth color/style path.
- This moves the constraint from a scalar penalty after the fact into the actuator itself.

Smoke:

- Remote task: `LANCET_EMA_Routed_Smoke`
- Variants: `ema_routed_w36_texton`, `ema_routed_w44_stylepush`
- Setting: 1 epoch, 30 batches, no eval.
- Result: both `train_ok`; peak VRAM around `9.93 GB`; no OOM/non-finite.

Full run:

- Remote task: `LANCET_EMA_Routed_Full`
- Output: `exp/vae_backend/ema_routed`
- Full eval epochs: 6/7/8.

Metrics:

| variant | epoch | clip_style | content_lpips | EC | peak MB |
|---|---:|---:|---:|---:|---:|
| `ema_routed_w36_texton` | 6 | 0.7157 | 0.5020 | 0.3564 | 9936 |
| `ema_routed_w36_texton` | 7 | 0.7133 | 0.5188 | 0.3432 | 9936 |
| `ema_routed_w36_texton` | 8 | 0.7143 | 0.5133 | 0.3476 | 9936 |
| `ema_routed_w44_stylepush` | 6 | 0.7063 | 0.5668 | 0.3060 | 9929 |
| `ema_routed_w44_stylepush` | 7 | 0.7017 | 0.5725 | 0.2999 | 9929 |
| `ema_routed_w44_stylepush` | 8 | 0.7024 | 0.5695 | 0.3024 | 9929 |

Seedream-gap diagnostic after pulling epoch-6 images locally:

| method | flat flood gap vs Seedream | highpass gap vs Seedream | phase cos | block std | readout |
|---|---:|---:|---:|---:|---|
| `t01_original_vae_e8` | +0.0048 | +0.0335 | -0.8097 | 0.0807 | still lowest artifact gap, but weak/washed style |
| `ema_dynamic_guard_w28/e32-like` | +0.0119 | +0.0453 | -0.7325 | 0.0983 | balanced EMA frontier, not enough style |
| `ema_support_w40_style_e6` | +0.0255 | +0.0537 | -0.7165 | 0.1128 | SConv style pressure, high flood/highpass mist |
| `ema_routed_w36_texton_e6` | +0.0225 | +0.0505 | -0.7222 | 0.1048 | router improves highpass gap vs SConv w40, but LPIPS still too high |
| `ema_routed_w44_stylepush_e6` | +0.0360 | +0.0494 | -0.7433 | 0.1149 | W44 collapses into flat flood and loses style |

Readout:

- Structural routing is directionally more useful than scalar edge/phase losses:
  - scalar edge/phase best stayed near `0.709`;
  - routed W36 reaches `0.7157`, near the previous support-SConv style ceiling.
- But the current gate is too coarse. It reduces highpass gap slightly compared with SConv W40 (`+0.0505` vs `+0.0537`) while still allowing too much flat-region change (`+0.0225`) and LPIPS `0.5020`.
- Stronger W44 pressure fails: style drops to `0.7063` while LPIPS reaches `0.5668`. That is a genuine negative result, not just undertraining.

Conclusion for EMA after this round:

- EMA is still better than KL-f4 and SDXL, but it is not yet better than the original VAE point on the target criterion.
- The no-warp EMA family has a hard clean-style ceiling around `0.708-0.716` depending on how much LPIPS we spend.
- The original unsafe EMA/t01-style route can cross `0.72`, but only by visible deformation (`~0.575 LPIPS`).
- The routed actuator proves the right direction is structural, but a source-derived edge/highpass scalar gate is not enough. It still cannot decide object-level texton placement like Seedream.

Next route if continuing EMA:

- Do not increase W44/W48 style pressure.
- The next architecture needs object/region conditioning, not just edge magnitude:
  - use semantic bins or style-pair assignment to route residuals by region;
  - keep `delta_low` as a tightly bounded color path;
  - generate highpass textons from a style bank only inside selected regions;
  - add a hard cap on flat-region lowpass drift, not only highpass drift.
- If we cannot add region-aware routing cheaply, the honest conclusion is that `sd-vae-ft-ema` is a useful diagnostic backend but does not surpass the original VAE at `256x256` under the current LANCET objective.

### SaMST / SaMam Reading Update

I read the local papers and code:

- `F:\SaMST.pdf`
- `F:\SaMam.pdf`
- `Related_Works/repos/external/SaMST`
- `Related_Works/repos/SaMam`

The high-level note is recorded in:

- `docs/experiments/2026-05-26-samam-samst-design-and-critique.md`

Main takeaway for this VAE-backend objective:

- SaMST and SaMam do **not** support simply increasing style loss or adding freer warp.
- Their transferable trick is constrained style-conditioned operator routing:
  - local textons through depthwise dynamic kernels;
  - global color/statistics through normalization or low-pass affine modulation;
  - feature selection through channel gates;
  - stability through identity / zero initialization.
- This matches our latest evidence:
  - scalar flat/edge losses mildly suppress one artifact but do not raise style;
  - SConv/support/routed branches raise style more effectively;
  - current gates are still too coarse and cause flat-region drift / highpass mist.

Updated plan:

1. Stop KL-f4 and SDXL drop-in sweeps unless a fundamentally different architecture is introduced.
2. Continue only EMA for one more architecture-level attempt.
3. Start from `ema_dynamic_frontier_w32` or `ema_sconv_support_w30_guard`.
4. Add region-aware style routing rather than higher global SWD:
   - low-pass color path tightly bounded;
   - high-pass texton path via zero-init style-conditioned depthwise kernels;
   - region/semantic gate from content low-frequency clusters or semantic bins, not only source-edge magnitude;
   - reject variants that gain style only by increasing Seedream-gap flat flood or highpass mist.

Current completion status against the goal:

- `CompVis KL-f4`: tested and currently rejected as a drop-in backend.
- `SDXL`: tested enough to deprioritize for 256x256 current operators.
- `sd-vae-ft-ema`: best replacement backend, not yet proven superior to original VAE.
- Goal target `clip_style > 0.72`, `content_lpips < 0.45`: not reached.
- Therefore the goal remains open; the only plausible remaining route is region-aware EMA operator routing.

### Region-Gated SConv Result And Hypothesis Revision

I added a first region-gated SConv actuator:

- `style_highpass_depthwise_region_gate`
- `style_highpass_depthwise_region_bins`
- `style_highpass_depthwise_region_gamma/floor/smooth_kernel`

The gate derives low-frequency content bins from the source latent and lets the style embedding generate a per-bin high-pass texton permission map. This was meant to test whether a simple region gate is enough to improve beyond edge/support gating.

Remote result:

| variant | epoch | status | clip_style | content_lpips | EC | readout |
|---|---:|---|---:|---:|---:|---|
| `ema_region_sconv_w30_guard_stable` | 0 | train_failed_1 | - | - | - | non-finite gradient at step 1 in dynamic output head |
| `ema_region_sconv_w40_style` | 6 | ok | 0.7171 | 0.5174 | 0.3460 | slight LPIPS improvement over support-W40, no real style breakthrough |
| `ema_region_sconv_w40_style` | 7 | ok | 0.7133 | 0.5321 | 0.3337 | worse |
| `ema_region_sconv_w40_style` | 8 | ok | 0.7143 | 0.5276 | 0.3374 | worse |

Comparison:

- previous `ema_sconv_support_w40_style` e6: `0.7168 / 0.5261`.
- region gate e6: `0.7171 / 0.5174`.

This is a real but small regularization effect, not a new route to `0.72+`.

Revised hypothesis:

> Low-frequency content bins are not semantic routing. They only provide weak spatial smoothing. They cannot solve object-level texton placement.

Also, the repeated non-finite failures in `w32_guard` / `w30_guard_stable` point to a separate issue:

> The free dynamic output head is an unstable gradient sink under guarded high-pass routing. Strong content/kinetic guards concentrate terminal-SWD gradients into the output-head generator at step 1, producing non-finite gradients.

Therefore, the next useful experiment should **not** be another weight sweep. The next code-level change should isolate the actuator:

1. Disable `dynamic_style_operator_head` for guarded EMA variants.
2. Keep a normal static output head for low-pass residual.
3. Move style capacity into constrained branches only:
   - bounded low-pass palette branch;
   - zero-init high-pass SConv/texton branch;
   - optional support/semantic gate.
4. Evaluate whether the constrained actuator can raise style without dynamic-head gradient singularity.

The sharper theory model:

| component | should carry | failure if overloaded |
|---|---|---|
| low-pass palette path | color/statistical shift | flat color flood, LPIPS damage |
| high-pass SConv path | textons / brush microstructure | noisy highpass mist |
| structure/warp path | ideally off for EMA 256 | fragmentation and phase anti-alignment |
| dynamic output head | only mild residual, or disabled | non-finite gradients / unconstrained artifacts |

This updates the plan again:

- Stop tuning region-gate floors/gammas.
- Test **static-head + constrained SConv** as an architectural ablation.
- If static-head SConv still caps at `~0.716`, conclude that EMA under current latent objective lacks the needed semantic style-routing capacity, and it is not superior to the original VAE for the requested target.

### Static-Head SConv Smoke

Implemented two architecture-isolation variants:

- `ema_static_sconv_w32_guard`
- `ema_static_sconv_w40_style`

Both remove `dynamic_style_operator_head` and keep style capacity in the constrained high-pass SConv branch.

Remote smoke:

| variant | max batches | status | peak MB | readout |
|---|---:|---|---:|---|
| `ema_static_sconv_w32_guard` | 30 | train_ok | 9948 | no non-finite; `|v|` about 0.323 |
| `ema_static_sconv_w40_style` | 30 | train_ok | 9938 | no non-finite |

This directly supports the revised diagnosis:

> The guarded branch instability was caused by the free dynamic output head acting as a gradient sink. Removing it makes the same family train stably.

Next action:

- Run full 8-epoch eval for `ema_static_sconv_w32_guard` and `ema_static_sconv_w40_style`.
- If style remains below `0.72`, the problem is not numerical stability but insufficient semantic/texton routing capacity.

Full result:

| variant | epoch | clip_style | content_lpips | EC | readout |
|---|---:|---:|---:|---:|---|
| `ema_static_sconv_w32_guard` | 6 | 0.7041 | 0.4419 | 0.3930 | LPIPS reaches target, style too weak |
| `ema_static_sconv_w32_guard` | 7 | 0.7044 | 0.4427 | 0.3926 | stable but flat |
| `ema_static_sconv_w32_guard` | 8 | 0.7050 | 0.4443 | 0.3918 | best guard epoch, still low style |
| `ema_static_sconv_w40_style` | 6 | 0.7063 | 0.4619 | 0.3801 | more style pressure barely helps |
| `ema_static_sconv_w40_style` | 7 | 0.7061 | 0.4637 | 0.3787 | no improvement |
| `ema_static_sconv_w40_style` | 8 | 0.7066 | 0.4651 | 0.3780 | no improvement |

Interpretation:

- Static output head fixes the numerical instability and content damage.
- But high-pass SConv alone is severely underpowered for CLIP-style.
- The dynamic output head was carrying broad low-frequency/statistical style, not just artifact.
- Therefore the missing carrier is a bounded low-pass palette/statistics branch, not more high-pass kernels.

### Theory Reset: What The VAE Backend Is Actually Testing

The replacement-VAE experiments should not be interpreted as "which VAE has nicer reconstructions". They test whether a latent space supplies the right carriers for three different style-transfer energies:

| energy | mathematical carrier | desired operator | current failure |
|---|---|---|---|
| palette / low-order style | low-frequency mean and covariance | bounded affine / low-pass residual | color flood if unconstrained |
| texton / brush style | local high-pass covariance | style-conditioned depthwise/local kernels | weak style if too constrained; mist if ungated |
| structural deformation | phase / edge-aligned displacement | ideally absent or strictly tangent | fragmentation and LPIPS blow-up |

Under this decomposition:

- KL-f4 gives more spatial samples but not a pretrained semantic/statistical latent manifold matched to our LANCET objective. It improves sampling topology in theory but loses the useful SD latent prior in practice.
- SDXL has a richer VAE but its 256 latent statistics are mismatched to the original operators; the style energy does not land in visible texture efficiently.
- EMA preserves SD15-style latent semantics and avoids MSE over-smoothing, so it is the only replacement backend with plausible useful carriers.

The current EMA evidence can be explained cleanly:

1. Dynamic output head variants reach `0.711-0.717`, and unsafe t01-like EMA can reach `0.7237`, because a free style-conditioned residual can inject broad style energy.
2. The same free residual causes LPIPS/content collapse, flat flood, highpass mist, and guarded non-finite gradients.
3. Static-head SConv guard is stable and reaches LPIPS `<0.45`, but style falls to about `0.705`, because high-pass texton kernels alone cannot carry enough global style.
4. Region bins help only slightly because low-frequency bins are not semantic objects.

This yields a stronger hypothesis:

> To exceed `0.72` style while keeping LPIPS near `0.45`, style energy must be split into two bounded carriers: a low-rank palette/statistics carrier and a high-pass texton carrier. A single free residual head is too unconstrained; a high-pass-only SConv head is too weak.

Therefore the next design should not be "more SWD". It should be a constrained two-carrier actuator:

1. **Palette carrier**
   - low-pass only;
   - rank/channel-limited;
   - bounded by content-region masks or global moment caps;
   - allowed to move CLIP-style through color/statistics without creating high-frequency mist.

2. **Texton carrier**
   - high-pass only;
   - zero-init style-conditioned depthwise kernels;
   - support/edge gated;
   - no coordinate warp.

3. **Residual safety**
   - dynamic output head off or severely low-rank;
   - static output head handles only mild base residual;
   - diagnostic acceptance requires style gain without Seedream-gap flat flood/highpass increase.

The next probe should be called something like `ema_dualcarrier_w36_styleguard`:

- `dynamic_style_operator_head=False`;
- keep `style_highpass_depthwise_head=True`;
- add a new bounded low-pass style affine/palette branch;
- terminal SWD around `36`, not `44+`;
- target: cross `0.712-0.716` first while keeping LPIPS `<0.48`; only then style-push toward `0.72`.

If this dual-carrier actuator still cannot beat `~0.716`, the conclusion becomes much firmer:

> EMA is cleaner and diagnostically useful, but the current 256x256 LANCET backend cannot make it superior to the original VAE. The original VAE remains better for style ceiling, while EMA is better for content-safe variants.

### Dual-Carrier Smoke

Implemented:

- `StyleLowpassAffineHead`: a bounded, zero-init low-pass affine carrier generated from the style embedding.
- Existing `StyleHighpassDepthwiseHead`: high-pass texton carrier.
- Dynamic output head disabled for the dual-carrier variants.

Variants:

| variant | intent |
|---|---|
| `ema_dualcarrier_w36_styleguard` | styleguard: recover style beyond static-SConv while keeping content |
| `ema_dualcarrier_w44_style` | style-push: test whether dual carriers can approach `0.72` without free residual |

Remote 30-batch smoke:

| variant | status | peak MB | readout |
|---|---|---:|---|
| `ema_dualcarrier_w36_styleguard` | train_ok | 10002 | stable, `|v|` about 0.335 |
| `ema_dualcarrier_w44_style` | train_ok | 10528 | stable, within 10.8GB target |

This supports the carrier decomposition: adding a bounded low-pass style path does not reintroduce the dynamic-head non-finite failure. Full eval is required to see whether it restores CLIP-style capacity.

### Theory Pass: What Must Be True To Reach `0.72 / 0.45`

The current evidence rules out a simple "increase style pressure" story. Three facts have to be explained together:

1. Unsafe EMA t01 can cross the style target (`0.7237`) but pays too much content LPIPS (`0.5755`).
2. Static SConv guard can reach the LPIPS target (`0.4419-0.4651`) but loses style (`~0.705`).
3. Seedream 4.5 reaches both high style and lower LPIPS, but its visible change is not uniformly smaller; it changes some styles strongly while keeping changes semantically organized.

This points to a routing problem, not a scalar-loss problem.

#### Carrier Model

For 256x256 EMA latent transfer, style energy appears to decompose into three carriers:

| carrier | latent object | visible role | safe operator | current failure |
|---|---|---|---|---|
| palette/statistics | low-pass channel mean/covariance | global color and style identity | bounded low-pass affine | free residual causes flat flood |
| texton/microtexture | high-pass local covariance | strokes, grain, brush pattern | zero-init depthwise/SConv | weak if isolated, mist if ungated |
| phase/geometry | edge phase and local coordinate support | structural deformation | ideally off or strictly tangent | fragmentation / LPIPS blow-up |

The important correction is that CLIP-style is not only high-pass texture. The static SConv experiments prove this: high-pass textons alone are content-safe but style-poor. The original dynamic head and unsafe EMA t01 had better CLIP-style because they also moved low-order palette/statistics. The problem is that this motion was not bounded to a low-pass carrier, so the decoder translated some of it into visible cross-edge artifacts.

#### Why Seedream Looks Like A Golden Target

The Seedream-gap diagnostics do not say "Seedream changes less everywhere." In the all-pair summary, Seedream has lower average residual than our stronger EMA variants, but for Hayao it changes much more than us. The difference is organization:

- Seedream has style-dependent amplitude: Hayao can move a lot, Van Gogh/Cezanne can move less.
- Our stronger EMA branches raise residuals more uniformly across styles and regions.
- Our edge/flat ratio is not catastrophically different, so the missing part is not just edge masking; it is semantic/texton placement and low-frequency route separation.

Therefore the model should learn a per-style carrier allocation, not just one global SWD/warp strength.

#### Hypotheses For The Dual-Carrier Run

The `ema_dualcarrier_*` run is the first clean test of this carrier model:

- If it improves from static SConv (`~0.706`) toward support/region SConv (`~0.716`) while keeping LPIPS below `~0.49`, the low-pass palette carrier is real and useful.
- If it reaches `>0.72` with LPIPS `<=0.50`, the next step is not more terminal SWD, but tighter semantic/per-style allocation to push LPIPS down.
- If it stays near `~0.706`, then the bounded low-pass affine is too weak; the style gain in dynamic-head variants came from a richer but unsafe residual family.
- If it reaches style but LPIPS blows up, then low-pass affine is still leaking through the decoder and needs moment caps or semantic gates.

#### Next Design If Dual-Carrier Is Insufficient

The next theory-driven design should be a **style-conditioned carrier allocator**:

1. Predict per-style scalar allocation weights for palette, texton, and geometry carriers.
2. Keep geometry/warp default near zero for EMA; allow it only on high-confidence structural supports.
3. Replace one global support gate with a semantic-confidence gate:
   - strong texton on confident foreground/object regions;
   - weaker texton on flat backgrounds;
   - low-pass palette allowed globally but moment-capped per channel.
4. Add diagnostics to log actual carrier energy:
   - low-pass delta norm;
   - high-pass delta norm;
   - edge-vs-flat delta ratio;
   - per-style residual amplitude.

This is a better search direction than changing terminal SWD alone, because it gives the model a way to match Seedream's behavior: style-dependent, region-organized change instead of uniform residual pressure.

### Dual-Carrier First Full Readout

`ema_dualcarrier_w36_styleguard` finished:

| epoch | clip_style | content_lpips | EC | peak VRAM |
|---:|---:|---:|---:|---:|
| 6 | 0.7056 | 0.4528 | 0.3861 | 10005 MB |
| 7 | 0.7059 | 0.4542 | 0.3853 | 10005 MB |
| 8 | 0.7060 | 0.4557 | 0.3843 | 10005 MB |

This result is important because it is cleanly negative. It lands almost exactly on the static-SConv frontier:

- content is good and close to the `<0.45` target;
- style remains stuck near `0.706`;
- adding the bounded low-pass affine carrier did not recover the missing dynamic-head style capacity.

Revised interpretation:

> The missing broad style carrier is not a simple low-pass affine field. The free dynamic head was probably carrying a richer **feature-remapping** family: local color covariance, channel mixing, and weak texture statistics together. A purely depthwise high-pass carrier plus channelwise low-pass affine is too separable.

This narrows the next design. The next bounded carrier should not be "more low-pass strength" alone. It should be one of:

1. low-rank **channel mixing** on the low/mid-frequency band, not just per-channel affine;
2. style-conditioned **AdaIN/moment matching** with explicit content reconstruction caps;
3. semantic/per-style carrier allocation that allows stronger low-frequency style only where Seedream-like residuals are acceptable.

`ema_dualcarrier_w44_style` is still needed as the pressure test. If it also stays near `0.706`, terminal SWD cannot activate the current bounded carriers. If it rises but LPIPS worsens, then carrier capacity exists but the allocation/gating is wrong.

Full pressure-test readout:

| variant | epoch | clip_style | content_lpips | EC | peak VRAM |
|---|---:|---:|---:|---:|---:|
| `ema_dualcarrier_w44_style` | 6 | 0.7066 | 0.4704 | 0.3742 | 10004 MB |
| `ema_dualcarrier_w44_style` | 7 | 0.7063 | 0.4722 | 0.3728 | 10004 MB |
| `ema_dualcarrier_w44_style` | 8 | 0.7066 | 0.4734 | 0.3721 | 10004 MB |

This confirms the first case: stronger terminal pressure worsens LPIPS but does not activate more style. The current bounded carrier family is too separable.

### Low/Mid-Frequency Channel-Mix Carrier

Implemented the next hypothesis in code:

- `StyleLowpassMixHead`
- config fields:
  - `style_lowpass_mix_head`
  - `style_lowpass_mix_strength`
  - `style_lowpass_mix_kernel`
  - `style_lowpass_mix_tanh_scale`
  - `style_lowpass_mix_hidden_mult`
  - `style_lowpass_mix_zero_init`
- variants:
  - `ema_lowmix_w36_styleguard`
  - `ema_lowmix_w44_style`

The theoretical distinction from `StyleLowpassAffineHead`:

- affine can only scale/shift each latent channel independently;
- lowmix can rotate low/mid-frequency channel covariance through a small style-conditioned matrix;
- because it operates only on low-pass centered content, it is still much more constrained than `dynamic_style_operator_head`;
- zero initialization keeps the branch identity-like at the start.

This is the closest bounded analogue to the useful SaMST/SaMam SCM/SAIN idea while staying in our latent ODE framework. If lowmix improves style over `~0.706` without jumping back to EMA t01 LPIPS, the missing carrier is channel covariance. If it does not, then the dynamic head was probably using spatially varying residual features beyond any simple low-frequency moment carrier.

Full lowmix readout:

| variant | epoch | clip_style | content_lpips | EC | peak VRAM |
|---|---:|---:|---:|---:|---:|
| `ema_lowmix_w36_styleguard` | 6 | 0.7058 | 0.4532 | 0.3859 | 10086 MB |
| `ema_lowmix_w36_styleguard` | 7 | 0.7056 | 0.4545 | 0.3849 | 10086 MB |
| `ema_lowmix_w36_styleguard` | 8 | 0.7061 | 0.4559 | 0.3842 | 10086 MB |
| `ema_lowmix_w44_style` | 6 | 0.7067 | 0.4709 | 0.3739 | 10240 MB |
| `ema_lowmix_w44_style` | 7 | 0.7062 | 0.4726 | 0.3725 | 10240 MB |
| `ema_lowmix_w44_style` | 8 | 0.7066 | 0.4738 | 0.3718 | 10240 MB |

This is another clean negative result. Low-frequency channel covariance mixing behaves almost identically to low-pass affine. The missing dynamic-head capacity is not global low/mid-frequency channel statistics alone.

### Theory Revision After Lowmix

The search has now isolated a sharper fact:

> Static output + high-pass SConv + bounded low-frequency moment carriers cannot exceed `clip_style~0.706`, even when terminal SWD is raised from 36 to 44.

So the dynamic-head style gain is probably not caused by:

- simple low-pass palette shift;
- per-channel low-pass affine;
- low-frequency channel covariance rotation;
- high-pass depthwise texton injection;
- stronger terminal SWD pressure.

The remaining plausible carrier is **spatially varying, content-conditioned mid-band remapping**. The free dynamic output head can apply a style-generated 3x3 operator to the content latent. That operator is unsafe because it sees the whole raw latent, but it has two capacities the bounded carriers lack:

1. local spatial kernel behavior, not only per-pixel channel mixing;
2. cross-channel remapping on local mid-frequency structure, not only depthwise high-pass filtering.

This reframes the next model:

| rejected carrier | why rejected |
|---|---|
| high-pass depthwise SConv only | content-safe but style-poor |
| low-pass affine | no style gain |
| low-pass channel mix | no style gain |
| stronger SWD | worsens LPIPS without style gain |

Next useful design:

1. Build a **mid-band style operator**:
   - input band = `lowpass_5(x) - lowpass_15(x)` or `x - lowpass_9(x)` with the very high-pass tail damped;
   - style-conditioned 3x3 depthwise kernel plus a tiny 1x1 channel mixer;
   - zero-init and tanh-bounded;
   - support/semantic gated.
2. Keep low-pass palette carrier weak or remove it.
3. Keep warp off.
4. Acceptance criterion:
   - if it rises above `0.711-0.716` while LPIPS stays `<0.50`, mid-band remapping is the missing safe style carrier;
   - if it stays near `0.706`, the EMA backend ceiling under safe operators is real and the original VAE remains the better style-ceiling backend.

This is theory-first and falsifiable: it tests exactly the capability that distinguishes the unsafe dynamic head from the safe but underpowered low/high separated carriers.
