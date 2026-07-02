# tok_pure_latent_spatial Remote Run Log

- Run dir: `./exp/inmortal-exp/aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2_c11`

## Current Position

- current newest settled evidence:
  - `c11`
- active remote lane now:
  - `c11`
- contract:
  - `tokenizer_family = pure_latent_spatial`
  - `transport_prediction_mode = endpoint`
  - `solver_family = euler_legacy`
  - `bridge.objective_mode = i2sb_endpoint`
- DINO policy:
  - retired from the active pure-latent lane
  - current smoke and runtime path do not require DINO sidecars
- important interpretation:
  - `c9` was still the wave-1 tokenizer isolation run
  - `c10` remained the same wave-1 tokenizer isolation run with a safer batch
  - `c11` is the next safer calibration attempt
  - none of `c9/c10/c11` is yet the final `solver_i2sb` lane
  - `c11` is the newest settled tokenizer-wave point
  - `c11` is the current calibration attempt
  - `c8` remains the last in-band warmup-stable reference point

## Launch 2026-06-12T09:33:00
- Config: `G:\GitHub\Latent_Style\SchrodingerBridge\configs\aaai2027\round2_pure_sde\aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2.json`
- Run name: `aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2`
- Run dir: `./exp/inmortal-exp/aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2`
- Remote cwd: `/mnt/i/Github/Latent_Style`
- Remote python: `/home/xy/venvs/samam312/bin/python`
- Health wait seconds: `30`
- Contract:
  - formal band `9.0-10.8 GiB`
  - hard stop `11.0 GiB`

## Execution Notes

- earlier same-day launches at smaller batch sizes were under-band:
  - `batch=46` observed around `6681 MiB`
  - `batch=68` observed around `9025 MiB`
- current launch retuned pure-latent tokenizer batch size to `70`
- first healthy VRAM read after retune:
  - `9221 MiB`

## Launch 2026-06-12T10:17:32
- Config: `G:\GitHub\Latent_Style\SchrodingerBridge\configs\aaai2027\round2_pure_sde\aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2.json`
- Run name: `aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2`
- Run dir: `./exp/inmortal-exp/aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2_c1`
- Remote cwd: `/mnt/i/Github/Latent_Style`
- Remote python: `/home/xy/venvs/samam312/bin/python`
- Health wait seconds: `30`
- Contract:
  - formal band `9.0-10.8 GiB`
  - hard stop `11.0 GiB`

## Launch 2026-06-12T10:28:43
- Config: `G:\GitHub\Latent_Style\SchrodingerBridge\configs\aaai2027\round2_pure_sde\aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2.json`
- Run name: `aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2_c2`
- Run dir: `./exp/inmortal-exp/aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2_c2`
- Remote cwd: `/mnt/i/Github/Latent_Style`
- Remote python: `/home/xy/venvs/samam312/bin/python`
- Health wait seconds: `30`
- Contract:
  - formal band `9.0-10.8 GiB`
  - hard stop `11.0 GiB`

## Corrected Relaunch Notes

- pre-correction lane:
  - old remote tokenizer run was stopped after confirmation that training still used the inherited `omf` objective
  - proof in old train log:
    - `t=1.000`
    - `sigma=0.000`
- corrected calibration `c1`:
  - save dir:
    - `./exp/inmortal-exp/aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2_c1`
  - batch:
    - `54`
  - corrected objective evidence:
    - `t=0.489`
    - `flow=0.1601`
    - `ot=0.2773`
  - failure:
    - runtime guard killed the lane at `2026-06-12 10:18:03 +08:00`
    - observed GPU:
      - `12082 MiB`
    - reason:
      - above the `11.0 GiB` hard cap
- corrected calibration `c2`:
  - save dir:
    - `./exp/inmortal-exp/aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2_c2`
  - batch:
    - `44`
  - training config additions:
    - `resume_prefer_local_checkpoint = false`
    - `remote_log_name = aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2_c2`
  - launch health policy:
    - launch-time minimum lowered to `7000 MiB`
    - runtime guard floor kept at `9216 MiB` with delayed warn-only policy after warmup
  - current corrected training evidence:
    - health check GPU at `30s`:
      - `7156 MiB`
    - later spot checks before the final retry showed:
      - `8148 MiB`
      - one transient sample near `9700 MiB`
    - numeric-debug samples:
      - `t_mean ≈ 0.495`
      - `flow ≈ 0.4820 -> 0.4525 -> 0.4262`
      - `terminal_swd ≈ 0.0168 -> 0.0143`
    - gradient state:
      - finite
  - final authoritative c2 outcome:
    - runtime guard killed the lane at `2026-06-12 10:32:54 +08:00`
    - observed GPU:
      - `11655 MiB`
    - decision:
      - `c2` is an exploded calibration, not the active lane

## Launch 2026-06-12T10:38:54
- Config: `G:\GitHub\Latent_Style\SchrodingerBridge\configs\aaai2027\round2_pure_sde\aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2.json`
- Run name: `aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2_c3`
- Run dir: `./exp/inmortal-exp/aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2_c3`
- Remote cwd: `/mnt/i/Github/Latent_Style`
- Remote python: `/home/xy/venvs/samam312/bin/python`
- Health wait seconds: `30`
- Contract:
  - formal band `9.0-10.8 GiB`
  - hard stop `11.0 GiB`

## c3 Notes

- batch:
  - `40`
- launch-health floor:
  - `6500 MiB`
- runtime guard floor:
  - `9216 MiB`
  - delayed warn-only after warmup
- current observed status:
  - health check GPU at `30s`:
    - `6639 MiB`
  - later spot check GPU:
    - `6777 MiB`
  - numeric-debug samples:
    - `t_mean ≈ 0.494 -> 0.501`
    - `flow ≈ 0.4926 -> 0.4523 -> 0.4190`
    - gradients remain finite
  - current interpretation:
    - corrected objective is healthy
    - lane was still under observation for post-warmup VRAM growth
      - max grad source still lands in `structured_style_tokenizer.*`
  - final authoritative c3 outcome:
    - runtime guard killed the lane at `2026-06-12 10:42:18 +08:00`
    - observed GPU:
      - `11185 MiB`
    - decision:
      - `c3` is an exploded calibration, not the active lane

## Launch 2026-06-12T10:38:17
- Config: `G:\GitHub\Latent_Style\SchrodingerBridge\configs\aaai2027\round2_pure_sde\aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2.json`
- Run name: `aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2_c3`
- Run dir: `./exp/inmortal-exp/aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2_c3`
- Remote cwd: `/mnt/i/Github/Latent_Style`
- Remote python: `/home/xy/venvs/samam312/bin/python`
- Health wait seconds: `30`
- Contract:
  - formal band `9.0-10.8 GiB`
  - hard stop `11.0 GiB`

## Launch 2026-06-12T10:57:25
- Config: `G:\GitHub\Latent_Style\SchrodingerBridge\configs\aaai2027\round2_pure_sde\aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2.json`
- Run name: `aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2_c4`
- Run dir: `./exp/inmortal-exp/aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2_c4`
- Remote cwd: `/mnt/i/Github/Latent_Style`
- Remote python: `/home/xy/venvs/samam312/bin/python`
- Health wait seconds: `20`
- Contract:
  - formal band `9.0-10.8 GiB`
  - hard stop `11.0 GiB`

## c4 Notes

- batch:
  - `36`
- launch-health floor:
  - `6000 MiB`
- runtime guard floor:
  - `9216 MiB`
  - delayed warn-only after warmup
- host-owned launcher status:
  - after launcher hardening, scheduled launch now stays alive through health check
  - current observed status:
    - health check GPU at `20s`:
      - `6360 MiB`
    - later spot-check GPU:
      - `6803 MiB`
  - numeric-debug samples:
    - `t_mean ≈ 0.523 -> 0.492 -> 0.499 -> 0.504`
    - `flow ≈ 0.4788 -> 0.4482 -> 0.4261 -> 0.4013 -> 0.3708 -> 0.3402 -> 0.3142`
    - gradients remain finite
  - current interpretation:
    - corrected objective is healthy
    - lane is under-band
  - final authoritative c4 outcome:
    - post-warmup runtime sample:
      - `7889 MiB`
    - runtime guard emitted:
      - `RUNTIME_UNDER_BAND_WARN`
    - decision:
      - `c4` is a valid corrected under-band calibration, not a formal lane

## c5 Notes

- batch:
  - `38`
- launch-health floor:
  - `6800 MiB`
- expected role:
  - bridge the gap between `c4` under-band and `c3` explosion
- observed issue:
  - launcher/task supervision is still noisy for this family
  - host-owned launch and fallback launch can both append wrapper preambles into the same log
  - repeated launch attempts were stopped to avoid duplicate remote consumers
- decision:
  - `c5` config is prepared and smoke-valid
  - active remote lane was not kept running at the end of that turn
  - next step is launcher stabilization before the next formal relaunch

## Launch 2026-06-12T11:24:11
- Config: `G:\GitHub\Latent_Style\SchrodingerBridge\configs\aaai2027\round2_pure_sde\aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2.json`
- Run name: `aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2_c8`
- Run dir: `./exp/inmortal-exp/aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2_c8`
- Remote cwd: `/mnt/i/Github/Latent_Style`
- Remote python: `/home/xy/venvs/samam312/bin/python`
- Health wait seconds: `20`
- Contract:
  - formal band `9.0-10.8 GiB`
  - hard stop `11.0 GiB`

## c8 Notes

- batch:
  - `37`
- launch-health floor:
  - `6600 MiB`
- host-owned launcher status:
  - health check now accepts a progressing training log even if the wrapper pid has already exited
  - no fallback was needed on the successful launch
- future relaunch note:
  - `freeze_mode=style_branch` on `pure_latent_spatial` no longer marks legacy `style_spatial_*` priors trainable
  - local verification shows only `8` structured-tokenizer parameters remain trainable on the pure path
- current observed status:
  - health GPU at `20s`:
    - `6916 MiB`
  - post-warmup GPU at `331s`:
    - `10431 MiB`
  - no `RUNTIME_UNDER_BAND_WARN`
  - no `RUNTIME_GUARD`
  - first corrected eval:
    - `epoch_0001`
    - transfer:
      - `clip_style = 0.702593`
      - `content_lpips = 0.535566`
    - all-pairs:
      - `clip_style = 0.718218`
      - `content_lpips = 0.531715`
  - numeric-debug samples remain finite
  - sampled-bridge evidence persists:
    - `t_mean` stays in the interior rather than collapsing to `1.0`
    - `flow` continues to decline over early optimizer steps
- current interpretation:
  - `c8` is the first corrected `tok_pure` lane that survives warmup inside the formal VRAM band
  - next gate is the first retained checkpoint and its remote `CLIP-S + LPIPS` eval

## Launch 2026-06-12T11:06:18
- Config: `G:\GitHub\Latent_Style\SchrodingerBridge\configs\aaai2027\round2_pure_sde\aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2.json`
- Run name: `aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2_c5`
- Run dir: `./exp/inmortal-exp/aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2_c5`
- Remote cwd: `/mnt/i/Github/Latent_Style`
- Remote python: `/home/xy/venvs/samam312/bin/python`
- Health wait seconds: `20`
- Contract:
  - formal band `9.0-10.8 GiB`
  - hard stop `11.0 GiB`

## Launch 2026-06-12T11:09:06
- Config: `G:\GitHub\Latent_Style\SchrodingerBridge\configs\aaai2027\round2_pure_sde\aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2.json`
- Run name: `aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2_c5`
- Run dir: `./exp/inmortal-exp/aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2_c5`
- Remote cwd: `/mnt/i/Github/Latent_Style`
- Remote python: `/home/xy/venvs/samam312/bin/python`
- Health wait seconds: `20`
- Contract:
  - formal band `9.0-10.8 GiB`
  - hard stop `11.0 GiB`

## Launch 2026-06-12T11:14:12
- Config: `G:\GitHub\Latent_Style\SchrodingerBridge\configs\aaai2027\round2_pure_sde\aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2.json`
- Run name: `aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2_c6`
- Run dir: `./exp/inmortal-exp/aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2_c6`
- Remote cwd: `/mnt/i/Github/Latent_Style`
- Remote python: `/home/xy/venvs/samam312/bin/python`
- Health wait seconds: `20`
- Contract:
  - formal band `9.0-10.8 GiB`
  - hard stop `11.0 GiB`

## Launch 2026-06-12T11:19:16
- Config: `G:\GitHub\Latent_Style\SchrodingerBridge\configs\aaai2027\round2_pure_sde\aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2.json`
- Run name: `aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2_c7`
- Run dir: `./exp/inmortal-exp/aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2_c7`
- Remote cwd: `/mnt/i/Github/Latent_Style`
- Remote python: `/home/xy/venvs/samam312/bin/python`
- Health wait seconds: `20`
- Contract:
  - formal band `9.0-10.8 GiB`
  - hard stop `11.0 GiB`

## Launch 2026-06-12T11:23:58
- Config: `G:\GitHub\Latent_Style\SchrodingerBridge\configs\aaai2027\round2_pure_sde\aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2.json`
- Run name: `aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2_c8`
- Run dir: `./exp/inmortal-exp/aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2_c8`
- Remote cwd: `/mnt/i/Github/Latent_Style`
- Remote python: `/home/xy/venvs/samam312/bin/python`
- Health wait seconds: `20`
- Contract:
  - formal band `9.0-10.8 GiB`
  - hard stop `11.0 GiB`

## Launch 2026-06-12T11:43:38
- Config: `G:\GitHub\Latent_Style\SchrodingerBridge\configs\aaai2027\round2_pure_sde\aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2.json`
- Run name: `aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2_c9`
- Run dir: `./exp/inmortal-exp/aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2_c9`
- Remote cwd: `/mnt/i/Github/Latent_Style`
- Remote python: `/home/xy/venvs/samam312/bin/python`
- Health wait seconds: `20`
- Contract:
  - formal band `9.0-10.8 GiB`
  - hard stop `11.0 GiB`

## c9 Notes

- batch:
  - `37`
- progress before eval:
  - flow:
    - about `0.4792 -> 0.1790`
  - terminal SWD:
    - about `0.0182 -> 0.0141`
- settled eval at `epoch_0001`:
  - transfer:
    - `clip_style = 0.702607`
    - `content_lpips = 0.535564`
  - all-pairs:
    - `clip_style = 0.718214`
    - `content_lpips = 0.531710`
- eval timing:
  - `wall_total = 110.14s`
  - `eval_total = 31.08s`
  - `generation = 10.45s`
  - `vae_decode = 57.08s`
- runtime-band evidence:
  - launcher log emitted:
    - `RUNTIME_UNDER_BAND_WARN`
  - warning sample:
    - `7570 MiB`
  - later, after eval and resume into `epoch_2`, runtime guard killed the lane:
    - `2026-06-12 11:57:21 +08:00`
    - `used = 11776 MiB`
    - `cap = 11000 MiB`
- decision:
  - the remote eval contract is now proven working on the pure-latent path
  - `c9` is useful evidence but not a promotable run
  - mark `c9` as `recalibration_needed`
  - next retry should move back to a safer batch point before solver-wave promotion

## Launch 2026-06-12T12:08:36
- Config: `G:\GitHub\Latent_Style\SchrodingerBridge\configs\aaai2027\round2_pure_sde\aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2.json`
- Run name: `aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2_c10`
- Run dir: `./exp/inmortal-exp/aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2_c10`
- Remote cwd: `/mnt/i/Github/Latent_Style`
- Remote python: `/home/xy/venvs/samam312/bin/python`
- Health wait seconds: `20`
- Contract:
  - formal band `9.0-10.8 GiB`
  - hard stop `11.0 GiB`

## c10 Notes

- batch:
  - `36`
- launch health:
  - `health_gpu_memory_used_mib = 6550`
- early runtime spot-check at `2026-06-12 12:10 +08:00`:
  - GPU:
    - `8866 MiB`
    - `98% util`
- later warmup evidence at `2026-06-12 12:13:37 +08:00`:
  - runtime guard emitted:
    - `RUNTIME_UNDER_BAND_WARN`
  - warning sample:
    - `7918 MiB`
  - train log status:
    - still inside `epoch_0001`
    - no hard-cap event yet
- initial interpretation:
  - still below the preferred `9.0 GiB` floor
  - materially safer than `c9` so far, but still not formal-band
  - no hard-cap violation observed during the first warmup window
- settled eval at `epoch_0001`:
  - transfer:
    - `clip_style = 0.699601`
    - `content_lpips = 0.515420`
  - all-pairs:
    - `clip_style = 0.716285`
    - `content_lpips = 0.513291`
- eval timing:
  - `wall_total = 109.83s`
  - `eval_total = 30.59s`
  - `generation = 10.58s`
  - `vae_decode = 56.98s`
- final failure:
  - after eval and resume into `epoch_2`, runtime guard killed the lane:
    - `2026-06-12 12:21:09 +08:00`
    - `used = 11508 MiB`
    - `cap = 11000 MiB`
- decision:
  - `c10` is the newest settled tokenizer-wave evidence
  - `c10` improves LPIPS versus `c9`, but still violates the hard cap after resume
  - mark `c10` as `recalibration_needed`

## Launch 2026-06-12T12:24:00
- Config: `G:\GitHub\Latent_Style\SchrodingerBridge\configs\aaai2027\round2_pure_sde\aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2.json`
- Run name: `aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2_c11`
- Run dir: `./exp/inmortal-exp/aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2_c11`
- Remote cwd: `/mnt/i/Github/Latent_Style`
- Remote python: `/home/xy/venvs/samam312/bin/python`
- Contract:
  - formal band `9.0-10.8 GiB`
  - hard stop `11.0 GiB`

## Launch 2026-06-12T12:27:03
- Config: `G:\GitHub\Latent_Style\SchrodingerBridge\configs\aaai2027\round2_pure_sde\aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2.json`
- Run name: `aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2_c11`
- Run dir: `./exp/inmortal-exp/aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2_c11`
- Remote cwd: `/mnt/i/Github/Latent_Style`
- Remote python: `/home/xy/venvs/samam312/bin/python`
- Health wait seconds: `20`
- Contract:
  - formal band `9.0-10.8 GiB`
  - hard stop `11.0 GiB`

## c11 Notes

- batch:
  - `34`
- launch health:
  - `health_gpu_memory_used_mib = 6176`
- later runtime spot-check:
  - `7653 MiB`
- settled eval at `epoch_0001`:
  - transfer:
    - `clip_style = 0.699700`
    - `content_lpips = 0.530487`
  - all-pairs:
    - `clip_style = 0.715840`
    - `content_lpips = 0.526602`
- eval timing:
  - `wall_total = 110.66s`
  - `eval_total = 32.56s`
  - `generation = 10.72s`
  - `vae_decode = 55.09s`
- post-resume evidence:
  - after eval and restore, the lane continued into `epoch_2`
  - observed GPU:
    - `9236 MiB`
    - `97% util`
  - this is the first tokenizer-wave lane that has crossed the `epoch_0001 eval + resume` gate while staying inside the preferred formal band
- longer `epoch_2` evidence:
  - later runtime sample:
    - `8556 MiB`
    - `97% util`
  - runtime guard emitted:
    - `RUNTIME_UNDER_BAND_WARN`
  - warning sample:
    - `8556 MiB`
  - no hard-cap event has been observed through the later `epoch_2` window so far
- current decision:
  - `c11` is the active tokenizer-wave reference
  - it has now cleared the immediate `epoch_1 -> eval -> epoch_2` stability gate
  - keep it running and watch whether later epochs still remain under `11.0 GiB`

- later progress:
  - `epoch_0002` eval completed
  - settled `epoch_0002` metrics:
    - transfer `0.704938 / 0.587250`
    - all-pairs `0.716840 / 0.582709`
  - after that eval and restore, the lane continued into `epoch_3`
  - no hard-cap event has appeared through the observed `epoch_3` window
  - `epoch_0003` eval completed
  - settled `epoch_0003` metrics:
    - transfer `0.705098 / 0.601646`
    - all-pairs `0.715398 / 0.596325`
  - after that eval and restore, the lane continued into `epoch_4`
  - no hard-cap event has appeared through the observed `epoch_4` window
