# SMoE Translator Remote Run

## Launch 2026-06-14 04:42 Asia/Shanghai

- Family: `smoe_translator_k070_e3`.
- Config: `SchrodingerBridge/configs/aaai2027/phase2_smoe_translator_k070_e3_seed42_b12a1.json`.
- Task: `phase2-phase2_smoe_translator_k070_e3_seed42_b12a1-train`.
- Remote cwd: `/mnt/i/Github/Latent_Style`.
- Remote python: `/home/xy/venvs/samam312/bin/python`.
- Parent checkpoint: `/mnt/i/Github/Latent_Style/exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1/epoch_0003.pt`.
- Data root: `/mnt/i/wikiarts_5_full_notest_latents_ema/train`.
- Test root: `/mnt/i/wikiart_distinct5_samam_512_classview/test`.
- Train log: `/mnt/i/Github/Latent_Style/exp/inmortal-exp/aaai2027_phase2_smoe_translator_k070_e3_seed42_b12a1_train.log`.
- Output root: `/mnt/i/Github/Latent_Style/exp/aaai2027_phase2_smoe_translator_k070_e3_seed42_b12a1`.

## First Health

- First heartbeat was inspected during launcher health window.
- Python PID: `456`.
- Initial GPU memory: `6969 MiB / 12288 MiB`.
- GPU utilization: `94%`.
- Log reached `Epoch 1/24`.
- Runtime guard: max memory `11000 MiB`; min-memory guard is warning-only because epoch-end full eval intentionally offloads the trainer.

## Decision

- The run is accepted as the SMoE matched-control lane despite using less than the preferred `9.0-10.8 GiB` formal band.
- Reason: increasing batch size would alter the tokenizer-only control schedule and contaminate the mechanism comparison.
- If the lower memory footprint persists, record it as an efficiency observation, not as a reason to change this lane.

## Epoch 1 Full Eval

- Checkpoint: `epoch_0001.pt`.
- Training time: `1450.6s` (`24.18min`).
- Eval wall time: `242.7s`; trainer log reports full eval completed in `265.1s`.
- Transfer: `CLIP-S=0.672379`, `LPIPS=0.333173`, `style - IDT=+0.032458`.
- All-pairs: `CLIP-S=0.703540`, `LPIPS=0.329736`, `style - IDT=+0.023417`.
- Matched delta vs `k070 epoch_0003`:
  - transfer: `+0.000558` style, `+0.018555` LPIPS.
  - all-pairs: `+0.000306` style, `+0.017186` LPIPS.
- Runtime observability from summary:
  - `translation_delta_from_identity=0.005202`.
  - `routing_entropy=1.676048`.
  - `effective_experts=5.401902`.
  - `spatial_abs=0.852155` on transfer.
- Read: early point gives only tiny style lift and a clear LPIPS regression. Keep running; this is not a closure point.

## Runtime Guard Incident And Relaunch

- At `2026-06-14 05:13:32 Asia/Shanghai`, the SMoE launcher guard stopped the first process with `used=11694MiB cap=11000MiB`, `rc=143`.
- Root cause: an older off-plan `phase2_i2sb_topo_anchor_sigma0p25_seed42_b30a1` task restarted at the same time and held about `8.8GiB`, so the two lanes overlapped.
- Action: stopped the I2SB PID `2221`; confirmed remote GPU returned to `388MiB / 12288MiB` and no Python training process remained.
- Relaunch: restarted the same SMoE config at `2026-06-14 05:21 Asia/Shanghai`; it resumed from local `epoch_0001.pt` at epoch 2, global step `1574`.
- Relaunch health: `6776MiB / 12288MiB`, utilization about `95%`; accepted as the same matched-control lane. No batch, loss, solver, tokenizer, or schedule parameter was changed.
- Interpretation: the guard stop is an orchestration/concurrency fault, not SMoE single-lane VRAM evidence. The run remains valid from `epoch_0001.pt` onward after the single-lane relaunch.

## Second Guard Incident And I2SB Task Quarantine

- At `2026-06-14 05:39:01 Asia/Shanghai`, the second SMoE attempt was also stopped by the runtime guard with `used=11772MiB cap=11000MiB`, `rc=143`.
- Root cause: another historical I2SB one-shot task, `phase2_i2sb_topo_anchor_sigma0p10_warm_vel2_seed42_b30a1`, started at `05:38:47` and occupied about `8.6GiB`.
- Action: stopped the I2SB process and disabled every scheduled task whose name contains `i2sb`, including old train tasks and old curve/watch helpers.
- Verification: after quarantine, remote GPU returned to `364MiB / 12288MiB`, no Python process remained, and no `i2sb` task had a future run time.
- Relaunch: restarted the same SMoE config at `2026-06-14 05:47 Asia/Shanghai`; it resumed again from local `epoch_0001.pt` at epoch 2, global step `1574`.
- Relaunch health: `6828MiB / 12288MiB`; accepted as the same matched-control lane. Still no batch, loss, solver, tokenizer, or schedule parameter change.
- Current operational rule: keep all I2SB tasks disabled until SMoE is closed or explicitly preempted; only the SMoE task may hold the remote GPU lane.

## Epoch 2 Full Eval

- Full eval completed at `2026-06-14 06:17:22 Asia/Shanghai`; training resumed into epoch 3.
- Checkpoint: `epoch_0002.pt`.
- Training time: `1468.3s` (`24.47min`) from epoch log `data+comp`; eval wall time from curve: `240.4s`.
- Transfer: `CLIP-S=0.670478`, `LPIPS=0.331323`, `style - IDT=+0.030557`.
- All-pairs: `CLIP-S=0.701436`, `LPIPS=0.327738`, `style - IDT=+0.021314`.
- Identity: `CLIP-S=0.825270`, `LPIPS=0.313400`.
- Matched delta vs `k070 epoch_0003`:
  - transfer: `-0.001343` style, `+0.016705` LPIPS.
  - all-pairs: `-0.001797` style, `+0.015189` LPIPS.
- Runtime observability from summary:
  - `translation_delta_from_identity=0.007511`.
  - `routing_entropy=1.663507`.
  - `effective_experts=5.321320`.
  - `spatial_abs=0.896592` on transfer.
- Convergence state from `round2_convergence.json`: `converged=false`, best transfer/all-pairs remains `epoch_0001`, newest checkpoint remains within the newest-2 patience window.
- Read: e2 recovers some LPIPS from e1 but gives up style and remains dominated by the matched parent. Continue to e3 before deciding whether to close SMoE-only as negative evidence or inspect identity scale/init.

## Epoch 3 Full Eval

- Full eval completed at `2026-06-14 06:46:41 Asia/Shanghai`; training resumed into epoch 4.
- Checkpoint: `epoch_0003.pt`.
- Training time: `1461.7s` (`24.36min`) from epoch log `data+comp`; eval wall time from curve: `244.4s`.
- Transfer: `CLIP-S=0.668568`, `LPIPS=0.323320`, `style - IDT=+0.028647`.
- All-pairs: `CLIP-S=0.699963`, `LPIPS=0.321329`, `style - IDT=+0.019840`.
- Identity: `CLIP-S=0.825543`, `LPIPS=0.313364`.
- Matched delta vs `k070 epoch_0003`:
  - transfer: `-0.003253` style, `+0.008702` LPIPS.
  - all-pairs: `-0.003271` style, `+0.008779` LPIPS.
- Runtime observability from summary:
  - `translation_delta_from_identity=0.009242`.
  - `routing_entropy=1.579277`.
  - `effective_experts=4.887480`.
  - `spatial_abs=0.877825` on transfer.
- Convergence state from `round2_convergence.json`: `converged=false`, best transfer/all-pairs remains `epoch_0001`, and `epoch_0003` is a new lower-LPIPS Pareto point on the candidate curve.
- Read: e3 is still dominated by the matched parent. The tokenizer keeps moving toward structure recovery while shedding style, so the mechanism read is negative unless later epochs recover style without losing LPIPS.

## Epoch 4 Full Eval

- Full eval completed at `2026-06-14 07:16:31 Asia/Shanghai`; training resumed into epoch 5.
- Checkpoint: `epoch_0004.pt`.
- Training time: `1477.0s` (`24.62min`) from epoch log `data+comp`; eval wall time from curve: `259.9s`.
- Transfer: `CLIP-S=0.669259`, `LPIPS=0.322103`, `style - IDT=+0.029339`.
- All-pairs: `CLIP-S=0.700884`, `LPIPS=0.317942`, `style - IDT=+0.020762`.
- Identity: `CLIP-S=0.827383`, `LPIPS=0.301297`.
- Matched delta vs `k070 epoch_0003`:
  - transfer: `-0.002561` style, `+0.007485` LPIPS.
  - all-pairs: `-0.002349` style, `+0.005392` LPIPS.
- Runtime observability from summary:
  - `translation_delta_from_identity=0.010881`.
  - `routing_entropy=1.752495`.
  - `effective_experts=5.804411`.
  - `spatial_abs=0.871335` on transfer.
- Convergence state from `round2_convergence.json`: `converged=false`, best transfer/all-pairs remains `epoch_0001`, and `epoch_0004` extends the low-LPIPS Pareto tail.
- Read: e4 slightly recovers style from e3 and improves LPIPS, but it remains dominated by `k070 epoch_0003`. Continue only under the formal curve rule; do not promote SMoE-only unless style reverses without structure loss.
