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

## Epoch 5 Full Eval

- Full eval completed at `2026-06-14 07:47:53 Asia/Shanghai`; training resumed into epoch 6.
- Checkpoint: `epoch_0005.pt`.
- Training time: `1564.0s` (`26.07min`) from epoch log `data+comp`; eval wall time from curve: `261.0s`.
- Transfer: `CLIP-S=0.670998`, `LPIPS=0.326648`, `style - IDT=+0.031077`.
- All-pairs: `CLIP-S=0.702158`, `LPIPS=0.323817`, `style - IDT=+0.022036`.
- Identity: `CLIP-S=0.826800`, `LPIPS=0.312494`.
- Matched delta vs `k070 epoch_0003`:
  - transfer: `-0.000822` style, `+0.012030` LPIPS.
  - all-pairs: `-0.001075` style, `+0.011268` LPIPS.
- Runtime observability from summary:
  - `translation_delta_from_identity=0.012272`.
  - `routing_entropy=1.802536`.
  - `effective_experts=6.149995`.
  - `spatial_abs=0.903199` on transfer.
- Convergence state from `round2_convergence.json`: `converged=false`, best transfer/all-pairs remains `epoch_0001`, and `epoch_0005` is another curve-level Pareto point.
- Read: e5 recovers style toward the parent but gives back structure. It is still dominated by `k070 epoch_0003`, so the mechanism remains negative pending formal closure.

## Epoch 6 Full Eval

- Full eval completed at `2026-06-14 08:19:03 Asia/Shanghai`; training resumed into epoch 7.
- Checkpoint: `epoch_0006.pt`.
- Training time: `1558.2s` (`25.97min`) from epoch log `data+comp`; eval wall time from curve: `256.6s`.
- Transfer: `CLIP-S=0.669504`, `LPIPS=0.335022`, `style - IDT=+0.029583`.
- All-pairs: `CLIP-S=0.700534`, `LPIPS=0.330586`, `style - IDT=+0.020411`.
- Identity: `CLIP-S=0.824652`, `LPIPS=0.312843`.
- Matched delta vs `k070 epoch_0003`:
  - transfer: `-0.002316` style, `+0.020404` LPIPS.
  - all-pairs: `-0.002700` style, `+0.018037` LPIPS.
- Runtime observability from summary:
  - `translation_delta_from_identity=0.013472`.
  - `routing_entropy=1.453477`.
  - `effective_experts=4.380065`.
  - `spatial_abs=0.885638` on transfer.
- Convergence state from `round2_convergence.json`: `converged=false`, best transfer/all-pairs remains `epoch_0001`, last Pareto remains `epoch_0005`, and `since_last_pareto=1`.
- Read: e6 is the first non-Pareto regression point and remains strongly dominated by `k070 epoch_0003`. Continue to collect tail evidence before formal closure.

## Epoch 7 Full Eval

- Full eval completed at `2026-06-14 08:49:20 Asia/Shanghai`; training resumed into epoch 8.
- Checkpoint: `epoch_0007.pt`.
- Training time: `1511.5s` (`25.19min`) from epoch log `data+comp`; eval wall time from curve: `249.8s`.
- Transfer: `CLIP-S=0.670730`, `LPIPS=0.333296`, `style - IDT=+0.030809`.
- All-pairs: `CLIP-S=0.701995`, `LPIPS=0.328815`, `style - IDT=+0.021872`.
- Identity: `CLIP-S=0.827052`, `LPIPS=0.310890`.
- Matched delta vs `k070 epoch_0003`:
  - transfer: `-0.001090` style, `+0.018678` LPIPS.
  - all-pairs: `-0.001239` style, `+0.016266` LPIPS.
- Runtime observability from summary:
  - `translation_delta_from_identity=0.014656`.
  - `routing_entropy=1.694707`.
  - `effective_experts=5.537481`.
  - `spatial_abs=0.854391` on transfer.
- Convergence state from `round2_convergence.json`: `converged=false`, best transfer/all-pairs remains `epoch_0001`, last Pareto remains `epoch_0005`, and `since_last_pareto=2`.
- Read: e7 rebounds from e6 but is still a non-Pareto tail point and remains dominated by the matched parent. Continue until either a new Pareto point appears or the patience threshold closes the family.

## Epoch 8 Full Eval

- Full eval completed at `2026-06-14 09:19:29 Asia/Shanghai`; training resumed into epoch 9.
- Checkpoint: `epoch_0008.pt`.
- Training time: `1514.0s` (`25.23min`) from epoch log `data+comp`; eval wall time from curve: `241.0s`.
- Transfer: `CLIP-S=0.669985`, `LPIPS=0.317808`, `style - IDT=+0.030065`.
- All-pairs: `CLIP-S=0.701901`, `LPIPS=0.315335`, `style - IDT=+0.021779`.
- Identity: `CLIP-S=0.829565`, `LPIPS=0.305445`.
- Matched delta vs `k070 epoch_0003`:
  - transfer: `-0.001835` style, `+0.003189` LPIPS.
  - all-pairs: `-0.001332` style, `+0.002785` LPIPS.
- Runtime observability from summary:
  - `translation_delta_from_identity=0.015515`.
  - `routing_entropy=1.681098`.
  - `effective_experts=5.435514`.
  - `spatial_abs=0.806139` on transfer.
- Convergence state from `round2_convergence.json`: `converged=false`, `epoch_0008` is a new candidate-curve Pareto point, and `since_last_pareto=0`.
- Read: e8 is the closest SMoE structural point so far but still loses both style and LPIPS against the matched parent. Because it is a new candidate-curve Pareto point, the formal closure patience resets.

## Epoch 9 Full Eval

- Full eval completed at `2026-06-14 09:48:45 Asia/Shanghai`; training resumed into epoch 10.
- Checkpoint: `epoch_0009.pt`.
- Training time: `1465.7s` (`24.43min`) from epoch log `data+comp`; eval wall time from curve: `242.5s`.
- Transfer: `CLIP-S=0.672774`, `LPIPS=0.327155`, `style - IDT=+0.032853`.
- All-pairs: `CLIP-S=0.704251`, `LPIPS=0.322688`, `style - IDT=+0.024128`.
- Identity: `CLIP-S=0.830159`, `LPIPS=0.304821`.
- Matched delta vs `k070 epoch_0003`:
  - transfer: `+0.000953` style, `+0.012536` LPIPS.
  - all-pairs: `+0.001017` style, `+0.010138` LPIPS.
- Runtime observability from summary:
  - `translation_delta_from_identity=0.016185`.
  - `routing_entropy=1.583672`.
  - `effective_experts=4.917878`.
  - `spatial_abs=0.826611` on transfer.
- Convergence state from `round2_convergence.json`: `converged=false`, `epoch_0009` is the new best transfer/all-pairs style point, and `best_in_newest_2=true`.
- Read: e9 is the first SMoE point with positive style delta against the parent, but the structure cost is still too large for promotion. Continue to see whether the line can keep the e9 style while moving back toward the e8/parent LPIPS band.

## Epoch 10 Full Eval

- Full eval completed at `2026-06-14 10:19:29 Asia/Shanghai`; training resumed into epoch 11.
- Checkpoint: `epoch_0010.pt`.
- Training time: `1528.6s` (`25.48min`) from epoch log `data+comp`; eval wall time from curve: `257.4s`.
- Transfer: `CLIP-S=0.670014`, `LPIPS=0.323925`, `style - IDT=+0.030093`.
- All-pairs: `CLIP-S=0.701628`, `LPIPS=0.320149`, `style - IDT=+0.021506`.
- Identity: `CLIP-S=0.828084`, `LPIPS=0.305047`.
- Matched delta vs `k070 epoch_0003`:
  - transfer: `-0.001806` style, `+0.009307` LPIPS.
  - all-pairs: `-0.001605` style, `+0.007600` LPIPS.
- Runtime observability from summary:
  - `translation_delta_from_identity=0.016836`.
  - `routing_entropy=1.671797`.
  - `effective_experts=5.370153`.
  - `spatial_abs=0.841168` on transfer.
- Convergence state from `round2_convergence.json`: `converged=false`, `epoch_0010` is a candidate-curve Pareto point, and `since_last_pareto=0`.
- Read: e10 recovers some LPIPS from e9 but loses the positive style delta, so it does not answer the promotion question. Continue while the curve is still creating Pareto points.

## Third Guard Incident: Safe-Rescan Task Overlap

- At `2026-06-14 10:36:08 Asia/Shanghai`, the SMoE launcher guard stopped the epoch-11 process with `used=11925MiB cap=11000MiB`, `rc=143`.
- Root cause: an old off-plan `phase2_vel_tok32_safe_rescan_r1_seed42_b20a1` scheduled task started at `10:35:51` and occupied the remote GPU concurrently with SMoE.
- Invalidated work: the interrupted epoch-11 partial has no checkpoint/eval and must not be counted as a retained point.
- Action: killed the safe-rescan process, confirmed the GPU returned to `533MiB / 12288MiB` with no Python process, then disabled old safe-rescan and structure/topogate scheduled tasks that can steal the formal lane.
- Disabled task set:
  - `exp-phase2_vel_tok32_safe_rescan_r1_seed42_b20a1-train`
  - `exp-phase2_vel_tok32_safe_rescan_r2_seed42_b20a1-train`
  - `exp-phase2_vel_tok32_safe_semantic_topogate_k085_appalign_seed42_b12a1-train`
  - `exp-phase2_vel_tok32_safe_semantic_topogate_k085_appalign_seed42_b16a1-train`
  - `phase2-structure-phase2_vel_tok32_safe_semantic_topogate_k085_seed42_b16a1-train`
  - `structure_reentry-phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1-train`
  - `structure_reentry-phase2_vel_tok32_safe_semantic_topogate_k085_seed42_b20a1-train`
- Relaunch: restarted the same SMoE scheduled task at `2026-06-14 10:46 Asia/Shanghai`.
- Relaunch health: PID `414`, resumed from `exp/aaai2027_phase2_smoe_translator_k070_e3_seed42_b12a1/epoch_0010.pt` at epoch 11/global step `15740`, initial GPU read `4549MiB / 12288MiB`.
- Decision: keep the SMoE lane valid after relaunch because no model, optimizer, batch, solver, tokenizer, loss, or schedule parameter changed. Treat the stop as orchestration/concurrency fault only.

## Epoch 11 Full Eval

- Full eval completed at `2026-06-14 11:17:19 Asia/Shanghai`; training resumed into epoch 12.
- Checkpoint: `epoch_0011.pt`.
- Training time: `1557.8s` (`25.96min`) from epoch log `data+comp`; eval wall time from curve: `253.4s`.
- Transfer: `CLIP-S=0.669667`, `LPIPS=0.327548`, `style - IDT=+0.029747`.
- All-pairs: `CLIP-S=0.701142`, `LPIPS=0.324272`, `style - IDT=+0.021019`.
- Identity: `CLIP-S=0.827041`, `LPIPS=0.311168`.
- Matched delta vs `k070 epoch_0003`:
  - transfer: `-0.002153` style, `+0.012929` LPIPS.
  - all-pairs: `-0.002091` style, `+0.011722` LPIPS.
- Runtime observability from summary:
  - `translation_delta_from_identity=0.017423`.
  - `routing_entropy=1.627435`.
  - `effective_experts=5.111550`.
  - `spatial_abs=0.896498` on transfer.
- Convergence state from `round2_convergence.json`: `converged=false`, best transfer/all-pairs remains `epoch_0009`, last Pareto remains `epoch_0010`, `since_last_pareto=1`, and `tail_flat=true`.
- Read: e11 is a non-Pareto tail point and worsens both style and LPIPS from e10. Continue because formal closure requires more post-Pareto failures after e10.

## Epoch 12 Full Eval

- Full eval completed at `2026-06-14 11:48:03 Asia/Shanghai`; training resumed into epoch 13.
- Checkpoint: `epoch_0012.pt`.
- Training time: `1535.2s` (`25.59min`) from epoch log `data+comp`; eval wall time from curve: `254.0s`.
- Transfer: `CLIP-S=0.670048`, `LPIPS=0.331773`, `style - IDT=+0.030127`.
- All-pairs: `CLIP-S=0.701154`, `LPIPS=0.328471`, `style - IDT=+0.021031`.
- Identity: `CLIP-S=0.825575`, `LPIPS=0.315265`.
- Matched delta vs `k070 epoch_0003`:
  - transfer: `-0.001772` style, `+0.017155` LPIPS.
  - all-pairs: `-0.002080` style, `+0.015922` LPIPS.
- Runtime observability from summary:
  - `translation_delta_from_identity=0.017843`.
  - `routing_entropy=1.635242`.
  - `effective_experts=5.161444`.
  - `spatial_abs=0.850654` on transfer.
- Convergence state from `round2_convergence.json`: `converged=false`, best transfer/all-pairs remains `epoch_0009`, last Pareto remains `epoch_0010`, `since_last_pareto=2`, and `tail_flat=true`.
- Read: e12 is the second post-e10 non-Pareto tail point and worsens LPIPS further. Continue until the formal patience rule can close or a new Pareto point appears.

## Epoch 13 Full Eval

- Full eval completed at `2026-06-14 12:19:26 Asia/Shanghai`; training resumed into epoch 14.
- Checkpoint: `epoch_0013.pt`.
- Training time: `1556.4s` (`25.94min`) from epoch log `data+comp`; eval wall time from curve: `268.5s`.
- Transfer: `CLIP-S=0.671565`, `LPIPS=0.336139`, `style - IDT=+0.031644`.
- All-pairs: `CLIP-S=0.702369`, `LPIPS=0.332186`, `style - IDT=+0.022246`.
- Identity: `CLIP-S=0.825586`, `LPIPS=0.316374`.
- Matched delta vs `k070 epoch_0003`:
  - transfer: `-0.000256` style, `+0.021521` LPIPS.
  - all-pairs: `-0.000865` style, `+0.019636` LPIPS.
- Runtime observability from summary:
  - `translation_delta_from_identity=0.018194`.
  - `routing_entropy=1.585930`.
  - `effective_experts=4.921388`.
  - `spatial_abs=0.836015` on transfer.
- Convergence state from `round2_convergence.json`: `converged=false`, best transfer/all-pairs remains `epoch_0009`, last Pareto remains `epoch_0010`, `since_last_pareto=3`, and `tail_flat=true`.
- Read: e13 nearly recovers parent transfer style but at the worst LPIPS cost in the post-e10 tail. One more non-Pareto retained checkpoint should satisfy the regular-family patience condition if the deep review contract is not contradicted.

## Epoch 14 Full Eval

- Full eval completed at `2026-06-14 12:51:38 Asia/Shanghai`; training resumed into epoch 15.
- Checkpoint: `epoch_0014.pt`.
- Training time: `1603.7s` (`26.73min`) from epoch log `data+comp`; eval wall time from curve: `266.5s`.
- Transfer: `CLIP-S=0.672185`, `LPIPS=0.324834`, `style - IDT=+0.032264`.
- All-pairs: `CLIP-S=0.703218`, `LPIPS=0.322686`, `style - IDT=+0.023095`.
- Identity: `CLIP-S=0.827351`, `LPIPS=0.314091`.
- Matched delta vs `k070 epoch_0003`:
  - transfer: `+0.000365` style, `+0.010216` LPIPS.
  - all-pairs: `-0.000015` style, `+0.010136` LPIPS.
- Runtime observability from summary:
  - `translation_delta_from_identity=0.018548`.
  - `routing_entropy=1.495014`.
  - `effective_experts=4.496835`.
  - `spatial_abs=0.819658` on transfer.
- Convergence state from `round2_convergence.json`: `converged=false`, best transfer/all-pairs remains `epoch_0009`, `epoch_0014` is a new candidate-curve Pareto point, and `since_last_pareto=0`.
- Read: e14 is useful evidence that SMoE can recover style after the tail, but it is still LPIPS-costly and not a promotion candidate. Continue because the new Pareto point resets the formal patience rule.
