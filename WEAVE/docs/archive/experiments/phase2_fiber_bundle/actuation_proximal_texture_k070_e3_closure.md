# Actuation Proximal Texture Closure

Date: 2026-06-16

## Status

- Family: `actuation_proximal_texture_k070_e3_b16a2bf16_vlen010`.
- Parent: `k070 epoch_0003`.
- Controlled switch delta: enable `model.proximal_mode=crossattn_texture`;
  disable `style_injection_mode` and `style_delta_mode`.
- Held fixed: tokenizer, solver, losses, TopoGate, parent, effective b32 via
  b16 accumulation-2, and transfer-only training eval.
- Closure status: `converged_not_promoted`.

## Evidence

- Live convergence curve: `full_eval_fast10`, deterministic `10` sources per
  style, `200` transfer pairs, no generated PNG/grid.
- Full confirmation curve: `full_eval_confirm`, deterministic `30` sources per
  style, `600` transfer pairs, no generated PNG/grid.
- Fast10 best: `epoch_0009`, transfer `0.680954 / 0.334124`.
- Fast10 final: `epoch_0014`, transfer `0.680484 / 0.335496`.
- Convergence: `epoch_0014` report has `converged=true`, `since_best=5`,
  `since_last_pareto=5`, and `tail_flat=true`; best and last Pareto remain
  `epoch_0009`.
- Full confirmation best: `epoch_0009`, transfer `0.674190 / 0.329931`.
- Full confirmation final: `epoch_0014`, transfer `0.673760 / 0.331171`.

## Matched Read

Against the parent transfer point `0.671820 / 0.314618`, the confirmed best
`epoch_0009` gains about `+0.002370` CLIP-S at a `+0.015313` LPIPS cost.

Against the stronger R16 full-board style frontier `0.674395 / 0.352223`, the
confirmed best is slightly lower style (`-0.000205`) but much lower LPIPS
(`-0.022292`). This is a useful structure-preserving point, but not a
style-frontier improvement and not a path toward the `0.74` style target by
itself.

## Decision

Close the proximal texture lane as `converged_not_promoted`.

The mechanism gives a small, real style response and stays within the
style-priority LPIPS budget, but it does not beat the existing style frontier.
It supports the `fiber.md` diagnosis that merely adding a local endpoint texture
residual is insufficient; the remaining bottleneck is still the generated
style-section/decoder actuation path.

## Next Action

Do not continue this lane or stack it with latent affine as a promoted model.
The next controlled experiment should attack the `dec_out` / generated-delta
rank bottleneck directly, with generated-delta rank and off-diagonal cosine
logged alongside CLIP-S/LPIPS.

## Artifacts

- Fast10 curve:
  `docs/experiments/phase2_fiber_bundle/eval/actuation_proximal_texture_k070_e3_b16a2bf16_vlen010/clip_lpips_curve_fast10.csv`
- Full confirmation curve:
  `docs/experiments/phase2_fiber_bundle/eval/actuation_proximal_texture_k070_e3_b16a2bf16_vlen010/clip_lpips_curve_confirm.csv`
- Fast10 convergence:
  `docs/experiments/phase2_fiber_bundle/eval/actuation_proximal_texture_k070_e3_b16a2bf16_vlen010/round2_convergence_fast10.json`
- Homepage plot data:
  `docs/experiments/phase2_fiber_bundle/plot_points.csv`
