# SMoE Translator Stage Plan

## Scope

This stage changes only the tokenizer mechanism:

- Parent checkpoint: `k070 epoch_0003`.
- Control: existing `pure_latent_spatial` parent and its all-checkpoint curve.
- Candidate: `tokenizer_family=smoe_translator`.
- Unchanged: solver, loss, topogate, appearance head, dataset, batch schedule, full-eval contract.

## Launch Contract

- Remote cwd: `/mnt/i/Github/Latent_Style`.
- Remote python: `/home/xy/venvs/samam312/bin/python`.
- Output root: `/mnt/i/Github/Latent_Style/exp/aaai2027_phase2_smoe_translator_k070_e3_seed42_b12a1`.
- Log path: `/mnt/i/Github/Latent_Style/exp/inmortal-exp/aaai2027_phase2_smoe_translator_k070_e3_seed42_b12a1_train.log`.
- VRAM: one lane only, formal cap `< 11.0 GiB`, exploded stop `> 11.3 GiB`.
- Eval: remote `CLIP-S + LPIPS` every retained epoch.

## Convergence Rule

- Do not close while the best checkpoint is within the newest two retained checkpoints.
- Close only after four later checkpoints fail to add a new transfer/all-pairs CLIP-S/LPIPS Pareto point and the last three checkpoints are near-flat.
- If the fast curve and visual/deep read disagree, extend by two retained checkpoints before closure.

## Observability Required

- `translation_delta_from_identity`
- `routing_entropy`
- `effective_experts`
- `spatial_abs`

## Decision Rule

- If early LPIPS sharply worsens, first inspect identity init and scale before treating the mechanism as negative.
- If LPIPS stays in-band and style improves against the matched control, promote SMoE as a tokenizer-side positive.
- If style does not improve, archive SMoE-only before moving to Fiberwise SWD.
