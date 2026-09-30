# 2026-05-30 Tokenizer Restart Design

## Baseline Choice

This restart uses documented baselines instead of the failed `high032` tokenizer run.

1. `t01_ws0p03_g6_nl0p05` is the raw-style/Pareto baseline.
   Evidence: `docs/experiments/2026-05-20-256-diffeomorphic-tangent-progress.md` reports `clip_style=0.7264`, `LPIPS=0.5170`, `clip_content=0.7570`, `DINO-SSM=0.0263`. The 2026-05-22 regression-fix note confirms the checkpoint can reproduce `clip_style=0.7264026194016138`.

2. EC best is the content-preserving reference.
   Evidence: `docs/repro_report_zh/00_总览与核心结论.md` records `K2_r00_balanced_default epoch3` with `CLIP-S=0.6980`, `CLIP-C=0.8727`, `LPIPS=0.3777`, `EC=0.4343`, and `Entropy gate 5.0 epoch1` with `CLIP-S=0.6916`, `CLIP-C=0.8804`, `LPIPS=0.3684`, `EC=0.4368`.

3. `t00_ws0p03_g6_nl0` is only a same-family sanity check, not the tokenizer baseline.
   Evidence: the same docs report `clip_style=0.7259`, `LPIPS=0.5166`, `clip_content=0.7602`, `DINO-SSM=0.0259`.

The tokenizer line should therefore be judged against two endpoints: preserve `t01` style strength, while learning a representation knob that can move toward EC-best content preservation without manual per-style hacks.

Current baseline policy:

- Primary style baseline: `t01_ws0p03_g6_nl0p05`, because it is the documented paper-facing strong-style operating point.
- Content-preserving reference: EC-best (`K2_r00_balanced_default epoch3` and `Entropy gate 5.0 epoch1`), because these define the low-LPIPS/high-content endpoint rather than the style endpoint.
- Sanity check only: `t00_ws0p03_g6_nl0`, because it is adjacent to `t01` in the same tangent sweep but is not the selected baseline for tokenizer work.
- Invalid baseline for this restart: failed high032/set-encoder tokenizer runs. They are negative ablations only.

## Representation Hypothesis

The tokenizer is not a larger embedding. Its job is to expose a small metric space for style control.

The first implementation should represent each style as three low-dimensional fields:

- `identity`: global color/moment displacement. This should affect broad color and contrast changes.
- `texture`: local brush/roughness amplitude. This should affect high-frequency style strength.
- `geometry`: stroke transport tendency. This should affect tangent/diffeomorphic behavior only through existing LANCET consumers.

Stage 1 should keep the parameter count small, roughly embedding-scale rather than transformer-scale. A reasonable target is below 20k parameters. This prevents the tokenizer from becoming a second backbone and makes frozen-backbone diagnostics meaningful.

## Stage-1 Tokenizer

Use a factorized token table:

```text
style_id -> identity token [d_id=24]
style_id -> texture token  [d_tex=32]
style_id -> geometry token [d_geo=24]
concat -> LayerNorm -> Linear -> style_code [style_dim=160]
```

Important constraints:

- No transformer in Stage 1.
- No target-latent or reference-image encoder in Stage 1. The tokenizer is a
  style-id/vocabulary representation module, not an evidence encoder.
- No external teacher or Seedream path.
- Keep the LANCET consumer interface unchanged: it still receives one `style_code`.
- Store debug tensors: per-field norm, pairwise cosine, and projected code norm.

This is intentionally close to `style_emb`, but it is no longer an opaque vector. It gives us separable axes that can be frozen, reinitialized, ablated, and measured.

The first code implementation is intentionally below transformer scale: three small style tables, one learned per-field gate vector, and one linear projector. This makes the first experiment a representation probe rather than another backbone-capacity experiment.

## Diagnostics Before Full Training

A tokenizer run is invalid unless these checks pass:

1. Parameter and gradient reachability:
   `style_tokenizer.identity`, `texture`, `geometry`, and projector must receive non-zero gradients in a batch16 smoke.

2. Style separability:
   per-style `style_code` cosine matrix should not collapse to all near-1.0 after initialization or after smoke.

3. Field usage:
   field norms should be non-zero and not dominated by a single field by more than roughly 10x during early training.

Only after those pass should we run a real base on the remote 3060.

Implementation note from the first smoke:

- Replacing the table alone was insufficient: the initial forward smoke produced finite images but zero tokenizer gradients.
- Root cause: in this clean baseline, `style_code` was mostly bypassed when skip routing was disabled and the decoder modulation module was instantiated but not applied.
- Fix: apply decoder-side `NormFreeModulation` before the output delta head. After this, all tokenizer fields and the projector receive non-zero gradients in a batch2 CPU shape smoke.

## Training Plan

1. Implement Stage-1 tokenizer on clean branch `codex/tokenizer-clean-c3058eab`.
2. Derive smoke configs from documented `t00/t01` settings, not from `high032`.
3. Run batch16 smoke for correctness and gradient diagnostics.
4. Run batch80 or calibrated remote batch for 8 epochs to establish a tokenizer+LANCET base.
5. If the base is within range, alternate:
   - freeze LANCET, train tokenizer only;
   - freeze tokenizer, train LANCET consumer;
   - compare movement against `t01/t00` and EC-best endpoints.

Success is not a smoke test. The real target remains `clip_style >= 0.73` with `LPIPS` near `0.45`, verified on strict evaluation and visual grids.

## Run 001: Factorized Tokenizer on t01 Settings

Code/config state used for the run: branch `codex/tokenizer-clean-c3058eab` at `4efa1c32a` before this result note was appended.

Config: `configs/tokenizer_t01_factorized_base.json`

Key implementation details:

- Replaced the opaque style table with `FactorizedStyleTokenizer(identity=24, texture=32, geometry=24)`.
- Kept the LANCET consumer interface as one projected `style_code`.
- Added decoder-side `NormFreeModulation` before the delta head because the first smoke found zero tokenizer gradients without it.
- Used t01-derived model/loss settings and explicit shared remote paths for `latent-256`, `style_data/overfit50`, and `eval_cache`.

Smoke:

- Remote Windows Python, batch 16, CUDA, one short epoch.
- Forward was finite.
- Backward debug showed non-zero gradients for `identity`, `texture`, `geometry`, field gates, and projector.

Full train:

- Remote 3060, Windows Python, batch 80, 8 epochs from scratch.
- Output directory: `exp/tokenizer_t01_factorized_base`.
- Final checkpoint: `epoch_0008.pt`.

Strict full_eval, 750 generated images:

| epoch | all CLIP-S | all LPIPS | all CLIP-C | transfer CLIP-S | transfer LPIPS | photo2art CLIP-S | photo2art LPIPS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 0.70917 | 0.44540 | 0.82241 | 0.68136 | 0.45646 | 0.65869 | 0.47874 |
| 7 | 0.70797 | 0.42961 | 0.82872 | 0.67996 | 0.43985 | 0.65391 | 0.45190 |
| 8 | 0.70916 | 0.43720 | 0.82501 | 0.68145 | 0.44806 | 0.65771 | 0.46710 |

Interpretation:

- This tokenizer is live and trainable; the previous failure mode of a silent style branch is fixed.
- It does not recover the documented t01 style endpoint (`clip_style=0.7264, LPIPS=0.5170`).
- It lands closer to the EC/content-preserving side than the t01 style side: LPIPS is much lower, but style strength collapses by about 0.017.
- Therefore the next tokenizer step should not simply increase token count. The immediate bottleneck is style actuation strength: the projected fields reach the decoder, but the current consumer path is too conservative to match t01's diffeomorphic style pull.

## Run 002: Alternating Freeze Probe

Purpose: separate tokenizer optimization from LANCET consumer optimization.

Code changes:

- Added `training.freeze_mode` for non-distill OMF training.
- Supported `tokenizer_only`, `style_branch`, and `backbone_only`.
- Added `training.resume_optimizer=false` so alternating freeze modes can resume model weights without loading incompatible optimizer parameter groups.

Tokenizer-only continuation:

- Config: `configs/tokenizer_t01_factorized_tokonly_e12.json`
- Resume: `tokenizer_t01_factorized_base/epoch_0008.pt`
- Freeze: LANCET frozen, tokenizer trainable.
- Result: no improvement.

| run | all CLIP-S | all LPIPS | all CLIP-C | transfer CLIP-S | transfer LPIPS | photo2art CLIP-S | photo2art LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| base e8 | 0.70916 | 0.43720 | 0.82501 | 0.68145 | 0.44806 | 0.65771 | 0.46710 |
| tokenizer-only e12 | 0.70847 | 0.43848 | 0.82306 | 0.68078 | 0.44889 | 0.65800 | 0.46818 |

Backbone-only continuation:

- Config: `configs/tokenizer_t01_factorized_backbone_e16.json`
- Resume: `tokenizer_t01_factorized_tokonly_e12/epoch_0012.pt`
- Freeze: tokenizer frozen, LANCET/style spatial branch trainable.
- Result: small style improvement, with expected LPIPS/content tradeoff.

| run | all CLIP-S | all LPIPS | all CLIP-C | transfer CLIP-S | transfer LPIPS | photo2art CLIP-S | photo2art LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| tokenizer-only e12 | 0.70847 | 0.43848 | 0.82306 | 0.68078 | 0.44889 | 0.65800 | 0.46818 |
| backbone-only e16 | 0.71260 | 0.44534 | 0.81980 | 0.68583 | 0.45689 | 0.66056 | 0.47175 |

Conclusion:

- The current tokenizer is not merely undertrained: tokenizer-only optimization cannot recover style.
- The LANCET consumer can extract slightly more style from the same frozen tokens, so the actuator path matters.
- The remaining gap to t01 is still large. The next representation step should remove `concat -> LayerNorm -> Linear` field mixing and use independent field projections. This keeps the parameter budget small while making field usage more identifiable.

## Run 003: Additive Field Projection Probe

Purpose: test whether removing the single `concat -> LayerNorm -> Linear` bottleneck helps the three fields stay identifiable.

Code/config:

- Config: `configs/tokenizer_t01_additive_base.json`
- Same t01-derived model/loss settings as `tokenizer_t01_factorized_base`.
- Changed only tokenizer projection:
  - `identity -> LayerNorm -> Linear(style_dim)`
  - `texture -> LayerNorm -> Linear(style_dim)`
  - `geometry -> LayerNorm -> Linear(style_dim)`
  - `style_code = (identity_code + texture_code + geometry_code) / sqrt(3)`
- Run location: remote 3060 Windows worktree `I:\Github\Latent_Style_TokenizerClean`.
- Batch size: 80. This keeps the comparison matched to Run 001, but only uses about 5.3GB VRAM, so it is a comparable diagnostic rather than the final 10GB-throughput profile.

Tokenizer diagnostics:

- The tokenizer is connected: identity, texture, geometry, field gates, and all three projectors receive non-zero gradients.
- The raw token fields are not collapsed at epoch 8 (`identity_texture_cos ~= -0.118`, `identity_geometry_cos ~= -0.043`, `texture_geometry_cos ~= 0.060`).
- The projected style-code fields are still substantially aligned (`identity_texture_code_cos ~= 0.445`, `identity_geometry_code_cos ~= 0.443`, `texture_geometry_code_cos ~= 0.390`). This means independent projectors did not produce an orthogonal actuation basis after projection; the consumer still sees a fairly shared direction in `style_code` space.

Full eval:

| run | all CLIP-S | all LPIPS | all CLIP-C | transfer CLIP-S | transfer LPIPS | photo2art CLIP-S | photo2art LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| concat base e8 | 0.70916 | 0.43720 | 0.82501 | 0.68145 | 0.44806 | 0.65771 | 0.46710 |
| additive e6 | 0.70846 | 0.45124 | 0.80889 | 0.68315 | 0.46114 | 0.65930 | 0.47702 |
| additive e7 | 0.70701 | 0.43451 | 0.81745 | 0.68077 | 0.44362 | 0.65321 | 0.44972 |
| additive e8 | 0.70844 | 0.44170 | 0.81346 | 0.68265 | 0.45150 | 0.65681 | 0.46589 |
| backbone-only e16 | 0.71260 | 0.44534 | 0.81980 | 0.68583 | 0.45689 | 0.66056 | 0.47175 |
| t01 target | 0.7264 | 0.5170 | 0.7570 | - | - | - | - |
| EC-best reference | 0.6980 | 0.3777 | 0.8727 | - | - | - | - |

Interpretation:

- Additive projection is not a positive move. It is at best neutral on style and worse than the previous `backbone_only` continuation.
- The result remains between EC-best and t01: LPIPS/content are much better than t01, but style is far below t01 by about `0.018`.
- The bottleneck is not only field mixing. The tokenizer path lacks a high-capacity style carrier/actuator that can preserve the original t01-style pull while making identity/texture/geometry measurable.
- Next probe should not be a larger black-box tokenizer. It should be a small carrier-plus-fields tokenizer: keep a full `style_dim` carrier table inside the tokenizer, add low-dimensional diagnostic fields as residual controls, and expose per-field projection norms/cosines. This restores the missing style-code rank while preserving tokenizer observability.

## Run 004 Plan: Carrier-Plus-Fields Tokenizer

Rationale:

- The `concat` and `additive` probes both use only `24+32+24` raw token dimensions before projection. That is a representation bottleneck compared with the original full-rank `style_dim=160` style conditioning.
- The failure mode is not tokenizer silence: gradients are live. The failure is insufficient style actuation/rank after the tokenizer projection.
- Therefore the next design should restore a full-rank style carrier inside the tokenizer while keeping fields as measurable residual controls.

Design:

```text
style_id -> carrier [style_dim=160]
style_id -> identity [24] -> projected identity_code [160]
style_id -> texture  [32] -> projected texture_code  [160]
style_id -> geometry [24] -> projected geometry_code [160]
residual_code = (identity_code + texture_code + geometry_code) / sqrt(3)
style_code = carrier + tokenizer_residual_gain * residual_code
```

This is not a return to external `style_emb`: the carrier is part of `FactorizedStyleTokenizer`, is logged with the tokenizer, and can be frozen or ablated with the other fields. It gives the backbone a full-rank style carrier so the first tokenizer base has a fair chance to preserve `t01` style strength.

Config:

- `configs/tokenizer_t01_carrier_base_b176.json`
- `tokenizer_projection_mode="carrier_residual"`
- `tokenizer_residual_gain=0.5`
- `batch_size=176`, selected after a batch160 calibration reached about 8.85GB and therefore sat just below the remote 3060 9-10.8GB formal-training target.

Acceptance criteria:

- Must reach non-zero gradients for `carrier`, all three raw fields, field gates, and all three field projectors.
- Must not regress below `concat base e8` (`0.70916 / 0.43720`) if it is merely capacity-restoring.
- To matter as a tokenizer base, it should move toward `backbone-only e16` or higher (`0.71260 / 0.44534`) while preserving LPIPS near `0.45`.
- The true target remains beyond the documented `t01` style endpoint: `clip_style >= 0.73` with LPIPS near `0.45`.

Run-004 calibration:

- `batch_size=160` reached about `8.85GB`, slightly below the formal-training target. It was stopped at epoch 2.
- `batch_size=176` reached about `9.6GB`, with all tokenizer gradients non-zero.
- Important protocol correction: batch176 has about half the optimizer steps per epoch compared with the earlier batch80 runs. Therefore batch176/8epoch is a VRAM-calibrated diagnostic, not an update-budget-matched comparison to Run 001.

Batch176/8epoch full eval:

| run | all CLIP-S | all LPIPS | all CLIP-C | transfer CLIP-S | transfer LPIPS | photo2art CLIP-S | photo2art LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| carrier b176 e6 | 0.70214 | 0.44095 | 0.81206 | 0.67634 | 0.45031 | 0.64744 | 0.45252 |
| carrier b176 e7 | 0.70340 | 0.44802 | 0.80897 | 0.67834 | 0.45713 | 0.64637 | 0.46342 |
| carrier b176 e8 | 0.70216 | 0.44755 | 0.81036 | 0.67663 | 0.45728 | 0.64494 | 0.45751 |

Interim interpretation:

- The b176/e8 result is worse than concat and additive, but it has only about half the update budget.
- Do not use this as final evidence against the carrier tokenizer.
- Next run: `configs/tokenizer_t01_carrier_base_b176_e16.json`, from scratch, batch176, 16 epochs. This keeps the 9-10.8GB VRAM discipline while restoring an optimizer-step budget close to the earlier batch80/8epoch tokenizer base.

Batch176/16epoch update-budget-matched full eval:

| run | all CLIP-S | all LPIPS | all CLIP-C | transfer CLIP-S | transfer LPIPS | photo2art CLIP-S | photo2art LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| carrier b176 e12 | 0.70506 | 0.46357 | 0.78630 | 0.68190 | 0.47350 | 0.65511 | 0.48638 |
| carrier b176 e14 | 0.70563 | 0.46507 | 0.79129 | 0.68185 | 0.47519 | 0.65689 | 0.48687 |
| carrier b176 e16 | 0.70469 | 0.46356 | 0.78949 | 0.68077 | 0.47323 | 0.65344 | 0.48141 |

Conclusion:

- Even after matching update budget, `carrier_residual` is worse than `concat base e8` and much worse than `backbone-only e16`.
- Training diagnostics explain the failure: the residual branch grows much larger than the carrier (`residual_code_norm ~= 2.78`, `carrier_norm ~= 0.285` at epoch16), and projected field cosines rise to `0.45-0.55`. The tokenizer fields remain trainable, but their projected actuation space collapses toward a shared direction.
- `carrier_residual` should be treated as a negative representation probe. Do not use it as the base for tokenizer-only refinement.

## Run 005 Plan: Freeze LANCET, Expand Tokenizer Only

The appropriate current tokenizer-LANCET base is `tokenizer_t01_factorized_backbone_e16`, not the carrier branch:

- It is the best verified tokenizer-line point so far: `all CLIP-S=0.71260`, `LPIPS=0.44534`.
- It is still below `t01`, but it has a working consumer path and LPIPS near the target range.

Next experiment:

- Config: `configs/tokenizer_t01_big_tokonly_from_backbone_e16.json`
- Initialize from `exp/tokenizer_t01_factorized_backbone_e16/epoch_0016.pt`.
- Freeze all LANCET/backbone parameters.
- Ignore old `style_tokenizer.*` checkpoint keys and train a larger tokenizer from scratch:
  - `identity_dim=48`
  - `texture_dim=64`
  - `geometry_dim=48`
  - `projection_mode="concat"`
- Use `resume_training_state=false` so the checkpoint is treated as initialization rather than a continuation epoch.

Hypothesis:

- If the bottleneck is tokenizer expressivity, this should move style above `0.71260` without changing the main actuator.
- If it fails, the remaining bottleneck is not tokenizer size; it is the consumer path/loss target that maps style codes into image-space edits.

Run 005 result:

| run | all CLIP-S | all LPIPS | all CLIP-C | transfer CLIP-S | transfer LPIPS | photo2art CLIP-S | photo2art LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| backbone-only e16 base | 0.71260 | 0.44534 | 0.81980 | 0.68583 | 0.45689 | 0.66056 | 0.47175 |
| big tokenizer-only e8 | 0.71197 | 0.43917 | 0.82560 | 0.68472 | 0.45051 | 0.65970 | 0.47238 |
| big tokenizer-only e10 | 0.71229 | 0.43867 | 0.82543 | 0.68511 | 0.44986 | 0.65981 | 0.47143 |
| big tokenizer-only e12 | 0.71191 | 0.43799 | 0.82585 | 0.68472 | 0.44911 | 0.65995 | 0.47157 |

Interpretation:

- Enlarging the tokenizer while freezing LANCET does not break the current style ceiling. Best style is `0.71229`, slightly below the frozen-backbone source point `0.71260`.
- It does improve LPIPS/content slightly, so the larger tokenizer can re-center the conditioning inside the existing actuator's basin.
- This is useful evidence: the current gap to `t01=0.7264` is not primarily raw tokenizer capacity. The bottleneck is the consumer/actuator path or the target distribution induced by the loss.
- Next representation step should keep the best tokenizer checkpoint as a content-preserving tokenizer variant, then unfreeze a narrow style-actuator subset instead of the whole backbone. Candidate trainable subset: tokenizer + decoder modulation + semantic/style spatial routing, with the rest frozen.

## Run 006 Plan: Frozen-Backbone Direct-Code Upper Bound

User decision: before unfreezing any actuator, first push tokenizer-only as far as possible.

Rationale:

- `big tokenizer-only` slightly improved LPIPS/content but did not improve style over `backbone-only e16`.
- That result still contains a tokenizer architectural bottleneck: `identity/texture/geometry -> concat projector`.
- To measure the frozen-backbone upper bound, remove the field bottleneck entirely and train a direct per-style `style_code` table inside `FactorizedStyleTokenizer`.

Design:

- Config: `configs/tokenizer_t01_direct_tokonly_from_backbone_e16.json`
- Base checkpoint: `exp/tokenizer_t01_factorized_backbone_e16/epoch_0016.pt`
- Freeze mode: `tokenizer_only`
- Ignore checkpoint `style_tokenizer.*` and reset tokenizer.
- Tokenizer mode: `tokenizer_projection_mode="direct_code"`
- Tokenizer parameters: exactly `num_styles * style_dim = 5 * 160 = 800` trainable code parameters.
- `tokenizer_init_std=0.2` so initial code norm is not near-zero.
- `learning_rate=5e-4`, `num_epochs=24`.

Interpretation rule:

- If direct-code tokenizer exceeds `0.71260`, the previous style-id tokenizer architectures were the bottleneck.
- If it plateaus near or below `0.71260`, the frozen single global-code interface is saturated for style-id-only deterministic lookup.
- This is not the tokenizer representation upper bound. It does not test stochastic tokenizers, sparse shared vocabularies, spatial tokens, per-layer tokens, or semantic-region-conditioned token injection.

## Representation Boundary Correction

Tokenizer and LANCET must be separated by role:

- Tokenizer represents "what the style is" as an executable control object, currently `style_code`.
- LANCET consumes that control object and performs the latent ODE/image edit.
- In tokenizer-only training, LANCET is frozen but still mounted in the forward/backward graph. Gradients pass through frozen LANCET into tokenizer parameters; LANCET weights are not updated.
- There are not two style encoders. The mainline has one tokenizer; any LANCET
  encoder blocks are content/actuation blocks, not style-evidence encoders.

Do not train a tokenizer encoder on per-sample `target_style` latent for the main benchmark. The standard evaluation/inference path only provides `target_style_id`; letting the tokenizer read the target latent during training creates a condition mismatch and leaks information that is unavailable at deployment. The main tokenizer object must therefore learn from style identity/learned vocab parameters into `style_code`, while richer future tokenizers should still respect the same available conditioning boundary.

Next valid tokenizer probes:

- direct `style_code` table as a control probe for the frozen single-code interface.
- sparse concept-atom tokenizer: style id -> logits over shared learnable atoms -> `style_code`.
- distributional tokenizer: style id -> mean/logvar over executable code with KL/entropy control, sampled during training and deterministic at eval.
- multi-token tokenizer only after LANCET is explicitly extended to consume layer-specific tokens, not by leaking target latents.

## Run 007 Plan: Frozen LANCET, Sparse Concept-Atom Tokenizer

This is the next valid representation probe after the direct-code control.

Design:

```text
style_id -> atom_logits [K]
weights = softmax(atom_logits / tau)
style_code = weights @ concept_atoms[K, style_dim]
```

Config:

- `configs/tokenizer_t01_concept_atoms_tokonly_from_backbone_e16.json`
- Base checkpoint: `exp/tokenizer_t01_factorized_backbone_e16/epoch_0016.pt`
- Freeze mode: `tokenizer_only`
- Ignore checkpoint `style_tokenizer.*` and reset tokenizer.
- `K=32`, `tau=0.12`, no target latent input.
- `tokenizer_init_std=0.2` so the initial atom mixture is not a near-uniform low-norm average.

Hypothesis:

- If concept atoms beat direct code or big concat, shared atoms give a better style metric space than independent style vectors.
- If atoms underperform but show low entropy collapse, the next change is entropy/temperature control, not LANCET changes.
- If atoms match direct code, the frozen single-code consumer is likely the limit and the next valid step is per-layer token consumption in LANCET.

Local smoke before remote training:

- Config loads: `configs/tokenizer_t01_concept_atoms_tokonly_from_backbone_e16.json`.
- Batch2 CPU OMF smoke with reduced SWD projections is finite.
- Trainable path under manual tokenizer-only freeze:
  - non-zero tokenizer gradients: `atom_logits.weight`, `concept_atoms`
  - non-tokenizer backbone grad count: `0`
- Debug state is populated:
  - `atom_entropy ~= 2.26`
  - `atom_effective_count ~= 9.6` out of `K=32`
  - `atom_max_prob ~= 0.39`
  - `style_code_norm ~= 1.11`
- No tokenizer path reads per-sample `target_style` latent. The target latent appears only in loss/evaluation targets and in existing optional moment/spatial paths when explicitly passed, not in tokenizer conditioning.

This smoke validates graph connectivity only. It is not evidence of metric improvement.

Remote trainer smoke:

- Remote path: `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge`
- Device: CUDA on the remote 3060.
- Entry: `SBTrainer` with the real concept-atom config, reduced to one batch for smoke.
- Resume/freeze path is valid: checkpoint `exp/tokenizer_t01_factorized_backbone_e16/epoch_0016.pt` is found and loaded.
- Trainable params:
  - `style_tokenizer.concept_atoms`
  - `style_tokenizer.atom_logits.weight`
- `all_trainable_tokenizer=true`; no LANCET/backbone params are trainable.
- One-batch loss is finite; peak smoke memory at batch16 is about `0.55GB`.
- Atom debug at trainer smoke:
  - `atom_entropy ~= 2.71`
  - `atom_effective_count ~= 15.0`
  - `atom_max_prob ~= 0.21`
  - `style_code_norm ~= 0.77`

This confirms the actual training entry obeys the intended tokenizer-only graph. A separate batch224 one-step memory calibration is still required before formal remote training.

Remote memory calibration:

- Batch224 one-step trainer smoke: `peak_gb ~= 7.37`, below the formal 9-10.8GB target.
- Batch288 one-step trainer smoke: `peak_gb ~= 9.47`, finite loss, `all_trainable_tokenizer=true`.
- Formal tokenizer-only concept-atom config is therefore set to `batch_size=288`.

Run 007 result:

- Remote path: `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge`.
- Training completed to `epoch_0024`.
- Strict full_eval was run for epochs 8, 16, and 24.

| run | all CLIP-S | all LPIPS | all CLIP-C | transfer CLIP-S | transfer LPIPS | photo2art CLIP-S | photo2art LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| concept atoms e8 | 0.70089 | 0.42239 | 0.83576 | 0.67254 | 0.42782 | 0.64701 | 0.44629 |
| concept atoms e16 | 0.70478 | 0.42587 | 0.83281 | 0.67685 | 0.43294 | 0.65196 | 0.45241 |
| concept atoms e24 | 0.70551 | 0.42661 | 0.83238 | 0.67773 | 0.43403 | 0.65304 | 0.45373 |
| backbone-only e16 base | 0.71260 | 0.44534 | 0.81980 | 0.68583 | 0.45689 | 0.66056 | 0.47175 |

Tokenizer diagnostics:

- Atom entropy fell from about `2.45` early to about `1.70` by epoch 24.
- Effective atom count fell from about `11.6` to about `5.5` out of `K=32`.
- `style_code_norm` rose from about `1.03` to about `2.29`.
- `concept_atoms` and `atom_logits.weight` retained non-zero gradients.

Conclusion:

- Sparse concept atoms are live, but this version is a negative probe.
- The low LPIPS/high content numbers mean the frozen LANCET can execute a
  conservative code, but pure shared atoms lose too much style pull.
- The likely failure is a representation-rank bottleneck: forcing every style
  through a convex mixture of shared atoms removes the per-style prototype
  capacity that the frozen single-code LANCET expects.
- This is not a tokenizer upper bound. It only rejects pure concept atoms as a
  replacement for the current single-code interface.

## Run 008 Plan: Direct Prototype + Shared Atom Residual Warmup

Purpose: test whether shared concept atoms add useful structure when they do not
replace each style's full-rank executable prototype. This is a tokenizer
initialization run, not a standalone performance run.

The previous pure-atom run had no per-style full-rank carrier. That makes the
representation elegant but too restrictive for a frozen LANCET trained to
consume one dense `style_code`. The next tokenizer should be nested: it can
degenerate to direct code if atoms are useless, while still exposing a shared
vocabulary if atoms help.

Design:

```text
style_id -> prototype_code [style_dim]
style_id -> atom_logits [K]
atom_residual = softmax(atom_logits / tau) @ concept_atoms[K, style_dim]
style_code = prototype_code + residual_gain * atom_residual
```

Boundary:

- Inputs remain only `target_style_id` and learned tokenizer parameters.
- No current batch `target_style` latent, reference image, or external model
  output enters the tokenizer.
- Target latents remain only as the loss-side style distribution.

Config:

- `configs/tokenizer_t01_direct_atom_residual_tokonly_from_backbone_e16.json`
- Base checkpoint: `exp/tokenizer_t01_factorized_backbone_e16/epoch_0016.pt`.
- Freeze mode: `tokenizer_only`.
- Ignore old `style_tokenizer.*` checkpoint keys and reset tokenizer.
- `tokenizer_projection_mode="direct_atom_residual"`.
- `K=32`, `tau=0.25`, `tokenizer_residual_gain=0.25`.
- `num_epochs=2`.
- Save dir: `exp/tokenizer_t01_direct_atom_residual_warmup_e2_from_backbone_e16`.
- Batch should reuse the concept-atom calibrated `batch_size=288` unless smoke
  memory contradicts it.

Interpretation:

- Do not evaluate this run as a final model. Its job is to move tokenizer weights
  out of random initialization and into a finite, executable style-control
  region.
- Acceptance is diagnostic: finite short training, non-zero gradients for
  `direct_code`, `concept_atoms`, and `atom_logits`, and sane debug values for
  prototype norm, atom residual norm, atom entropy, and prototype-residual
  cosine.
- The real test is Run 009: whether a fresh LANCET trained from scratch can read
  the warm-started fixed tokenizer vocabulary.

Remote smoke:

- Config loads on remote Windows Python.
- Batch2 real OMF smoke on CUDA is finite: loss about `6.36`.
- Trainable parameters are exactly:
  - `style_tokenizer.concept_atoms`
  - `style_tokenizer.direct_code.weight`
  - `style_tokenizer.atom_logits.weight`
- All three receive non-zero gradients.
- Initial debug state:
  - `prototype_norm ~= 2.53`
  - `atom_residual_norm ~= 0.58`
  - `atom_entropy ~= 3.24`
  - `atom_effective_count ~= 25.4`
  - `prototype_residual_cos ~= -0.03`

Warmup result:

- The first accidental launch used the old 24-epoch setting and was stopped at
  `epoch_0008`; it is not part of the main evidence for this route.
- The corrected Stage A warmup runs only 2 epochs.
- Output: `exp/tokenizer_t01_direct_atom_residual_warmup_e2_from_backbone_e16/epoch_0002.pt`.
- Runtime was about one minute on the remote 3060 with batch 288.
- Final epoch metrics: loss `8.0729`, terminal SWD `7.9375`, kinetic `0.1639`.
- Numeric debug at epoch 2:
  - `style_code_norm ~= 2.56`
  - `prototype_norm ~= 2.56`
  - `atom_residual_norm ~= 0.56`
  - `atom_entropy ~= 3.22`
  - `atom_effective_count ~= 25.0 / 32`
  - `prototype_residual_cos ~= 0.02`
  - non-zero gradients for `direct_code.weight`, `concept_atoms`, and
    `atom_logits.weight`

Interpretation of warmup:

- This is a valid initializer: finite, connected, and not atom-collapsed.
- It is not a trained tokenizer claim and should not be evaluated as a final
  stylization model.
- The next evidence comes from whether LANCET can learn to consume this fixed
  vocabulary.

## Run 009 Plan: Trained Tokenizer, Fresh LANCET Consumer

Purpose: test whether a learned tokenizer can act as a stable style vocabulary
for a newly trained LANCET, instead of only being optimized inside an already
formed LANCET loss landscape.

This route is different from the earlier alternating continuation:

```text
Stage A:
existing LANCET frozen
style_id -> Tokenizer(theta_T) -> style_code
content, style_code -> frozen LANCET -> endpoint -> SWD/OMF

Stage B:
load only style_tokenizer.* from Stage A
initialize LANCET(theta_L) from scratch
freeze Tokenizer(theta_T)
content, fixed style_code -> fresh LANCET(theta_L) -> endpoint -> SWD/OMF
```

Why this is valid:

- The tokenizer still only sees `target_style_id`; no target latent or reference
  image enters the conditioning path.
- Stage A learns an executable style vocabulary through the old LANCET.
- Stage B asks whether a new LANCET can learn to render that fixed vocabulary
  better than the old consumer.
- This isolates representation stability from actuator training, rather than
  letting both modules move and hide the failure mode.

Implementation requirement:

- Use `training.resume_include_prefixes=["style_tokenizer."]` so Stage B loads
  only tokenizer weights from the Stage A checkpoint.
- Use `freeze_mode="backbone_only"` so tokenizer parameters are frozen while
  LANCET trains.
- Do not use `resume_ignore_prefixes` in Stage B; include-prefix filtering is
  the authority.

Config:

- Stage A warmup: `configs/tokenizer_t01_direct_atom_residual_tokonly_from_backbone_e16.json`
- Stage B: `configs/tokenizer_t01_direct_atom_residual_frozen_tok_fresh_lancet_e16.json`

Implementation smoke:

- Added `training.resume_include_prefixes` for checkpoint filtering.
- A remote smoke checkpoint confirmed that Stage B loads only
  `style_tokenizer.*`.
- In Stage B smoke, tokenizer parameters are all loaded equal to Stage A,
  tokenizer trainable count is zero, and non-tokenizer LANCET parameters are
  trainable.
- A second remote smoke using the real Stage A warmup checkpoint confirmed:
  - tokenizer trainable count is zero;
  - LANCET trainable parameter count is non-zero;
  - one batch loss is finite (`~9.13`);
  - tokenizer gradient count is zero;
  - non-tokenizer backbone gradients are present.

Interpretation:

- If Stage B beats the Stage A frozen-LANCET endpoint, the tokenizer vocabulary
  is useful but the previous actuator was limiting.
- If Stage B regresses badly, the tokenizer was overfit to the old LANCET's
  actuation geometry and is not a portable style representation.
- If Stage B matches the best tokenizer-line result but cannot approach
  `t01=0.7264`, the next required architecture change is not tokenizer size; it
  is a richer LANCET consumer interface, such as layer-specific style tokens.
