# 2026-06-17 Implementation Audit: Was The Previous 616 Implementation Wrong?

## Short answer

Partly.

- The previous 616 line was **not a fake run**. The core `pure_vertical_flow` target projection and `sinkhorn_unbalanced` machinery were really implemented and really used by the active 616 configs.
- But several pieces were only **weak or proxy implementations** of the theory written in `docs/616/design.md`.
- The biggest mismatch was not a silent config bug. It was that some mechanisms were implemented as **diagnostic approximations**, then temporarily promoted into the active loop before we had fully checked whether they matched the intended geometry.
- A later audit also found a real **logging bug**: `transport_stats_*` was
  present in the live `loss_dict` and in `numeric_debug.jsonl`, but the epoch
  CSV writer in
  [utils/training.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/utils/training.py)
  forgot to map those fields in `append_training_log()`. That made some
  `training_*.csv` rows look like the mechanism was inactive when it was not.

## Practical verdict

To avoid mixing categories, the previous 616 line should now be read in three buckets:

### A. Real implementation mistakes

- `transport_stats_*` was computed but not fully written into the training CSV.
- old sampled-bridge `velocity_abs` was mislabeled and tracked the teacher target magnitude rather than the model prediction magnitude.

These issues affected observability and interpretation. They did **not** mean the remote runs were secretly training a different mechanism.

### B. Theory-to-implementation mismatches

- the retained OT control used a **raw-latent structural proxy** rather than the stronger internal-feature geometry described in `docs/616/design.md`
- `pure_vertical_flow` was applied to the supervision target before it was applied to bridge noise
- OT target geometry was still stochastic whenever `coupling_target_mode = "sample"`
- the clean 616 line still used the mixed `appearance_plus_structure` coupling objective in several retained controls

These are the main reasons the earlier 616 evidence was weaker or noisier than intended.

### C. Valid negative evidence

- replacing the OT proxy with encoder/tokenizer-native structure descriptors did **not** produce a promotable win
- standalone bridge-noise vertical projection improved structure-side probes but reduced style materially
- the first combined `terminal_affine + bridge-noise vertical` line produced only a soft positive, not a frontier jump

So the current bottleneck should no longer be explained mainly as "the previous implementation was simply wrong." Part of it was wrong or incomplete, but part of it was a real negative result after correction.

## What was implemented correctly

### 1. Unbalanced OT was real

The 616 configs that launched the OT/vertical lane do use:

- `bridge.coupling_solver = "sinkhorn_unbalanced"`
- `bridge.training_target_projection_mode = "pure_vertical_flow"`

Relevant files:

- [phase616_ot_vertical_scratch_b8a2_e24.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_ot_vertical_scratch_b8a2_e24.json)
- [phase616_clean_ot_vertical_k085_b8a2_e24.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_ot_vertical_k085_b8a2_e24.json)

Implementation path:

- unbalanced Sinkhorn branch in [losses.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/losses.py:385)
- target projection branch in [losses.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/losses.py:699)

So the active 616 OT lane was not accidentally falling back to Hungarian or balanced Sinkhorn.

### 2. Pure vertical target projection was real

`pure_vertical_flow` is not just a config label. It really replaces the target low-frequency component with the source low-frequency anchor and keeps only target high-frequency fiber:

- [losses.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/losses.py:732)

The training bridge then uses `objective_target`, not the unprojected target:

- [losses.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/losses.py:2084)
- [losses.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/losses.py:2091)

So this part was aligned with the intended “remove horizontal contamination from the supervision target” idea.

### 3. The clean contract did keep old hacks off in the active 616 line

The current 616 configs already keep these off:

- `pre_integrate_moment_match = false`
- `output_moment_match = false`
- `output_appearance_alignment_mode = none`
- `proximal_mode = off`
- `style_delta_mode = none`
- full-eval RGB/latent postprocess off

Relevant guard:

- [style_families.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/style_families.py:372)

## What was implemented only as a weaker proxy

### 1. The retained `self_affinity_gw` OT descriptor was a raw-latent proxy, not the intended internal feature geometry

This was the largest mismatch.

The old retained control `self_affinity_gw` computes self-affinity from pooled raw latent components:

- lowpass
- edge map from lowpass
- high-frequency magnitude

Code:

- [losses.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/losses.py:524)

That is a useful structural proxy, but it is not the same as “let OT see the model’s internal semantic structure.”

We later added and probed the two more faithful versions:

- `encoder_self_affinity_gw`
- `tokenizer_aux_self_affinity_gw`

Code:

- [losses.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/losses.py:579)
- [losses.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/losses.py:595)
- [losses.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/losses.py:644)

Round-4 result:

- The proxy control was narrower than the theory target.
- But after replacing it with true internal feature maps, neither candidate promoted over the retained proxy.

So the diagnosis is:

- **yes, the earlier implementation was weaker than intended**
- **no, correcting it did not reveal a hidden win**

See:

- [clean_ot_probe_round4_featuremaps.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/616/clean_ot_probe_round4_featuremaps.md)

### 2. The base/fiber split is still a simple handcrafted split, not a learned or fully geometry-aware projector

Current implementation supports:

- avg-pool lowpass split
- wavelet-style 2x down/up split

Code:

- [losses.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/losses.py:462)
- [losses.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/losses.py:471)

This matches the 616 probe plan, but it is still a hand-designed proxy for the vertical subspace. It is not yet a principled learned projector or a real connection operator.

So this is **not wrong**, but it is still an approximation layer in the current phase.

## What remained outside the 616 clean path

### 1. Photometry / DC-stat alignment was not in the audited 616 clean runtime path

There are two existing stat-alignment mechanisms:

- pre-integrate moment match
- output moment match

Code:

- [model.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/model.py:1860)
- [model.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/model.py:1941)

And eval-only postprocess still exists separately:

- RGB affine in [run_evaluation.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/utils/run_evaluation.py:714)
- latent affine in [run_evaluation.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/utils/run_evaluation.py:799)

But `docs/616/design.md` asks for something more specific:

- a global-local decoupled transport
- analytical style-stat track
- neural texture/fiber track
- integrated into runtime, not eval-only postprocess

That mechanism was **not** implemented in the audited 616 training/eval runs.

So here the answer is:

- **yes, previous implementation did not match the 616 photometry design**
- but it was mostly because the new mechanism was not built yet, not because the old code was silently misbehaving

Status update on 2026-06-17:

- a clean runtime `transport_stats_mode` path has now been added with:
  - `none`
  - `terminal_affine`
  - `normalized_solver`
- the new path is wired through:
  - model runtime transport
  - training objective pair construction
  - trainer scalar observability
  - inference-side bank loading
- a new offline bank builder exists:
  - [build_phase616_style_stats_bank.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/build_phase616_style_stats_bank.py)

This means the earlier 616 evidence should still be read as “stats-track absent”, while the current branch now has the first clean implementation ready for controlled experiments.

### 2. The lowpass solver corrector still existed in code and old configs, even though 616 should treat it as retired

The old corrector remains implemented:

- [model.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/model.py:1377)

It is not active in the current clean 616 configs, but it was still possible to reintroduce by config drift. To close that hole, the clean-contract validator was tightened on 2026-06-17 to also require:

- `model.solver_corrector_mode = "none"`

Files updated:

- [style_families.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/style_families.py:372)
- [config_schema.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/config_schema.py:830)

## Net conclusion

The previous 616 implementation was not “running the wrong experiment” in a broad sense. The active lane did have:

- clean contract
- unbalanced OT
- projected vertical target

But two important caveats are now fixed in our understanding:

1. The promoted `self_affinity_gw` OT descriptor was a structural proxy, not the intended internal-feature geometry.
2. The photometry/stat-track part of the 616 theory still has not entered the clean runtime path; older moment/affine machinery should not be mistaken for that mechanism.
3. The 2026-06-17 stats-track CSV rows underreported `transport_stats_*`
   because of a writer bug. The authoritative evidence for those runs is
   `numeric_debug.jsonl` plus eval outputs, not the raw epoch CSV alone.

## Additional branch-level finding from the 2026-06-17 audit

While integrating the new stats-track, the current working branch briefly contained a real implementation error:

- `src/model.py` had an `IndentationError` caused by the new stats-track methods being inserted across the tail of `clear_runtime_caches()`
- this was a **current-branch code defect**, not evidence that earlier remote experiments had secretly run a different compiled artifact
- it is now fixed, and the affected paths compile again:
  - `model.py`
  - `trainer.py`
  - `utils/inference.py`
- `build_phase616_style_stats_bank.py`

## Additional line-by-line findings after the first audit pass

### 1. The feature-map OT audit was narrower than first claimed, but not for the reason initially suspected

The first re-read overcalled one issue. After checking the actual block math, the
encoder probe is more content-side than it first appeared.

Relevant paths:

- [losses.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/losses.py:585)
- [losses.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/losses.py:601)
- [lancet_blocks.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/lancet_blocks.py:125)

What is true:

- `_ot_encoder_feature_map()` runs the encoder-side residual blocks with
  `gate=0.0`.
- In `CrossAttnAdaGN.forward()`, `gate=0.0` returns the normalized input path,
  so the style-modulated branch is effectively bypassed.
- That means the encoder feature-map OT probe is already close to a content-only
  structure descriptor under the current code.

What was still ambiguous before cleanup:

- the OT probe helpers still *constructed* style code tensors even though the
  encoder probe did not need them
- the tokenizer-side probe was routed indirectly through the runtime sidecar
  machinery, which made it harder to see whether we were really auditing
  content-side routing or style-conditioned maps

What is true for the tokenizer-side probe:

- for `pure_latent_spatial` and `smoe_translator`, `aux_16` is the tokenizer
  routing attention map
- that routing map comes from content queries against universal keys, while the
  style-specific values affect `spatial_map` / `global_code` rather than the
  routing attention itself

So the corrected interpretation is:

- the round-4 feature-map OT audit was **not** secretly measuring heavily
  target-style-conditioned encoder geometry
- but the code path was still too implicit and easy to misread
- the cleanup now makes the content-side intent explicit by using a neutral OT
  style code and calling the latent-native tokenizer directly for routing maps

## Additional implementation mismatches verified on 2026-06-17 afternoon

These findings matter because they can make a mechanism look weaker or noisier
than the 616 theory intended, even when the run is otherwise "real."

### 1. `pure_vertical_flow` had been applied to the training target, but not to bridge noise by default

The active clean OT configs did enable:

- `training_target_projection_mode = "pure_vertical_flow"`

But the bridge-noise path still defaulted to:

- `training_bridge_noise_projection_mode = "none"`

Relevant files:

- [config_schema.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/config_schema.py:479)
- [losses.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/losses.py:1192)
- [phase616_clean_ot_probe_lowedge_b16a1_vlen010_e6.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_ot_probe_lowedge_b16a1_vlen010_e6.json:11)

Interpretation:

- the earlier 616 OT line was a **vertical-target** implementation
- it was **not yet a full vertical-bridge implementation**
- this is a meaningful source of structural contamination because the sampled
  bridge state can still receive full-frequency Brownian noise even after the
  supervision target itself has been vertically projected

So if a run looked structurally noisy, that does not automatically falsify the
vertical-geometry idea. It may simply mean the bridge-noise half of the design
was still off.

### 2. OT target geometry was still stochastic by default because `coupling_target_mode = "sample"`

The bridge config default remained:

- `coupling_target_mode = "sample"`

and the target selection branch in training used row-wise multinomial sampling.

Relevant files:

- [config_schema.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/config_schema.py:456)
- [losses.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/losses.py:1251)

This means the 616 OT probes were not using a deterministic barycentric target
unless a config explicitly overrode the default.

Interpretation:

- this is **not a code bug**
- but it is a mismatch with the stated 616 goal of first making the OT
  supervision as clean and interpretable as possible
- fast 60-step probes therefore contained an avoidable amount of supervision
  randomness on top of the intended mechanism change

So when reading earlier OT probe variance, some of that variance should be
attributed to sampled target geometry rather than to the OT descriptor itself.

### 3. The retained OT control was still a mixed appearance+structure objective, not a pure structure OT repair

The clean OT control family inherited the default mixed composition
`appearance_plus_structure`, and some later probe configs kept that composition
explicitly to preserve continuity with the retained line.

Relevant files:

- [losses.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/losses.py:991)
- [phase616_clean_ot_probe_selfaffgw_mix_faststep60_e1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_ot_probe_selfaffgw_mix_faststep60_e1.json:4)
- [phase616_clean_ot_probe_lowedge_b16a1_vlen010_e6.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_ot_probe_lowedge_b16a1_vlen010_e6.json:3)

Interpretation:

- the earlier 616 OT results are best understood as evidence about a
  **mixed OT repair line**
- they are **not yet pure evidence** for the stronger metric-mismatch claim
  from `docs/616/design.md`, because pointwise appearance cost was still part
  of the coupling objective

This does not make those runs invalid. It does mean they should not be used as
the final argument for or against structure-only OT.

### 4. One real observability bug was present: `velocity_abs` reported the teacher target, not the model prediction

In the sampled bridge path, the scalar previously written as `velocity_abs`
actually tracked `target_velocity.abs().mean()` instead of
`pred_velocity.abs().mean()`.

Relevant file:

- [losses.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/losses.py:2441)

This has now been corrected, and a separate `target_velocity_abs` scalar has
been added to the training log path:

- [losses.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/losses.py:2441)
- [trainer.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/trainer.py:1397)
- [utils/training.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/utils/training.py:148)

Interpretation:

- old `velocity_abs` curves in sampled-bridge 616 runs should be treated as
  mislabeled
- they still carry some magnitude information, but they are not valid evidence
  about the model's own predicted field strength

## Resulting operational decision

Because of the four points above, the next OT diagnostic step should not be a
new descriptor family by default. It should be a cleaner matched 2x2 probe
that isolates:

1. `appearance_plus_structure` vs `structure_only`
2. `sample` vs deterministic barycentric target geometry

Only after that cleaner table exists should we read later OT gains or failures
as evidence about the 616 theory itself.

### 2. The `phase616` config family is materially clean; earlier regressions were not caused by old losses silently leaking back in

The second-pass config audit checked the active 616 bases and the fast matched
probes against the enforced clean contract:

- [phase616_cleanbase_i2sb_k085_b8a2_e24.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_cleanbase_i2sb_k085_b8a2_e24.json)
- [phase616_clean_ot_probe_lowedge_b16a1_vlen010_e6.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_ot_probe_lowedge_b16a1_vlen010_e6.json)
- [style_families.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/style_families.py:372)

What is explicitly held off in the 616 base:

- `style_delta_mode = "none"`
- `proximal_mode = "off"`
- `output_appearance_alignment_mode = "none"`
- `solver_corrector_mode = "none"`
- `pre_integrate_moment_match = false`
- `output_moment_match = false`
- `w_content_lowpass_anchor = 0.0`
- `w_content_edge_anchor = 0.0`
- `cycle_consistency_weight = 0.0`
- eval-side RGB / latent postprocess both off

That means the weak 616 outcomes should be interpreted as one of:

- the mechanism under test was only a proxy form of the theory
- observability was incomplete or partially broken
- the mechanism was a genuine negative result

and **not** as "the active 616 probe was accidentally dominated by old
content-anchor / cycle / appearance-alignment machinery."

### 2. `terminal_affine` is a terminal remap, not a training-pair transport rewrite

The new stats-track family has two very different semantics:

- `normalized_solver`
- `terminal_affine`

Only `normalized_solver` changes the actual training pair seen by the bridge:

- [model.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/model.py:653)

`terminal_affine` does **not** rewrite `content/target` before the bridge loss.
It leaves training pairs unchanged and only applies the bank-driven affine
restore at the transport output:

- [model.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/model.py:635)
- [model.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/model.py:2158)

So if we read the first stats probe as "a fully integrated global-local stats
transport mechanism", that reading would be wrong. The correct reading is:

- `terminal_affine` = output-side photometry remap with clean runtime wiring
- `normalized_solver` = actual normalized transport track

The positive signal from `terminal_affine` is still useful, but it should be
classified as a lighter intervention than the full 616 stats-track theory.

### 3. The first stats probe round had a real observability gap

The first matched stats probe did compute transport-stats scalars in the model
and trainer path, but the CSV schema was still missing the
`transport_stats_*` columns at write time:

- [utils/training.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/utils/training.py:15)

That means the round was valid for transfer-level evidence, but not yet a fully
authoritative white-box closure for the stats-track mechanism itself.

This has now been corrected by adding the full `transport_stats_*` field family
to `TRAIN_LOG_COLUMNS`.

## Decision after audit

### Retained as valid evidence

- `sinkhorn_unbalanced`
- `pure_vertical_flow`
- wavelet-vs-avg split probes
- round-4 feature-map OT audit results

### Reclassified

- raw-latent `self_affinity_gw`:
  retained as the current best OT control, but now explicitly labeled a **proxy descriptor**

### Still missing relative to 616 design

- runtime analytical stats transport for photometry
- clean style-stat bank feeding runtime transport
- stronger global/local decoupled integration path

## Immediate follow-up

Before launching the next 616 training family, compare only mechanisms that are actually aligned with this audit:

1. OT: keep `self_affinity_gw` as control unless a new internal-feature OT candidate beats it on matched delta.
2. Vertical geometry: continue the split-operator probe with wavelet control, because that is a real unresolved approximation.
3. Photometry: do not rely on eval affine or moment-match hacks as if they were the 616 answer; implement the runtime stats-track explicitly.

## Addendum after round-5 tokenizer-entropy OT closure

Round-5 directly tested the strongest remaining suspicion about the earlier OT
implementation:

- maybe the retained proxy looked best only because we had never let OT see the
  tokenizer's routing complexity and topology explicitly

That suspicion is now weaker.

See:

- [clean_ot_probe_round5_tokenentropy.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/616/clean_ot_probe_round5_tokenentropy.md)

Matched result summary:

- tokenizer-native entropy+affinity slightly improved hubness and transfer LPIPS
- but it slightly regressed transfer style
- it also worsened train-side structural-drift / leakage probes
- and it was materially slower

So the refined answer to “was the previous implementation wrong?” is now:

- **yes, partly**: the older retained OT surface was a proxy and therefore not a
  complete realization of the 616 OT theory
- **but not in the decisive way we hoped**: replacing that proxy with a closer
  tokenizer-native implementation still did not produce a promotable lane

That means the style ceiling in the current 616 branch should no longer be read
mainly as “the OT implementation was wrong.” It is more likely dominated by the
next unresolved approximation layers:

- handcrafted base/fiber split
- incomplete stats-track integration
- solver / stochastic geometry still not carrying enough style actuation
