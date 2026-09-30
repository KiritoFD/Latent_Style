# Next Mechanism After QEdgePattn

Date: 2026-06-09

This note records the current mechanism recommendation after the accumulated read on:

- `Hold4TwoStage`
- `EdgeGated`
- `QEdgeGated + CrossAttnTexture`

It is not the final paper claim.
It is the current execution recommendation for the next mechanism family.

## Current evidence base

Used evidence:

- local best-few `IntroStyle + DINO`
- training-side `full_eval` curves
- image-backed `full_eval_fresh_localreview`
- local CPU-only `VLM`

Most important current reads:

1. `Hold4TwoStage` is now closed negative.
2. `EdgeGated` is theory-positive but not promotable.
3. `QEdgePattn` is better than `LBM-Knee`, but still clearly below `Seedream`.

## What QEdgePattn did prove

`QEdgePattn` is not a null result.

It proved that:

- stronger structure leashing is useful
- explicit late texture injection is better than asking the carrier path to do everything
- the family can produce real local wins over `LBM-Knee`

This matters because it means the current direction is not random.

## What QEdgePattn failed to prove

`QEdgePattn` still failed to become a promoted family.

Why:

- image-backed best-few still follows:
  - higher style
  - worse `LPIPS`
- local CPU-only `VLM` still places the line clearly below `Seedream`
- top-vote wins remain sparse even after the local review batch is no longer tiny

So the current failure is no longer well explained by:

- not enough epochs
- not enough image-backed closure
- not enough schedule smoothing

The family has now had enough exposure to reject those softer explanations.

## Narrowed bottleneck

The current bottleneck is best described as:

- insufficient target-specific style recovery capacity after structure has been stabilized

Not:

- lack of structure control
- lack of optimization stability
- lack of post-train closure

In short:

- we already know how to keep the model from fully collapsing
- we do not yet know how to recover enough target-specific style energy without paying too much structure cost

## Recommended next family

The next family should therefore be a stronger late style-recovery family.

The design requirement is:

- do not ask the same transport/carrier path to both:
  - preserve geometry
  - inject high-style target-specific texture

The next mechanism should instead increase the capacity of the late branch itself.

## Concrete mechanism rule

The next family should satisfy all of:

1. Keep a strong structure leash.
2. Keep transport-side geometry discipline.
3. Increase the expressivity of the late style branch.
4. Make the late branch more target-specific, not just more energetic.

## What not to do next

Do not spend the next round on:

- more hold/release schedule variants
- more plain carrier-scale tweaks
- more tiny qedge threshold nudges
- more “same mechanism but slightly different loss weight” work

Those directions now have diminishing information value.

## What to do next

Move to one of these stronger late style-recovery variants:

1. A stronger explicit target-style residual branch.
2. A more expressive late cross-attention texture head.
3. A multi-stage style branch where:
   - one sub-branch handles coarse target-style palette/statistics
   - another sub-branch handles localized high-frequency texture

The key is that the branch must be:

- later
- more target-specific
- more expressive than the current `QEdgePattn` head

Current implementation landing:

- a concrete branch-capacity candidate is now implemented as:
  - `proximal_mode = dualpath_texture`
- code path:
  - [model.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/model.py)
  - [config_schema.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/config_schema.py)
- current dual-path behavior:
  - a low-frequency coarse style branch from `NormFreeModulation + conv head`
  - a high-frequency cross-attention texture branch
  - additive late residual:
    - `lowpass(coarse) * proximal_coarse_gain`
    - plus `highpass(texture) * proximal_texture_gain`
- ready-to-run config:
  - [inmortal_knee_e13_spatial_carriergate_bodydecoder_qedgegated_dualpath_seed42_b8a2.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_knee_e13_spatial_carriergate_bodydecoder_qedgegated_dualpath_seed42_b8a2.json)
- ready-to-run launcher:
  - [launch_remote_knee_spatial_carriergate_bodydecoder_qedgegated_dualpath_train.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_knee_spatial_carriergate_bodydecoder_qedgegated_dualpath_train.py)
- ready-to-run best-few pull/local-review path:
  - [pull_remote_dualpath_bestfew_localreview.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/pull_remote_dualpath_bestfew_localreview.py)
  - [run_local_dualpath_bestfew_review.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_local_dualpath_bestfew_review.py)
- next stronger follow-up candidate now also exists:
  - `proximal_mode = dualpath_spatialtexture`
  - config:
    - [inmortal_knee_e13_spatial_carriergate_bodydecoder_qedgegated_dualpath_spatial_seed42_b8a2.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_knee_e13_spatial_carriergate_bodydecoder_qedgegated_dualpath_spatial_seed42_b8a2.json)
  - launcher:
    - [launch_remote_knee_spatial_carriergate_bodydecoder_qedgegated_dualpath_spatial_train.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_knee_spatial_carriergate_bodydecoder_qedgegated_dualpath_spatial_train.py)

Current eval-side read of the branch-capacity rounds:

- earlier dual-path family comparison:
  - [dualpath_vs_qedgepattn_early_curve_20260609.md](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/dualpath_vs_qedgepattn_early_curve_20260609.md)
- current spatialtexture fresh curve:
  - [dualpath_spatial_fresh_curve_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/dualpath_spatial_fresh_curve_20260609.csv)
- current early-read note:
  - [2026-06-09-dualpath-spatialtexture-early-read.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-dualpath-spatialtexture-early-read.md)

Current interpretation of that comparison:

- `dualpath_texture` had already shown:
  - lower style than `QEdgePattn`
  - but much lower `LPIPS`
- `dualpath_spatialtexture` has now extended that same picture into a real early curve:
  - transfer style stays in a very narrow band around `0.6916 to 0.6929`
  - `LPIPS` rises from about `0.401` toward `0.440`
  - all-pairs `CLIP-style` stays in a similarly narrow band around `0.715 to 0.718`

So the current unresolved question is no longer:

- `can a stronger branch avoid the old LPIPS blow-up`

It is now:

- `can a stronger branch escape this conservative low-style basin at all`

And the current evidence says:

- not yet

That does not fully close the family, because later paper-facing non-CLIP review can still refine the read.

But it does raise the bar for the next move:

- a new branch family must show more than just:
  - `stable`
  - `clean`
  - `low LPIPS`

It has to actually reopen target-specific style energy.

## Newly selected follow-up after the spatialtexture read

Current next execution branch:

- `dualpath_spatialtexture + Sinkhorn proximal routing`

Reason:

- the current failure mode is no longer best explained by raw branch capacity
- the late branch already has:
  - coarse style path
  - spatial coarse prior
  - high-frequency texture path
- yet the branch still settles into a conservative basin

So the next mechanism target is now:

- not more capacity alone
- but more selective target-style routing inside the proximal branch itself

The concrete hypothesis is:

- current softmax proximal attention is too diffuse
- it lets target-style evidence average out across many latent locations
- that encourages the same washed-out, generic painterly basin seen in visual review

The new mechanism therefore replaces the proximal attention routing with a
near doubly-stochastic routing rule:

- `proximal_attn_routing_mode = sinkhorn`

Expected benefit:

- less diffuse texture routing
- stronger localized target-style assignment
- better chance of reopening style specificity without giving up the existing
  structure leash

Concrete landed packet:

- config:
  - [inmortal_knee_e13_spatial_carriergate_bodydecoder_qedgegated_dualpath_spatial_sinkhorn_seed42_b8a2.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_knee_e13_spatial_carriergate_bodydecoder_qedgegated_dualpath_spatial_sinkhorn_seed42_b8a2.json)
- launcher:
  - [launch_remote_knee_spatial_carriergate_bodydecoder_qedgegated_dualpath_spatial_sinkhorn_train.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_knee_spatial_carriergate_bodydecoder_qedgegated_dualpath_spatial_sinkhorn_train.py)

## Post-sinkhorn next move

Current sinkhorn read:

- the first closed sweep is still basically a near-tie with predecessor
- the first full local non-CLIP point (`epoch_0001`) is near-negative
- later local image-backed point (`epoch_0009`) is not rescuing structure so far

So the next branch should not keep pushing routing alone.

Current next execution branch:

- `qedge + pattn + target-specific style-signature losses`

Reason:

- routing-only changes did not materially reopen target-specific style
- the next move should add explicit style-side pressure, not only structural or
  routing changes
- the current codebase already has a coherent bundle for that:
  - style contrastive alignment
  - residual style-direction alignment
  - spectral amplitude matching
  - style-energy floor

This is still a mechanism move rather than a micro-tuning move because it
changes what signal the branch is asked to satisfy, not just how much of the
same signal it sees.

Concrete landed packet:

- config:
  - [inmortal_knee_e13_spatial_carriergate_bodydecoder_qedgegated_pattn_stylesig_seed42_b8a2.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_knee_e13_spatial_carriergate_bodydecoder_qedgegated_pattn_stylesig_seed42_b8a2.json)
- launcher:
  - [launch_remote_knee_spatial_carriergate_bodydecoder_qedgegated_pattn_stylesig_train.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_knee_spatial_carriergate_bodydecoder_qedgegated_pattn_stylesig_train.py)

## Current operational conclusion

Use the current evidence as:

- `QEdgePattn` is a valid positive-over-`Knee` family
- but it is also a valid negative result against the claim that the current late branch is already strong enough
- `dualpath_texture` and `dualpath_spatialtexture` together now suggest that simply increasing late-branch capacity is still not sufficient by itself
- the next round should still remain a branch-family round rather than a schedule round
- but the branch now likely needs to be:
  - more explicitly target-specific
  - not just broader or cleaner
