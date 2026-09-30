# Current Mainline Read

Date: 2026-06-09

This note records the current strongest evidence-based interpretation before the next fresh packet closes.

## Stable conclusions

1. `Seedream-4.5` is still the current external style ceiling.

2. `LBM-Knee e13` is still the strongest internal balanced point.
   - It nearly matches `LBM-K e1` on full750 `IntroStyle`
   - but keeps a cleaner `DINO` structure score
   - and avoids the generic painterly-fog failure of `LBM-PS-v2`

3. `LBM-PS-v2 e13` is now downgraded.
   - full750 `IntroStyle` is weaker than `LBM-Knee`
   - full750 `DINO` is worse than `LBM-Knee`
   - local VLM interim also rates it far below both `Seedream` and the internal balanced line

4. `Hold4Mid`-anchored reopen lines are not paying off enough.
   - the clean bodydecoder rerun stayed weak
   - `Hold4Mid + CarrierGate` local review also stayed below `LBM-Knee`

## Live bet

The current best open question is no longer:

- `Can we reopen style from the strongest geometry anchor?`

It is now:

- `Can we reopen style from the strongest current internal balanced point?`

Therefore the current remote mainline should be:

- `Knee e13 + Spatial CarrierGate Body+Decoder + Quantile Edge-Gated Structure Leash`

not:

- more plain `CarrierGate Injection`
- more spatial-carrier micro-variants on the same `Knee` line
- more `Trust`-style soft leash continuations
- more `LBM-PS-v2`-like style-heavy fog lines

## Closed read on the plain carrier line

- fresh `epoch_0002` local review:
  - `IntroStyle target = 0.1092`
  - `IntroStyle delta-IDT = -0.0393`
  - `DINO = 0.0269`
- fresh `epoch_0012` local review:
  - `IntroStyle target = 0.1074`
  - `IntroStyle delta-IDT = -0.0419`
  - `DINO = 0.0268`

Interpretation:

- absolute style can rise slightly above `LBM-Knee`
- but directional style evidence stays negative
- and structure is already worse than `LBM-Knee`
- this closes plain `Knee + carrier_gate` as not good enough

## Current live packets

1. closed-to-local-review:
   - `aaai2027_inmortal_knee_e13_spatial_carriergate_bodydecoder_seed42_b8a2`
   - note:
     - [2026-06-09-knee-e13-spatial-carriergate-bodydecoder.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-knee-e13-spatial-carriergate-bodydecoder.md)
   - state:
     - train complete
     - `fresh_localreview` complete through `epoch_0012`
     - local best-few `IntroStyle + DINO` review already closed it as `near-negative / do not promote`

2. current remote active lane:
   - primary training-side line:
     - `aaai2027_inmortal_knee_e13_spatial_carriergate_bodydecoder_qedgegated_dualpath_seed42_b8a2`
   - note:
     - [2026-06-09-knee-e13-spatial-carriergate-bodydecoder-qedgegated-dualpath.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-knee-e13-spatial-carriergate-bodydecoder-qedgegated-dualpath.md)
   - intent:
     - keep the stronger structure leash
     - replace the single late style branch with a higher-capacity dual-path branch
     - explicitly split coarse style recovery from high-frequency texture recovery
   - state:
      - remote training is live
      - first-health passed
      - current first-health GPU memory is `8133 MiB`
      - this is below the formal `< 11.0 GiB` cap
      - matching post-train watcher is now also alive
      - current live combined GPU memory is still only about `8142 MiB`
      - retained checkpoints have already started landing:
        - `epoch_0001`
        - `epoch_0002`
        - `epoch_0003`
        - `epoch_0004`
        - `epoch_0005`
        - `epoch_0006`
        - `epoch_0007`
        - `epoch_0008`
        - `epoch_0009`
        - `epoch_0010`
        - `epoch_0011`
        - `epoch_0012`
      - latest checked live progress is already into:
        - `epoch 12`
      - latest early training read:
        - `loss 8.4354 -> 8.0421`
        - `flow 0.8279 -> 0.7959`
        - `terminal_swd` reached `5.8750` at `epoch 5`, rebounded to `6.3438` at `epoch 6`, eased back down to `5.9688` by `epoch 9`, reopened upward to `6.1875` at `epoch 10`, softened slightly to `6.1563` at `epoch 11`, and improved further to `5.7188` at `epoch 12`
      - this is now the true active branch-capacity round
      - deferred `full_eval` has now started for this family
      - current visible evaluator process is already working on:
        - `full_eval/epoch_0001`
      - first completed eval-side points are now available locally:
        - `epoch_0001 = 0.6927 / 0.4037`
        - `epoch_0002 = 0.6919 / 0.4218`
        - `epoch_0003 = 0.6918 / 0.4306`
        - `epoch_0004 = 0.6921 / 0.4348`
        - `epoch_0005 = 0.6917 / 0.4348`
        - `epoch_0006 = 0.6922 / 0.4377`
      - relative to `QEdgePattn epoch_0001`, the new line currently reads:
        - lower on cheap style
        - better on `LPIPS`
      - the first few eval-side points are now internally consistent:
        - they keep the same more geometry-conservative shape
        - but they have not yet shown stronger target-style recovery
      - the packet has now reached the end of its planned `12 epoch` budget
      - `full_eval` directory has now appeared for this family, indicating the run is transitioning from pure training into deferred eval
      - first image-backed `full_eval_fresh_localreview` packet is now locally mirrored for:
        - `epoch_0001`
      - first local non-CLIP read is now available for that point:
        - `IntroStyle target = 0.1076`
        - `IntroStyle margin = -0.0509`
        - `DINO = 0.0263`
      - current interpretation:
        - still not promotable on the first image-backed point
        - the line still looks like a conservative operating point rather than a real style-ceiling rescue

3. previous remote line now in local-review closure:
   - primary training-side line:
     - `aaai2027_inmortal_knee_e13_spatial_carriergate_bodydecoder_qedgegated_pattn_seed42_b8a2`
   - note:
     - [2026-06-09-knee-e13-spatial-carriergate-bodydecoder-qedgegated-pattn.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-knee-e13-spatial-carriergate-bodydecoder-qedgegated-pattn.md)
   - intent:
     - keep the more selective qedgegated structure leash
     - but stop forcing the carrier branch itself to do all the style work
     - hand more style-specific high-frequency burden to the explicit `crossattn_texture` proximal head
   - state:
      - remote training is live
      - first-health passed
      - the new line has started cleanly after the old qedgegated line was stopped
      - the fresh-eval watcher is armed for the packet
      - retained checkpoints have now landed through `epoch_0012`
      - current runtime read from the latest local status poll:
        - `run.py` has exited
        - watcher alive
        - training log currently records `epochs 1..12`
        - per-epoch wall time is about `145s`
        - current remote WSL-side checkpoint band is low-VRAM safe
        - first image-backed `full_eval_fresh_localreview` artifacts have now started landing
        - the packet has now reached the end of its planned `12 epoch` budget
        - visible in-process `full_eval` artifacts have already landed through `epoch_0012`
        - current early in-process mean trend is:
          - `epoch_0001: 0.7259 / 0.4477`
          - `epoch_0004: 0.7241 / 0.4835`
          - `epoch_0010: 0.7249 / 0.5024` for all-pairs
          - `epoch_0010: 0.7105 / 0.5068` for transfer
      - one early watcher attempt at `11:51 +08:00` misfired before the train process was visible and failed during fresh-epoch inference
      - the valid watcher restart at `11:56 +08:00` is now polling correctly
      - watcher robustness patch has now landed locally and the remote stale watcher instance was replaced with a single refreshed watcher process
      - because `fresh_localreview` still lagged after train exit, a manual image-backed closure was launched on the same remote surface
      - first visible image-backed epoch dir:
        - `full_eval_fresh_localreview/epoch_0001`
      - earliest local image-backed best-few packet is now already available:
        - `epoch_0001`
        - `epoch_0003`
        - `epoch_0007`
      - image-backed `full_eval_fresh_localreview` has now effectively landed through `epoch_0012`
      - local CPU-only `VLM` for the active line is now genuinely accumulating:
        - `QEdgePattn_e01` and `QEdgePattn_e03` have both moved into multi-case bands
        - `QEdgePattn_e01` has now started to pick up sparse overall wins
        - `QEdgePattn_e03` has now also picked up its first sparse overall win
        - current local VLM still places the active-line points above `LBM-Knee` but below `Seedream`
        - within the active family, `e01` currently looks more alive than `e03`
        - newer local evidence now sharpens that split:
          - `e01` has fewer but denser wins
          - `e03` has broader coverage but weaker win density
      - `full_eval_fresh_localreview` has now materially progressed through the run:
        - image-backed summaries are already visible through `epoch_0012`
      - first image-backed local non-CLIP read is now available for `DualPathFresh_e01`
      - current first `VLM` batch read:
        - `140+` completed cases
        - sparse overall/style/structure/artifact wins over `LBM-Knee`
        - still clearly below `Seedream`
      - the higher-style image-backed point is now also in the same local `VLM` path:
        - `DualPathFresh_e09`
      - current dualpath family-level read is:
        - `e01` has the broader evidence and stronger structure/artifact breadth
        - `e09` has slightly stronger style-win density so far, but on a still much smaller batch
      - current interpretation:
        - the first image-backed local read is stronger than `Knee`
        - but still does not rescue the family into promoted territory
        - specificity is wrong and structure is still too weak
      - later image-backed best-few closure is now also available:
        - `epoch_0011`
        - `epoch_0012`
      - closure read:
        - `IntroStyle` margins remain negative:
          - about `-0.0354`
          - and `-0.0340`
        - `DINO` worsens further to about:
          - `0.0330`
        - so the family is now closed as:
          - `not promotable`
          - `negative for mainline promotion`

4. previous remote line now judged by early eval trend:
   - `aaai2027_inmortal_knee_e13_spatial_carriergate_bodydecoder_qedgegated_seed42_b8a2`
   - note:
     - [2026-06-09-knee-e13-spatial-carriergate-bodydecoder-qedgegated.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-knee-e13-spatial-carriergate-bodydecoder-qedgegated.md)
   - state:
     - first retained `full_eval` summaries landed through `epoch_0004`
     - early signal:
       - `epoch_0001 = 0.7048 / 0.4521`
       - `epoch_0004 = 0.7074 / 0.4878`
       - `epoch_0010 = 0.7106 / 0.5068`
     - read:
       - `style-up`
       - `LPIPS-worse`
       - enough to justify moving to the prepared qedgegated+pattn follow-up

5. previous remote line now judged locally:
   - `aaai2027_inmortal_knee_e13_spatial_carriergate_bodydecoder_edgegated_seed42_b8a2`
   - note:
     - [2026-06-09-knee-e13-spatial-carriergate-bodydecoder-edgegated.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-knee-e13-spatial-carriergate-bodydecoder-edgegated.md)
     - [2026-06-09-edgegated-bestfew-localreview.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-edgegated-bestfew-localreview.md)
      - [2026-06-09-edgegated-vlm-interim.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-edgegated-vlm-interim.md)
   - state:
     - training has completed
     - post-train fresh-eval watcher has produced the image-backed `fresh_localreview` packet
     - local best-few non-CLIP review is now available
     - current read:
      - `theory-positive`
      - still `not promotable`
      - current CPU-only VLM interim on `EdgeGated_e03 vs Knee vs Seedream` is:
        - `Seedream 292 / 309`
        - `EdgeGated_e03 17 / 309`
        - `EdgeGated_e03` also has `48` structure subwins and `61` artifact-control subwins
      - companion CPU-only VLM triplets on `EdgeGated_e01` and `EdgeGated_e12` are now also live and writing first cases

6. concurrent remote packetization task:
   - `aaai2027_inmortal_xpred_kmanifold_pattn_anisostokes_queue_clamphold4twostage_reseed_from_e13_seed42_b8a2`
   - active remote task:
     - `Hold4TwoStage best-few image rerun` was used to prepare the local review packet
   - note:
      - [2026-06-08-inmortal-anisostokes-queue-clamphold4twostage-from-e13.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-08-inmortal-anisostokes-queue-clamphold4twostage-from-e13.md)
      - [2026-06-09-hold4twostage-vlm-interim.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-hold4twostage-vlm-interim.md)
      - [2026-06-09-hold4twostage-bestfew-localreview.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-hold4twostage-bestfew-localreview.md)
   - stage signal:
      - `Hold4TwoStage_e12`: `Seedream 565 / 585`
      - `Hold4TwoStage_e02`: `Seedream 517 / 532`
      - `Hold4TwoStage_e20`: `Seedream 500 / 519`
   - state:
     - train and `full_eval` already exist
     - local best-few `IntroStyle + DINO` review now closes the family as negative
     - local CPU-only VLM interim on `Hold4TwoStage_e12 vs Knee vs Seedream` is also negative-leaning:
       - [2026-06-09-hold4twostage-vlm-interim.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-hold4twostage-vlm-interim.md)

## Current heavy-review state

- local full750 `IntroStyle + DINO + VLM` remains the paper-facing interpretation stack
- remote is now only responsible for handing over:
  - retained point images
  - eval CSVs
  - compact best-few handoff rows
- current best-few handoff file for the most recently closed image-backed family:
  - [knee_spatial_carriergate_bodydecoder_edgegated_bestfew_handoff_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/knee_spatial_carriergate_bodydecoder_edgegated_bestfew_handoff_20260609.csv)
- current active-line image-backed local handoff root:
  - [qedgegated_pattn_bestfew_localreview_20260609](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/qedgegated_pattn_bestfew_localreview_20260609)
- local VLM review is still running
- current method summary at the latest checked point:
  - `Seedream_repaired750`: `709 / 713` wins
  - `LBM-Knee_e13`: `0 / 713` wins
  - `LBM-PS-v2_e13`: `3 / 713` wins

## Current implication

- the plain `Knee + carrier_gate` family is not enough
- the only reason the family stays alive is the stronger spatial `body+decoder` variant now running
- the current full `epoch_0001..0012` fresh-localreview curve is still flat rather than clearly improving
- best LPIPS only occurs at:
  - `epoch_0003 = 0.4387`
- best CLIP-style only occurs at:
  - `epoch_0008 = 0.7038`
- local best-few review now gives the decisive read:
  - `IntroStyle target` rises above `LBM-Knee`
  - but `DINO` degrades from `0.0217` to about `0.0283`
  - and `IntroStyle` specificity margin also weakens
- so this is a `near-negative` family, not a promoted win
- the edge-gated follow-up now gives the next decisive read:
  - `IntroStyle target` remains only around `0.1085 to 0.1088`
  - `DINO` improves only slightly versus the plain spatial-carrier line:
    - from about `0.0284` to about `0.0281`
  - but it still sits far to the worse side of `LBM-Knee` on `DINO`
  - and it gives back part of the style gain
- so the edge-gated line is:
  - `theory-positive`
  - but still `not promotable`
- the qedgegated line now gives a stronger warning:
  - `epoch_0001 = 0.7048 / 0.4521`
  - `epoch_0004 = 0.7074 / 0.4878`
  - so the current trajectory is already:
    - `style-up`
    - `LPIPS-worse`
- the `Hold4TwoStage` family is now also negative-leaning on local best-few review:
  - weaker than `LBM-Knee` on `IntroStyle`
  - and much worse on `DINO`
- early VLM triplet evidence is now consistent with that same conclusion:
  - even after the larger local CPU-only completed set, `Seedream` still dominates decisively
  - and `Hold4TwoStage` still does not emerge as a hidden win over `LBM-Knee`
- the next meaningful improvement likely requires:
  - a stronger target-specific spatial branch
  - or another mechanism family beyond plain late carrier injection
  - more specifically:
    - a more selective structure leash than the current edge-gated pressure
    - or a late style-recovery head that does not make the carrier branch do all the work
- current next-step recommendation note:
  - [2026-06-09-next-mechanism-after-qedgepattn.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-next-mechanism-after-qedgepattn.md)
  - the first concrete branch-capacity candidate is already implemented:
    - `proximal_mode = dualpath_texture`
- that next explicit training-side candidate is now live:
  - [2026-06-09-knee-e13-spatial-carriergate-bodydecoder-qedgegated.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-knee-e13-spatial-carriergate-bodydecoder-qedgegated.md)
- if qedgegated continues the same `style-up / LPIPS-worse` trajectory, the next prepared follow-up is:
  - [2026-06-09-knee-e13-spatial-carriergate-bodydecoder-qedgegated-pattn.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-knee-e13-spatial-carriergate-bodydecoder-qedgegated-pattn.md)
  - this follow-up is now the true active remote mainline
  - it has already progressed through `epoch_0012`
  - image-backed `fresh_localreview` closure has now materially landed
  - later image-backed points did not reverse the same style-vs-LPIPS tradeoff

Current decision board:

- [2026-06-09-stage-compare-current.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-stage-compare-current.md)

## Local review burden for the live line

The packet has now effectively been judged by:

1. full750 `IntroStyle`
2. full750 `DINO`
3. local `Qwen xopqwen36v35b` panel review

The packet only stays alive if it can:

- beat `LBM-Knee` on `IntroStyle`
- without moving to the worse side of `LBM-Knee` on `DINO`
- and without looking like a weaker version of `Seedream` under the VLM review
