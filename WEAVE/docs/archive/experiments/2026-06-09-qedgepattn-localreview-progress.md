# QEdgePattn Local Review Progress

Date: 2026-06-09

This note records the current local-review state for:

- `aaai2027_inmortal_knee_e13_spatial_carriergate_bodydecoder_qedgegated_pattn_seed42_b8a2`

It is specifically about the image-backed local review path, not about the raw training-side curve.

## Current remote closure state

Remote training is already complete.

Current remote closure stage:

- training-side `full_eval` landed through `epoch_0012`
- manual image-backed `full_eval_fresh_localreview` has now effectively completed
- visible image-backed epochs already present remotely:
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

Final image-backed curve read:

- best transfer `LPIPS`
  - `epoch_0001 = 0.7047 / 0.4519`
- best transfer `CLIP-style`
  - `epoch_0011 = 0.7106 / 0.5073`

The later image-backed epochs therefore do not overturn the original tradeoff.

They continue the same shape:

- style rises slightly
- `LPIPS` keeps worsening

## Current local image-backed handoff

Local handoff root:

- [qedgegated_pattn_bestfew_localreview_20260609](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/qedgegated_pattn_bestfew_localreview_20260609)

Current local image-backed best-few pair already pulled:

- `epoch_0001`
- `epoch_0003`
- `epoch_0007`

Current handoff CSV:

- [full_eval_fresh_localreview_bestfew_handoff.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/qedgegated_pattn_bestfew_localreview_20260609/full_eval_fresh_localreview_bestfew_handoff.csv)

Current read from that handoff:

- best transfer `LPIPS`
  - `epoch_0001 = 0.7047 / 0.4519`
- best transfer `CLIP-style`
  - `epoch_0007 = 0.7102 / 0.5023`

So the image-backed best-few has shifted to a later higher-style point, but it still has not overturned the same early tradeoff pattern:

- style rises
- LPIPS remains substantially worse

## Current non-CLIP local review preparation

Prepared local IntroStyle manifest:

- [full_eval_fresh_localreview_bestfew_introstyle_manifest.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/qedgegated_pattn_bestfew_localreview_20260609/full_eval_fresh_localreview_bestfew_introstyle_manifest.csv)

Current evidence caveat:

- the best-few image-backed packet is already local
- but the best-few local `IntroStyle / DINO` outputs are not yet fully materialized as standalone CSVs
- so the current family judgment still leans primarily on:
  - image-backed `VLM`
  - plus the image-backed scalar `CLIP/LPIPS` handoff rows
- detached local `CPU-only` `DINO` backfill is now running to close the structure-axis gap on the best-few packet

Prepared CPU-only `VLM` triplet manifests:

- [vlm_manifest_qedgepattn_e01_vs_knee_vs_seedream_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_manifest_qedgepattn_e01_vs_knee_vs_seedream_20260609.csv)
- [vlm_manifest_qedgepattn_e03_vs_knee_vs_seedream_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_manifest_qedgepattn_e03_vs_knee_vs_seedream_20260609.csv)
- current compact staging board:
  - [qedgepattn_vlm_triplets_compare_20260609.md](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/qedgepattn_vlm_triplets_compare_20260609.md)

Current CPU-only `VLM` state:

- `QEdgePattn_e01`
  - current completed cases:
    - `584`
  - current interim means:
    - style specificity: `2.70`
    - structure preservation: `3.74`
    - artifact control: `3.22`
  - current vote read:
    - now `23 / 584` on overall wins
    - now `21 / 584` on style wins
    - current subwins:
      - structure `45 / 584`
      - artifact `61 / 584`
    - still behind `Seedream` on the main vote stack
    - but now consistently above `LBM-Knee` on mean local scores across a materially larger small batch
- `QEdgePattn_e03`
  - current completed cases:
    - `609`
  - current interim means:
    - style specificity: `2.71`
    - structure preservation: `3.60`
    - artifact control: `2.96`
  - current vote read:
    - now `18 / 609` on overall wins
    - now `16 / 609` on style wins
    - current subwins:
      - structure `20 / 609`
      - artifact `33 / 609`
    - still clearly behind `Seedream`
    - but its mean local scores remain above `LBM-Knee`

First image-backed local `DINO` read:

- `QEdgePattnBestFew epoch_0001`
  - `745` pairs
  - `DINO = 0.02635`

This is currently:

- worse than `LBM-Knee full750 DINO = 0.02171`
- but better than the first landed `DualPath e01 DINO = 0.02902`

Current runtime note:

- both CPU-only `VLM` jobs are now stably alive as detached local background processes
- local logs show occasional remote API `500` / connection-reset failures
- the built-in retry path is recovering from those events and continuing forward
- so the correct read is:
  - slower than ideal
  - but not stalled

This means the local non-CLIP review chain for the active line is now genuinely started, not just planned.

## Current implication

At this moment the active line is in a better position than before:

- there is already a real image-backed local best-few packet
- there is already a real CPU-only `VLM` triplet path for those best-few points

But the current evidence still remains provisional.

Why:

- only the earliest image-backed pair is pulled locally so far
- `IntroStyle + DINO` full local read has not yet been run on that pair in this turn
- the local `VLM` triplet evidence for `QEdgePattn_e01/e03` has only just started
- and the first two cases already suggest the likely same pattern:
- and the first small VLM batch now strengthens the same pattern:
  - `QEdgePattn` can look better than `LBM-Knee`
  - but is still clearly behind `Seedream`
  - `QEdgePattn_e01` now shows that pattern across `584` completed cases and keeps real top-vote wins
  - `QEdgePattn_e03` now shows that pattern across `609` completed cases and keeps sparse but real top-vote wins

Current family-level read:

- `QEdgePattn_e01` is now the more alive point in this family
- `QEdgePattn_e03` is no longer a zero-win line, and now has a much broader evidence base
- but `e01` still looks sharper on win density, while `e03` mainly strengthens confidence that the family remains below `Seedream`
- this split matters:
  - it suggests the family is not completely flat
  - but the improvement is still too weak and too sparse to count as a promoted win

## Current stage conclusion

This family now has a reasonably complete local read:

1. image-backed best-few closure is available
2. CPU-only `VLM` has moved beyond probe scale
3. the mechanism still fails to close the `Seedream` gap

So the current conclusion is:

- `QEdgePattn` is a real positive-over-`Knee` family
- but it is not a paper-facing promoted replacement
- the remaining bottleneck still looks like:
  - insufficient target-specific style recovery capacity
  - not merely insufficient structure control

Current next-step recommendation:

- [2026-06-09-next-mechanism-after-qedgepattn.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-next-mechanism-after-qedgepattn.md)

So the current gate remains:

- extend image-backed closure enough to see whether best-few changes
- keep local `VLM` accumulating on `QEdgePattn_e01/e03`
- then decide whether this family is:
  - still just `style-up / LPIPS-worse`
  - or actually rescued by the image-backed local read
