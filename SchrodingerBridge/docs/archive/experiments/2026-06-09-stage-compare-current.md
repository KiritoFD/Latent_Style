# Stage Compare Current

Date: 2026-06-09

This note is the current decision board for the still-relevant families under the remote/local split.

## Closed or de-prioritized

`Knee + spatial carrier body+decoder`

- local best-few review already closed it as:
  - `near-negative / do not promote`
- reason:
  - `IntroStyle` target rose a little
  - but `DINO` degraded too much
  - and specificity margin weakened

`Hold4TwoStage`

- local `IntroStyle + DINO` best-few read is negative-leaning
- local CPU-only `VLM` triplets now reinforce that read:
  - `e12`: `Seedream 596 / 616`, candidate `20 / 616`
  - `e02`: `Seedream 550 / 566`, candidate `15 / 566`
  - `e20`: `Seedream 530 / 549`, candidate `18 / 549`
- current decision:
  - effectively de-prioritized

## Current live family

`Knee + spatial carrier body+decoder + edge-gated structure leash`

Why it is still live:

- it is the smallest theory-aligned correction to the previous spatial carrier failure
- it keeps the same style-routing family
- and adds the missing explicit structure leash

Current remote state:

- train completed through `epoch_0012`
- image-backed `fresh_localreview` packet is now available through `epoch_0012`

Current best-few handoff by `CLIP/LPIPS`:

- `epoch_0001`
  - best transfer `CLIP-style`
  - `0.7040 / 0.4397`
- `epoch_0003`
  - best transfer `LPIPS`
  - `0.7036 / 0.4391`
- `epoch_0012`
  - latest
  - `0.7036 / 0.4402`

Interpretation:

- on `CLIP/LPIPS`, this family still looks like an early plateau line
- local CPU-only VLM is now at least giving weak partial encouragement:
  - [2026-06-09-edgegated-vlm-interim.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-edgegated-vlm-interim.md)
  - current `340` completed cases give:
    - `Seedream 321 / 340`
    - `EdgeGated_e03 19 / 340`
  - and `EdgeGated_e03` already has:
    - `50` structure subwins
    - `64` artifact-control subwins
  - companion triplets:
    - `EdgeGated_e01`: `Seedream 235 / 245`, candidate `10 / 245`
    - `EdgeGated_e12`: `Seedream 235 / 244`, candidate `9 / 244`
- CPU-only triplet staging board:
  - [edgegated_vlm_triplets_compare_20260609.md](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/edgegated_vlm_triplets_compare_20260609.md)
- so unlike the earlier branches, it has not yet been ruled out by non-CLIP local review

Current active follow-up read:

- the newer `qedgegated + pattn` line now has a materially longer training-side `full_eval` curve
- current completed in-process points through `epoch_0010` still read as:
  - `style-up`
  - `LPIPS-worse`
- so the live question is no longer whether the old warning was noise
- it is whether the later image-backed `fresh_localreview` packet can overturn that same pattern on `IntroStyle + DINO`
- local CPU-only `VLM` for the active line is now also moving past probe status:
  - [qedgepattn_vlm_triplets_compare_20260609.md](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/qedgepattn_vlm_triplets_compare_20260609.md)
  - `QEdgePattn_e01`: `2 / 83` overall wins
  - `QEdgePattn_e03`: `3 / 185` overall wins
  - read:
    - active-line points can beat `LBM-Knee`
    - but are still far from challenging `Seedream`
    - and within the active family itself:
      - `e01` still looks sharper on win density
      - `e03` now adds broader evidence that the family still sits below `Seedream`
- combined image-backed plus early non-CLIP local read now supports:
  - keep the family as evidence that the direction is better than `Knee`
  - but treat it as `non-promoted / still below Seedream`
- local CPU-only `VLM` for the active line is now also moving past probe status:
  - [qedgepattn_vlm_triplets_compare_20260609.md](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/qedgepattn_vlm_triplets_compare_20260609.md)
  - `QEdgePattn_e01`: `2 / 37` overall wins
  - `QEdgePattn_e03`: `0 / 34` overall wins
  - read:
    - active-line points can beat `LBM-Knee`
    - but are still far from challenging `Seedream`
    - and within the active family itself:
      - `e01` currently looks more alive than `e03`

## Immediate next decision gate

The next meaningful question is:

- does local best-few review on the edge-gated line show:
  - any real `IntroStyle` gain over `LBM-Knee`
  - without repeating the previous `DINO` collapse

If yes:

- keep the family alive

If no:

- stop iterating on this spatial branch family
- move to a new `late style-recovery head`
