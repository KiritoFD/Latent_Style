# Midcycle Decision From Full750 IntroStyle-DINO and Local VLM

Date: 2026-06-09

Evidence bundle:

- [local_finalists_introstyle_dino_full750_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/local_finalists_introstyle_dino_full750_20260609.csv)
- [local_finalists_introstyle_vs_dino_full750_20260609.png](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/local_finalists_introstyle_vs_dino_full750_20260609.png)
- [vlm_distinct5_finalists_interim_means_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_distinct5_finalists_interim_means_20260609.csv)
- [2026-06-09-local-vlm-full750-interim.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-local-vlm-full750-interim.md)

## Current ranking

External ceiling:

- `Seedream-4.5`
  - full750 `IntroStyle target = 0.1201`
  - full750 `DINO = 0.0291`
  - local VLM interim:
    - `49 / 49` completed cases currently select it as `best_overall`
    - mean scores so far:
      - style specificity: `4.98`
      - structure preservation: `5.00`
      - artifact control: `5.00`
    - caution:
      - the interim VLM run is still early and front-loaded toward `Early_Renaissance` source cases
      - so this is strong but not yet the final full-distribution verdict

Best internal balanced point:

- `LBM-Knee e13`
  - full750 `IntroStyle target = 0.1073`
  - full750 `DINO = 0.0217`
  - this is the cleanest current tradeoff between:
    - style movement
    - structure preservation
    - not collapsing into generic painterly fog

Internal near-tie but weaker:

- `LBM-K e1`
  - full750 `IntroStyle target = 0.1077`
  - full750 `DINO = 0.0251`
  - interpretation:
    - style is almost tied with `Knee`
    - but structure is slightly worse

Rejected style-heavy point:

- `LBM-PS-v2 e13`
  - full750 `IntroStyle target = 0.0993`
  - full750 `DINO = 0.0303`
  - local VLM interim means:
    - style specificity: `1.22`
    - structure preservation: `1.98`
    - artifact control: `1.14`
  - interpretation:
    - this is now strongly supported as a bad target-specific frontier point
    - it is worse than `Knee` on both:
      - structure
      - visual final-review quality

Interesting but not headline:

- `SaMST e15`
  - full750 `IntroStyle target = 0.1018`
  - full750 `DINO = 0.0172`
  - interpretation:
    - structure retention is stronger than expected
    - but style still does not exceed the best internal LBM point

## Decision

1. `Seedream` remains the current style ceiling.

2. `LBM-Knee` remains the current paper-facing internal operating point.

3. `LBM-PS-v2` should not be treated as the main style frontier anymore under the new evidence stack.
   - raw `CLIP` overstated it
   - `IntroStyle + DINO + VLM` all move against it

4. New mechanism burden:
   - beat `LBM-Knee` on `IntroStyle`
   - without sliding rightward on the DINO axis
   - and without triggering the VLM failure pattern currently hitting `LBM-PS-v2`

## Immediate implication for training

- do not keep spending remote GPU time on families that are already below:
  - `LBM-Knee` on full750 `IntroStyle`
  - and worse than `LBM-Knee` on `DINO`
- if a live lane looks likely to land near `0.69 / 0.51` in `CLIP/LPIPS`, it is probably not the highest-value consumer of the single formal GPU lane

## Next best local heavy review target

- `Hold4TwoStage best`
  - because it is the strongest open question inside the hold family
  - and it is the most plausible candidate for:
    - stronger structure than `Seedream`
    - better style than `Hold4Mid`
    - without `LBM-PS-v2` fog
