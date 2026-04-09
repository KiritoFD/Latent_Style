

## 2. The "Big Bang" & "Signal Separation" Era (Feb 8 - Mar 13)

**Context**: This period covers the birth of `Cycle-NCE` (Feb 8), the massive signal separation experiments (Feb 13), the SWD revolution (Feb 17), and the Gating/NCE improvements (March).

### 2.1 The Overfit50 Era (Feb 8 - Feb 16)
*This phase was dominated by the `overfit50` family (28 experiments), trying to find the balance between Style and Structure in latent space.*

**The "Peak" of Feb 13 (The Gram Era):**
From the experiment inventory of Feb 13, the top runners using Gram/Struct losses were:
- **`overfit50-upscale`**: Style=0.593, LPIPS=0.0 (Low Confidence content metric).
- **`full_300_distill_low_only_v1`**: Style=0.551, LPIPS=0.697 (Solid baseline).
- **`overfit50-style-force-balance-v1`**: Style=0.549, LPIPS=0.764.

*Note*: Many top performers had `content_lpips=0.0` due to old evaluator output format.

### 2.2 The Death of Gram & Rise of SWD (Feb 17 - Feb 22)
*This was the most critical turning point. The project moved from Gram Matrices to Sliced Wasserstein Distance (SWD).*

- **Feb 17, 01:49 (`84b525f`)**: "SWD has weak effect, GRAM is completely useless."
- **Feb 17, 22:52 (`d3526ef`)**: "diff-gram continued ablation, got 0.07." (The last stand of Gram).
- **Feb 21, 16:16 (`025b77e`)**: "diff-gram performed extremely poorly on SDXL-FP32." -> Gram abandoned.

### 2.3 The Architecture Stabilization (Feb 24 - Mar 13)
*After the Loss War, the model architecture entered a "feature rich" period: style injection, texture maps, and eventually gating.*

- **Feb 24 (`adb274a`)**: `patch 1,3,5` (SWD Patch sizes tuning).
- **Feb 25 (`fae58a0`)**: `SpatiallyAdaptiveAdaMixGN` appeared (577 lines). Result: "Effect not good, SNR improved with conv1".
- **Feb 26 (`f7b328c`)**: `GlobalDemodulatedAdaMixGN` introduced. "Brushstrokes changed obviously." (617 lines).
- **Mar 8 (`4992e06`)**: **NCE Loss is effective!** Model simplified to 498 lines (`MSContextualAdaGN`).
- **Mar 9 (`5c7c2a2`)**: Attempted `HybridStyleBank` (690 lines). Result: "CLIP prior pollution", reverted.
- **Mar 11 (`80ef230`)**: **Reverted to Decoder-D Configs** (686 lines). This marks the transition to the "Decoder-D" era.

### 2.4 Key Architecture Parameters (from Feb 13 Inventory)
| Parameter | Status in Feb 13 |
|---|---|
| `use_decoder_adagn` | True (AdaGN was standard) |
| `use_style_delta_gate` | False (Gating not yet invented) |
| `use_style_spatial_tv` | True (Spatial TV was active) |
| `loss__w_stroke_gram` | 0.7 (Gram weight was high) |
| `loss__w_distill` | 1.0 (Distillation key) |
| `loss__w_cycle` | 3.0 - 10.0 (Cycle consistency was heavy) |

## Summary of Feb 8 - Mar 13
The project evolved from a messy mix of Gram/Struct/Cycle losses to a streamlined **AdaGN + SWD** model.
- **Feb 1-7**: Thermal project (Pre-scratch).
- **Feb 8**: Cycle-NCE born (251 lines).
- **Feb 9-16**: Gram Matrix experiments (`Model.py` bloated to 889 lines).
- **Feb 17-22**: The Purge! Gram dies, SWD takes over (`Model.py` shrinks to 338 lines).
- **Feb 24-Mar 5**: Refinement of Style Injection and TV (`Model.py` grows to 791 lines).
- **Mar 7-13**: Gating, NCE, and "HybridStyleBank" experiment (`Model.py` stabilizes around 686 lines).
