# Experiment Summary for Method and Next Plan

Date: 2026-07-13
Primary metric: DINO-S.
Primary evidence: `docs/model_probe/target_hf_delta_eval_summary.json`, `docs/713/HF_ARCHITECTURE_PROBE_2026-07-13.md`.

## Method-level Takeaway

The training target is not the main failure. The target already asks for style in high-frequency bands:

```text
LL       = 0.7 * content_LL + 0.3 * AdaIN(content_LL -> style_LL)
LH/HL/HH = target_style bands
```

The main failure was the **conditioning path**. The baseline used `target_style` to build the training target, but the model mostly read `style_id -> style_memory -> cross-attention`. That means the target-style image latent affected the supervision but did not have a strong, explicit path into the HF velocity predictor.

The useful architectural fix is therefore:

```text
target image -> DWT HF -> pooled per-subband code -> HF residual velocity -> LH/HL/HH
```

The LL path should remain protected. Raw spatial target maps should not pass through, because they leak geometry.

## Probe Results

All listed runs use the same 6-epoch fine-tune recipe from the `brk_a_ll03_10ep` checkpoint family and evaluate with AdaIN 1.5 unless noted.

| Run | Route | DINO-S | DINO-C | CLIP-S | LPIPS | Off DINO-S | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| `target_hf_delta_ft15_epoch0006` | HF delta, AdaIN 1.0 | 0.482656 | 0.791748 | 0.717485 | 0.295013 | 0.398592 | Path connected, modest. |
| `target_hf_delta_ft15_epoch0006_adain15` | HF delta, AdaIN 1.5 | 0.484984 | 0.796570 | 0.717586 | 0.292892 | 0.400533 | Better with endpoint stats. |
| `target_hf_delta_strong_ft6` | stronger pooled HF delta | 0.487036 | 0.799077 | 0.717586 | 0.295459 | 0.401948 | First usable architecture improvement. |
| `target_hf_spatial_ft6` | raw spatial HF maps | **0.490074** | 0.404308 | **0.748291** | 0.538240 | n/a | Reject: content collapse. |
| `target_hf_subband_ft6` | per-subband pooled HF residual | **0.488624** | 0.798123 | **0.720880** | 0.296553 | 0.403917 | Primary current architecture. |
| `target_hf_subband_nomem_ft6` | subband route, style memory disabled | 0.484903 | 0.794833 | 0.716728 | **0.294348** | 0.401335 | Rejected; style memory is useful coarse prior. |
| `target_hf_subband_memres_ft6` | subband route, target-HF residualized against style memory | 0.486561 | 0.793519 | 0.719228 | 0.297730 | 0.402490 | Rejected; explicit prior subtraction below subband-only. |
| `target_hf_subband_texture_ft6` | subband pooled + stationary texture stats | 0.488420 | **0.798815** | 0.719357 | **0.296046** | **0.404302** | Conservative alternate. |
| `target_hf_content_anchor_ft6` | content-energy placement residual | 0.484393 | 0.795462 | 0.717251 | 0.298162 | 0.399538 | Safe but not competitive. |
| `target_hf_multitoken_ft6` | stationary-stat multi-token residual | 0.483562 | 0.794129 | 0.718699 | 0.297979 | 0.398793 | Rejected; code removed. |
| `target_hf_subband_deep_energy_ft6` | deep residual + RMS bound | 0.482631 | 0.794932 | 0.717588 | 0.297529 | 0.397683 | Rejected; code removed. |
| `target_hf_subband_film_head_ft6` | pure subband FiLM into HF heads | 0.482591 | 0.791672 | 0.717951 | 0.299591 | 0.398305 | Rejected; config removed. |

## What This Proves

1. **The network has enough capacity to use target-HF information.** Raw spatial HF pushes DINO-S/CLIP-S up strongly.
2. **Raw target spatial information is unsafe.** The same route destroys DINO-C and LPIPS, so it leaks target layout rather than only style.
3. **Coordinate-free HF codes are the usable middle ground.** Pooled subband codes improve style while retaining content.
4. **Extra stationary statistics are not automatically better.** Texture stats help off-diagonal style and content slightly, but do not beat simple subband pooling on all-pairs DINO-S.
5. **More placement engineering is not the next lever.** Content-anchor placement is safe but weaker.
6. **More stationary-stat tokens are not enough.** The 2026-07-14 multi-token route underperformed subband-only on all tracked metrics, so the next gain should come from better orientation-specific residual structure rather than wider statistic-token conditioning.
7. **Style memory is not just a bad shortcut.** Removing it makes the target-HF probe cleaner but hurts DINO-S/DINO-C/CLIP-S/off-DINO-S, so the next design should decompose coarse memory prior and image-specific HF residual instead of deleting memory.
8. **Explicitly subtracting memory is too blunt.** Memory-residualized target-HF partly recovers from no-memory but remains below subband-only and hurts DINO-C/LPIPS, so do not algebraically remove the class prior.

## Method Framing

For the paper, the method should be explained as three separable pathways:

| Pathway | Role | Evidence |
|---|---|---|
| Structure path | LL mostly preserves content. | Unlocking/spatial shortcuts harm content. |
| Style statistics path | Endpoint AdaIN and target-HF codes inject style. | AdaIN scaling and HF route probes improve DINO-S. |
| Transport path | Rectified flow produces content-aware motion. | Latent-WCT alone is weak; raw stats alone are insufficient. |

Cross-attention should be described as auxiliary style memory, not as the main style injector.

## Current Recommended Checkpoints

| Role | Checkpoint family | Reason |
|---|---|---|
| Paper main baseline | `brk_a_ll03_10ep` | Stable 10-epoch result, low cost, current main table. |
| Best architecture probe | `target_hf_subband_ft6/epoch_0006.pt` | Highest usable all-pairs DINO-S with good content. |
| Conservative alternate | `target_hf_subband_texture_ft6/epoch_0006.pt` | Best off-DINO-S and DINO-C balance. |
| Fallback | `target_hf_delta_strong_ft6/epoch_0006.pt` | Simpler route, already beats baseline. |
| Reject | `target_hf_spatial_ft6` | Content collapse despite high style. |

## Next Plan

### A. Paper-facing next step

Do not promote the raw spatial route. If incorporating the HF route into the paper, use `target_hf_subband_ft6` as an architecture improvement and clearly state it is a probe/follow-up unless fully re-run under the final main-table protocol.

### B. Architecture next step

Increase coordinate-free HF capacity without target spatial leakage:

1. Orientation-specific residual depth for LH/HL/HH.
2. Energy normalization against existing HF head output.
3. Stronger but compact subband residual head.
4. Keep LL disconnected from target image features except the existing mild LL target blend.
5. Keep style memory as a bounded coarse prior, then make target-HF carry residual orientation/style details.

### C. Evaluation next step

For any promoted architecture:

1. Re-run the full D5-512 protocol.
2. Re-run P2A-256 and R5-WikiArt.
3. Report DINO-S as primary style, CLIP-S as secondary.
4. Include DINO-C and LPIPS to reject content-collapse wins.
5. Re-measure generation-only timing.

### D. Cleanup next step

Before more experiments, stabilize evidence:

1. Commit current probe summaries and this experiment summary.
2. Move scratch supplement build products out of `aaai2027_v4/` or add them to ignore policy.
3. Decide whether the large config/tool deletions are a cleanup commit or should remain out-of-band.

## One-sentence Conclusion

The model was not failing because the target was style-weak; it was failing because the target image's useful HF style signal did not have a clean non-spatial route into the HF velocity heads.
