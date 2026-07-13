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
| `target_hf_subband_ablate_residual` | same checkpoint, subband residual zeroed at inference | 0.485770 | 0.788810 | 0.720464 | 0.300980 | 0.403276 | Causal check: residual path is real and content-helpful. |
| `target_hf_subband_scale_1p25` | same checkpoint, residual scaled 1.25x at inference | 0.487311 | 0.788688 | 0.720082 | 0.300671 | 0.404491 | No balanced gain; slight off-style gain costs content. |
| `target_hf_subband_scale_1p5` | same checkpoint, residual scaled 1.5x at inference | 0.487485 | 0.779830 | 0.721106 | 0.305438 | 0.406744 | Style-biased, content cost too high. |
| `target_hf_subband_scale_hh1p5` | same checkpoint, only HH residual scaled 1.5x | 0.487815 | 0.783560 | 0.720560 | 0.303415 | 0.406092 | HH is more aligned, but boosting it still costs content. |
| `target_hf_subband_nomem_ft6` | subband route, style memory disabled | 0.484903 | 0.794833 | 0.716728 | **0.294348** | 0.401335 | Rejected; style memory is useful coarse prior. |
| `target_hf_subband_memres_ft6` | subband route, target-HF residualized against style memory | 0.486561 | 0.793519 | 0.719228 | 0.297730 | 0.402490 | Rejected; explicit prior subtraction below subband-only. |
| `target_hf_subband_texture_ft6` | subband pooled + stationary texture stats | 0.488420 | **0.798815** | 0.719357 | **0.296046** | **0.404302** | Conservative alternate. |
| `target_hf_content_anchor_ft6` | content-energy placement residual | 0.484393 | 0.795462 | 0.717251 | 0.298162 | 0.399538 | Safe but not competitive. |
| `target_hf_multitoken_ft6` | stationary-stat multi-token residual | 0.483562 | 0.794129 | 0.718699 | 0.297979 | 0.398793 | Rejected; code removed. |
| `target_hf_subband_deep_energy_ft6` | deep residual + RMS bound | 0.482631 | 0.794932 | 0.717588 | 0.297529 | 0.397683 | Rejected; code removed. |
| `target_hf_subband_film_head_ft6` | pure subband FiLM into HF heads | 0.482591 | 0.791672 | 0.717951 | 0.299591 | 0.398305 | Rejected; config removed. |
| `target_hf_subband_basis_ft6` | target-HF selects low-rank content-derived residual basis | 0.482840 | 0.793659 | 0.718310 | 0.297061 | 0.398561 | Rejected; safe but too weak; code/config removed. |
| `target_hf_subband_pairstats_ft6` | target-HF plus current-vs-target HF discrepancy statistics | 0.483765 | 0.794304 | 0.718318 | 0.297092 | 0.399385 | Rejected; dynamic global statistics are too coarse; code/config removed. |
| `target_hf_subband_diraux_ft6` | direct residual-direction auxiliary loss | 0.486150 | 0.793859 | 0.718929 | 0.297425 | 0.402097 | Rejected; improved direction probe but hurt the image frontier; code/config removed. |
| `target_hf_subband_timewindow_norm` | inference-only early/late residual windows | 0.48660-0.48664 | 0.79361-0.79365 | 0.71933-0.71938 | 0.297480 | 0.40254-0.40256 | Rejected; temporal localization underperforms full-path residual; temporary hook code removed. |
| `target_hf_subband_mixer_ft6` | cross-orientation pooled-code mixer | 0.486666 | 0.793705 | 0.719392 | 0.297500 | 0.402582 | Rejected; live but did not improve residual direction or metrics; code/config removed. |
| `target_hf_subband_current_delta_ft6` | target-current pooled HF code difference | 0.486683 | 0.793621 | 0.719366 | 0.297567 | 0.402626 | Rejected; slightly stronger target-specific info flow but no residual-direction or metric gain; code/config removed. |

## What This Proves

1. **The network has enough capacity to use target-HF information.** Raw spatial HF pushes DINO-S/CLIP-S up strongly.
2. **Raw target spatial information is unsafe.** The same route destroys DINO-C and LPIPS, so it leaks target layout rather than only style.
3. **Coordinate-free HF codes are the usable middle ground.** Pooled subband codes improve style while retaining content.
4. **The trained subband residual has causal value.** Zeroing only the three subband residual delta modules at inference lowers DINO-S, DINO-C, and LPIPS, so the route is not a dead branch and does not merely trade content for style.
5. **Simple residual amplification is not the answer.** Scaling the trained residual to 1.25x or 1.5x raises off-DINO-S slightly but lowers all-pairs DINO-S and content metrics, so the bottleneck is residual direction/conditioning rather than scalar magnitude.
6. **The residual direction is a real bottleneck, but direct auxiliary supervision is too invasive.** Direction decomposition shows mean MSE improvement `0.0319`, mean cosine `0.1575`, and orthogonal fraction `0.9818`. A direct residual-direction auxiliary improves the probe (`cos=0.3222`, MSE improvement `0.1167`) but lowers DINO-S/DINO-C/LPIPS, so direction alignment must not compete with the main transport objective.
7. **The residual is not only an endpoint texture patch.** Normalized early-only and late-only residual windows both score around DINO-S `0.4866`, below full residual `0.4886`, so timing alone is not the route bottleneck.
8. **Extra stationary statistics are not automatically better.** Texture stats help off-diagonal style and content slightly, but do not beat simple subband pooling on all-pairs DINO-S.
9. **More placement engineering is not the next lever.** Content-anchor placement is safe but weaker.
10. **More stationary-stat tokens are not enough.** The 2026-07-14 multi-token route underperformed subband-only on all tracked metrics, so the next gain should come from better orientation-specific residual structure rather than wider statistic-token conditioning.
11. **Style memory is not just a bad shortcut.** Removing it makes the target-HF probe cleaner but hurts DINO-S/DINO-C/CLIP-S/off-DINO-S, so the next design should decompose coarse memory prior and image-specific HF residual instead of deleting memory.
12. **Explicitly subtracting memory is too blunt.** Memory-residualized target-HF partly recovers from no-memory but remains below subband-only and hurts DINO-C/LPIPS, so do not algebraically remove the class prior.
13. **Low-rank content-basis residuals are too restrictive.** Letting target-HF choose coefficients over content-derived residual bases prevents coordinate leakage, but drops DINO-S/off-DINO-S/DINO-C; the branch loses useful image-specific HF style rather than improving direction.
14. **Current-target global HF discrepancy statistics are too coarse.** Pair statistics are available at train and inference time and avoid spatial leakage, but they still underperform target-only subband pooling, so the next gain is not in adding more global statistic descriptors.
15. **The target image is much stronger as supervision than as condition.** The gradient/info-flow probe separates those roles: condition-path gradients on LH/HL/HH are only about `2.5%/1.3%/0.5%` of the target-construction gradients under the actual FM-HF objective.
16. **The current subband route is clean but narrow.** Single-band interventions are almost perfectly diagonal (`LH->LH`, `HL->HL`, `HH->HH`) and LL leakage is near zero. This protects content, but it also means target-specific style response is small.
17. **Simple route widening is not sufficient.** Cross-orientation mixing and target-current code deltas are both live and safe, but neither changes the residual direction or improves DINO-S/CLIP-S/content together.

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
3. Better-conditioned compact subband residual head; do not just multiply the learned residual.
4. Do not add a direct residual-direction auxiliary loss in the current form; it improves probe alignment but hurts the image frontier.
5. Do not time-gate the residual route as a standalone fix; normalized early/late windows both underperform the full residual.
6. Keep LL disconnected from target image features except the existing mild LL target blend.
7. Keep style memory as a bounded coarse prior, then make target-HF carry residual orientation/style details.
8. Do not replace the residual with a low-rank content-derived basis unless a new probe shows the target-HF coefficient path is not underpowered.
9. Do not add current-target global discrepancy statistics unless a direction probe shows they improve residual alignment without weakening image metrics.
10. Do not add simple cross-orientation code mixing or target-current pooled-code deltas as-is; both were tested and removed after worse full eval.

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
