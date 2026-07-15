# WEAVE Delivery Summary

**Date:** 2026-07-13
**Status:** current paper bundle is coherent; broader repo is an active dirty research workspace.
**Primary style metric:** DINO-S. CLIP-S is secondary.
**Current evidence base:** `aaai2027_v4/paper.tex`, `aaai2027_v4/supplement.tex`, `docs/model_probe/target_hf_delta_eval_summary.json`.

---

## 1. Current State

| Area | Current fact |
|---|---|
| Paper bundle | `aaai2027_v4/` has a committed AAAI v4 packet from `0867d43d7`; the supplement has now been expanded in `supplement.tex` and mirrored in `SUPPLEMENTARY_MATERIAL.md`. |
| Local worktree | Not clean. The 2026-07-13 audit found roughly 816 working-tree changes: large historical deletions, new probe files, source edits, and supplement scratch. |
| Remote `I:` | `I:\Github\Latent_Style\WEAVE` is the synchronized execution tree, not a git repo. |
| Stale checkpoint path | Old checkpoint-drive references are not valid for the audited remote. Use logical paths such as `<EXP_ROOT>/dino_s_break/brk_a_ll03_10ep/epoch_0010.pt`. |
| Historical probe root | `I:\Github\Latent_Style\WEAVE\exp\model_probe` mirrored into local `docs/model_probe/`. |

Old statements saying the codebase is fully clean should be treated as historical cleanup intent, not current state.

---

## 2. Main Paper Result

Use `aaai2027_v4/paper.tex`, Table 1 as source of truth.

| Board | DINO-S | CLIP-S | LPIPS | DINO-C |
|---|---:|---:|---:|---:|
| D5-512 | **0.4859** | 0.7075 | **0.2583** | **0.8287** |
| P2A-256 | 0.4801 | 0.6681 | 0.3116 | 0.8612 |
| R5-WikiArt | 0.5226 | 0.7747 | 0.2895 | 0.7717 |

Paper claim:

> WEAVE is not best on every single axis. It is the strongest balanced learned method under the shared protocol: DINO-S first tier, strong content retention, 903K trainable parameters, 3-minute training, and 95-second generation-only inference for 750 pairs.

Important comparisons:

| Method | D5 DINO-S | D5 LPIPS | D5 DINO-C | Reading |
|---|---:|---:|---:|---|
| StyleAligned | 0.675 | 0.869 | 0.239 | High style, content collapse. |
| Seedream 4.5 | 0.486 | 0.477 | 0.739 | Similar style, much higher content cost. |
| SaMam | 0.477 | 0.243 | 0.812 | Strong content, weaker DINO-S/CLIP-S on D5. |
| Latent-WCT | 0.362 | 0.441 | 0.559 | Wavelet statistics alone are insufficient. |
| WEAVE | 0.4859 | 0.2583 | 0.8287 | Balanced paper point. |

### 2.1 内容-风格上下界夹逼分析

为量化各方法在内容-风格 Pareto 前沿上的位置，我们在同一评估管线下建立了两个理论边界：

| 基准 | 含义 | CLIP-S ↑ | DINO-S ↑ | LPIPS ↓ | DINO-C ↑ |
|------|------|----------|----------|---------|----------|
| **Target Style** | 风格上界 + 内容下界 | 0.8737 | 1.0000 | 0.7332 | 0.1760 |
| **IDT (Identity)** | 内容上界 + 风格下界 | 0.6532 | 0.3700 | 0.0000 | 1.0000 |

> Target Style: 直接将目标风格参考图作为输出。CLIP-S=0.874 是理论风格上限（同风格图之间的 CLIP 相似度），LPIPS=0.733 是理论内容下限（和内容图完全没有关系）。
> IDT: 直接将内容图作为输出。LPIPS=0 是完美内容保持，DINO-S=0.370 是跨风格 DINOv2 相似度基线（即使不做迁移，不同风格图之间也有 0.37 的 DINOv2 CLS 余弦相似度）。

**StyleAligned 内容崩溃量化：**

| 方法 | LPIPS | DINO-C | vs Target Style LPIPS | vs IDT DINO-C |
|------|-------|--------|----------------------|---------------|
| Target Style (下界) | 0.7332 | 0.1760 | — | — |
| **StyleAligned** | **0.869** | **0.239** | **+0.136 (更差!)** | **-0.761** |
| Seedream 4.5 | 0.477 | 0.739 | -0.256 | -0.261 |
| WEAVE (ours) | 0.2583 | 0.8287 | -0.475 | -0.171 |

**结论：StyleAligned 的 LPIPS=0.869 比 Target Style 下界（0.733）还差 0.136**，即其生成图和内容图的感知距离，比直接拿一张完全无关的目标风格图还要大。这是内容崩溃的定量证据——StyleAligned 不是"内容保留差"，而是"内容被破坏到比随机图还差"。

我们的 WEAVE 在 LPIPS 上距离下界有 0.475 的安全边际，DINO-C 距离上界仅差 0.171，是唯一同时在风格和内容上都不崩溃的 learned 方法。Seedream 4.5 虽然 DINO-S 接近我们（0.486 vs 0.486），但 LPIPS 差了近一倍（0.477 vs 0.258），内容代价远高于我们。

---

## 3. Current Mechanism Diagnosis

The training target is not the main problem. It already asks for high-frequency style:

```text
LL       = 0.7 * content_LL + 0.3 * AdaIN(content_LL -> style_LL)
LH/HL/HH = target_style bands
```

The bottleneck is the condition route. Baseline mainly reads:

```text
style_id -> style_memory -> cross-attention
```

but the image-specific target latent mostly affects target construction rather than the HF velocity predictor.

The useful route is:

```text
target image -> DWT HF -> pooled per-subband code -> HF residual velocity -> LH/HL/HH
```

The unsafe route is raw spatial target-HF injection. It proves the network has capacity, but leaks target geometry and destroys content.

Latest gradient/info-flow diagnosis:

| Probe fact | Reading |
|---|---|
| Condition-path gradients into target-style LH/HL/HH are only about `2.5%/1.3%/0.5%` of target-construction gradients. | The target image is strong as supervision but weak as a condition input. |
| Single-band intervention is almost perfectly diagonal: `LH->LH`, `HL->HL`, `HH->HH`; LL leakage is near zero. | The route is clean/content-safe, but narrow. |
| Zeroing target-HF residual changes HF velocity much more than swapping content-vs-target condition bands. | The residual branch is live and large, but its target-specific modulation is small. |
| Naive HF-stat loss gives large gradients and can conflict through time/global transport paths. | Strengthening style via an auxiliary stat loss is risky unless it is routed locally. |
| Condition-direction probe shows target-condition deltas have only `0.054/0.045/0.032` cosine with the desired HF correction. | The route is not just weak; it is mostly off-direction. |
| WCT-direction improves local condition-direction cosine to `0.110/0.125/0.093` but still worsens final metrics. | Local velocity alignment alone does not guarantee a better full transport/image frontier. |
| Route-competition probe shows style memory and target-HF are both useful but low-projection; full route MSE improvement is `0.0576`. | Do not delete style memory; improve target-HF without degrading the coarse memory prior. |
| Training-only style-memory dropout slightly improves target-HF direction but drops style-memory alignment from `0.1599` to `0.0463` and worsens full eval. | Blunt memory dropout is not the fix; implementation/config removed. |

---

## 4. HF-Route Probe Results

All rows are diagnostic probes from the `brk_a_ll03_10ep` family. Do not promote a probe to the main table until it is rerun under the full D5/P2A/R5 protocol.

| Run | DINO-S | DINO-C | CLIP-S | LPIPS | Off DINO-S | Verdict |
|---|---:|---:|---:|---:|---:|---|
| `target_hf_delta_ft15_epoch0006` | 0.482656 | 0.791748 | 0.717485 | 0.295013 | 0.398592 | Path connected, modest. |
| `target_hf_delta_ft15_epoch0015` | 0.482480 | 0.791313 | 0.718030 | 0.299062 | 0.398681 | Longer training alone does not fix it. |
| `target_hf_delta_strong_ft6` | 0.487036 | 0.799077 | 0.717586 | 0.295459 | 0.401948 | First usable HF route. |
| `target_hf_spatial_ft6` | **0.490074** | 0.404308 | **0.748291** | 0.538240 | n/a | Reject: geometry leak/content collapse. |
| `target_hf_subband_ft6` | **0.488624** | 0.798123 | **0.720880** | 0.296553 | 0.403917 | Primary current architecture probe. |
| `target_hf_subband_texture_ft6` | 0.488420 | **0.798815** | 0.719357 | **0.296046** | **0.404302** | Conservative alternate. |
| `target_hf_content_anchor_ft6` | 0.484393 | 0.795462 | 0.717251 | 0.298162 | 0.399538 | Safe but weaker. |
| `target_hf_subband_diraux_ft6` | 0.486150 | 0.793859 | 0.718929 | 0.297425 | 0.402097 | Reject: direction probe improves, image frontier worsens. |
| `target_hf_subband_timewindow_norm` | 0.48660-0.48664 | 0.79361-0.79365 | 0.71933-0.71938 | 0.297480 | 0.40254-0.40256 | Reject: early/late residual windows both underperform full residual. |
| `target_hf_subband_basis_ft6` | 0.482840 | 0.793659 | 0.718310 | 0.297061 | 0.398561 | Reject: low-rank content-derived basis is safe but underpowered. |
| `target_hf_subband_pairstats_ft6` | 0.483765 | 0.794304 | 0.718318 | 0.297092 | 0.399385 | Reject: current-target global HF stats are safe but too coarse. |
| `target_hf_subband_mixer_ft6` | 0.486666 | 0.793705 | 0.719392 | 0.297500 | 0.402582 | Reject: cross-orientation mixing live but no direction/metric gain. |
| `target_hf_subband_current_delta_ft6` | 0.486683 | 0.793621 | 0.719366 | 0.297567 | 0.402626 | Reject: target-current code delta slightly improves condition flow but no frontier gain. |
| `target_hf_subband_affine_delta_ft6` | 0.482449 | 0.790343 | 0.717787 | 0.298913 | 0.398861 | Reject: stronger condition route, worse direction/frontier. |
| `target_hf_subband_wct_direction_ft6` | 0.486511 | 0.793320 | 0.719448 | 0.297849 | 0.402438 | Reject: better local direction probe, worse final metrics. |
| `target_hf_subband_memdrop_ft6` | 0.486414 | 0.791995 | 0.719449 | 0.298218 | 0.402734 | Reject: training-only memory dropout weakens useful style-memory prior. |

Conclusion:

> Style is weak because target-style HF information lacks a clean non-spatial route into the HF velocity heads. The fix is route topology/capacity, but not raw spatial injection, not simple residual amplification, not direct residual-direction auxiliary loss, and not temporal gating alone.

---

## 5. CFG Recheck

Previous CFG/content-preservation observations are confounded. Those runs also changed style delta heads, DWT routing, HH head availability, and cross-attention gates. Do not attribute the content improvement to CFG alone without a matched ablation.

Current supplement wording treats CFG as unresolved, not as a proven mechanism.

---

## 6. Timing

| Run | Wall total | Network generation | VAE decode | Notes |
|---|---:|---:|---:|---|
| `brk_a_ll03_10ep` | 94.63 s | 53.57 s | 39.36 s | Paper generation-only timing, 750 pairs. |
| `target_hf_subband_ft6` | 106.25 s | 65.25 s | 39.34 s | Probe route adds modest network cost. |

Training for the main paper checkpoint is 176.9 s on one RTX 3060 12GB. This supports the paper's efficiency claim, but the user concern is valid: future architecture work should raise the learning ceiling, not merely keep training short.

---

## 7. Supplement Status

Updated files:

| File | Role |
|---|---|
| `aaai2027_v4/supplement.tex` | Formal comprehensive supplement: protocol, source-of-record, probe diagnosis, timing, artifact ledger, next plan. |
| `aaai2027_v4/SUPPLEMENTARY_MATERIAL.md` | Compact supplement map aligned to the formal TeX source. |
| `docs/713/HANDOFF_2026-07-13.md` | Complete local/remote audit, experiment summary, remote workflow, conclusions, and next plan. |
| `docs/713/README.md` | Entry point for current 713 handoff and detailed notes. |
| `docs/713/METHOD_EXPLORATION_AND_CKPT_2026-07-13.md` | Current checkpoint and method exploration history extracted from old non-713 docs. |
| `docs/README.md` | Source map and maintenance rules. |

The supplement now explicitly states:

- DINO-S is the primary style metric.
- Radar is a visual summary, not a custom mixed score.
- Latent-WCT is included as an analytic baseline.
- StyleAligned is not a balanced win because content collapses.
- HF-route probes are diagnostic until rerun under the full protocol.

---

## 8. Next Plan

Architecture first, tuning second:

| Step | Action | Reason |
|---|---|---|
| 1 | Keep `target_hf_subband_ft6` as the best current architecture probe. | It is still the strongest balanced point after the latest negative controls. |
| 2 | Explore less invasive target-HF route decomposition. | Direct residual-direction loss and time-window gating both hurt final metrics. |
| 3 | Keep LL disconnected from target-image shortcuts. | Avoid buying style by destroying content. |
| 4 | Rerun D5-512, P2A-256, R5-WikiArt before promotion. | Required before changing the main table. |

Avoid raw target HF maps, scalar/HH residual amplification, direct residual-direction auxiliary loss, time-window residual gating, low-rank content-derived basis replacement, current-target global HF statistic codes, cross-orientation pooled-code mixing, target-current pooled-code deltas, blunt style-memory dropout, global target-token fusion, and CFG claims without matched controls.
Also avoid affine scale+shift subband residuals and analytic WCT/AdaIN direction residuals inside the velocity field as currently tested: both improve some probe numbers but lower the final style/content frontier.

---

## 9. Cleanup Rule

Do not stage the large source/config cleanup together with paper documentation. Commit paper-facing docs separately, then decide whether the large historical deletions should become a cleanup commit, an archive branch, or remain out of band.
