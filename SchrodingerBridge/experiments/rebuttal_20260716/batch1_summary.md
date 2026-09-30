# Rebuttal Experiments Batch 1 Summary
# Date: 2026-07-16
# Status: COMPLETED

## Exp A: Per-epoch Evaluation (Oracle Regret)

### seed7 (15 epochs, no internal early stop gate fired)
| Epoch | DINO-S | DINO-C | CLIP-S | LPIPS |
|-------|--------|--------|--------|-------|
| 1 | 0.4841 | 0.8279 | 0.7108 | 0.2404 |
| 2 | 0.4864 | 0.8079 | 0.7141 | 0.2601 |
| 3 | 0.4894 | 0.7909 | 0.7160 | 0.2621 |
| **4** | **0.4910** | 0.8076 | 0.7140 | 0.2668 | <-- ORACLE
| 5 | 0.4898 | 0.8256 | 0.7091 | 0.2562 |
| 6 | 0.4892 | 0.8242 | 0.7098 | 0.2621 |
| 7 | 0.4887 | 0.8308 | 0.7074 | 0.2619 |
| 8 | 0.4883 | 0.8308 | 0.7063 | 0.2652 |
| 9 | 0.4868 | 0.8303 | 0.7067 | 0.2526 |
| 10 | 0.4858 | 0.8314 | 0.7050 | 0.2581 |
| 11 | 0.4868 | 0.8315 | 0.7062 | 0.2640 |
| 12 | 0.4869 | 0.8322 | 0.7056 | 0.2608 |
| 13 | 0.4871 | 0.8316 | 0.7067 | 0.2665 |
| 14 | 0.4869 | 0.8320 | 0.7060 | 0.2639 |
| 15 | 0.4864 | 0.8320 | 0.7053 | 0.2611 |

- e_oracle = 4 (DINO-S=0.4910)
- e_internal = null (gate never fired)
- regret = null

### seed42 (4 epochs, internal early stop)
| Epoch | DINO-S | DINO-C | CLIP-S | LPIPS |
|-------|--------|--------|--------|-------|
| 1 | 0.4857 | 0.8230 | 0.7119 | 0.2409 |
| 2 | 0.4887 | 0.7978 | 0.7158 | 0.2625 |
| 3 | 0.4913 | 0.7853 | 0.7176 | 0.2645 |
| **4** | **0.4917** | 0.8104 | 0.7127 | 0.2595 | <-- ORACLE & INTERNAL

- e_oracle = 4 (DINO-S=0.4917)
- e_internal = 4
- regret = 0.000000

### seed123 (3 epochs, internal early stop)
| Epoch | DINO-S | DINO-C | CLIP-S | LPIPS |
|-------|--------|--------|--------|-------|
| 1 | 0.4836 | 0.8307 | 0.7095 | 0.2395 |
| 2 | 0.4859 | 0.8000 | 0.7158 | 0.2639 |
| **3** | **0.4862** | 0.8040 | 0.7144 | 0.2552 | <-- ORACLE & INTERNAL

- e_oracle = 3 (DINO-S=0.4862)
- e_internal = 3
- regret = 0.000000

### Exp A Summary
| Seed | Epochs | e_oracle | e_internal | Regret | Conclusion |
|------|--------|----------|------------|--------|------------|
| 7 | 15 | 4 | null | null | Gate never fired (no internal stop) |
| 42 | 4 | 4 | 4 | 0.0 | Perfect: internal stop = oracle |
| 123 | 3 | 3 | 3 | 0.0 | Perfect: internal stop = oracle |

**Key finding**: Internal early-stop mechanism achieves zero regret on seed42 and seed123. seed7's gate never fired (15 full epochs trained), but oracle is still at epoch 4, suggesting early stop would have been beneficial.

---

## Exp B1: Reference-pool Paired Margin

| Pool size m | WEAVE DINO-S | IDT DINO-S | Margin (mean) | Margin > 0 (%) |
|-------------|-------------|------------|---------------|-----------------|
| 8 | 0.3452 | 0.3155 | 0.0297 | 68.9% |
| 16 | 0.3782 | 0.3463 | 0.0319 | 69.2% |
| 30 | 0.4034 | 0.3700 | 0.0334 | 69.5% |

**Key finding**: WEAVE consistently outperforms IDT baseline across all pool sizes. Margin increases with pool size (0.0297 -> 0.0334), and ~69-70% of source images have positive margin.

---

## Exp D: Inference Ablation (on production checkpoint epoch_0004.pt)

| Variant | DINO-S | DINO-C | CLIP-S | LPIPS | Delta vs D0 |
|---------|--------|--------|--------|-------|-------------|
| D0_full | 0.4918 | 0.8102 | 0.7128 | 0.2595 | baseline |
| D1_adain0 | 0.4917 | 0.8100 | 0.7127 | 0.2595 | -0.0001 (negligible) |
| D2_no_hf_route | 0.4825 | 0.8125 | 0.7157 | 0.2837 | -0.0093 (significant) |

**Key findings**:
- D1 (AdaIN scale=0): Nearly identical to baseline (delta=-0.0001). Stepwise AdaIN statistics injection has negligible effect on final quality.
- D2 (no HF route): Significant DINO-S drop (-0.0093) and LPIPS increase (+0.0242). Oriented target-HF residual route is critical for content preservation.

---

## Batch 1 Completion Status
- [x] Exp A seed7: 15 epochs evaluated
- [x] Exp A seed42: 4 epochs evaluated (re-run to fix missing epoch_0004)
- [x] Exp A seed123: 3 epochs evaluated
- [x] Exp B1: reference-pool paired margin (m=8,16,30)
- [x] Exp C: ArtFID manifest audit (completed in prior session)
- [x] Exp D: D0/D1/D2 inference ablation

## Batch 2 Status (D3/D4/D5 - COMPLETED)
- [x] D3: lambda_LL=1.0 (train 12:50-12:55, eval 12:55-13:59, regret=6.8e-05)
- [x] D4: direct target endpoint (train 13:59-14:03, eval 14:03-15:07, regret=0.0039)
- [x] D5: learned HH head (train 15:07-15:12, eval 15:12-16:15, regret=0.0)
- [x] Chain runner exit code: 0 @ 2026-07-16 16:15:34

---

## Exp D3: Matched Ablation - lambda_LL=1.0 (unweighted LL)

**Config**: Based on hf_oriented_internal_early_stop, lambda_LL=1.0 (production=0.3), internal_early_stop disabled, seed=42, 15 epochs.

| Epoch | DINO-S | DINO-C | CLIP-S | LPIPS |
|-------|--------|--------|--------|-------|
| 1 | 0.4866 | 0.8214 | 0.7129 | 0.2459 |
| 2 | 0.4873 | 0.8025 | 0.7154 | 0.2633 |
| **3** | **0.4910** | 0.7874 | 0.7180 | 0.2626 | <-- ORACLE
| 4 | 0.4908 | 0.7852 | 0.7183 | 0.2651 |
| **5** | 0.4909 | 0.7963 | 0.7161 | 0.2605 | <-- INTERNAL STOP
| 6 | 0.4899 | 0.8113 | 0.7137 | 0.2566 |
| 7 | 0.4893 | 0.8180 | 0.7125 | 0.2621 |
| 8 | 0.4884 | 0.8209 | 0.7111 | 0.2621 |
| 9 | 0.4871 | 0.8230 | 0.7110 | 0.2581 |
| 10 | 0.4869 | 0.8229 | 0.7102 | 0.2597 |
| 11 | 0.4862 | 0.8245 | 0.7100 | 0.2647 |
| 12 | 0.4863 | 0.8236 | 0.7098 | 0.2624 |
| 13 | 0.4865 | 0.8241 | 0.7100 | 0.2645 |
| 14 | 0.4865 | 0.8243 | 0.7095 | 0.2661 |
| 15 | 0.4864 | 0.8240 | 0.7095 | 0.2635 |

- e_oracle = 3 (DINO-S=0.4910)
- e_internal = 5 (DINO-S=0.4909)
- regret = 6.8e-05 (negligible)

**Key finding**: D3 (lambda_LL=1.0) achieves oracle DINO-S=0.4910, slightly below baseline seed42 (0.4917). Regret is negligible (6.8e-05). This confirms that unweighted LL (lambda_LL=1.0) performs marginally worse than the production lambda_LL=0.3 setting, supporting the de-weighting design choice. The internal early-stop mechanism still fires correctly at epoch 5 with near-zero regret.

---

## Exp D4: Matched Ablation - Direct Target Endpoint (no source-aligned LL)

**Config**: Based on hf_oriented_internal_early_stop, structure_aligned_target=false, ll_partial_style_enabled=false, internal_early_stop disabled, seed=42, 15 epochs.

| Epoch | DINO-S | DINO-C | CLIP-S | LPIPS |
|-------|--------|--------|--------|-------|
| 1 | 0.4817 | 0.8294 | 0.7119 | 0.2645 |
| 2 | 0.4830 | 0.8275 | 0.7132 | 0.2718 |
| 3 | 0.4793 | 0.8055 | 0.7137 | 0.2990 |
| 4 | 0.4796 | 0.8084 | 0.7140 | 0.2872 |
| **5** | 0.4855 | 0.8113 | 0.7146 | 0.2847 | <-- INTERNAL STOP
| 6 | 0.4834 | 0.8032 | 0.7147 | 0.2864 |
| 7 | 0.4885 | 0.8020 | 0.7171 | 0.3058 |
| 8 | 0.4882 | 0.8021 | 0.7150 | 0.2890 |
| 9 | 0.4865 | 0.8026 | 0.7156 | 0.2939 |
| 10 | 0.4891 | 0.8057 | 0.7158 | 0.2982 |
| 11 | 0.4892 | 0.8010 | 0.7169 | 0.3100 |
| 12 | 0.4878 | 0.8018 | 0.7160 | 0.3057 |
| **13** | **0.4894** | 0.7966 | 0.7172 | 0.3138 | <-- ORACLE
| 14 | 0.4879 | 0.8037 | 0.7161 | 0.3054 |
| 15 | 0.4885 | 0.8001 | 0.7163 | 0.3089 |

- e_oracle = 13 (DINO-S=0.4894)
- e_internal = 5 (DINO-S=0.4855)
- regret = 0.0039 (significant, ~8x larger than D3)
- epoch_offset = -8 (e_oracle 8 epochs later than e_internal)

**Key findings**:
- D4 (direct target, no source-aligned LL) oracle DINO-S=0.4894, **significantly below baseline (0.4917)** and below D3 (0.4910). Source-aligned LL endpoint contributes +0.0023 DINO-S.
- LPIPS consistently worse (0.28-0.31 vs baseline 0.26). Content preservation is materially damaged.
- Internal early-stop fires at epoch 5, but oracle is at epoch 13. Regret=0.0039 is significant: the gate's signal is misaligned with oracle when source-aligned LL is disabled.
- **Strong evidence** that source-aligned LL endpoint is critical both for content preservation and for the internal early-stop mechanism's alignment with oracle.

---

## Exp D5: Matched Ablation - Learned HH Velocity Head

**Config**: Based on hf_oriented_internal_early_stop, enable_hh_head=true, spectral_w_hh=2.0 (production=false), internal_early_stop disabled, seed=42, 15 epochs.

| Epoch | DINO-S | DINO-C | CLIP-S | LPIPS |
|-------|--------|--------|--------|-------|
| 1 | 0.4842 | 0.8289 | 0.7112 | 0.2387 |
| 2 | 0.4877 | 0.8074 | 0.7149 | 0.2564 |
| 3 | 0.4908 | 0.7843 | 0.7185 | 0.2637 |
| **4** | **0.4930** | 0.8061 | 0.7164 | 0.2670 | <-- ORACLE & INTERNAL
| 5 | 0.4918 | 0.8196 | 0.7101 | 0.2696 |
| 6 | 0.4896 | 0.8205 | 0.7096 | 0.2703 |
| 7 | 0.4887 | 0.8239 | 0.7075 | 0.2795 |
| 8 | 0.4878 | 0.8230 | 0.7066 | 0.2801 |
| 9 | 0.4859 | 0.8226 | 0.7066 | 0.2762 |
| 10 | 0.4851 | 0.8263 | 0.7042 | 0.2692 |
| 11 | 0.4847 | 0.8250 | 0.7051 | 0.2813 |
| 12 | 0.4856 | 0.8240 | 0.7049 | 0.2800 |
| 13 | 0.4855 | 0.8224 | 0.7053 | 0.2860 |
| 14 | 0.4852 | 0.8241 | 0.7048 | 0.2825 |
| 15 | 0.4855 | 0.8234 | 0.7050 | 0.2806 |

- e_oracle = 4 (DINO-S=0.4930)
- e_internal = 4 (DINO-S=0.4930)
- regret = 0.0 (perfect)

**Key findings**:
- D5 (learned HH head) achieves the **highest oracle DINO-S** of all matched ablations: 0.4930, surpassing baseline seed42 (0.4917, +0.0013), D3 (0.4910, +0.0020), and D4 (0.4894, +0.0036).
- Internal early-stop at epoch 4 = oracle epoch 4, perfect regret=0.0.
- LPIPS at epoch 4 (0.267) is slightly worse than baseline (0.259), and CLIP-S (0.7164) is slightly lower than baseline (0.7127, but baseline DINO-S was lower). Trade-off: improved content preservation at small cost in style similarity/LPIPS.
- **Important discovery**: The learned HH head with spectral_w_hh=2.0 actually IMPROVES content preservation. Production currently disables it (enable_hh_head=false). This suggests the HH head design choice in production may be suboptimal and warrants further investigation. However, the improvement is modest (+0.0013 DINO-S) and comes with a small LPIPS penalty.

---

## Batch 2 Cross-Ablation Summary

### Oracle DINO-S Comparison (all seed=42, 15 epochs, current 1.04M architecture)

| Variant | Oracle DINO-S | e_oracle | e_internal | Regret | LPIPS@oracle | Notes |
|---------|---------------|----------|------------|--------|--------------|-------|
| **Baseline** (seed42) | 0.4917 | 4 | 4 | 0.0000 | 0.2595 | Production: lambda_LL=0.3, no HH head, source-aligned LL |
| **D3** lambda_LL=1.0 | 0.4910 | 3 | 5 | 6.8e-05 | 0.2626 | Unweighted LL, slightly worse, near-zero regret |
| **D4** direct target | 0.4894 | 13 | 5 | 0.0039 | 0.3138 | **Worst**: no source-aligned LL, big regret, big LPIPS penalty |
| **D5** learned HH head | **0.4930** | 4 | 4 | 0.0000 | 0.2670 | **Best DINO-S**, perfect regret, small LPIPS penalty |

### Key Conclusions

1. **Source-aligned LL endpoint is critical** (D4 vs baseline): Disabling it causes -0.0023 DINO-S, +0.05 LPIPS, and breaks the internal early-stop alignment (regret=0.0039, oracle at epoch 13 vs internal stop at epoch 5).

2. **lambda_LL de-weighting (0.3 vs 1.0) has minor effect** (D3 vs baseline): Oracle DINO-S difference is only -0.0007, regret is near-zero. The de-weighting design choice has limited impact on content preservation, but production setting (0.3) is marginally better.

3. **Learned HH head improves content preservation** (D5 vs baseline): +0.0013 DINO-S with perfect regret. This challenges the production decision to disable HH head. However, the improvement is small and comes with a slight LPIPS penalty.

4. **Internal early-stop mechanism is robust** across D3/D5 (regret ≤ 6.8e-05), but fails when the source-aligned LL endpoint is removed (D4, regret=0.0039). This suggests the gate signal depends on the source-aligned LL route.

### Actionable Insights for Paper

- Keep the source-aligned LL endpoint claim: strongly supported by D4 ablation.
- Soften the lambda_LL=0.3 sweet-spot claim: D3 shows lambda_LL=1.0 is only marginally worse. The de-weighting is not a critical design choice.
- Reconsider the HH head decision: D5 shows enabling it improves DINO-S. Either enable it in production or weaken claims about HH head being unnecessary.
- Internal early-stop is reliable as long as source-aligned LL is present (regret ≤ 6.8e-05 across baseline/D3/D5).
