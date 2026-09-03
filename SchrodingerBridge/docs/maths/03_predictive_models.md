# Predictive Models for Direction A

Reviewed `2026-05-19`.

## Core Hypothesis

**Direction A (重装甲核爆拉力)**: Sinkhorn double-stochastic routing (topological armor) protects LPIPS while extreme terminal SWD weight (0.60-0.80) forcibly pulls Style past 0.72.

The key insight is that Sinkhorn and extreme SWD form a complementary pair — one constrains the *attention mechanism* (preventing local over-painting), the other constrains the *endpoint distribution* (forcing global style shift). They operate on different objects and are not competing.

## 1. Sinkhorn as Topological Armor

### Definition

The raw attention matrix `A_raw = Q K^T / (sqrt(C) * temp)` at spatial resolution 16×16 (256 positions) is transformed by iterative row and column normalization:

```
A⁰ = softmax(A_raw, dim=-1)
A^{k+1} = normalize_rows(A^k) then normalize_cols(A^k)
```

After convergence, `A` is doubly stochastic:
```
sum_j A_{ij} = 1  (each content position gets total weight 1 from style positions)
sum_i A_{ij} = 1  (each style position distributes total weight 1 to content positions)
```

### Why This Protects LPIPS

**Theorem (informal)**: Under doubly-stochastic attention, the painted feature `P = A V` satisfies:

```
Var[P] <= Var[V]  (variance is not amplified)
max_spread(P) <= max_spread(V)  (extreme values are bounded)
```

**Proof sketch**: For any row `i`:
```
P_i = sum_j A_{ij} V_j
```
Since `sum_j A_{ij} = 1` and `A_{ij} >= 0`, `P_i` is a convex combination of `{V_j}`. Convex combinations cannot produce values outside the range of `{V_j}`. Similarly, the column constraint prevents any single `V_j` from dominating all `P_i`.

**Contrast with softmax**: Softmax routing can produce arbitrary concentration. A single content position can attend to a single style position with weight ~1.0, while all other content positions compete for the remaining mass. This creates local "style hot spots" that overwrite content structure (high LPIPS).

**Prediction 1**: Sinkhorn LPIPS will be strictly lower than softmax LPIPS at equal terminal SWD weight.

**Prediction 2**: The LPIPS gap between Sinkhorn and softmax widens as terminal SWD weight increases. At sw=0.60, softmax LPIPS > 0.60 while Sinkhorn LPIPS < 0.48.

### Why Sinkhorn Previously Capped Style

Historical experiments showed Sinkhorn capping style at ~0.707 (vs softmax reaching 0.714). Reason: the doubly-stochastic constraint forces uniform style distribution — color variance is "spread thin" across all spatial positions. High-frequency style details are attenuated.

**This is precisely why Sinkhorn + extreme SWD is the right combination**: Sinkhorn provides the armor, then extreme SWD provides the raw pulling force to overcome the variance attenuation.

## 2. Terminal SWD at Extreme Weights

### Current Baseline

- `terminal_swd_weight = 0.15` (config.json)
- Raw SWD loss: ~0.01-0.02
- Weighted contribution: ~0.0015-0.003 — negligible relative to kinetic (~0.035)

### Scaling Analysis

The total loss is:
```
L = w_kin * E||v||^2 + w_swd * SWD(z_1, Z_style)
```

At equilibrium, the gradient norms from the two terms should balance:
```
||d/dtheta (w_kin * kinetic)|| ≈ ||d/dtheta (w_swd * SWD)||
```

This implies:
```
w_swd / w_kin ≈ ||d(kinetic)/dtheta|| / ||d(SWD)/dtheta||
```

If kinetic gradients are ~10x larger than SWD gradients (empirically observed), then:
- At `w_swd / w_kin = 0.15/1.5 = 0.1` → SWD gradient is 1% of kinetic gradient → SWD has negligible effect
- At `w_swd / w_kin = 0.60/1.5 = 0.4` → SWD gradient is 4% → still small
- At `w_swd / w_kin = 0.80/1.0 = 0.8` → SWD gradient is 8% → meaningful
- At `w_swd / w_kin = 0.80/0.5 = 1.6` → SWD gradient is 16% → dominant

**Prediction 3**: Terminal SWD weight must be scaled by ~5-10x from current 0.15 to see a meaningful effect on style. The effective range is 0.60-0.80 with kinetic at 0.5-1.0.

### Saturation Limit

SWD weight cannot be increased indefinitely. There is a saturation point where:
1. The endpoint distribution is already close to the target → further SWD reduction yields no style gain
2. The model enters a "forced mode" where endpoint variance collapses → all outputs look identical (style ~0.65, LPIPS ~0.90)

**Prediction 4**: The saturation point for terminal_swd_weight is ~1.0 with kinetic at 0.5. Beyond this, style either plateaus or collapses.

## 3. Combined Prediction

### Expected Pareto Frontier

With Sinkhorn routing + kinetic in {1.5, 1.0, 0.5} + terminal SWD weight in {0.15, 0.30, 0.60, 0.80}:

| Config | Predicted Style | Predicted LPIPS | Confidence |
|--------|:-:|:-:|:-:|
| sinkhorn, kin=1.5, sw=0.15 | 0.700-0.710 | 0.43-0.46 | high (repro) |
| sinkhorn, kin=1.0, sw=0.30 | 0.710-0.718 | 0.44-0.47 | medium |
| sinkhorn, kin=1.0, sw=0.60 | 0.718-0.725 | 0.45-0.49 | medium |
| sinkhorn, kin=0.5, sw=0.60 | 0.722-0.730 | 0.47-0.52 | low (approaching collapse) |
| **sinkhorn, kin=0.5, sw=0.80** | **0.725-0.735** | **0.48-0.54** | **lowest but target** |

### Target Regime

The goal `clip_style > 0.72` and `content_lpips < 0.45` is predicted at:
- `sinkhorn, kin=1.0, sw=0.60` — if LPIPS stays low enough
- `sinkhorn, kin=0.5, sw=0.60` — if style needs more push

The "golden path" prediction is `sinkhorn + kin=1.0 + sw=0.60` achieving `style=0.722, LPIPS=0.46`.

## 4. Failure Modes

### Failure Mode 1: Sinkhorn + high SWD → style plateaus early

**Symptom**: Style saturates at 0.710-0.715 despite increasing SWD weight.

**Root cause**: The doubly-stochastic constraint is too tight — it prevents the spatial variance needed for high CLIP style scores even with endpoint pressure.

**Mitigation**: Increase `semantic_attn_temperature` to make routing less sharp (allow more mixing), or reduce `semantic_sinkhorn_iters` to 1-2 (partial normalization).

### Failure Mode 2: High SWD overpowers kinetic → LPIPS climbs anyway

**Symptom**: LPIPS rises past 0.50 even with Sinkhorn routing.

**Root cause**: SWD gradient magnitude at extreme weights overwhelms kinetic constraint, and the model finds a path that sacrifices content despite Sinkhorn's protection.

**Mitigation**: Instead of reducing SWD weight, increase kinetic weight to restore balance. The ratio `w_swd / w_kin` is the true control variable, not absolute values.

### Failure Mode 3: Both metrics degrade together

**Symptom**: Style drops AND LPIPS rises.

**Root cause**: The model enters a pathological regime where neither distribution matching nor content preservation works. This typically means the learning rate is wrong, the model is unstable, or the data is corrupted.

**Mitigation**: Fall back to the known-good baseline config and verify reproducibility before trying again.

## 5. Strategic Implications

### What Direction A Tests

- Is the current style ceiling (0.714) a kinetic bottleneck or a SWD bottleneck?
- If extreme SWD + Sinkhorn pushes past 0.72, the bottleneck was SWD.
- If style plateaus regardless of SWD weight, the bottleneck is kinetic (motion budget).
- If LPIPS collapses regardless of Sinkhorn, the bottleneck is architectural (attention mechanism).

### What Comes After

- **Direction A succeeds** (style > 0.72, LPIPS < 0.48): Lock the regime, then run Direction B (temperature annealing) to further improve LPIPS.
- **Direction A partially succeeds** (style > 0.72, LPIPS 0.48-0.55): Try Direction C (frequency-decoupled SWD) to repair content.
- **Direction A fails** (style < 0.72): The bottleneck is not SWD. Try lowering kinetic before trying new loss terms.
