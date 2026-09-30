# Style Ceiling Liberation Plan (from live probes)

Date: 2026-07-13  
Checkpoint: `exp/dino_s_break/brk_a_ll03_10ep/epoch_0010.pt`  
Probes:
- `docs/model_probe/brk_a_ll03_internal_flow.{json,md}`
- `docs/model_probe/style_ceiling_oracle.{json,md}`

## Live findings (hard numbers)

### A. Gradient / information-flow probe

1. **Loss is almost pure Flow Matching**
   - total loss ≈ 1.028
   - weighted: HL 0.529 + LH 0.468 + LL 0.031 + HH 0.0
   - HH head disabled → no HH learning path

2. **Style pathways are weak**
   - `style_memory` grad/param = 0.0020 (lowest among main groups)
   - `style_memory / head_hf` grad-norm ratio ≈ **0.07**
   - cross-attn gates ≈ **0.056–0.061** (almost shut)
   - CA delta abs ≈ 0.04–0.06 while CA out std ≈ 1.7–3.1 → residual style injection is tiny

3. **`style_latent` does not condition velocity at all**
   - sensitivity `target_style_latent_only_fixed_id`: **all bands delta_rms = 0**
   - only discrete `style_id` moves velocity (LL delta/base ≈ 0.39, LH ≈ 0.29)
   - conclusion: learned transport is **domain-token conditioned**, not reference-latent conditioned

### B. Oracle / reachability probe (32 samples)

1. **Content→style energy is LL-dominated**
   - delta energy share: **LL 68.3% / LH 10.6% / HL 11.3% / HH 9.8%**

2. **After best practical model (AdaIN=1.5), residual is still LL-dominated**
   - remain energy share: **LL 65.3%**
   - remain stat L2: LL **1.21** vs LH 0.079 / HL 0.087 / HH 0.108
   - closed fraction of style-stat gap:
     - LL **11.6%**
     - LH **35.9%**
     - HL **36.3%**
     - HH **23.9%**

3. **AdaIN only helps mid/high-frequency stats; full transfer is flat**
   - transfer_ratio full: no_adain 0.126 → a1.0 0.122 → a1.5 0.122 → a2.0 0.114
   - LH transfer: 0.137 → 0.291 → **0.334** → 0.044 (over-scale collapses)
   - LL transfer stuck ≈ **0.11–0.12** for all AdaIN scales

4. **Oracle proves the ceiling is LL, not network capacity**
   - perfect SAT α=0 (LL locked + exact style HF): transfer **0.0045**
   - SAT α=0.3 (current train): transfer **0.303**
   - SAT α=0.5: **0.502**
   - SAT α=1.0 (full LL appearance match + style HF): **1.000**
   - single-band swap on model output then AdaIN1.5:
     - swap **LL** → transfer **0.911**
     - swap LH/HL/HH → transfer ≈ **0.128**

### One-sentence diagnosis

> Current WEAVE already extracts most of the style that lives in HF statistics; the remaining ~65% residual is LL appearance that SAT intentionally withholds. Decoder upgrades cannot liberate this, because the target set itself excludes it and style_latent never enters the velocity field.

---

## Liberation strategy (do these, not decoder AdaLN)

### Priority 0 — Stop losing bets

Do **not** spend more budget on:
- decoder AdaIN/AdaLN/Q-gate
- width/depth scale-up alone
- joint SWD/contrastive as primary style engine
- α>2 endpoint overdrive as “training breakthrough”

These conflict with the measured gradient geometry.

### Priority 1 — **LL Appearance Residual Head (LAH)**  [recommended first exp]

**Idea:** keep SAT structure lock, but add a *separate* low-rank LL appearance residual that is trained/used outside the main velocity MSE fight.

#### Design
1. Freeze Stage-A WEAVE (`brk_a_ll03_10ep`) backbone + velocity heads.
2. Add tiny module `LAH`:
   - input: style tokens / style LL stats (μ,σ or 4×4 cov)
   - output: residual `Δll_app ∈ R^{B×4×1×1}` or low-spatial map `B×4×h×w` with h,w ≤ 4
   - apply: `ll_out = ll_flow + α_app * Δll_app` with spatial broadcast / bilinear upsample
3. Train only LAH with appearance losses:
   - `L_mu_std = ||μ(ll_out)-μ(ll_style)|| + ||σ(ll_out)-σ(ll_style)||`
   - optional channel Gram / tiny WCT match
   - content regularizer: `||ll_out - ll_content||_spatial_highpass` or LPIPS-proxy on lowpass only
4. Inference: keep Endpoint AdaIN on HF; add LAH on LL with scale sweep `{0.2,0.4,0.6,0.8}`.

#### Why this matches probes
- oracle swap LL alone jumps transfer 0.12 → **0.91**
- style_latent currently unused → LAH finally consumes reference stats
- does not disturb 92% FM path (frozen)

#### Acceptance
- DINO-S ≥ **0.492** on D5-512 (clear >0.486)
- DINO-C ≥ 0.74, LPIPS ≤ 0.34
- if DINO-S↑ only with DINO-C collapse → reduce α_app / strengthen content reg

#### Minimal config sketch
```json
{
  "bridge": {
    "structure_aligned_target": true,
    "ll_partial_style_enabled": true,
    "ll_partial_alpha": 0.3,
    "ll_appearance_head_enabled": true,
    "ll_appearance_train_only": true,
    "ll_appearance_loss_weight": 1.0,
    "ll_appearance_content_reg": 0.2
  },
  "training": {
    "freeze_backbone": true,
    "num_epochs": 3,
    "batch_size": 96,
    "lr": 2e-4
  }
}
```

### Priority 2 — **Two-stage FM → Appearance finetune**

If LAH works partially:

1. Stage-A: current SAT FM (content-safe transport)
2. Stage-B: unfreeze only `head_ll` + style_memory + last block CA, with:
   - reduced FM weight on LH/HL (`w_lh=w_hl=0.3`)
   - strong LL appearance match weight
   - stop-grad on HF reconstruction target (keep HF from Stage-A prediction)

This is safer than full joint training because Stage-A already placed HF correctly.

### Priority 3 — **Reference-latent style path (fix the zero sensitivity bug/feature)**

Probe showed `style_latent` sensitivity = 0. That is both a bug and a design smell.

Implement true reference conditioning:
- tokenize `style_latent` HF patches into CA keys/values (not only style_id memory)
- force nonzero path with unit test:
  - swap style_latent, freeze style_id → velocity must change on LH/HL
- train with style_id dropout so model cannot ignore latent path

This liberates **instance-level** style (brush of a painting), not only family mean.

### Priority 4 — **Only if product allows external priors**

Add frozen DINOv2 CLS cosine loss on final latent-decoded preview / feature proxy:
- direct attack on eval metric
- expected to break 0.48 immediately
- paper narrative changes from “prior-free” to “tiny student + frozen teacher”

Use only after P1/P2 fail to hit 0.49+.

---

## Experiment ladder (1–2 GPU days)

| ID | Change | Train | Eval | Kill criterion |
|----|--------|-------|------|----------------|
| L0 | repro brk_a @ adain 1.0/1.5/2.0 | 0 | D5 full | baseline sanity |
| L1 | LAH frozen-backbone, α_app sweep | 3 ep | D5 | DINO-S <0.486 and no Pareto win |
| L2 | LAH + HF AdaIN joint scale grid | 0 (infer) | D5 | CLIP-S drop >0.015 w/o DINO-S +0.005 |
| L3 | Stage-B head_ll appearance finetune | 5 ep | D5 | DINO-C <0.72 |
| L4 | style_latent CA path + unit sensitivity | 5 ep | D5+qual | style_latent sensitivity still ~0 |
| L5 | optional DINO feature loss | 3 ep | D5 | only if L1–L4 fail |

### Probe gates (run every exp, cheap)
1. `probe_baseline_internal_flow.py`  
   - require: style path grad/param not << 1e-3 if style modules are trained  
   - require: `style_latent` sensitivity > 0 if L4 enabled
2. `probe_style_ceiling_oracle.py`  
   - require: remain_energy_share LL decreases (target <0.50)  
   - require: transfer_ratio_ll > 0.25  
   - require: LH transfer stays ≥ 0.30 at operating point

---

## What “liberated” means quantitatively

Current practical point (paper main): DINO-S ≈ 0.484–0.486 with strong content.

Liberation success bar:
1. **Hard:** DINO-S ≥ 0.495 on same D5-512 protocol
2. **Pareto:** DINO-C ≥ 0.74 and LPIPS ≤ 0.34
3. **Mechanism:** remain LL energy share ≤ 0.50 and LL transfer_ratio ≥ 0.25

If (1) only via DINO loss and (2) fails → report as metric hacking, not architecture liberation.

---

## Implementation notes in this repo

Already runnable probes:
```bash
python tools/probe_baseline_internal_flow.py \
  --config configs/exp_brk_a_ll03_10ep.json \
  --checkpoint exp/dino_s_break/brk_a_ll03_10ep/epoch_0010.pt \
  --data-root G:/wikiart27_latents_compact/train \
  --output docs/model_probe/brk_a_ll03_internal_flow.json

python tools/probe_style_ceiling_oracle.py \
  --config configs/exp_brk_a_ll03_10ep.json \
  --checkpoint exp/dino_s_break/brk_a_ll03_10ep/epoch_0010.pt \
  --data-root G:/wikiart27_latents_compact/train \
  --output docs/model_probe/style_ceiling_oracle.json
```

Next code to add for L1 (small surface):
- `src/model.py`: `LLAppearanceHead` + apply after integrate, before decode
- `src/flow.py` or new `appearance_loss.py`: mu/std match on LL
- `src/config_schema.py`: flags above
- freeze logic in `src/trainer.py` (`requires_grad_` patterns already exist)

Do **not** reintroduce Round10–12 decoder AdaLN.
