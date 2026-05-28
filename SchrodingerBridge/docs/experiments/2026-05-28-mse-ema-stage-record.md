# 2026-05-28 MSE / EMA Stage Record

## Executive Verdict

Question:

```text
Is MSE better than EMA, considering the current experiments as a whole?
```

Answer:

```text
MSE is slightly better than EMA on matched style efficiency, but it is not
globally or decisively better as the final backend.
```

The clean matched controls changed only:

```text
vae_model = mse
latent_root = latent-256
```

Across five matched EMA-to-MSE controls, MSE gives:

```text
mean delta clip_style = +0.00194
mean delta LPIPS      = -0.00270

among MSE rows with LPIPS <= 0.50:
mean delta clip_style = +0.00224
mean delta LPIPS      = -0.00358
```

This falsifies the earlier strong claim that MSE is simply unusable. However,
the best content-safe MSE row is still:

```text
mse_transport_texton_w34_guard e6 = 0.718588 clip_style / 0.483008 LPIPS
```

It is close, but still below the `0.72` gate and below the `0.73` target. The
MSE row that crosses `0.72` is:

```text
mse_guard_w20_lowwarp e7 = 0.725233 / 0.553443
```

That is the same high-style / high-damage regime as the old unsafe frontier and
SaMST-like tradeoff. It is not a paper-ready improvement.

Practical decision:

- Keep EMA as the clean diagnostic/content-safe backend.
- Promote MSE texton as a serious near-miss/control candidate, not as a
  wholesale backend replacement and not as the paper mainline without visual
  proof.
- Do not spend more time on generic "EMA vs MSE" ideology. The current blocker
  is the style actuator/operator, especially how style energy enters without
  LPIPS drift.

## Evidence Against An Absolute MSE Ban

The latest theory objection says MSE should be absolutely banned because it
may act as a low-pass latent prior and "cheat" CLIP-style. That is a useful
failure hypothesis, but the current matched evidence does not justify an
absolute ban.

What the evidence proves:

- In same-config controls, MSE is slightly better than EMA on the current
  operators. The gain is small but repeated: plain4, dynamic guard, and texton
  all improve style, and bodyblend improves LPIPS.
- MSE is not sufficient. The best content-safe row remains below the `0.72`
  gate, and the row that crosses `0.72` pays the same unsafe LPIPS cost as the
  old high-style frontier.
- The "MSE is just unusable" claim is therefore falsified for this code path.
  The weaker claim "MSE may be a fragile metric carrier beyond the current
  frontier" remains open and must be tested visually/per-style.

Operational rule:

```text
Do not ban MSE. Do not trust MSE by scalar metrics alone.
Use MSE texton as a near-target pressure/control line, while EMA remains the
clean theoretical and diagnostic line.
```

Additional acceptance burden for any MSE-promoted row:

| check | requirement |
|---|---|
| style gate | `clip_style > 0.72`, preferably `>0.73` |
| content gate | `LPIPS <= 0.50`, ideally `<=0.47` |
| visual gate | first-image grid must not show fog, blocky repaint, or fractured contours |
| per-style gate | Hayao must improve without only buying the global mean through Van Gogh/Monet |
| structural gate | EC/grid diagnostics cannot degrade into the old lowwarp regime |

## Gates

Current decision gates:

| gate | clip_style | content_lpips | meaning |
|---|---:|---:|---|
| hard target | `>0.73` | `<=0.50` | desired next-stage result |
| strong region | `>0.72` | near `0.40-0.50` | enough to claim broad win over SaMST if visual grid holds |
| SaMST protocol A 800 | `0.725312` | `0.539039` | high style but weak structure/content |
| current best content anchor | `0.71073` | `0.40735` | very good LPIPS, style short |

Seedream remains an external diagnostic reference only:

| reference | clip_style | content_lpips | status |
|---|---:|---:|---|
| Seedream 4.5 all-pairs API | `0.7532` | `0.3644` | golden diagnostic, not training supervision |
| Seedream 4.5 transfer subset | `0.7326` | `0.3822` | golden diagnostic, not training supervision |

## Backend Status

| backend | best useful status | verdict |
|---|---|---|
| SDXL VAE | stable but `clip_style ~0.667` | not a current 256x256 mainline backend |
| CompVis KL-f4 | fair tests around `clip_style ~0.66`, content acceptable | reject as drop-in; revisit only with a different f4/wavelet architecture |
| SD15 EMA | cleanest content-safe diagnostics; texton/bodyblend reach `0.714-0.716` under LPIPS budget | keep as diagnosis and content anchor |
| SD15 MSE | matched controls show small style gain; texton reaches `0.718588 / 0.483008` | keep as near-miss candidate/control, not proven final backend |

## Matched MSE vs EMA Controls

| family | EMA best | MSE best | delta clip | delta LPIPS | readout |
|---|---:|---:|---:|---:|---|
| plain4 W20 anchor | `0.700700 / 0.421500` | `0.703597 / 0.419917` | `+0.002897` | `-0.001583` | MSE small gain |
| dynamic guard W28 | `0.707800 / 0.447700` | `0.710273 / 0.446022` | `+0.002473` | `-0.001678` | MSE small gain |
| transport texton W34 | `0.714510 / 0.482610` | `0.718588 / 0.483008` | `+0.004078` | `+0.000398` | best MSE-positive row |
| bodyblend W28 | `0.715800 / 0.497200` | `0.715295 / 0.485741` | `-0.000505` | `-0.011459` | MSE improves LPIPS, not style |
| guard W20 lowwarp | `0.724500 / 0.552600` | `0.725233 / 0.553443` | `+0.000733` | `+0.000843` | both are over LPIPS budget |

Interpretation:

- MSE improves the style slope a little in the content-safe region.
- The largest useful MSE lift is texton: `+0.004078` clip at unchanged LPIPS.
- MSE does not repair the unsafe high-style route. Lowwarp still crosses
  `0.72` by paying `LPIPS > 0.54`.
- Therefore MSE is a useful carrier/backend control, not the missing theory by
  itself.

Artifacts:

```text
SchrodingerBridge/exp/vae_backend_256_mse_controls/vae_backend_256_results.csv
SchrodingerBridge/exp/analysis/mse_backend_controls_20260528/mse_backend_matched_comparison.csv
SchrodingerBridge/exp/analysis/mse_backend_controls_20260528/mse_backend_matched_comparison.md
SchrodingerBridge/exp/analysis/mse_backend_controls_20260528/mse_representative_per_target_style.csv
SchrodingerBridge/exp/analysis/mse_backend_controls_20260528/grids/
```

## MSE Per-Style Readout

Best content-safe MSE row:

```text
mse_transport_texton_w34_guard e6 = 0.718588 / 0.483008
```

Per target style:

| target | clip_style | content_lpips | readout |
|---|---:|---:|---|
| Van Gogh | `0.759225` | `0.488793` | strong |
| Monet | `0.729284` | `0.452744` | strong |
| Cezanne | `0.729082` | `0.471978` | strong |
| photo | `0.699480` | `0.485490` | acceptable |
| Hayao | `0.675871` | `0.516032` | still the main weakness |

This matters. MSE texton is not uniformly solving the problem. It mostly
improves the already texture-rich art styles. Hayao remains structurally
different: flat color planes and clean contours are not being generated as a
separate visual grammar.

High-style MSE row:

```text
mse_guard_w20_lowwarp e7 = 0.725233 / 0.553443
```

Per target style:

| target | clip_style | content_lpips | readout |
|---|---:|---:|---|
| Van Gogh | `0.754928` | `0.553286` | style high, content poor |
| photo | `0.734421` | `0.550714` | style high, content poor |
| Cezanne | `0.724394` | `0.534315` | style high, content poor |
| Monet | `0.721141` | `0.567628` | style high, content poor |
| Hayao | `0.691279` | `0.561272` | Hayao improves but by deformation/content drift |

This confirms the old frontier behavior: high style is available, but current
operators buy it through global repainting and content loss.

## Current Experiment Families

### 1. Old High-Style Frontier

Representative older decision-tree/tangent rows can cross `0.72`, but LPIPS is
well above the new target:

| row | clip_style | content_lpips | verdict |
|---|---:|---:|---|
| `s20_temp_var1p0_temp0p06_e0008` | `0.726547` | `0.575385` | style high, content failure |
| `s11_comp_var1p0_kin1p75_swd40_e0008` | `0.726423` | `0.585453` | style high, content failure |
| `SaMST_protocol_a_800` | `0.725312` | `0.539039` | external baseline, high style but weak content |
| old tangent/t00 family | around `0.7259` | around `0.5166` | useful historical frontier, still over preferred LPIPS |

These are not enough for the current goal. They prove style capacity exists,
but not style-content separation.

### 2. EMA Post-VAE Mainline

Important EMA rows:

| row | clip_style | content_lpips | verdict |
|---|---:|---:|---|
| `ema_plain4_w20_anchor e6` | `0.7007` | `0.4215` | clean, style weak |
| `ema_dynamic_guard_w28 e6` | `0.7078` | `0.4477` | better style, still short |
| `ema_dynamic_frontier_w32 e6` | `0.7093` | `0.4690` | saved representative, not enough style |
| `ema_transport_adain_w34_guard e6` | `0.71343` | `0.49859` | content-safe but below target |
| `ema_transport_texton_w34_guard e6` | `0.71451` | `0.48261` | best EMA texton balance |
| `ema_bodyblend_w28_guard e6` | `0.7158` | `0.4972` | best EMA body style/content balance |
| `ema_guard_w20_lowwarp e7` | `0.7245` | `0.5526` | high style, over LPIPS budget |

EMA conclusion:

- EMA is much better than SDXL/KL-f4 for current operators.
- EMA gives the cleanest content-safe diagnostics.
- EMA has not produced a `>0.72 && <0.50` row.
- Stronger EMA style pressure mostly follows the same LPIPS-damaging slope.

### 3. Style Embedding / Tokenizer Spiral

Current rollback anchor:

```text
m02_embspatial_highpass_style = 0.71073 / 0.40735 / EC 0.84967
```

This is visually/content safe, but below the style target.

Tokenizer experiments so far:

| branch | best row | result | conclusion |
|---|---|---:|---|
| band gate | `bg00_band_anchor` | `0.71289 / 0.44403` | safe but style-neutral |
| grammar/band gate over m02 | `ag02_m02_g56_texture_anchor` | `0.710955 / 0.407269` | stable, marginal only |
| stat vocab / stat reader | `sr00/sr01` | around `0.7105-0.7107`, LPIPS lower | content safe, style neutral |
| texton/prototype carrier | `tc/pc` branches | around `0.7103-0.7106` | non-hazy but no style lift |
| reference memory diagnostic | `rm01_lowfreq_match_k8` | `0.715447 / 0.477220` | source availability helps, but protocol-risk diagnostic |
| Fisher memory residual | `rf00/rf01` | `0.7066-0.7079`, LPIPS `0.431-0.436` | atom separability fixed, residual consumption fails |
| Fisher/depthwise operator tokenizer | `fo01/fo11` | around `0.71030 / 0.40877` | operator active, frozen consumer does not amplify style |

Tokenizer conclusion:

- The tokenizer is not yet a style amplifier.
- Better style atoms or better token coordinates alone are insufficient.
- The bottleneck has moved to operator consumption: style coordinates must bind
  to an executable flow/operator path that the backbone actually uses.

### 4. Seedream Side Probe

Seedream is used only for diagnosis, not mainline supervision.

The external-teacher style-adapter side probe was deliberately separated:

| teacher pairs | ok pairs | clip_style | content_lpips | conclusion |
|---:|---:|---:|---:|---|
| 16 | 16 | `0.70347` | `0.50339` | pipeline works only |
| 120 | 116 | `0.70111` | `0.49938` | more teacher refs did not raise benchmark style |

Conclusion: Seedream teacher distillation does not prove a hidden easy adapter
route. It remains useful as a visual diagnostic for the missing operator:
region-organized repainting, flat color planes, contour locking, and less
texture fog.

## Theory Update

The current evidence changes the VAE story:

1. The claim "MSE is bad because MSE smooths the latent" is too strong.
   Matched controls show MSE gives small but consistent style gains on several
   current carriers.
2. The claim "non-MSE / higher-capacity VAE is automatically better" is false.
   SDXL and KL-f4 both failed as drop-in 256x256 backends under current
   operators.
3. EMA is still valuable because it gives clean content-safe readouts. But EMA
   alone does not solve style delivery.
4. MSE may match the old f8 LANCET operator assumptions better than EMA in the
   texton branch. This is an engineering fact to exploit, not a theoretical
   reason to abandon EMA. It may also be a metric-fragile carrier, so any
   MSE-promoted row needs stronger visual and per-style proof than an EMA row.
5. The true unresolved problem is operator separation:

```text
where style enters  !=  what style statistics enter  !=  how strongly they move
```

Safe routers preserve content but suppress style. High-style routes inject
style too globally and damage content. The missing component is an executable
style operator that can add organized local style statistics while respecting
semantic/content boundaries.

Hayao remains the clearest diagnostic. Texture-rich styles already rise well
under MSE texton; Hayao needs a separate flat-color / edge-contour grammar.

## Current Recommended Mainline

Short-term experiment priority:

1. Preserve `mse_transport_texton_w34_guard e6` as the current best
   content-safe MSE pressure/control candidate.
2. Use `m02_embspatial_highpass_style` as the rollback/content anchor.
3. Treat EMA/m02 as the clean theory baseline and MSE/texton as a near-target
   stress test. Compare all future runs against both:

```text
MSE texton candidate: 0.718588 / 0.483008
EMA m02 content anchor: 0.710730 / 0.407350
```

Next experiments should be narrow and theory-driven:

| priority | hypothesis | acceptable outcome |
|---|---|---|
| MSE texton guarded operator | MSE texton has the best measured style/content slope; test whether a structured operator can lift it without exposing MSE fragility | `clip_style > 0.72`, LPIPS `<=0.50`, visual/per-style pass |
| EMA operator-consumer spiral | Fisher/depthwise coordinates are active but under-consumed; train the consumer while freezing tokenizer | style rises above `ag02/m02` without haze |
| Hayao flat/contour branch | Hayao is not a texton-weighting problem; it needs flat-color plus edge-contour grammar | Hayao clip rises without global LPIPS damage |

Do not continue:

- SDXL/KL-f4 drop-in sweeps;
- blind terminal SWD/patch scalar increases;
- memory residual variants that inject selected atoms as generic residuals;
- Seedream-supervised training as a mainline result.

## Bottom Line

If the decision is "which backend should get the next serious run", the answer
is:

```text
Run a near-target MSE-texton pressure test only if it is designed to falsify the
metric-cheating concern. Keep EMA/m02 as the diagnostic, rollback, and theory
mainline.
```

If the decision is "has MSE solved the problem", the answer is:

```text
No. MSE improves the current style/content frontier slightly, but the remaining
gap still requires an operator/tokenizer design change.
```
