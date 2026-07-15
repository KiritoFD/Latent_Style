# Full Phase Experiment Summary

Date: 2026-06-10

This note is the current paper-facing phase summary for:

- authoritative experiment registries
- baseline and internal model settings
- current numerical results
- local non-CLIP / visual conclusions
- current image failure modes
- next-step mechanism plan

It is not a replacement for the row-level CSV registries.
It is the shortest readable document that says what has been run, what the
settings were, what the results currently say, and what we should do next.

## 1. Authoritative Sources

Full row-level authoritative registries for the current phase:

- paper-facing mixed registry:
  - [aaai2027_results_master.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/aaai2027_results_master.csv)
- mechanism-sweep registry:
  - [aaai2027_inmortal_results_master.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/aaai2027_inmortal_results_master.csv)
- Distinct5 same-cost / convergence inventory:
  - [2026-06-04-distinct5_same_cost_inventory.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-04-distinct5_same_cost_inventory.csv)
- current mainline non-CLIP board:
  - [current_mainline_evidence_board_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/current_mainline_evidence_board_20260609.csv)
- current four-way external-baseline blind review:
  - [vlm_lbmpsv2_vs_seedream_vs_samst_vs_samam_20260610_snapshot6.method_summary.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_lbmpsv2_vs_seedream_vs_samst_vs_samam_20260610_snapshot6.method_summary.csv)
  - [vlm_lbmpsv2_vs_seedream_vs_samst_vs_samam_20260610_snapshot6.method_summary.md](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_lbmpsv2_vs_seedream_vs_samst_vs_samam_20260610_snapshot6.method_summary.md)

## 2. Stable Evaluation Contract

Current evaluation split is stable:

- fast screen:
  - `CLIP-S + LPIPS`
- paper-facing style axis:
  - `IntroStyle`
- paper-facing structure axis:
  - `DINO`
- final visual audit:
  - local frozen-snapshot four-way `VLM`

This means:

- `CLIP` is still valid and useful
- but it is not the sole paper-facing style judge
- `IntroStyle` and `VLM` are now necessary to detect target-style identity gaps

## 3. External Baselines And References

### 3.1 Current non-CLIP style / structure board

Authoritative source:

- [current_mainline_evidence_board_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/current_mainline_evidence_board_20260609.csv)

Current important rows:

| Label | Run | IntroStyle Target | Delta-IDT | Margin | DINO | Read |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `IDT` | `IDT` | `0.0992` | `0.0000` | `-0.1305` | `0.0000` | no-op reference |
| `LBM-K` | `LBM-K_e1` | `0.1077` | `0.0085` | `-0.0510` | `0.0251` | older internal anchor |
| `LBM-Knee` | `LBM-Knee_e13` | `0.1073` | `0.0080` | `-0.0373` | `0.0217` | current internal balanced anchor |
| `LBM-PS-v2` | `LBM-PS-v2_e13` | `0.0993` | `0.0001` | `-0.0326` | `0.0303` | style-heavy but downgraded under non-CLIP |
| `SaMST e15` | `SaMST_e15` | `0.1018` | `0.0026` | `-0.0705` | `0.0172` | real style signal, weak structure |
| `Seedream-4.5` | `Seedream_repaired750` | `0.1201` | `0.0209` | `-0.0347` | `0.0291` | external style ceiling |

Immediate read:

- `Seedream` is strongest on `IntroStyle`
- `LBM-Knee` is still the best current internal balanced point
- `LBM-PS-v2` is not paper-safe despite some internal style-heavy reads
- `SaMST` shows style activation but poor structural behavior

### 3.2 Current four-way blind VLM board

Authoritative source:

- [vlm_lbmpsv2_vs_seedream_vs_samst_vs_samam_20260610_snapshot6.method_summary.md](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_lbmpsv2_vs_seedream_vs_samst_vs_samam_20260610_snapshot6.method_summary.md)

Snapshot size:

- `205` valid cases

Current four-way ranking:

| Method | Wins | WinRate | StyleWins | StructWins | ArtifactWins | MeanStyle | MeanStruct | MeanArtifact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Seedream_repaired750` | `116` | `0.566` | `113` | `108` | `96` | `4.185` | `4.444` | `4.151` |
| `SaMAM_2250` | `89` | `0.434` | `88` | `92` | `108` | `3.912` | `4.371` | `4.302` |
| `SaMST_e15` | `0` | `0.000` | `4` | `0` | `0` | `3.444` | `3.312` | `2.980` |
| `LBM-PS-v2_e13` | `0` | `0.000` | `0` | `5` | `1` | `1.683` | `1.941` | `1.610` |

Immediate read:

- `Seedream` still wins overall
- `SaMAM_2250` is the strongest non-Seedream external baseline
- `SaMAM_2250` is better than `Seedream` on the cleaner-image axes:
  - artifact wins
  - mean artifact control
- `SaMST_e15` is style-active but not overall competitive
- `LBM-PS-v2_e13` is clearly not paper-facing

### 3.3 Non-CLIP style classifier probe

Authoritative source:

- [distinct5_nonclip_style_probe.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/distinct5_nonclip_style_probe.csv)

Current relevant rows:

| Method | All-Pairs Target Acc | Transfer Target Acc | Identity Source Acc | Transfer Target Prob | Transfer Source Prob | Margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `LBM-K_e1` | `0.3080` | `0.1667` | `0.8733` | `0.1524` | `0.7069` | `-0.5544` |
| `LBM-Knee_e13` | `0.3427` | `0.2367` | `0.7667` | `0.2123` | `0.5633` | `-0.3511` |
| `LBM-PS-v2_e13` | `0.2893` | `0.2717` | `0.3600` | `0.2696` | `0.3064` | `-0.0368` |
| `SaMST_e15` | `0.3867` | `0.2483` | `0.9400` | `0.2405` | `0.5953` | `-0.3548` |
| `Seedream_repaired750` | `0.4920` | `0.3783` | `0.9467` | `0.3758` | `0.4774` | `-0.1016` |

Immediate read:

- `Seedream` is still strongest
- `LBM-PS-v2` can raise target probability without producing globally strong images
- this is exactly the kind of `CLIP/non-CLIP mismatch` that motivated the newer evaluation stack

## 4. Main Internal Paper Anchors

Authoritative source:

- [aaai2027_results_master.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/aaai2027_results_master.csv)

Current important paper-facing internal anchors:

| Experiment | Model Setting | Selected Result |
| --- | --- | --- |
| `LBM F_e1` | `distinct5_512_ema_variant_f_annealed_prototype_ot_queue_e3`, batch `44`, epoch `1` | transfer `0.6644 / 0.3245`, full `0.6969 / 0.3186` |
| `LBM H_e1` | `distinct5_512_ema_variant_h_hard_explore_queue_e3`, batch `44`, epoch `1` | transfer `0.6653 / 0.3281`, full `0.6974 / 0.3213` |
| `LBM H_e2` | same `H` family, batch `44`, epoch `2` | transfer `0.6684 / 0.3561`, full `0.6994 / 0.3484` |
| `LBM K_e1` | `distinct5_512_ema_variant_k_content_adaptive_vq_queue_e3`, batch `44`, epoch `1` | transfer `0.6712 / 0.3723`, full `0.7010 / 0.3623` |
| `LBM K_longer_e5` | `longer_train_k_seed42_b44_e8`, batch `44`, epoch `5` | transfer `0.6670 / 0.3588`, full `0.6969 / 0.3504`; closed negative |

Immediate read:

- `F/H/K` gave the old compact tradeoff anchors
- `K` is not the current answer anymore
- those anchors are still useful as historical comparators, not as current frontier claims

## 5. External Pixel / Latent Baselines

Authoritative sources:

- [aaai2027_results_master.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/aaai2027_results_master.csv)
- [2026-06-04-distinct5_same_cost_inventory.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-04-distinct5_same_cost_inventory.csv)

Current relevant baseline rows:

| Experiment | Model Setting | Selected Result | Current Reading |
| --- | --- | --- | --- |
| `SaMST e5` | pixel baseline, Distinct5-512, epoch `5` | transfer `0.6989 / 0.6335`, full `0.7276 / 0.6271` | same-cost-ish baseline; style-active but structurally weak |
| `SaMST e15` | pixel baseline, Distinct5-512, epoch `15` | transfer `0.6957 / 0.6319`, full `0.7247 / 0.6255` | saturated pixel baseline |
| `SaMAM step_2250` | pixel baseline, Distinct5-512, remote WSL segmented lane | transfer `0.5523 / 0.3605`, full `0.5811 / 0.3538` | manuscript-safe same-cost anchor; weak style on CLIP, but visually strong on cleanliness |
| `SaMAM step_3000` | same run, later audit point | transfer `0.6646 / 0.3271`, full `0.6978 / 0.3221` | audit-only stronger later point; not manuscript-safe boundary |
| `SaMST-latent b50_fast` | latent baseline, same-cost fast retained point | transfer `0.6104 / 0.7296` | collapse / not paper-safe |
| `SaMST-latent b300_fast` | latent baseline, later same-cost retained point | transfer `0.6104 / 0.7296` | operationally identical collapse |
| `SaMST-latent convergence batch1050_fast` | latent convergence retained point | transfer `0.6820 / 0.8318` | still structurally unusable |
| `SaMAM-latent convergence step1500_fast` | latent convergence retained point | transfer `0.6547 / 0.1635`, full `0.6920 / 0.1634` | numerically unusual; not yet a paper-facing promoted non-CLIP winner |

Immediate read:

- pixel `SaMAM` is much more important than first expected
- pixel `SaMST` is useful mainly as a style-aggressive negative control
- latent baselines are still not paper-facing winners under the current broader reading

## 6. Inmortal Mechanism Sweep

Authoritative source:

- [aaai2027_inmortal_results_master.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/aaai2027_inmortal_results_master.csv)

Current most important mechanism families:

| Family | Setting | Selected Result | Current Reading |
| --- | --- | --- | --- |
| `K_spatial` | spatial kinetic split, batch `16`, epoch `6` | transfer `0.6615 / 0.3504` | kinetic-only, no ceiling lift |
| `K_manifold` | manifold-adaptive kinetic, batch `16`, epoch `6` | transfer `0.6629 / 0.3349` | slightly cleaner, still not enough |
| `K_spectral_b12` | spectral kinetic, batch `12`, epoch `2` | transfer `0.6788 / 0.3637` | stable but not frontier |
| `XPred_Barycenter` | endpoint/barycentric target, batch `16`, epoch `5` | transfer `0.7255 / 0.6149` | strong style, poor LPIPS |
| `XPred_Kmanifold` | x-pred + K-manifold, batch `32`, epoch `7` | transfer `0.7259 / 0.6863` | stronger than bary, still too destructive |
| `XPred_Kmanifold_Pattn_base` | add cross-attn proximal, batch `16`, epoch `6` | transfer `0.7289 / 0.6370` | first promoted style frontier |
| `XPred_Kmanifold_Pattn_longer` | longer training, batch `16`, epoch `11` | transfer `0.7289 / 0.6211` | style flat, LPIPS modestly better |
| `XPred_Kmanifold_Pattn_Stokes_finetune` | late Stokes, batch `16`, epoch `13` | transfer `0.7274 / 0.6033` | better LPIPS frontier, current important internal style/LPIPS tradeoff |
| `XPred_Kmanifold_Pattn_Stokes002_finetune` | weaker Stokes, batch `16`, epoch `13` | transfer `0.7307 / 0.6183` | raw style peak, but LPIPS regresses |
| `AnisoStokesQueue e13` | queue + aniso+stokes, batch `8`, epoch `13` | transfer `0.7102 / 0.4603`, full `0.7303 / 0.4559` | strong low-LPIPS successor, not strongest style |
| `Clamp reseed` | hard clamp recovery, batch `8`, epoch `3` | transfer `0.7022 / 0.4867` | positive closure, but not headline frontier |
| `ClampRelease reseed` | clamp + release, batch `8`, epoch `3` | transfer `0.7007 / 0.4754` | positive incremental recovery |
| `Hold4Mid` | clamp-hold-release mid anchor, batch `8`, epoch `8` | transfer `0.6679 / 0.2877` | geometry anchor; very low LPIPS, low style |
| `Hold4SlowMid` | slower hold4mid, batch `8`, epoch `12` | transfer `0.6673 / 0.2898` | near-tie negative versus hold4mid |
| `stylesig` | same structure leash + target-specific style losses, batch `8`, live | first eval `epoch_0001 -> epoch_0005`: style `0.7046 -> 0.7094`, LPIPS `0.4513 -> 0.4915` | currently looks like style-up / structure-down, not a clean rescue |

Immediate read:

- we already know how to create more style pressure
- we already know how to create lower-LPIPS geometry anchors
- we still do **not** know how to obtain `Seedream`-level target-specific style
  while keeping `SaMAM`-level cleanliness

## 7. Current Image Failure Modes

Relative to strong baselines, the current internal images still fail in a fairly consistent way.

### 7.1 Relative to `Seedream`

Main gap:

- not enough target-style specificity

Typical failure:

- the image looks generically painterly
- but not enough like the actual target style family

### 7.2 Relative to `SaMAM`

Main gap:

- not clean enough

Typical failure:

- weaker edge stability
- softer local geometry
- more visible local texture blending artifacts
- more “stylized but slightly broken” images

### 7.3 Relative to `SaMST`

Main gap:

- we are safer, but we have not turned safety into enough target-specific identity

Typical failure:

- less collapsed than `SaMST`
- but still not specific enough to beat strong external baselines visually

## 8. Current Theory Conclusion

The evidence says the main unsolved problem is not:

- `more style`

It is:

- more target-specific style
- without paying away structure and cleanliness

That means:

- a branch that raises internal style metrics but still loses the visual board
  to `SaMAM` is not solving the real paper-facing problem

## 9. Current Next-Step Plan

1. Keep `stylesig` running through more eval points.
   - it now has real eval evidence, so we should close the early curve properly
   - but the current curve is not yet promising

2. Keep local evaluation on frozen snapshots.
   - live `VLM` remains useful
   - but transport/runtime failures now exist
   - current error mix is mostly:
     - `ConnectionAbortedError 10053`
     - `HTTP 500`

3. Keep `Seedream + SaMAM + SaMST` as the mandatory external board.

4. Promotion rule for the next internal family:
   - it must improve target-specific style enough to challenge `Seedream`
   - without losing the cleaner-image advantages that currently make `SaMAM`
     such a strong external anchor

## 10. Bottom Line

The phase is no longer ambiguous.

What is already known:

- `Seedream` is still the overall visual ceiling
- `SaMAM` is a strong and serious external comparator
- `SaMST` is a useful style-aggressive negative control
- current internal branches can move style or LPIPS, but still do not close the
  real target-style-identity gap

What remains unsolved:

- how to produce images that are:
  - as target-specific as `Seedream`
  - while staying as clean and structurally usable as `SaMAM`
