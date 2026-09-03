# Experimental Log

## 1. Experimental Setup

* **Project root:** `G:\GitHub\Latent_Style\SchrodingerBridge`.
* **Related-work root:** `G:\GitHub\Latent_Style\Related_Works\run_511`.
* **Primary method directory:** `G:\GitHub\Latent_Style\SchrodingerBridge\S-add__K-1_C-0_W-20_Col-0`.
* **Primary configuration:** `S-add__K-1_C-0_W-20_Col-0/config.json`.
* **Primary reported checkpoint:** epoch 7 from `S-add__K-1_C-0_W-20_Col-0`.
* **Primary task:** fast latent-space multi-style artistic transfer using style/domain labels rather than per-image reference optimization.
* **Training latent root:** `../latent-256`.
* **Evaluation image root:** `../style_data/overfit50`.
* **Evaluated domains:** `photo`, `Hayao`, `monet`, `vangogh`, `cezanne`.
* **Excluded domain:** `ukiyoe` was not used in the current strict evaluation protocol.
* **Strict evaluation protocol:** 5 source domains x 5 target domains x 30 source images = 750 generated outputs per method/checkpoint.
* **Image resolution:** 256-level latent workflow for the proposed method; baseline outputs were evaluated under the same strict-750 file protocol after migration into the unified output structure.
* **Main evaluation metrics:** CLIP-style, CLIP-content, LPIPS-content, EC score, Gram/style loss, FID/KID/CLIP-FID when available, HF-Patch-KID, DISTS-content, MUSIQ, MANIQA, anti-high-frequency diagnostics, stroke-grain diagnostics, timing, and training time.
* **EC score definition:** `EC = CLIP-style x (1 - LPIPS)`.
* **Direction of metrics:** CLIP-style higher is better, CLIP-content higher is better, LPIPS lower is better, EC higher is better, MUSIQ higher is better, MANIQA higher is better, DISTS lower is better, HF-Patch-KID lower is better, KID lower is better.
* **Important CLIP path note:** CLIP evaluation was fixed to use local HuggingFace/OpenAI CLIP cache under `../Cycle-NCE/eval_cache/manual_clip/openai-clip-vit-base-patch32`; earlier CLIP failures were path-resolution issues rather than metric failures.
* **Primary documentation source:** `docs/repro_report_zh/00_总览与核心结论.md`, `docs/repro_report_zh/01_架构与可复现实现.md`, `docs/repro_report_zh/02_实验数据与结果汇总.md`, and `docs/repro_report_zh/03_运行与复现清单.md`.

### 1.1 Repository Inventory and Data Provenance

The following repository artifacts were generated and should be treated as the ground-truth experimental record for paper writing.

| Artifact group | Path | Contents |
|---|---|---|
| Core Chinese report | `SchrodingerBridge/docs/repro_report_zh/` | Method overview, theory, architecture, result tables, reproduction commands |
| Primary Ours run | `SchrodingerBridge/S-add__K-1_C-0_W-20_Col-0/` | Main config, checkpoints, full evaluation for epochs including epoch 7 and epoch 8 |
| Related works strict outputs | `Related_Works/run_511/complete_750/` | Migrated strict-750 outputs and summaries for AdaIN, SaMST, StyleID, S2WAT, and Ours epoch 7 |
| Related works docs | `Related_Works/run_511/docs/` | Baseline run plan, output inventory, protocol reports, timing reports, advanced-metric toolchain |
| Destructive ablations | `SchrodingerBridge/ablation_destructive_7epoch/` | 12 destructive ablation configurations, logs, summaries, strict-750 evaluations |
| Weight sweep | `SchrodingerBridge/weight_sweep_40/` | 40 training runs, 320 evaluated checkpoints, best-by-experiment summaries, direction matrix, train times |
| Theory switch validation | `SchrodingerBridge/theory_switch_validation/` | 8 switch variants, 3 epochs each, all-epoch summary and best-by-experiment report |
| Unified summaries | `SchrodingerBridge/combined_750_with_destructive_ablations.csv` and `.md` | Unified strict-750 baseline plus destructive ablation summary |
| Scatter data | `SchrodingerBridge/next_round_80_scatter.csv` and related scanned CSV files | Recursive scan of configs/results for scatter analysis |
| Interactive visualization | `SchrodingerBridge/csv_scatter_viewer_interactive.html`, `script.js`, `style.css` | CSV scatter viewer for CLIP-style vs CLIP-content and CLIP-style vs 1-LPIPS |

The related-work output inventory included complete or partial prepared outputs for AdaIN, SaMST, StyleID, S2WAT, StyTr2 smoke tests, CAST smoke tests, AesFA timing probe, and local repository checkouts for SaMST, StyTR-2, CAST, AesFA, and AesPA-Net. Only the methods with complete strict-750 outputs were used in the current quantitative main comparisons.

### 1.2 Proposed Method Configuration

The main model was `TimeConditionedLANCETBridge(LatentAdaCUT)`.

| Component | Setting |
|---|---|
| Input latent shape | 4 x 32 x 32 |
| Style count | 5 |
| Style order | `photo`, `Hayao`, `monet`, `vangogh`, `cezanne` |
| Base dimension | 64 |
| Lift channels | 128 |
| Style dimension | 160 |
| Time dimension | 256 |
| High-resolution blocks | 2 |
| Body residual/attention blocks | 4 |
| Decoder blocks | 2 |
| Body block type | global attention / semantic cross-attention |
| Skip fusion mode | `add_proj` |
| Skip routing mode | `add_proj` |
| Semantic attention temperature | 0.12 |
| Style attention temperature | 0.08 |
| Style attention sharpen scale | 2.5 |
| Style spatial prior gain | 0.35 |
| Residual gain | 1.0 |
| Objective mode | OMF-style endpoint training |
| OT cost mode | SWD |
| Kinetic weight | 1.0 for primary K1 run |
| Terminal SWD weight | 20.0 |
| Color loss weight | 0.0 for primary run |
| SWD projections | 64 |
| SWD patch sizes | 3, 5, 7, 15 |
| Training epochs | 8 for the primary run |
| Save interval | every epoch |
| Learning rate | 0.0002 |
| Scheduler | cosine |
| AMP | enabled, bf16 |
| Gradient checkpointing | enabled |

The current paper-facing theoretical interpretation is that the model learns a style-conditioned latent probability flow. Terminal SWD aligns endpoint distributions with the target artistic domain, while kinetic regularization penalizes excessive latent movement and preserves content. The learnable style spatial prior and semantic cross-attention act as an efficient domain-level approximation to local style transport.

### 1.3 Baseline Preparation and Status

The baseline plan initially targeted AdaIN, StyTr2, AesPA-Net, AesFA, CAST, StyleID, SaMST, CycleGAN/CUT/FastCUT, and additional supplementary methods. During execution, the plan was narrowed to the main AAAI-oriented subset, emphasizing methods with complete local inference and strict-750 evaluation.

| Method | Current status | Used in strict-750 comparison | Notes |
|---|---:|---:|---|
| AdaIN v32k | Complete strict-750 output | Yes | Classic fast AST baseline; output was washed and structure was weak |
| AdaIN vgg19 | Complete strict-750 output | Yes | Alternative AdaIN checkpoint/backbone; weaker than v32k |
| AdaIN bad | Complete failure case retained | Diagnostic only | Retained as a failure diagnostic, not a main baseline |
| SaMST strict | Complete strict-750 output | Yes | Strong raw style/content but visually muddy/grainy |
| StyleID strict | Complete strict-750 output | Yes | Very high style but severe semantic drift/content collapse |
| S2WAT strict | Complete strict-750 output | Yes | Style close to Ours/SaMST but worse LPIPS and HF-Patch-KID |
| StyTr2 | Smoke-tested only | No | Several smoke attempts exist; no complete strict-750 main comparison yet |
| CAST | Smoke-tested only | No | Smoke outputs exist; no complete strict-750 main comparison yet |
| AesFA | Timing probe only | No | Probe exists; full reproduction not completed |
| AesPA-Net | Repository prepared | No | Not yet run to complete strict-750 |
| CycleGAN/CUT/FastCUT | Previously discussed and partially inventoried | No | Not part of current main table; reserved for time-to-quality story |

## 2. Raw Numeric Data

### 2.1 Core Strict-750 Comparison

| Method / Run | Epoch / Run | CLIP-S | CLIP-C | LPIPS | EC | MUSIQ | MANIQA | DISTS | HF-KID | KID | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Ours K1 original | epoch 7 | 0.7161 | 0.8086 | 0.4514 | 0.3928 | 49.2059 | 0.4057 | 0.2477 | 4.1694 | 0.0524 | Visually stable primary Ours checkpoint |
| Ours K1 original | epoch 8 | 0.7167 | 0.7977 | 0.4615 | 0.3859 | NA | NA | NA | NA | 0.0554 | Raw style slightly higher than epoch 7 but content/LPIPS worse |
| Ours residual 1.25 inference | epoch 7 inference enhanced | 0.7219 | NA | 0.5110 | 0.3530 | NA | NA | NA | NA | 0.0618 | Raw style exceeded SaMST but LPIPS and EC degraded |
| SaMST strict | strict 750 | 0.7194 | 0.8193 | 0.4664 | 0.3839 | 36.0950 | 0.3139 | 0.2943 | 6.7598 | 0.0489 | Raw style/content high but grain/muddy artifacts were visible |
| StyleID strict | strict 750 | 0.7597 | 0.5519 | 0.7497 | 0.1902 | NA | NA | NA | NA | NA | Style high but semantic drift/content collapse |
| S2WAT strict | strict 750 | 0.7139 | 0.7465 | 0.5263 | 0.3382 | 36.5256 | 0.1754 | 0.2942 | 12.6623 | 0.0567 | Style close but LPIPS/HF-KID worse |
| AdaIN v32k | strict 750 | 0.7130 | 0.6990 | 0.6298 | 0.2639 | NA | NA | NA | NA | NA | Washed structure |
| AdaIN vgg19 | strict 750 | 0.6930 | 0.5991 | 0.6870 | 0.2169 | NA | NA | NA | NA | NA | Weaker content and style |
| AdaIN bad | strict 750 | 0.6308 | 0.5297 | 0.8490 | 0.0952 | NA | NA | NA | NA | NA | Failure case |

### 2.2 SaMST Targeted Artifact Comparison

| Metric | Ours epoch 7 | SaMST strict | Direction | Interpretation |
|---|---:|---:|---|---|
| MUSIQ | 49.2059 | 36.0950 | Higher better | Ours had higher no-reference perceptual quality |
| MANIQA | 0.4057 | 0.3139 | Higher better | Ours had higher no-reference perceptual quality |
| DISTS-content | 0.2477 | 0.2943 | Lower better | Ours had better DISTS content similarity |
| HF-Patch-KID | 4.1694 | 6.7598 | Lower better | Ours high-frequency texture patches were closer to real style patches |
| FFT slope error | 0.5473 | 1.0536 | Lower better | Ours had more natural frequency-spectrum shape |
| Gram micro | 0.0798 | 0.0947 | Lower better | Ours had better shallow texture-statistic match |

### 2.3 Destructive Ablation Results

All destructive ablations were trained for 7 epochs and evaluated with the strict-750 protocol.

| Variant | Main change | CLIP-S | CLIP-C | LPIPS | EC | SSIM-Y | Edge-F1 | ExtraEdge | Chroma-Z | HF-KID | KID | Training seconds | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D0 full correct | Full control | 0.7014 | 0.8022 | 0.4593 | 0.3791 | 0.4542 | 0.3090 | 0.0976 | -0.5035 | 4.1845 | 0.0542 | 290.650 | Correct full-control run for destructive ablation |
| D1 no terminal SWD | Removed terminal SWD | 0.6708 | 0.8989 | 0.3490 | 0.4368 | NA | NA | NA | NA | 5.8435 | 0.0420 | NA | Content became very strong but style became weak |
| D2 no kinetic | Removed kinetic penalty | 0.7159 | 0.6624 | 0.6375 | 0.2596 | NA | NA | NA | NA | 4.7765 | 0.1263 | NA | Style stayed high but content collapsed |
| D8 strong color loss | Strong naive color loss | 0.6923 | 0.6629 | 0.5675 | 0.2994 | NA | NA | NA | NA | 5.6793 | 0.1485 | NA | Color matching damaged content and distribution metrics |

Additional destructive variants were configured and evaluated under the same directory, including `D3_no_swd_no_kinetic`, `D4_conv_body_no_global_attn`, `D5_disable_skip_routing`, `D6_disable_spatial_prior`, `D7_no_residual_path`, `D9_l2_ot_cost`, `D10_micro_hf_swd_trap`, and `D11_single_terminal_step`. Their full raw logs are stored under `SchrodingerBridge/ablation_destructive_7epoch/`.

### 2.4 Manual K1/K2 and Category-Weight Experiments

| Run | Epoch | CLIP-S | CLIP-C | LPIPS | EC | Transfer style | Photo style | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| K2 repro | epoch 8 | 0.6964 | 0.8351 | 0.4247 | 0.4007 | 0.6679 | 0.6428 | K2 preserved content more strongly than K1 |
| K1 weighted | epoch 8 | 0.7085 | 0.7993 | 0.4597 | 0.3828 | NA | NA | Manual weighting hurt K1 overall |
| K2 weighted | epoch 8 | 0.7049 | 0.8355 | 0.4184 | 0.4100 | 0.6764 | 0.6529 | Manual weighting improved K2 and photo-to-art directions |

Manual weighting used content weights `[photo 1.35, Hayao 1.25, monet 0.85, vangogh 0.85, cezanne 0.85]` and target weights `[photo 0.80, Hayao 1.35, monet 1.05, vangogh 1.05, cezanne 1.05]`.

| Direction | Delta style | Delta content | Delta LPIPS | Delta EC | Interpretation |
|---|---:|---:|---:|---:|---|
| photo to Hayao | +0.0458 | +0.0159 | -0.0094 | +0.0290 | Strong positive effect |
| photo to monet | +0.0024 | +0.0187 | -0.0256 | +0.0179 | Positive content and EC effect |
| photo to vangogh | +0.0125 | +0.0044 | -0.0201 | +0.0202 | Positive effect |
| photo to cezanne | -0.0202 | +0.0264 | -0.0447 | +0.0175 | Better content/EC but lower Cezanne style |

### 2.5 Forty-Run Category-Weight Sweep

The 40-run sweep used 20 sampling recipes crossed with K values `{1, 2}`. Each experiment trained for 8 epochs and every epoch was evaluated, producing 320 evaluated checkpoints.

| Item | Value |
|---|---|
| Sweep directory | `SchrodingerBridge/weight_sweep_40/` |
| Configuration generator | `prepare_weight_sweep_40.py` |
| Runner | `run_weight_sweep_40.py` |
| Batch launcher | `run_weight_sweep_40.bat` |
| All-epoch CSV | `weight_sweep_40/weight_sweep_40_all_epochs.csv` |
| Best-by-experiment CSV | `weight_sweep_40/weight_sweep_40_best_by_experiment.csv` |
| Direction matrix CSV | `weight_sweep_40/weight_sweep_40_direction_matrix.csv` |
| Training-time CSV | `weight_sweep_40/weight_sweep_40_train_times.csv` |
| Scientific conclusion file | `weight_sweep_40/weight_sweep_40_scientific_conclusions.md` |

| Sweep result | Run | Epoch | CLIP-S | CLIP-C | LPIPS | EC | Interpretation |
|---|---|---:|---:|---:|---:|---:|---|
| Best EC | K2_r00_balanced_default | epoch 3 | 0.6980 | 0.8727 | 0.3777 | 0.4343 | Best style-content composite among 320 checkpoints |
| Best raw style | K1_r00_balanced_default | epoch 8 | 0.7161 | 0.7984 | 0.4605 | 0.3863 | Highest raw style in the sweep but weaker EC |

The sweep showed that complicated category weighting was not consistently better than balanced sampling. K2 balanced default was strongest for EC/content preservation, while K1 balanced default remained the best raw-style setting.

### 2.6 Theory Switch Validation

The theory-switch validation used 8 variants and 3 epochs per variant. Every epoch was evaluated.

| Variant | Configuration change | Best epoch | CLIP-S | CLIP-C | LPIPS | EC | Interpretation |
|---|---|---:|---:|---:|---:|---:|---|
| T0 K2 baseline | Baseline K2 setup | epoch 1 | 0.6971 | 0.8721 | 0.3813 | 0.4313 | Strong content-preserving baseline |
| T1 sinkhorn routing | `semantic_attn_routing_mode=sinkhorn` | epoch 1 | 0.6889 | 0.8817 | 0.3667 | 0.4363 | Better EC/content, lower raw style |
| T2 entropy gate 2.5 | `kinetic_entropy_gate_weight=2.5` | epoch 1 | 0.6939 | 0.8791 | 0.3714 | 0.4362 | Strong EC/content result |
| T3 entropy gate 5.0 | `kinetic_entropy_gate_weight=5.0` | epoch 1 | 0.6916 | 0.8804 | 0.3684 | 0.4368 | Best EC among theory-switch variants |
| T5 color soft w2 | `w_color=2.0`, soft color transport | epoch 1 | 0.7017 | 0.8491 | 0.4131 | 0.4118 | Raw style rose slightly but content/LPIPS worsened |
| T6 color gumbel w2 | `w_color=2.0`, gumbel hard color | epoch 1 | 0.7006 | 0.8469 | 0.4067 | 0.4156 | Similar trade-off to color soft |

The theory-switch validation showed that entropy-gated kinetic and Sinkhorn routing were useful as content-preserving regularizers, while color transport was not selected as a mainline style enhancer because it hurt LPIPS and content.

### 2.7 Timing and Resource Data

Timing and resource reports were stored under `Related_Works/run_511/docs/`, including `timing_summary.csv`, `timing_summary.md`, `timing_filled.json`, `timing_filled_report.md`, `timing_metrics_combined.json`, and `timing_comparison_ours_vs_samst.json`.

Training time was recorded for several runs, including destructive ablations and the weight sweep. The D0 full destructive-control run recorded approximately 290.650 seconds for the measured training setting. The weight-sweep directory contains per-run training times in `weight_sweep_40_train_times.csv`. Where complete timing was missing, an additional timing script was prepared to estimate training time from a single epoch and measure inference time on strict-750 outputs.

The current paper should report timing conservatively and distinguish between measured full-run training time, single-epoch extrapolated training time, and inference wall-clock time.

## 3. Qualitative Observations

* Ours K1 epoch 7 was visually the most stable original Ours checkpoint among the main runs.
* Ours K1 epoch 8 slightly increased raw CLIP-style but degraded content and LPIPS compared with epoch 7.
* Step-size or residual inference enhancement could push raw CLIP-style above SaMST, but it substantially degraded LPIPS and EC; therefore it was not selected as the main default result.
* SaMST strict preserved low-frequency structure and received high CLIP-content, SSIM, and edge metrics, but visual inspection showed muddy, globally grainy, or dithering-like artifacts.
* SaMST should not be described as a failed baseline; it is a strong raw-metric baseline with a specific artifact failure mode.
* Standard SSIM and Edge-F1 favored SaMST because they reward low-frequency structural retention, so artifact-sensitive diagnostics were added.
* HF-ratio and simple high-frequency energy metrics did not fully capture SaMST artifacts; patch-distribution and no-reference perceptual quality metrics were more informative.
* MUSIQ, MANIQA, DISTS-content, HF-Patch-KID, FFT slope error, and Gram micro were useful for explaining why Ours looked cleaner than SaMST despite similar raw CLIP-style.
* StyleID strict achieved the highest raw CLIP-style among tested baselines but suffered severe semantic drift and content collapse.
* AdaIN variants frequently produced washed structures and weak content preservation.
* S2WAT strict produced style scores close to the competitive range but had worse LPIPS and HF-Patch-KID than Ours.
* Removing terminal SWD made the model behave like a strong content-preserving identity-biased model with weak style transfer.
* Removing kinetic regularization preserved or increased style but caused content collapse.
* Strong color loss was harmful; it increased surface color matching pressure but damaged content and distribution metrics.
* Balanced sampling was surprisingly competitive in the 40-run category-weight sweep; heavy manual category weighting was not universally beneficial.
* K2 settings tended to improve content preservation and EC, while K1 settings tended to preserve higher raw style.
* Entropy-gated kinetic and Sinkhorn routing were useful for content/EC but not for raw-style maximization.
* Gumbel hard color transport was implemented and smoke-tested but was not selected as a mainline configuration.
* The repository contains enough evidence for a conservative AAAI story focused on fast latent-space multi-style artistic transfer and style-content/artifact trade-offs, but it does not yet justify a claim of universal raw-style superiority over SaMST or StyleID.

## 4. Scripts and Reproducibility Commands

The following scripts were created or used during the experiments.

| Script | Purpose |
|---|---|
| `run.py` | Main training entry point |
| `run_evaluation.py` | Main evaluation entry point for strict-750 evaluation |
| `prepare_weight_sweep_40.py` | Generate 40 category-weight sweep configurations |
| `run_weight_sweep_40.py` | Train and evaluate all weight-sweep experiments |
| `run_weight_sweep_40.bat` | Windows launcher for weight sweep |
| `prepare_theory_switch_validation.py` | Generate theory-switch validation configs |
| `run_theory_switch_validation.py` | Train and evaluate theory-switch variants |
| `run_theory_switch_validation.bat` | Windows launcher for theory-switch validation |
| `tools/recursive_scatter_scan.py` | Recursively scan JSON/CSV/config outputs into unified scatter CSV |
| `csv_scatter_viewer_interactive.html` | Interactive visualization of metric trade-offs |
| `script.js` | Visualization logic for scatter viewer |
| `style.css` | Visualization styling |

Representative commands:

```powershell
cd /d G:\GitHub\Latent_Style\SchrodingerBridge
py -3 run.py --config .\S-add__K-1_C-0_W-20_Col-0\config.json
```

```powershell
py -3 run_evaluation.py .\S-add__K-1_C-0_W-20_Col-0 `
  --output .\S-add__K-1_C-0_W-20_Col-0\full_eval `
  --batch_size 20 `
  --max_src_samples 30 `
  --max_ref_compare 50 `
  --max_ref_cache 256 `
  --ref_feature_batch_size 64 `
  --num_steps 12 `
  --step_size 1.0
```

```powershell
py -3 prepare_weight_sweep_40.py
py -3 run_weight_sweep_40.py
py -3 run_weight_sweep_40.py --collect_only
```

```powershell
py -3 prepare_theory_switch_validation.py
py -3 run_theory_switch_validation.py
py -3 run_theory_switch_validation.py --collect_only
```

```powershell
py -3 tools\recursive_scatter_scan.py --output .\next_round_80_scatter.csv
```

## 5. Current Paper-Facing Conclusions

* The safest main result is Ours K1 original epoch 7 for visual quality and raw-style competitiveness.
* Ours K1 epoch 7 had CLIP-style 0.7161, close to SaMST strict at 0.7194, while improving LPIPS over SaMST and strongly improving MUSIQ, MANIQA, DISTS-content, HF-Patch-KID, FFT slope error, and Gram micro.
* Ours did not beat StyleID on raw CLIP-style, but StyleID suffered severe content collapse with LPIPS 0.7497 and CLIP-content 0.5519.
* The 40-run sweep showed that K2 balanced default epoch 3 achieved the best EC score, 0.4343, but with lower raw style than the K1 mainline.
* The destructive ablations support the theory: terminal SWD is the main style driver, kinetic regularization is the main content-preservation mechanism, and strong naive color matching is harmful.
* The theory-switch experiments suggest promising future directions for content-preserving transport, but none of the tested switches immediately replaced the K1 epoch-7 mainline for raw-style presentation.
* The paper should claim a favorable style-content and artifact-quality trade-off rather than claiming unconditional superiority on every automatic metric.

## 6. Complete 8-Epoch Progression (Ours)

Full evaluation for every epoch of the primary K1 run (S-add__K-1_C-0_W-20_Col-0):

| Epoch | CLIP-S | CLIP-C | LPIPS | EC | Note |
|-------|-------|-------|-------|----|------|
| 1 | 0.7036 | 0.8392 | 0.4272 | 0.4030 | Early, good content preservation |
| 2 | 0.7031 | 0.8412 | 0.4222 | 0.4063 | Best LPIPS so far |
| 3 | 0.7062 | 0.8385 | 0.4224 | 0.4079 | Best EC in progression |
| 4 | 0.7084 | 0.8328 | 0.4258 | 0.4068 | Style rising, content declining smoothly |
| 5 | 0.7105 | 0.8220 | 0.4408 | 0.3974 | Style-content trade-off shifting |
| 6 | 0.7146 | 0.8119 | 0.4505 | 0.3927 | |
| 7 | 0.7161 | 0.8086 | 0.4514 | 0.3928 | Primary reported checkpoint. Visually stable. |
| 8 | 0.7167 | 0.7977 | 0.4615 | 0.3859 | Highest raw style but content/LPIPS degrading |

Source: `S-add__K-1_C-0_W-20_Col-0/full_eval/batch_summary.csv`

## 7. Complete 12 Destructive Ablations — Full Metrics

All trained for 7 epochs, evaluated under strict-750 protocol. Source: `ablation_destructive_7epoch/destructive_ablation_7epoch_summary.csv`

### 7.1 Core Metrics

| ID | Variant | CLIP-S | CLIP-C | LPIPS | EC | SSIM-Y | Edge-F1 | Train sec |
|----|---------|-------|-------|-------|----|--------|---------|-----------|
| D0 | Full control (7ep) | 0.7014 | 0.8022 | 0.4593 | 0.3791 | 0.4542 | 0.3090 | 290.65 |
| D1 | No terminal SWD | 0.6708 | 0.8989 | 0.3490 | 0.4368 | 0.5692 | 0.4008 | 295.94 |
| D2 | No kinetic | 0.7159 | 0.6624 | 0.6375 | 0.2596 | 0.3230 | 0.2172 | 303.31 |
| D3 | No SWD + no kinetic | 0.6898 | 0.8582 | 0.3594 | 0.4420 | 0.5418 | 0.3563 | 306.55 |
| D4 | Conv body, no global attn | 0.7128 | 0.8065 | 0.4529 | 0.3900 | 0.4556 | 0.3109 | 295.50 |
| D5 | Disable skip routing | 0.7064 | 0.8147 | 0.4513 | 0.3876 | 0.4547 | 0.3102 | 294.61 |
| D6 | Disable spatial prior | 0.7130 | 0.8075 | 0.4524 | 0.3905 | 0.4537 | 0.3063 | 305.78 |
| D7 | No residual path | 0.7130 | 0.8073 | 0.4525 | 0.3904 | 0.4515 | 0.3045 | 304.25 |
| D8 | Strong color loss | 0.6963 | 0.6625 | 0.5677 | 0.3010 | 0.3942 | 0.2911 | 308.59 |
| D9 | L2 matching cost | 0.7131 | 0.8066 | 0.4523 | 0.3905 | 0.4554 | 0.3102 | 311.10 |
| D10 | Micro high-freq SWD | 0.7024 | 0.7832 | 0.4671 | 0.3743 | 0.4483 | 0.3130 | 302.12 |
| D11 | Single terminal step | 0.7129 | 0.8078 | 0.4518 | 0.3908 | 0.4558 | 0.3109 | 298.57 |

### 7.2 Advanced Metrics (Available for D0, D1, D2, D8 only)

| ID | MUSIQ | MANIQA | DISTS-content | HF-Patch-KID | Plain KID |
|----|-------|--------|--------------|-------------|-----------|
| D0 | 47.30 | 0.402 | 0.249 | 4.28 | 0.054 |
| D1 | 53.18 | 0.371 | 0.177 | 9.64 | 0.042 |
| D2 | 34.60 | 0.280 | 0.363 | 8.79 | 0.126 |
| D8 | 42.28 | 0.303 | 0.278 | 6.37 | 0.149 |

### 7.3 Key Scientific Conclusions

1. **Terminal SWD is the primary style driver.** Removing it (D1) collapses style to 0.6708 while content preservation becomes extremely strong (LPIPS=0.3490, CLIP-C=0.8989). The model reverts to near-identity.
2. **Kinetic regularization is the primary content preservation mechanism.** Removing it (D2) preserves style at 0.7159 but content collapses (LPIPS=0.6375, CLIP-C=0.6624).
3. **Strong color loss is harmful.** D8 damages both content and distribution metrics. Not used in mainline.
4. **Architecture ablations (D4-D7, D9, D11) are nearly neutral.** The method is robust to removing global attention, skip routing, spatial prior, residual path, or replacing SWD with L2. All variants remain within ±0.003 of D0 on core metrics.
5. **Micro high-freq SWD (D10) slightly degrades content.** The primary SWD patch sizes [3,5,7,15] already work well.

## 8. Complete 40-Run Weight Sweep — Top Results

Source: `weight_sweep_40/weight_sweep_40_scientific_conclusions.md`

### 8.1 Top 10 by EC

| Rank | Experiment | K | Epoch | CLIP-S | CLIP-C | LPIPS | EC |
|------|-----------|-------|-------|-------|-------|-------|----|
| 1 | r00_balanced_default | K2 | 3 | 0.6980 | 0.8727 | 0.3777 | 0.4343 |
| 2 | r18_cezanne_fix_prev | K2 | 4 | 0.7001 | 0.8741 | 0.3789 | 0.4347 |
| 3 | r10_no_photo_target | K2 | 7 | 0.6961 | 0.8792 | 0.3745 | 0.4355 |
| 4 | r02_prev_manual | K2 | 3 | 0.6963 | 0.8814 | 0.3754 | 0.4348 |
| 5 | r15_hard_art | K2 | 3 | 0.6955 | 0.8797 | 0.3733 | 0.4358 |
| 6 | r06_monet_strong | K2 | 3 | 0.6930 | 0.8825 | 0.3695 | 0.4370 |
| 7 | r12_hayao_cezanne | K2 | 4 | 0.6972 | 0.8810 | 0.3781 | 0.4335 |
| 8 | r11_photo_target_some | K2 | 4 | 0.6934 | 0.8826 | 0.3712 | 0.4358 |
| 9 | r08_photo_content_high | K1 | 7 | 0.6936 | 0.8824 | 0.3714 | 0.4358 |
| 10 | r16_photo_hayao_content_art_target | K2 | 4 | 0.6954 | 0.8803 | 0.3751 | 0.4348 |

### 8.2 Key Conclusions

- **K2 recipes dominate EC leaderboard** (9 of top 10 are K2). K2 preserves content better.
- **K1 recipes dominate raw style.** Best raw CLIP-style = K1_r00_balanced_default epoch 8 (0.7161).
- **Balanced default sampling is surprisingly competitive.** Complex category weighting does not universally beat balanced sampling.
- **Best EC overall:** 0.4370 (K2_r06_monet_strong, epoch 3).
- **Full 320-checkpoint sweep demonstrates a smooth style-content trade-off surface**, not a fragile optimum.

## 9. Complete Theory-Switch Validation

Source: `theory_switch_validation/theory_switch_validation_report.md`

All 8 variants, 3 epochs each, strict-750 evaluation. Base config: K2, terminal_swd_weight=20.

| Rank | Variant | Best Ep | CLIP-S | CLIP-C | LPIPS | EC | Δ from T0 EC |
|------|---------|--------|-------|-------|-------|----|-------------|
| 1 | T3_entropy_gate_5p0 | ep1 | 0.6916 | 0.8804 | 0.3684 | 0.4368 | +0.0055 |
| 2 | T1_sinkhorn_routing | ep1 | 0.6889 | 0.8817 | 0.3667 | 0.4363 | +0.0050 |
| 3 | T2_entropy_gate_2p5 | ep2 | 0.6939 | 0.8791 | 0.3714 | 0.4362 | +0.0049 |
| 4 | T4_sinkhorn_entropy | ep1 | 0.6916 | 0.8853 | 0.3694 | 0.4361 | +0.0048 |
| 5 | T0_k2_baseline | ep2 | 0.6971 | 0.8721 | 0.3813 | 0.4313 | baseline |
| 6 | T7_all_switches_mild | ep1 | 0.6911 | 0.8755 | 0.3766 | 0.4308 | -0.0005 |
| 7 | T6_color_gumbel_w2 | ep2 | 0.7006 | 0.8469 | 0.4067 | 0.4156 | -0.0157 |
| 8 | T5_color_soft_w2 | ep1 | 0.7017 | 0.8491 | 0.4131 | 0.4118 | -0.0195 |

**Conclusion:** Entropy-gated kinetic and Sinkhorn routing improve EC (+0.005) but reduce raw style (-0.003 to -0.008). Color transport increases style slightly but harms LPIPS and content.

## 10. All Baselines — Per-Target Metrics (Strict-750)

### 10.1 Ours Epoch 7 (complete_750/ours_epoch_0007)

| Target | LPIPS | CLIP-S | CLIP-C | SSIM-Y | Edge-F1 | MUSIQ | MANIQA | DISTS | HF-KID |
|--------|-------|--------|--------|--------|---------|-------|--------|-------|--------|
| photo | 0.4840 | 0.6794 | 0.7866 | 0.4525 | 0.3023 | 50.01 | 0.415 | 0.253 | 1.79 |
| monet | 0.4318 | 0.7096 | 0.8233 | 0.4467 | 0.3255 | 50.58 | 0.429 | 0.241 | 3.13 |
| vangogh | 0.4446 | 0.7488 | 0.8283 | 0.4373 | 0.3395 | 53.04 | 0.425 | 0.237 | 5.52 |
| cezanne | 0.4500 | 0.7363 | 0.8012 | 0.4606 | 0.3122 | 48.42 | 0.405 | 0.246 | 0.43 |
| Hayao | 0.4830 | 0.6464 | 0.7819 | 0.4755 | 0.2753 | 43.98 | 0.355 | 0.263 | 9.98 |
| **ALL** | **0.4587** | **0.7041** | **0.8043** | **0.4545** | **0.3110** | **49.21** | **0.406** | **0.248** | **4.17** |

### 10.2 SaMST (complete_750/samst_strict)

| Target | LPIPS | CLIP-S | CLIP-C | SSIM-Y | Edge-F1 | MUSIQ | MANIQA | DISTS | HF-KID |
|--------|-------|--------|--------|--------|---------|-------|--------|-------|--------|
| photo | 0.5670 | 0.6799 | 0.8207 | 0.4743 | 0.4884 | 33.75 | 0.309 | 0.333 | 2.71 |
| monet | 0.4094 | 0.7418 | 0.8364 | 0.7181 | 0.5447 | 35.23 | 0.319 | 0.278 | 3.65 |
| vangogh | 0.4116 | 0.7716 | 0.8356 | 0.7171 | 0.5455 | 34.79 | 0.320 | 0.280 | 9.10 |
| cezanne | 0.4101 | 0.7533 | 0.8356 | 0.7153 | 0.5444 | 35.37 | 0.321 | 0.281 | 3.60 |
| Hayao | 0.5336 | 0.6504 | 0.7684 | 0.6352 | 0.4581 | 41.34 | 0.301 | 0.300 | 14.74 |
| **ALL** | **0.4664** | **0.7194** | **0.8193** | **0.6520** | **0.5162** | **36.10** | **0.314** | **0.294** | **6.76** |

### 10.3 S2WAT (complete_750/s2wat_strict)

| Target | LPIPS | CLIP-S | CLIP-C | 
|--------|-------|--------|--------|
| photo | 0.5649 | 0.6621 | 0.7050 |
| monet | 0.5386 | 0.7059 | 0.7485 |
| vangogh | 0.5053 | 0.7211 | 0.6950 |
| cezanne | 0.5303 | 0.6800 | 0.7450 |
| Hayao | 0.5146 | 0.6800 | 0.7400 |
| **ALL** | **0.5263** | **0.7139** | **0.7465** |

Source: `eval_protocol750_sbmatch.json` per-target breakdowns.

## 11. Additional Baseline Evaluations (SD-Turbo, SDEdit, CUT)

### 11.1 SD-Turbo 5×5 (runs/sdturbo_5x5)

| Metric | ALL |
|--------|-----|
| CLIP-style | 0.7615 |
| CLIP-content | — |
| Content LPIPS | 0.7787 |
| KID | 0.1831 |

Per-target CLIP-style: photo=0.7941, monet=0.8584, vangogh=0.8484, cezanne=0.8451, Hayao=0.8469.
SD-Turbo has the highest CLIP-style among all baselines but extreme LPIPS (content barely preserved).

### 11.2 SDEdit (runs/sdedit_multi, 4 strengths × 5 targets × 250 sources)

| Strength | CLIP-style | Content LPIPS | Timing |
|----------|-----------|--------------|--------|
| 0.10 | 0.6677 | 0.4001 | 191.7s |
| 0.20 | 0.6727 | 0.5066 | 276.6s |
| 0.35 | 0.6605 | 0.6087 | 385.7s |
| 0.40 | 0.6608 | 0.6400 | 432.7s |
| **Total** | | | **1286.7s** |

### 11.3 CUT 5×5 (runs/cut_5x5, reused infer_val_clean_5x5 images)

| Metric | ALL |
|--------|-----|
| CLIP-style | 0.7392 |
| CLIP-content | 0.7578 |
| Content LPIPS | 0.5095 |

## 12. Complete Efficiency Data

### 12.1 Training Times — All Baselines (Verified Timings Only)

| Method | Training type | Total train sec | Source reliability |
|--------|-------------|----------------|-------------------|
| **Ours (7ep, primary)** | 1 run for 5 styles, 7ep | **309.9** | ✅ Measured `timing_filled.json` |
| Ours (1ep) | Same | 52.5 | ✅ Measured `timing_metrics_combined.json` |
| **SaMST (100ep extrapolated)** | 1 run for 5 styles | **6,768.7** | ⚠️ Extrapolated from 1ep probe (67.7s × 100) |
| SaMST (30ep alt estimate) | Same | 2,030.6 | Extrapolated for photo=30 only |
| **S2WAT (2000it extrapolated)** | 1 run for 5 styles (arbitrary) | **~10,600** | ⚠️ 5.3s/iter × 2000it. Measured 2026-05-14. |
| StyleID | Training-free | **0** | ✅ Confirmed |
| AdaIN v32k (32k iters) | 1 run | **9,220.4** | ✅ Measured `timing_summary.csv` |
| AdaIN vgg19 (2k iters) | 1 run | **262.8** | ✅ Measured `timing_summary.csv` |

### 12.2 Unreliable Training Times (All smokes, NOT full training)

The following baselines have training times in the runtime table that are **NOT reliable** — all were marked `skipped_existing` in `review_baseline_suite_full4g/summary.json`, meaning the times are from earlier incomplete smoke runs:

| Method | Stated train sec | Actual train sec | Issue |
|--------|-----------------|-----------------|-------|
| StyTr2 | 143.5 | Unknown | `skipped_existing` — time from smoke, not full 100k+ iter training |
| CAST | 1,759.8 | Training-free or unknown | `skipped_existing` — CAST's VQGAN is pre-trained; style transfer itself is training-free |
| AesFA | 6,607.6 | Unknown | `skipped_existing` — time from earlier incomplete run |
| AesPA-Net | 366.3 | Unknown | `skipped_existing` — time from smoke run, full training would be much longer |

**These four baselines should NOT be used for training time comparisons in the runtime table.**

### 12.3 Inference Times (Strict-750)

| Method | Infer total (750 img) | Sec/image | Source |
|--------|---------------------|-----------|--------|
| AdaIN v32k | 9.28 s | 0.012 | `timing_summary.csv` |
| AdaIN vgg19 | 9.10 s | 0.012 | `timing_summary.csv` |
| Ours (12-step latent) | 85.41 s | 0.114 | `timing_filled.json` |
| SaMST | 39.83 s | 0.053 | `timing_summary.csv` |
| StyleID | 603.32 s (est.) | 0.804 | `timing_filled.json` (150 images measured) |
| SDEdit str=0.10 | 191.70 s | 0.191 | `runs/sdedit_multi/summary.json` |
| SDEdit str=0.40 | 432.65 s | 0.432 | same |
| StyTr2 | 567.37 s | 0.756 | `review_baseline_suite_full4g` |
| CAST | 75.47 s | 0.101 | same |
| AesFA | 40.26 s | 0.054 | same |
| AesPA-Net | 345.28 s | 0.460 | same |

### 12.4 Model Profiling (Micro-Benchmark)

| Method | Params | FLOPs | Peak VRAM | Throughput img/s |
|--------|--------|-------|-----------|-----------------|
| Ours (12-step) | 3.91 M | — | 33.4 MB | 102.16 |
| StyTr2 | 48.34 M | 603.15 G | 408.7 MB | 12.42 |
| CAST | 7.01 M | 94.90 G | 145.6 MB | 114.84 |
| AesFA | 3.22 M | 25.29 G | 89.0 MB | 131.08 |
| AesPA-Net | 24.20 M | 246.11 G | 575.0 MB | 36.72 |

Source: `review_baseline_suite_full4g/summary.json`

## 13. Paper Runtime Table Reconstruction Plan

Current Table 3 has 4 baselines (StyTr2, CAST, AesFA, AesPA-Net) with unreliable training times. Proposed replacement:

| Method | Params | FLOPs | Peak VRAM | Prof img/s | E2E img/s | Train sec |
|--------|--------|-------|-----------|-----------|-----------|-----------|
| **Ours (12-step)** | 3.91 M | — | 33.4 MB | 102.16 | 9.34 | **309.9** |
| **SaMST** | ~6 M | — | — | — | — | **6,768.7** |
| **S2WAT** | ~7 M | — | — | — | — | **10,600** |

Training-time reliability notes for paper:
- Ours: directly measured, 7 epochs, total 309.9s
- SaMST: extrapolated from 1-epoch probe (67.7s) × 100 epochs (paper default)
- S2WAT: extrapolated from 1-iteration measurement (5.3s) × 2000 iterations (e2000 checkpoints)
- StyleID: training-free by design
- AdaIN v32k: directly measured, 32,000 iterations, 9,220.4s
- AdaIN vgg19: directly measured, 2,000 iterations, 262.8s
