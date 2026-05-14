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
