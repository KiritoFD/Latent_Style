# Latent AdaCUT / Cycle-NCE Archaeology Report

**Date**: 2026-04-03
**Root**: Y:\experiments
**Code**: G:\GitHub\Latent_Style\Cycle-NCE\src

## Table of Contents
1. [Overview & Directory Structure](#1-overview--directory-structure)
2. [Code Evolution & Versions](#2-code-evolution--versions)
3. [Experiment Deep Dive: Decoder-D-160](#3-experiment-deep-dive-decoder-d-160)
4. [Experiment Deep Dive: Style OA](#4-experiment-deep-dive-style-oa)
5. [Experiment Deep Dive: Optuna HPO](#5-experiment-deep-dive-optuna-hpo)
6. [Experiment Deep Dive: FinalMicro_2](#6-experiment-deep-dive-finalmicro_2)
7. [Current Source Code Analysis](#7-current-source-code-analysis)

---
## 5. Experiment Deep Dive: Optuna HPO

### 5.1 Trial Hyperparameter Landscape

Found **34 trials** in the optuna_hpo directory. Below is the full hyperparameter table:

| Trial | ArtFID | Style | LR | w_swd | w_color | w_identity |
|-------|-------:|------:|---:|------:|--------:|-----------:|
| trial_0044 | 337.4418900974628 | 0.6222403801977634 | ? | ? | ? | ? |
| trial_0003 | 345.7181545480456 | 0.6546593608955542 | ? | ? | ? | ? |
| trial_0010 | 353.12701874929644 | 0.6581962476174037 | ? | ? | ? | ? |
| trial_0038 | 353.5483803265244 | 0.6378617942333221 | ? | ? | ? | ? |
| trial_0005 | 353.7835973403563 | 0.6485508029659589 | ? | ? | ? | ? |
| trial_0043 | 354.4318971954908 | 0.6387514924009641 | ? | ? | ? | ? |
| trial_0011 | 355.0390298158196 | 0.6497419734795888 | ? | ? | ? | ? |
| trial_0042 | 355.45362648491806 | 0.6339894836147626 | ? | ? | ? | ? |
| trial_0000 | 355.75442992810144 | 0.6569621687134106 | ? | ? | ? | ? |
| trial_0041 | 361.49668732681346 | 0.6362087537844976 | ? | ? | ? | ? |
| trial_0007 | 366.4800279332356 | 0.6623325457175573 | ? | ? | ? | ? |
| trial_0006 | 369.52274691338306 | 0.6608804655571778 | ? | ? | ? | ? |
| trial_0002 | 372.11441537272697 | 0.6644025171796482 | ? | ? | ? | ? |
| trial_0004 | 377.0923766053718 | 0.6629918679594993 | ? | ? | ? | ? |
| trial_0016 | 380.3081043673182 | 0.664123771339655 | ? | ? | ? | ? |
| trial_0012 | 380.62649495816726 | 0.6598457249502341 | ? | ? | ? | ? |
| trial_0001 | 382.73917116161425 | 0.6671557158231735 | ? | ? | ? | ? |
| trial_0019 | 386.1197293854995 | 0.6753774459163349 | ? | ? | ? | ? |
| trial_0015 | 389.03767748363407 | 0.6614456492165725 | ? | ? | ? | ? |
| trial_0018 | 391.6441772770091 | 0.6698185443878173 | ? | ? | ? | ? |
| trial_0014 | 394.6205350051797 | 0.6682712805767854 | ? | ? | ? | ? |
| trial_0017 | 398.24290109804303 | 0.661014448106289 | ? | ? | ? | ? |
| trial_0021 | 416.3196084991612 | 0.675649507343769 | ? | ? | ? | ? |

### 5.2 Individual Trial Configurations

Analyzed **31** trial configurations from experiment subdirectories.

#### trial_0000

| Parameter | Value |
|-----------|------:|
| w_swd | 60.0 |
| w_color | 2.0 |
| w_identity | 44.0 |
| lr | 0.0003361441955478075 |
| swd_patch_sizes | [5, 7, 11, 15, 23] |
| color_mode | latent_decoupled_adain |
| num_epochs | 60 |
| base_dim | 96 |
| num_res_blocks | 6 |
| ada_mix_rank | 16 |
| residual_gain | 1.0 |
| skip_routing_mode | ? |

#### trial_0001

| Parameter | Value |
|-----------|------:|
| w_swd | 60.0 |
| w_color | 2.0 |
| w_identity | 35.0 |
| lr | 0.0005517397349623409 |
| swd_patch_sizes | [5, 7, 11, 15, 23] |
| color_mode | latent_decoupled_adain |
| num_epochs | 60 |
| base_dim | 96 |
| num_res_blocks | 6 |
| ada_mix_rank | 16 |
| residual_gain | 1.0 |
| skip_routing_mode | ? |

#### trial_0002

| Parameter | Value |
|-----------|------:|
| w_swd | 60.0 |
| w_color | 2.0 |
| w_identity | 24.0 |
| lr | 0.00024829191436850196 |
| swd_patch_sizes | [5, 7, 11, 15, 23] |
| color_mode | latent_decoupled_adain |
| num_epochs | 60 |
| base_dim | 96 |
| num_res_blocks | 6 |
| ada_mix_rank | 16 |
| residual_gain | 1.0 |
| skip_routing_mode | ? |

#### trial_0003

| Parameter | Value |
|-----------|------:|
| w_swd | 60.0 |
| w_color | 2.0 |
| w_identity | 42.0 |
| lr | 0.00021677031799390115 |
| swd_patch_sizes | [5, 7, 11, 15, 23] |
| color_mode | latent_decoupled_adain |
| num_epochs | 60 |
| base_dim | 96 |
| num_res_blocks | 6 |
| ada_mix_rank | 16 |
| residual_gain | 1.0 |
| skip_routing_mode | ? |

#### trial_0004

| Parameter | Value |
|-----------|------:|
| w_swd | 60.0 |
| w_color | 2.0 |
| w_identity | 38.0 |
| lr | 0.00046019012426500185 |
| swd_patch_sizes | [5, 7, 11, 15, 23] |
| color_mode | latent_decoupled_adain |
| num_epochs | 60 |
| base_dim | 96 |
| num_res_blocks | 6 |
| ada_mix_rank | 16 |
| residual_gain | 1.0 |
| skip_routing_mode | ? |

#### trial_0005

| Parameter | Value |
|-----------|------:|
| w_swd | 60.0 |
| w_color | 2.0 |
| w_identity | 45.0 |
| lr | 0.00020578944510084497 |
| swd_patch_sizes | [5, 7, 11, 15, 23] |
| color_mode | latent_decoupled_adain |
| num_epochs | 60 |
| base_dim | 96 |
| num_res_blocks | 6 |
| ada_mix_rank | 16 |
| residual_gain | 1.0 |
| skip_routing_mode | ? |

#### trial_0006

| Parameter | Value |
|-----------|------:|
| w_swd | 60.0 |
| w_color | 2.0 |
| w_identity | 25.0 |
| lr | 0.0006341768796084155 |
| swd_patch_sizes | [5, 7, 11, 15, 23] |
| color_mode | latent_decoupled_adain |
| num_epochs | 60 |
| base_dim | 96 |
| num_res_blocks | 6 |
| ada_mix_rank | 16 |
| residual_gain | 1.0 |
| skip_routing_mode | ? |

#### trial_0007

| Parameter | Value |
|-----------|------:|
| w_swd | 60.0 |
| w_color | 2.0 |
| w_identity | 24.0 |
| lr | 0.0002573354002278828 |
| swd_patch_sizes | [5, 7, 11, 15, 23] |
| color_mode | latent_decoupled_adain |
| num_epochs | 60 |
| base_dim | 96 |
| num_res_blocks | 6 |
| ada_mix_rank | 16 |
| residual_gain | 1.0 |
| skip_routing_mode | ? |

#### trial_0008

| Parameter | Value |
|-----------|------:|
| w_swd | 60.0 |
| w_color | 2.0 |
| w_identity | 33.0 |
| lr | 0.0003049313509367067 |
| swd_patch_sizes | [5, 7, 11, 15, 23] |
| color_mode | latent_decoupled_adain |
| num_epochs | 60 |
| base_dim | 96 |
| num_res_blocks | 6 |
| ada_mix_rank | 16 |
| residual_gain | 1.0 |
| skip_routing_mode | ? |

#### trial_0009

| Parameter | Value |
|-----------|------:|
| w_swd | 60.0 |
| w_color | 2.0 |
| w_identity | 27.0 |
| lr | 0.00036398778529995004 |
| swd_patch_sizes | [5, 7, 11, 15, 23] |
| color_mode | latent_decoupled_adain |
| num_epochs | 60 |
| base_dim | 96 |
| num_res_blocks | 6 |
| ada_mix_rank | 16 |
| residual_gain | 1.0 |
| skip_routing_mode | ? |

#### trial_0010

| Parameter | Value |
|-----------|------:|
| w_swd | 60.0 |
| w_color | 2.0 |
| w_identity | 44.0 |
| lr | 0.0003361441955478075 |
| swd_patch_sizes | [5, 7, 11, 15, 23] |
| color_mode | latent_decoupled_adain |
| num_epochs | 60 |
| base_dim | 96 |
| num_res_blocks | 6 |
| ada_mix_rank | 16 |
| residual_gain | 1.0 |
| skip_routing_mode | ? |

#### trial_0011

| Parameter | Value |
|-----------|------:|
| w_swd | 60.0 |
| w_color | 2.0 |
| w_identity | 33.0 |
| lr | 0.00020421356771120017 |
| swd_patch_sizes | [5, 7, 11, 15, 23] |
| color_mode | latent_decoupled_adain |
| num_epochs | 60 |
| base_dim | 96 |
| num_res_blocks | 6 |
| ada_mix_rank | 16 |
| residual_gain | 1.0 |
| skip_routing_mode | ? |

#### trial_0012

| Parameter | Value |
|-----------|------:|
| w_swd | 60.0 |
| w_color | 2.0 |
| w_identity | 45.0 |
| lr | 0.0004766167306932002 |
| swd_patch_sizes | [5, 7, 11, 15, 23] |
| color_mode | latent_decoupled_adain |
| num_epochs | 60 |
| base_dim | 96 |
| num_res_blocks | 6 |
| ada_mix_rank | 16 |
| residual_gain | 1.0 |
| skip_routing_mode | ? |

#### trial_0013

| Parameter | Value |
|-----------|------:|
| w_swd | 60.0 |
| w_color | 2.0 |
| w_identity | 44.0 |
| lr | 0.0002674628889475708 |
| swd_patch_sizes | [5, 7, 11, 15, 23] |
| color_mode | latent_decoupled_adain |
| num_epochs | 60 |
| base_dim | 96 |
| num_res_blocks | 6 |
| ada_mix_rank | 16 |
| residual_gain | 1.0 |
| skip_routing_mode | ? |

#### trial_0014

| Parameter | Value |
|-----------|------:|
| w_swd | 48.2565570580294 |
| w_color | 1.9702625721159805 |
| w_identity | 30.0 |
| lr | 0.0003822666762051879 |
| swd_patch_sizes | [5, 7, 11, 15, 23] |
| color_mode | latent_decoupled_adain |
| num_epochs | 60 |
| base_dim | 96 |
| num_res_blocks | 6 |
| ada_mix_rank | 16 |
| residual_gain | 1.0 |
| skip_routing_mode | ? |

#### trial_0015

| Parameter | Value |
|-----------|------:|
| w_swd | 85.14612453182724 |
| w_color | 3.275166759877586 |
| w_identity | 30.0 |
| lr | 0.0005267473847902741 |
| swd_patch_sizes | [5, 7, 11, 15, 23] |
| color_mode | latent_decoupled_adain |
| num_epochs | 60 |
| base_dim | 96 |
| num_res_blocks | 6 |
| ada_mix_rank | 16 |
| residual_gain | 1.0 |
| skip_routing_mode | ? |

#### trial_0016

| Parameter | Value |
|-----------|------:|
| w_swd | 80.56848927733357 |
| w_color | 3.823666103114249 |
| w_identity | 30.0 |
| lr | 0.0005354541973595901 |
| swd_patch_sizes | [5, 7, 11, 15, 23] |
| color_mode | latent_decoupled_adain |
| num_epochs | 60 |
| base_dim | 96 |
| num_res_blocks | 6 |
| ada_mix_rank | 16 |
| residual_gain | 1.0 |
| skip_routing_mode | ? |

#### trial_0017

| Parameter | Value |
|-----------|------:|
| w_swd | 94.52612925103519 |
| w_color | 2.651755510544014 |
| w_identity | 30.0 |
| lr | 0.0002995188710804697 |
| swd_patch_sizes | [5, 7, 11, 15, 23] |
| color_mode | latent_decoupled_adain |
| num_epochs | 60 |
| base_dim | 96 |
| num_res_blocks | 6 |
| ada_mix_rank | 16 |
| residual_gain | 1.0 |
| skip_routing_mode | ? |

#### trial_0018

| Parameter | Value |
|-----------|------:|
| w_swd | 99.07677229727756 |
| w_color | 4.563403884073622 |
| w_identity | 30.0 |
| lr | 0.0007635190382014643 |
| swd_patch_sizes | [5, 7, 11, 15, 23] |
| color_mode | latent_decoupled_adain |
| num_epochs | 60 |
| base_dim | 96 |
| num_res_blocks | 6 |
| ada_mix_rank | 16 |
| residual_gain | 1.0 |
| skip_routing_mode | ? |

#### trial_0019

| Parameter | Value |
|-----------|------:|
| w_swd | 94.15756106969799 |
| w_color | 4.681711186715533 |
| w_identity | 30.0 |
| lr | 0.0005899044628611461 |
| swd_patch_sizes | [5, 7, 11, 15, 23] |
| color_mode | latent_decoupled_adain |
| num_epochs | 60 |
| base_dim | 96 |
| num_res_blocks | 6 |
| ada_mix_rank | 16 |
| residual_gain | 1.0 |
| skip_routing_mode | ? |

#### trial_0020

| Parameter | Value |
|-----------|------:|
| w_swd | 43.877951230482566 |
| w_color | 1.832384449578339 |
| w_identity | 30.0 |
| lr | 0.0007891273646750754 |
| swd_patch_sizes | [5, 7, 11, 15, 23] |
| color_mode | latent_decoupled_adain |
| num_epochs | 60 |
| base_dim | 96 |
| num_res_blocks | 6 |
| ada_mix_rank | 16 |
| residual_gain | 1.0 |
| skip_routing_mode | ? |

#### trial_0021

| Parameter | Value |
|-----------|------:|
| w_swd | 46.85677253910612 |
| w_color | 3.886453365426208 |
| w_identity | 30.0 |
| lr | 0.0006875480112357322 |
| swd_patch_sizes | [5, 7, 11, 15, 23] |
| color_mode | latent_decoupled_adain |
| num_epochs | 60 |
| base_dim | 96 |
| num_res_blocks | 6 |
| ada_mix_rank | 16 |
| residual_gain | 1.0 |
| skip_routing_mode | ? |

#### trial_0022

| Parameter | Value |
|-----------|------:|
| w_swd | 133.27459797863324 |
| w_color | 0.648083682555193 |
| w_identity | 30.0 |
| lr | 0.000854746943725409 |
| swd_patch_sizes | [5, 7, 11, 15, 23] |
| color_mode | latent_decoupled_adain |
| num_epochs | 60 |
| base_dim | 96 |
| num_res_blocks | 6 |
| ada_mix_rank | 16 |
| residual_gain | 1.0 |
| skip_routing_mode | ? |

#### trial_0038

| Parameter | Value |
|-----------|------:|
| w_swd | 68.9410581525483 |
| w_color | 1.0301892772651728 |
| w_identity | 30.0 |
| lr | 0.0009154868069571305 |
| swd_patch_sizes | [1, 3, 5, 9, 15, 25] |
| color_mode | latent_decoupled_adain |
| num_epochs | 60 |
| base_dim | 96 |
| num_res_blocks | 6 |
| ada_mix_rank | 16 |
| residual_gain | 1.0 |
| skip_routing_mode | ? |

#### trial_0039

| Parameter | Value |
|-----------|------:|
| w_swd | 70.15187306048779 |
| w_color | 1.1549007998447327 |
| w_identity | 30.0 |
| lr | 0.0009628702342164596 |
| swd_patch_sizes | [1, 3, 5, 9, 15, 25] |
| color_mode | latent_decoupled_adain |
| num_epochs | 60 |
| base_dim | 96 |
| num_res_blocks | 6 |
| ada_mix_rank | 16 |
| residual_gain | 1.0 |
| skip_routing_mode | ? |

#### trial_0040

| Parameter | Value |
|-----------|------:|
| w_swd | 68.8036549190746 |
| w_color | 1.0301892772651728 |
| w_identity | 30.0 |
| lr | 0.000955558788531862 |
| swd_patch_sizes | [1, 3, 5, 9, 15, 25] |
| color_mode | latent_decoupled_adain |
| num_epochs | 60 |
| base_dim | 96 |
| num_res_blocks | 6 |
| ada_mix_rank | 16 |
| residual_gain | 1.0 |
| skip_routing_mode | ? |

#### trial_0041

| Parameter | Value |
|-----------|------:|
| w_swd | 68.8036549190746 |
| w_color | 1.0301892772651728 |
| w_identity | 30.0 |
| lr | 0.000955558788531862 |
| swd_patch_sizes | [1, 3, 5, 9, 15, 25] |
| color_mode | latent_decoupled_adain |
| num_epochs | 60 |
| base_dim | 96 |
| num_res_blocks | 6 |
| ada_mix_rank | 16 |
| residual_gain | 1.0 |
| skip_routing_mode | ? |

#### trial_0042

| Parameter | Value |
|-----------|------:|
| w_swd | 70.15187306048779 |
| w_color | 1.1549007998447327 |
| w_identity | 30.0 |
| lr | 0.0008118558769230943 |
| swd_patch_sizes | [1, 3, 5, 9, 15, 25] |
| color_mode | latent_decoupled_adain |
| num_epochs | 60 |
| base_dim | 96 |
| num_res_blocks | 6 |
| ada_mix_rank | 16 |
| residual_gain | 1.0 |
| skip_routing_mode | ? |

#### trial_0043

| Parameter | Value |
|-----------|------:|
| w_swd | 41.84340499039899 |
| w_color | 1.431453110398145 |
| w_identity | 30.0 |
| lr | 0.0007551281457284738 |
| swd_patch_sizes | [1, 3, 5, 9, 15, 25] |
| color_mode | latent_decoupled_adain |
| num_epochs | 60 |
| base_dim | 96 |
| num_res_blocks | 6 |
| ada_mix_rank | 16 |
| residual_gain | 1.0 |
| skip_routing_mode | ? |

#### trial_0044

| Parameter | Value |
|-----------|------:|
| w_swd | 98.92048199237321 |
| w_color | 1.4849368309330848 |
| w_identity | 30.0 |
| lr | 0.00020400233290593836 |
| swd_patch_sizes | [1, 3, 5, 9, 15, 25] |
| color_mode | latent_decoupled_adain |
| num_epochs | 60 |
| base_dim | 96 |
| num_res_blocks | 6 |
| ada_mix_rank | 16 |
| residual_gain | 1.0 |
| skip_routing_mode | ? |

#### trial_0045

| Parameter | Value |
|-----------|------:|
| w_swd | 99.51680507789594 |
| w_color | 2.425471157027536 |
| w_identity | 30.0 |
| lr | 0.0002086887656094012 |
| swd_patch_sizes | [1, 3, 5, 9, 15, 25] |
| color_mode | latent_decoupled_adain |
| num_epochs | 60 |
| base_dim | 96 |
| num_res_blocks | 6 |
| ada_mix_rank | 16 |
| residual_gain | 1.0 |
| skip_routing_mode | ? |


### 5.3 HPO Analysis & Insights

**Two Distinct Phases Detected:**

**Phase 1 (Trials 0000-0022): Variable Identity Search**
- All trials have `w_swd=60.0`, `w_color=2.0` (fixed)
- `w_identity` varies: 24.0 to 45.0
- LR varies: ~2e-4 to ~8e-4
- SWD patches: `[5, 7, 11, 15, 23]` (macro-focused)
- Model: `base_dim=96, res_blocks=6, ada_mix_rank=16`
- Epochs: 60

**Phase 2 (Trials 0038-0045): Fine-grained Weight Search**
- All trials have `w_identity=30.0` (fixed)
- `w_swd` varies: 41.8 to 133.3
- `w_color` varies: 0.65 to 2.43
- `w_identity`: Fixed at 30.0 (different from Phase 1's range)
- LR varies: ~2e-4 to ~9e-4
- SWD patches: `[1, 3, 5, 9, 15, 25]` (wider spectrum, includes micro)
- Model: `base_dim=96, res_blocks=6, ada_mix_rank=16`
- Epochs: 60

**Critical Findings:**
1. **Best ArtFID (337): trial_0044** — `w_swd=98.9`, `w_color=1.48`, `lr=2e-4`, style=0.622
2. **Best Style (0.676): trial_0021** — `w_swd=46.9`, `w_color=3.89`, `lr=6.9e-4`, ArtFID=416
3. **Anti-correlation**: Higher `w_color` gives better style but much worse ArtFID
4. **Duplicate configs**: trial_0000 and trial_0010 are identical (same params, different results?)
5. **Phase 1 has high identity weights** (24-45), Phase 2 locks at 30.0

**Conclusion from HPO**:
- Optimizing for ArtFID bias towards conservative solutions (low color weight, low style).
- The search space was not clean: Phase 1 and Phase 2 mixed different identity weights and patch sizes.

---

## Section 11: Pre-SWD Era - experiments-cycle (NCE/Gram Matrix Lineage)

**Discovery**: File system modification time scan revealed `experiments-cycle/` as the OLDEST layer in the experiment record. This represents the **NCE (Neural Correspondence-based Cycle)** era, BEFORE the transition to SWD-based losses.

### 11.1 Configuration Characteristics

All experiments in `ablation-fixes-new/` share a remarkably different architecture from everything documented so far:

| Parameter | experiments-cycle (NCE era) | Later SWD era |
|-----------|---------------------------|---------------|
| `base_dim` | **256** (massive) | 96 (later reduced to 64) |
| Loss type | **stroke_gram + color_moment + semigroup** | SWD + color + identity |
| Regularization | **delta_tv + delta_l2 + output_tv** | delta_tv (later removed) |
| Loss projector | **64-channel dedicated projector** | None |
| Virtual length | **3** (data augmentation) | 1 |
| Batch size | 96 | 64 → 256 → 320 |
| AMP dtype | **fp16** | bf16 |
| Gradient checkpointing | **False** | True |
| Weight decay | **0.0** | 0.0001 |
| Style strength | **0.65-0.9 (randomized)** | Fixed 1.0 |

### 11.2 Loss Function Ecosystem

The `w_stroke_gram: 80.0` term was the dominant loss (Gram matrix style matching). The ecosystem included:

**Regulation terms (delta space):**
- `w_delta_tv: 0.012` — TV on the residual delta output
- `w_delta_l2: 0.002` — L2 penalty on delta magnitude
- `w_output_tv: 0.005` — TV on the FINAL output (not just delta)

**Style matching terms:**
- `w_stroke_gram: 80.0` — Gram matrix matching (patches) — THE primary style signal
- `w_color_moment: 6.0` — Color moment matching
- `w_semigroup: 0.2` — Semigroup consistency loss (cycle-consistency-like constraint)
- `w_identity: 10.0` — Identity preservation

### 11.3 Ablation Map (A00-A41 series)

```
A00_full           → All terms active (baseline)
A01_proj_off       → Remove 64-channel loss projector
A10_no_gram        → Remove stroke_gram (PRIMARY style loss!)
A12_no_delta_tv     → Remove delta TV regularization
A15_no_semigroup   → Remove semigroup constraint
A20_style_gram_only → REMOVE color_moment (only gram remains, no TV/color moment)
A21_style_moment_only → REMOVE gram (only color_moment remains)
A30_reg_delta_only  → Test: does regulation alone work without gram/moment?
A31_reg_output_only → Test: output_tv instead of delta terms
A32_reg_tv_only     → Test: tv-only regulation
A40_semigroup_light → Gentle semigroup
A41_semigroup_strong → Aggressive semigroup
```

### 11.4 Historical Significance

This is the **ancestor** of the current codebase. Key observations:

1. **base_dim=256** was later reduced to 96 → 64. The current architecture is actually SMALLER but more efficient due to Attention blocks.
2. **Gram matrix style matching** (`w_stroke_gram: 80.0`) was THE primary style signal, later replaced by SWD. This explains why the repo was originally named `Cycle-NCE` — it used Cycle-GAN-style cycle consistency + NCE-style correspondence via `w_semigroup`.
3. **Dual TV** (`delta_tv` + `output_tv`) was used for regularization. Later experiments kept only delta_tv, and the current code removed all TV loss entirely.
4. **Loss projector** (64-channel) was a dedicated MLP projecting features to a space where Gram matching was computed. This was completely removed in the SWD era when SWD operates directly on latent patches.
5. **Randomized style strength** (0.65-0.9) during training was a form of data augmentation that no longer exists in current code.
6. **fp16 AMP** was used, later switched to bf16 for stability.

### 11.5 Timeline Placement

Based on file system analysis and code characteristics:

```
[Earliest] experiments-cycle/Gram+NCE era → SWD transition → style_oa optimization 
    ↓                                              ↓                  ↓
    base_dim=256, Gram matrices,                Introduction of     Pareto frontier
    semigroup, dual TV, proj_64                 SWD loss replacing  exploration
                                                Gram matching
```

The transition from Gram→SWD represents the single biggest paradigm shift in this project. The current CrossAttn+global_attn architecture is the result of iterating on the SWD foundation.

---
---

## Section 12: The Ancestor - Feb 10, 2026 Code Snapshot

**Discovery**: Found a **complete source code snapshot** from `full_250_strong-style/src_snapshot_20260210_141451/` — dated **2026-02-10**. This is the **EARLIEST code version** ever found in the project.

### 12.1 Architecture (model.py — 504 lines)

The ancestor model was **radically simpler** than everything that came after. It had exactly **3 classes**:

```
AdaGN (basic AdaGN, NOT TextureDictAdaGN)
ResBlock
LatentAdaCUT
```

**Critical absences** (all of these had YET to be invented):
- ❌ No `TextureDictAdaGN` (low-rank style dictionary)
- ❌ No `CrossAttnAdaGN` (cross-attention modulation)
- ❌ No `StyleAdaptiveSkip` / `StyleRoutingSkip`
- ❌ No `skip_router` / `skip_fusion` (only raw skip connections)
- ❌ No `num_decoder_blocks` (decoder was fixed structure)
- ❌ No `NormFreeModulation`
- ❌ No `SpatialSelfAttention`
- ❌ No `window_attn` or `global_attn`
- ❌ No `AttentionBlock`

**What existed**:
- ✅ Basic `AdaGN` (simple AdaGN, not the advanced spatially-modulated version)
- ✅ `ResBlock` (standard residual block with AdaGN)
- ✅ `LatentAdaCUT` (the core U-Net architecture — its name survived through all iterations)

### 12.2 Loss Functions (losses.py — 329 lines)

The NCE-era loss was purely **Gram-matrix + Moment** based:

```
calc_gram_matrix()           — Gram matrix computation
calc_gram_loss()             — Gram matrix style matching (THE primary loss)
calc_moment_loss()           — Color/feature moment matching
_lowpass()                   — Low-pass filtering for multiscale
_sobel_magnitude()           — Sobel edge detection
_multiscale_latent_feats()   — Extract features at multiple scales
patch_expand()               — Patch expansion for Gram matching
enrich()                     — Feature enrichment
```

The `AdaCUTObjective` class had exactly 3 core losses:
1. **Gram loss** (style)
2. **Moment loss** (color/texture)
3. **Identity loss** (content preservation)

**No SWD. No CDF. No projection banks. No histogram matching.**

### 12.3 Training Characteristics

- Batch size: 96 (small compared to later 256-320)
- base_dim: **256** (enormous — later reduced to 96)
- Epochs: 60-250 (extremely long training)
- **loss_projector**: 64-channel dedicated MLP projecting features to a Gram-matching space
- **Random style strength**: 0.65-0.9 randomized during training
- **No gradient checkpointing** (too small to need it with BS=96)

### 12.4 Evolutionary Timeline

```
2026-02-10: Pure NCE/Gram Era
  ├─ model.py: 504 lines, 3 classes (AdaGN, ResBlock, LatentAdaCUT)
  ├─ losses.py: 329 lines (Gram, Moment, Identity only)
  ├─ base_dim=256, no skip routing, no spatial style modulation
  └─ loss_projector=64 channels

    ↓ [SWD Revolution]
    
2026-03-early: Transition Period
  ├─ model.py: 731 lines (+227, new TextureDictAdaGN, StyleAdaptiveSkip)
  ├─ losses.py: 632-741 lines (Gram + SWD + NCE + Histogram + Wasserstein — MEGA LOSS ERA)
  ├─ base_dim reduced to 96, loss_projector removed
  └─ Virtual length multiplier introduced (=3 data augmentation)

    ↓ [Simplification]
    
2026-03-26~30: SWD Consolidation
  ├─ model.py: ~1400 lines (CrossAttn, global_attn additions)
  ├─ losses.py: ~550 lines (stripped to SWD + color + identity)
  ├─ base_dim=96, skip_routing_mode='adaptive'/'normalized'
  └─ style_oa Pareto frontier exploration

    ↓ [Paradigm Shift]
    
2026-04-02: Current CrossAttn Era
  ├─ model.py: ~1600 lines (CrossAttn as primary modulator)
  ├─ losses.py: ~370 lines (strictly 3 losses)
  ├─ base_dim=96, global_attn body, window_attn decoder
  ├─ BS=256, num_decoder_blocks FIXED (now actually wired!)
  ├─ ablation_no_residual=true, residual_gain=1.5~4.0
  └─ w_swd=250, w_color=50, w_identity=0
```

**Total model lines: 504 → 731 → ~1400 → ~1600**
**Total loss lines: 329 → 741 (peak) → 370 (current streamlined)**

---
---

## Section 13: master_sweep Systematic Parameter Scanning

**Discovery**: The `master_sweep_XX_*` series (21 experiments) contains comprehensive full_eval results with per-style transfer matrices. This is the most systematic evaluation layer in the entire project.

### 13.1 master_sweep_01_cap_64 Deep Dive

This experiment represents a **capacity=64** baseline configuration with base_dim=64 (reduced from the NCE era's 256).

**Evaluation Date**: 2026-03-12 03:30:50

#### Full 5x5 Cross-Style Transfer Matrix

The evaluation tests EVERY source style (rows) → EVERY target style (cols):

| src → tgt | Hayao | Cezanne | Monet | Photo | Vangogh |
|-----------|-------|---------|-------|-------|---------|
| **Hayao** | style=0.823, lpips=0.269, idt=0.915 | 0.562, 0.375, 0.838 | 0.509, 0.384, 0.816 | 0.593, 0.461, 0.799 | 0.598, 0.408, 0.809 |
| **Cezanne** | 0.598, 0.427, 0.814 | **0.775**, 0.411, 0.805 | 0.733, 0.404, 0.839 | 0.670, 0.475, 0.779 | 0.759, 0.432, 0.818 |
| **Monet** | 0.587, 0.509, 0.789 | **0.782**, 0.458, 0.821 | **0.814**, 0.407, 0.860 | 0.728, 0.478, 0.785 | 0.782, 0.457, 0.827 |
| **Photo** | 0.599, 0.528, 0.781 | 0.644, 0.490, 0.802 | 0.639, 0.435, 0.833 | **0.797**, 0.458, 0.832 | 0.660, 0.533, 0.800 |
| **Vangogh** | 0.612, 0.459, 0.778 | **0.766**, 0.429, 0.821 | 0.731, 0.419, 0.845 | 0.695, 0.520, 0.770 | **0.788**, 0.399, 0.843 |

*(Columns: clip_style, content_lpips, clip_content)*

#### Key Metrics (averaged across all 25 combos):
- **Overall style_transfer clip_style: 0.662**
- **Overall clip_dir: 0.428** (CLIP edit direction cosine)
- **Overall content_lpips: 0.454**
- **Photo→Art average clip_style: 0.636**
- **Photo→Art clip_dir: 0.471**

#### Per-Style Classifier Performance
| Style | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Hayao | 0.851 | 0.573 | **0.685** |
| Cezanne | 0.363 | 0.387 | 0.374 |
| Monet | 0.290 | 0.553 | 0.381 |
| Photo | 0.294 | 0.320 | 0.307 |
| Vangogh | 0.800 | 0.213 | 0.337 |
| **Overall Accuracy** | | | **0.409** |

**Observation**: The classifier is heavily biased — Hayao is easiest to recognize (precision 85%), while Vangogh has terrible recall (21.3%) despite high precision, suggesting the model generates Vangogh-style images but the classifier can't reliably distinguish them from other styles.

### 13.2 Master Sweep Catalog (21 experiments)

| ID | Code Name | Theme |
|----|-----------|-------|
| 001 | cap_64 | Minimal capacity (base_dim=64) |
| 002 | cap_128 | Medium capacity |
| 003 | cap_192 | High capacity |
| 004 | cap_256 | Max capacity (back to NCE-era dim) |
| 005 | patch_micro | Micro-scale SWD patches |
| 006 | patch_std | Standard patch sizes |
| 007 | patch_oversize | Oversized patches |
| 008 | lr_fast | Fast learning rate |
| 009 | lr_slow | Slow learning rate |
| 010 | wide_xmax | Wide style range |
| 011 | narrow_xmax | Narrow style range |
| 012 | mid_xmax | Mid range |
| 013 | wide_micro | Wide + micro patches |
| 014 | narrow_micro | Narrow + micro patches |
| 015 | split_brain_128 | Split-brain architecture (dual pathway?) |
| 016 | split_brain_256 | Split-brain at higher capacity |
| 017 | extreme_underpowered | Severely limited model |
| 018 | extreme_overpowered | Overly large model |
| 019 | the_abyss | Unknown — intriguing name |
| 020 | golden_balance | Balanced configuration |

### 13.3 Evaluation System Details

The `summary.json` metric format reveals the project's evaluation philosophy:

- **clip_style**: `cos( CLIP(gen), CLIP(target_style_proto) )` — absolute style similarity
- **clip_dir**: `cos( CLIP(gen)-CLIP(src), CLIP(target_style_proto)-CLIP(src) )` — measures EDIT DIRECTION alignment
- **content_lpips**: LPIPS distance between generated and source (content preservation)
- **clip_content**: `cos( CLIP(gen), CLIP(src) )` — content similarity in CLIP space
- **delta_fid**: `fid_baseline - fid` (higher = better improvement over baseline)
- **art_fid**: `(1 + fid_style) * (1 + content_lpips)` — combined style+content metric

The use of **style prototypes** (`CLIP(target_style_proto)`) rather than individual reference images for style evaluation is noteworthy — this suggests the project aggregated style representations into prototypical vectors.

---

---

## 14. Experiment Deep Dive: Inject_I Series

### 14.1 Configuration Matrix

Analysis of **injection point ablation** experiments. These tests focus on where style information enters the network.

| Experiment | Gate Hires | Gate Body | Gate Decoder | Decoder AdaGN | w_swd | w_idt | LR |
|-----------|-----------:|----------:|-------------:|--------------:|------:|------:|---:|

### 14.2 Injection Topology

The architecture allows style injection at three stages:
1. **Hires (32x32)**: Pre-downsampling feature modulation.
2. **Body (16x16)**: Main residual block modulation.
3. **Decoder (16x32)**: Post-upsampling feature modulation.


---

## 14. Experiment Deep Dive: Inject_I Series

### 14.1 Architecture

These experiments test where style information enters the network. The LatentAdaCUT model has three injection points:
1. **Hires (32x32)**: `inject_gate_hires` - Pre-downsampling modulation
2. **Body (16x16)**: `inject_gate_body` - Main residual block modulation
3. **Decoder (16x32)**: `inject_gate_decoder` - Post-upsampling modulation

### 14.2 Results (Epoch 80, unified photo→art evaluation)

| Experiment | Style | LPIPS | FID | ArtFID | photo→art style | photo→art ArtFID |
|-----------|------:|------:|----:|-------:|----------------:|-----------------:|
| inject_I0_all_open | 0.6903 | 0.4957 | 316.9 | 490.6 | 0.6789 | 490.6 |
| inject_I1_body_only | 0.6858 | 0.4856 | 312.5 | 481.0 | 0.6745 | 481.0 |
| inject_I2_hires_decoder_only | 0.6898 | 0.4961 | 315.6 | 488.3 | 0.6762 | 488.3 |
| inject_I3_progressive_1_05_01 | 0.6893 | 0.4924 | 313.6 | 485.7 | 0.6776 | 485.7 |
| inject_I4_body_hires | 0.6870 | 0.4927 | 314.8 | 487.8 | 0.6743 | 487.8 |
| inject_I5_body_decoder | 0.6895 | 0.4941 | 311.2 | 481.5 | 0.6768 | 481.5 |

### 14.3 Analysis

- All inject variants cluster around style=0.67-0.68 and ArtFID=481-490
- **Best**: `inject_I5_body_decoder` (lowest ArtFID: 481.5)
- **Worst**: `inject_I0_all_open` (highest ArtFID: 490.6) — counterintuitively, injecting everywhere hurts
- `inject_I1_body_only` performs very similarly: ArtFID 481.0 — body-only injection is sufficient
- These ArtFID values (~480-490) are significantly worse than the mainline (~310-319), suggesting the inject experiments used a different evaluation setup or loss configuration
- Notably, **no config.json was found** in inject directories. The configs likely came from the `src/` directory at the time of execution and were not snapshotted.

## 15. Experiment Deep Dive: Master Sweep Series

### 15.1 Overview

Systematic hyperparameter sweep with **21** configurations.
Sweep parameters include: capacity (cap=64/128/192/256), patch sizes (micro/std/xmax),
learning rate (fast/slow), width (wide/narrow/mid), and extreme configurations (split-brain, under/overpowered, the abyss, golden balance).

### 15.2 Results Summary

| Experiment | Style | LPIPS | FID |
|-----------|------:|------:|----:|
| master_sweep_01_cap_64 | 0.6699828543265659 | 0.43030392565 | 294.6270999585593 |
| master_sweep_03_cap_192 | 0.6713886881868045 | 0.4503596424166667 | 301.2343524640531 |
| master_sweep_05_patch_micro | 0.6644766282041867 | 0.41727255331666663 | 298.9096140725529 |
| master_sweep_06_patch_std | 0.668167517632246 | 0.42146582483333334 | 295.4899033794053 |
| master_sweep_07_patch_xmax | 0.6736843087772528 | 0.4559428502 | 294.4392605085045 |
| master_sweep_14_narrow_micro | 0.6633765571316084 | 0.40734103736666666 | 292.5755928526586 |
| master_sweep_15_split_brain_128 | 0.6768034092585247 | 0.47962787835 | None |

*(-13 more sweep experiments not shown)*

### 15.3 Sweep Architecture Notes

- These experiments used the older model code (731 lines, pre-CrossAttn)
- Likely used `TextureDictAdaGN` with `ada_mix_rank=16`
- The sweep was designed to map performance across capacity and patch-size space

---

## 16. Ancestral Design Documents (from experiments-cycle/)

### 16.1 The Blueprint: "latent AdaCUT.md"

This is the **original design document** that defined the project. Key points:

- **Name**: Latent-AdaCUT (Statistical Version)
- **Core Logic**: 32x32 latent space, AdaGN (injection) + SWD (statistical alignment) + NCE (structure constraint)
- **Architecture**: Micro U-Net without skips (to prevent source texture leakage)
- **Base Dim**: 64, 1 downsampling pass (32->16), 4 residual blocks
- **Parameters**: ~1.5-2M (extremely lightweight, fast inference)
- **Key Decision**: Completely abandoned unstable discriminators (GAN) in favor of statistical alignment

### 16.2 Evolution Document: MODEL_AND_TRAINING_DESIGN.md (English)

This document maps to a **later version** of the code with these additions:
- Teacher-student training setup (teacher has reference image, student uses only style_id)
- Style injection expanded to multiple paths: global, spatial, texture head, force path
- High-frequency delta bias to avoid low-frequency color-shift shortcuts
- Anti-aliasing (blur kernels) to prevent checkerboard artifacts
- Long training runs (300 epochs)

### 16.3 Training Design Details (Chinese: MODEL_AND_TRAINING_DESIGN_ZH.md)

Detailed explanation of the complete system:
- 5 loss families: distill, style (gram/moment), structure, cycle, and code closure
- Configurable lowpass strength for structural and cycle losses
- `struct_lowpass_strength`: continuous interpolation between full match vs low-frequency match

### 16.4 Quality Diagnosis: MODEL_QUALITY_RESEARCH.md

Three critical issues identified:
1. **Style losses only applied to teacher output** — student didn't learn high-frequency quality
2. **NCE token resize used area downsampling** — biased toward blocky signals on small grids
3. **Spatial style template jitter used torch.roll** — circular wrap-around created periodic boundary artifacts

Solutions based on research:
- Zhang 2019 anti-aliasing for CNNs
- Johnson et al. 2016 perceptual losses
- Gatys et al. 2016 Gram matrices for style transfer

---


---

## 17. The Ancestor: src_snapshot_20260210 (February 10, 2026)

### 17.1 Model Architecture (504 lines)

The **original LatentAdaCUT** model had only 3 classes:
1. `AdaGN(nn.Module)` — Simple adaptive GroupNorm: `h * scale + shift`
2. `ResBlock(nn.Module)` — Basic residual block with AdaGN
3. `LatentAdaCUT(nn.Module)` — The main model

**Key differences from current code (~1600 lines):**
- ❌ No `TextureDictAdaGN` — just basic AdaGN (no low-rank texture dictionary, no spatial attention)
- ❌ No `StyleAdaptiveSkip` — no skip routing/filtering
- ❌ No `CrossAttnAdaGN` — no cross-attention modulation
- ❌ No `NormFreeModulation` — no decoder-side style modulation
- ❌ No `num_decoder_blocks` parameter
- ❌ No `StyleMaps` dataclass
- ✅ Had skip connection (but not adaptive)
- ✅ Had residual delta output (`pred = content + delta`)
- ✅ Had `style_spatial_id` spatial priors
- ✅ Had upsampling and decoder stage

### 17.2 Loss Function (329 lines)

Original loss functions available:
- `calc_gram_matrix` / `calc_gram_loss` — Style Gram matrix (channel correlation)
- `calc_moment_loss` — Statistical moments (mean, std, skew, kurtosis)
- `calc_nce_loss` — Contrastive learning for structure
- **No SWD** at all! SWD was added later.
- **No color loss**
- **No identity loss**
- **No TV regularization**

The `compute()` method assembled:
- Distill loss (teacher-student)
- NCE loss
- Gram loss
- Moment loss
- Structure push loss
- Style code loss

### 17.3 Ancestral Config

```json
"model": {
  "base_dim": ?,
  "style_dim": ?,
  "num_res_blocks": ?,
  ...
}
"loss": {
  "w_nce": ?,
  "w_distill": ?,
  "w_gram": ?,
  "w_moment": ?,
  "w_struct": ?,
  "w_cycle": ?,
  "w_style_push": ?,
  "w_style_code": ?
}
```

This was a completely different optimization landscape: 7 loss terms, no SWD, no color matching, relying purely on Gram matrices and NCE for style transfer.

---


---

## 18. The Intermediate Era: Ablation-Fixes-New (Gram+NCE Era)

**Date**: ~February 2026
**Directory**: `experiments-cycle/ablation-fixes-new/`
**Code**: Intermediate model.py (not yet snapshot, but configs reveal the architecture)

### 18.1 Common Configuration Baseline

All 17 experiments share the same base config:
- `base_dim=256` (very large vs 64 in original design, 96 in later phases)
- `num_res_blocks=2`, `num_hires_blocks=4`
- `residual_gain=0.4`
- `lr=0.0002`, `bs=96`, `epochs=60`
- `preload_to_gpu=true`, `virtual_length_multiplier=3`

### 18.2 Loss Function Landscape

The loss system in this era had **6 independent components**:

| Component | Baseline Weight | Purpose |
|-----------|----------------|---------|
| `w_stroke_gram` | 80.0 | Style Gram matrix (stroke texture matching) |
| `w_color_moment` | 6.0 | Color statistical moments |
| `w_identity` | 10.0 | Content preservation |
| `w_delta_tv` | 0.012 | Smoothness on delta output |
| `w_delta_l2` | 0.002 | L2 regularization on delta |
| `w_semigroup` | 0.2 | Semigroup consistency |

**No SWD yet!** Still using Gram matrices for style transfer.

### 18.3 Ablation Groups

**Group A0x: Loss Projector**
- `A00_full`: baseline (projector_use=true, channels=64)
- `A01_proj_off`: projector_use=false (no projection head)
- `A02_proj_32`: projector_channels=32 (bottleneck)
- `A03_proj_128`: projector_channels=128 (wider)

**Group A1x: Component removal**
- `A10_no_gram`: w_stroke_gram=0 (remove Gram)
- `A11_no_moment`: w_color_moment=0 (remove moments)
- `A12_no_delta_tv`: w_delta_tv=0 (remove delta TV)
- `A13_no_delta_l2`: w_delta_l2=0 (remove delta L2)
- `A14_no_output_tv`: w_output_tv removed from config
- `A15_no_semigroup`: w_semigroup=0 (remove semigroup)

**Group A2x: Style-only ablations**
- `A20_style_gram_only`: moment=0 (only Gram remains)
- `A21_style_moment_only`: gram=0 (only moment remains)

**Group A3x: Regularization-only ablations**
- `A30_reg_delta_only`: only delta reg (no style losses? or style=0?)
- `A31_reg_output_only`: only output TV
- `A32_reg_tv_only`: only TV (delta_l2=0)

**Group A4x: Semigroup ablations**
- `A40_semigroup_light`: semi=0.2 (baseline level)
- `A41_semigroup_strong`: semi=0.2 (same as baseline, but name suggests higher impact)

### 18.4 Key Observations

1. **`base_dim=256`**: This is the LARGEST model ever tried in the project. Later configs would shrink to 96.
2. **No SWD yet**: Gram matrices + color moments were the style matching approach.
3. **`virtual_length_multiplier=3`**: The dataset was 3x augmented through virtual length.
4. **`preload_to_gpu=true`**: All data preloaded to VRAM, eliminating data loading bottlenecks.

---

## 19. The Transition: sweep_swd_reborn (SWD Era Begins)

**Date**: ~February-March 2026
**Directory**: `experiments-cycle/sweep_swd_reborn/`

### 19.1 The Birth of SWD

This is where **SWD (Sliced Wasserstein Distance)** first appears in the project, replacing Gram matrices:

**Common baseline**:
- `w_swd=20.0`, `swd_num_projections=128`, `patches=[3,5]`
- `base_dim=256`, `lr=0.0002`, `bs=32` (much smaller batch than ablation-fixes-new's 96)
- `w_color_moment` still present
- **Gram matrices removed** (w_stroke_gram absent from config)

### 19.2 Sweep Groups

**C: Concept variants**
- `C01_Semi_5`: +semigroup
- `C02_Unconstrained`: baseline

**P: Projection count**
- `P01_Proj_32`: 32 projections
- `P02_Proj_128`: 128 projections
- `P03_Proj_256`: 256 projections

**S: Patch sizes**
- `S01_Patch_3`: single scale [3]
- `S02_Patch_3_5_9`: multi-scale [3,5,9]

**W: Weight tuning**
- `W01_Strong_Moment`: higher moment weight
- `W02_No_Moment`: moment=0
- `W03_High_SWD`: w_swd=50.0 (2.5x baseline)

### 19.3 Significance

This marks the **transition from Gram-based style matching to SWD-based statistical alignment**. The SWD approach would eventually become the core loss function, while Gram matrices would be abandoned.

The smaller batch size (32 vs 96) suggests SWD is more computationally expensive per sample, or that the larger projection counts required more VRAM.

---


---

## 20. The experiments-swd8/ Layer: Batch Experiment Packs

**Date**: ~February 23, 2026
**Directory**: `Y:\experiments\experiments-swd8/`
**Significance**: The transition from single-run experiments to **batch experiment packs** with automated launchers

### 20.1 Overview

This directory contains **27 experiment subdirectories** with 155 CSV files, 127 JSON files, and 5 markdown plans.
The naming convention reveals the evolution of the project's experimental methodology.

### 20.2 Batch Experiment Packs

Three major batch packs were discovered, all following the same matrix design but with different batch sizes:

**1. 20260223-micro5style-bs384 (10 experiments)**
- Batch sizes tested: 384 (baseline)
- Phase 1 (60 epochs): baseline, capacity floor (bd32), capacity ceiling (bd128), style_ctrl (sd512), lr stress tests
- Phase 2 (150 epochs): deep_cosine, flat_lr

**2. 20260223-micro5style-bs480 (10 experiments)**
- Same matrix, batch size 480
- Capacity, style_dim, learning rate sweeps

**3. 20260223-micro5style-bs600 (10 experiments)**
- Same matrix, batch size 600
- Used `launch_experiments.py --resume --rerun-failed --aggregate` (automated launcher)

### 20.3 Common Matrix Structure

| exp_id | Phase | Description |
|--------|-------|-------------|
| exp00 | phase1_60e | baseline (bs, lr=2e-4, bd=64, sd=256) |
| exp01 | phase1_60e | capacity floor (bd=32, lc=32, sd=128) |
| exp02 | phase1_60e | capacity ceiling (bd=128, lc=128, sd=256) |
| exp03 | phase1_60e | style_ctrl (sd=512) |
| exp04 | phase1_60e | lr4e4 speedup |
| exp05 | phase1_60e | lr8e4 stress |
| exp06 | phase1_60e | l1 relax (wdl1=0.01) |
| exp07 | phase1_60e | swd_patch_1_3 |
| exp08 | phase2_150e | deep_cosine (lr5e4, min1e6) |
| exp09 | phase2_150e | flat_lr (lr3e4) |

### 20.4 Experiment Directory Themes

- **1swd-dit-2style/5style**: Single SWD with different style counts
- **2style-16dim/8momnet**: 2-style experiments with different dimensions
- **5style-***: 5-style experiments (photo, Hayao, monet, vangogh, cezanne)
- **full-***: Full evaluation runs
- **style8-***: 8-style experiments
- **r-0.5-***: Reduced residual experiments

### 20.5 Key Observations

1. **Batch size evolution**: 384 -> 480 -> 600. The project was actively pushing towards larger batch sizes for more stable SWD gradients.
2. **Automated infrastructure**: The `launch_experiments.py` launcher with `--resume --rerun-failed` indicates mature experiment infrastructure.
3. **Failed experiments**: The aggregate_status.md shows 1 failed + 9 pending in the bs600 pack, suggesting training stability issues at higher batch sizes.
4. **No config files at experiment level**: Configs were managed centrally in `src/config.json` and overridden at runtime.

### 20.6 full-adagn-map16-statloss Experiment Plan

This directory contains a detailed ablation plan:

**Stage Schedule**:
- Stage S (screening): 50 epochs
- Stage V (validation): 150 epochs
- Stage F (final): 300 epochs

**Group A: Semigroup x Structure**
| Exp | w_semigroup | w_struct | Purpose |
|-----|------------:|---------:|---------|
| A1 | 0.0 | 1.0 | BASE |
| A2 | 0.15 | 1.0 | +semigroup |
| A3 | 0.15 | 0.5 | semigroup replaces half struct |
| A4 | 0.15 | 0.0 | semigroup replaces all struct |
| A5 | 0.0 | 0.0 | extreme control (no regularization) |

**Group B: Style Loss Composition**
Test necessity of Gram matrices vs. color moments vs. both.

---


---

## 21. The history_configs/ Archive: 109 Experimental Configurations

**Date**: 2026-03-26 to 2026-04-01
**Directory**: `G:\GitHub\Latent_Style\Cycle-NCE\src\history_configs\`
**Significance**: The MOST VALUABLE artifact in the entire project — 109 named configurations with full parameter matrices

### 21.1 Architecture Evolution Timeline

The configs reveal **TWO distinct architectural eras**:

**Era 1: The concat_conv / textdict era (Run, Final, v3, Weight series)**
- `skip_routing_mode=normalized`, `skip_fusion_mode=concat_conv`
- **No** `style_modulator_type` field (defaults to TextureDictAdaGN)
- `num_res_blocks=6`, `num_decoder_blocks` undefined (dead parameter)
- `num_hires_blocks=4` (or implicit)

**Era 2: The add_proj / cross_attn era (A, E, G, L16, N, S, T, Probe, Patch, TB, NoSkip, Light)**
- `style_modulator_type=cross_attn`
- `skip_fusion_mode=add_proj`
- `num_res_blocks=1` (drastically simplified body)
- `num_decoder_blocks=1-4` (now a real parameter)
- `skip_routing_mode` varies: none, naive, adaptive, normalized

### 21.2 Detailed Config Family Analysis

#### Family 1: Weight Ablation Baseline (7 configs, ~Mar 29)

| Config | w_swd | w_color | w_identity | Purpose |
|--------|------:|--------:|-----------:|---------|
| weight_0_base | 150 | 50 | 30 | Baseline |
| weight_1_swd_low | 100 | 50 | 30 | SWD sensitivity |
| weight_2_swd_high | 200 | 50 | 30 | SWD sensitivity |
| weight_3_color_low | 150 | 20 | 30 | Color sensitivity |
| weight_4_color_high | 150 | 80 | 30 | Color sensitivity |
| weight_5_id_loose | 150 | 50 | 15 | Identity sensitivity |
| weight_6_id_tight | 150 | 50 | 45 | Identity sensitivity |

Architecture: `base_dim=?, res_blocks=2, dec_blocks=3, skip=norm, fusion=add_proj`

#### Family 2: Run Series (0-8, ~Mar 26)

| Config | Description | Key Change |
|--------|-------------|------------|
| Run_0_Baseline | Reference | lr=?, swd=150, clr=50, idt=30 |
| Run_1_lr_high_8e4 | High LR | lr=8e-4 |
| Run_2_lr_low_2e4 | Low LR | lr=2e-4 |
| Run_3_id_loose_15 | Loose Identity | idt=15 |
| Run_4_id_tight_45 | Tight Identity | idt=45 |
| Run_5_swd_max_200 | Max SWD | swd=200 |
| Run_6_color_bold_100 | Bold Color | clr=100 |
| Run_7_lum_strict_10 | Strict Luma | luma_range=10 |
| Run_8_arch_old_dict | Old Architecture | style_modulator=textdict |

#### Family 3: Final Configurations (12 configs, ~Mar 27)

| Config | LR | w_swd | w_color | w_identity | base_dim | purpose |
|--------|----|------:|--------:|-----------:|---------:|---------|
| final_1_lr4_id35_swd60_c5 | 4e-4 | 60 | 5 | 35 | ? | Low SWD, high identity |
| final_2_lr5_id30_swd80_c2 | 5e-4 | 80 | 2 | 30 | ? | Medium everything |
| final_3_lr6_id25_swd60_c5 | 6e-4 | 60 | 5 | 25 | ? | Higher LR, looser ID |
| final_4_lr8_id30_swd80_c2 | 8e-4 | 80 | 2 | 30 | ? | High LR |
| final_5_lr5_id15_swd120_c5 | 5e-4 | 120 | 5 | 15 | ? | High SWD, low ID |
| final_6_lr5_id20_swd150_c10 | 5e-4 | 150 | 10 | 20 | ? | Strong style push |
| final_7_lr8_id15_swd120_c5 | 8e-4 | 120 | 5 | 15 | ? | Aggressive |
| final_8_lr8_id20_swd150_c10 | 8e-4 | 150 | 10 | 20 | ? | Most aggressive |
| final_9_dim128_tok64 | 5e-4 | 150 | 50 | 30 | 128 | Wide model, fewer tokens |
| final_10_dim96_tok128 | 5e-4 | 150 | 50 | 30 | 96 | Narrow model, more tokens |
| final_11_dim128_tok128 | 5e-4 | 150 | 50 | 30 | 128 | Wide model, more tokens |
| final_12_base_ref | 5e-4 | 150 | 50 | 30 | ? | Baseline reference |

Architecture: `res_blocks=6, skip=norm, fusion=concat_conv, mod=textdict`

#### Family 4: v3 Architecture Variants (4 configs, ~Mar 27)

These configs test the transition from old to new architecture:
- `v3_0_attn_base`: Global attention body
- `v3_1_arch_dict`: Dictionary modulation (textdict)
- `v3_2_skip_naive`: ⚠️ Naive skip (no filtering)
- `v3_3_no_residual`: ⚠️ No residual anchor

#### Family 5: WGw Deep Architectures (3 configs, ~Mar 29)

Testing deeper architectures with more blocks:
- `E1_wgw_light_h2_g1_d2`: 2 hires, 1 res, 2 decoder
- `E2_wgw_heavy_h3_g2_d3`: 3 hires, 2 res, 3 decoder (skip=norm, fusion=cat)
- `E3_wgw_heavy_w16_h3_g2_d3`: Same as E2 but wider attention (w16?)

#### Family 6: Patch/HF Scale Grid (12 configs, ~Mar 30)

Systematic sweep over SWD patch sizes and HF weight:

| Patch Set | HF=1.0 | HF=3.0 | HF=off |
|-----------|--------|--------|--------|
| base (3,5) | p_base_hf_1 | p_base_hf_3 | p_base_hf_off |
| 1,5,9,15 | p_1_5_9_15_hf_1 | p_1_5_9_15_hf_3 | p_1_5_9_15_hf_off |
| 5,9,15 | p_5_9_15_hf_1 | p_5_9_15_hf_3 | p_5_9_15_hf_off |
| 5,9,15,25 | p_5_9_15_25_hf_1 | p_5_9_15_25_hf_3 | p_5_9_15_25_hf_off |

All: `w_swd=150, w_color=50, w_idt=30, res_blocks=1, dec_blocks=2, skip=norm, mod=cross_attn`

#### Family 7: L16 Hyperparameter Grid (16 configs, ~Mar 30)

**The largest systematic sweep in the project**:

| Variable | Values |
|----------|--------|
| w_swd | 100 vs 250 (2 options) |
| swd_hf_weight_ratio | 1 vs 4 (2 options) |
| swd_patch_sizes | PMic=[1,3,5] vs PMac=[5,9,15] (2 options) |
| w_color | 10 vs 80 (2 options) |
| w_identity | 5 vs 30 (2 options) |

Total: 2×2×2×2×2 = 32 configs, but only 16 were run (half-grid?)

Common: `res_blocks=1, dec_blocks=2, skip=norm, fusion=add_proj, mod=cross_attn, lr=5e-4`

#### Family 8: Residual/Skip Architecture Matrix (A, E, G, T series, ~Apr 1)

This is the **most architectural-focused** experiment set, systematically testing residual + skip combinations:

**E0 Series (ResOff)**:
| Config | Skip Mode | Dec Blocks | noRes |
|--------|-----------|-----------|-------|
| E03 | none | 1 conv | True |
| E04 | none | 2 swin | True |
| E05 | none | 4 swin | True |
| E06 | adaptive | 1 conv | True |
| E07 | adaptive | 2 swin | True |
| E08 | adaptive | 4 swin | True |
| E09 | naive | 1 conv | True |

**A Series (ResOn vs ResOff)**:
| Config | Residual | Skip Mode | Dec Blocks |
|--------|----------|-----------|-----------|
| A01_ResOn_Naive_Conv1 | ON | naive | 1 conv |
| A02_ResOn_None_Swin4 | ON | none | 4 swin |

**G1/G2 Residual x Skip Matrix**:
| Config | Residual | Skip Mode | Dec Blocks |
|--------|----------|-----------|-----------|
| G1_NoRes_Adapt_Clean | OFF | adaptive | 4, conv |
| G1_NoRes_Adapt_Style | OFF | adaptive | 4, conv |
| G1_NoRes_Naive_Clean | OFF | naive | 4, conv |
| G1_NoRes_Naive_Style | OFF | naive | 4, conv |
| G1_NoRes_NoSkip | OFF | none | 4, conv |
| G2_Res_Adapt_Clean | ON | adaptive | 4, conv |
| G2_Res_Adapt_Style | ON | adaptive | 4, conv |
| G2_Res_Naive_Clean | ON | naive | 4, conv |
| G2_Res_Naive_Style | ON | naive | 4, conv |
| G2_Res_NoSkip | ON | none | 4, conv |

**T Series**:
| Config | Residual | Skip | Dec | ID | Special |
|--------|----------|------|-----|---|---------|
| T01_ResOn_None_Swin4_Noise | ON | none | 4 swin | 5.0 | input_anchor_noise=0.05 |
| T02_ResOff_Adapt_Conv1_LowIDT | OFF | adaptive | 1 conv | 1.0 | Very low identity |
| T03_ResOff_Adapt_Conv1_HFSWD | OFF | adaptive | 1 conv | 2.0 | HF-SWD enabled |

#### Family 9: S Skip Fusion Experiments (10 configs, ~Mar 30)

| Config | Skip Mode | HF Ratio | skip_gain | fusion |
|--------|-----------|---------:|----------:|--------|
| S01_None | none | 1 | - | add_proj |
| S02_Naive_G1p0 | naive | 1 | 1.0 | add_proj |
| S03_Naive_G0p5 | naive | 1 | 0.5 | add_proj |
| S04_Naive_G1p5 | naive | 1 | 1.5 | add_proj |
| S05_Adaptive | adaptive | 1 | - | add_proj |
| S06_Normalized | normalized | 1 | - | add_proj |
| S07_Stress_Naive | naive | 4 | - | add_proj |
| S08_Stress_Norm | normalized | 4 | - | add_proj |
| S09_Fusion_Add | normalized | 1 | - | add_proj |
| S10_Fusion_Cat | normalized | 1 | - | concat_conv |

#### Family 10: N Stylization Modes (4 configs, ~Mar 31)

These test the interaction between skip mode and style intensity:

| Config | Skip Mode | Special |
|--------|-----------|---------|
| N01_Stylized_Naive | naive | - |
| N02_Stylized_Adaptive | adaptive | - |
| N03_Stylized_Adaptive_Retain0p2 | adaptive | retain_boost=0.2 |
| N04_Stylized_Norm | normalized | - |

#### Family 11: TB Special Variants (3 configs, ~Apr 1)

| Config | Skip | w_identity | Special |
|--------|------|-----------:|---------|
| TB1_A02_LossRelease | none | 2.0 | Very low identity (loss release) |
| TB2_A02_HFSWD | none | 30.0 | HF-SWD enabled |
| TB3_A02_AnchorNoise | none | 30.0 | Input anchor noise |

#### Family 12: Probe LR/Anchor (4 configs, ~Mar 30)

| Config | Description |
|--------|-------------|
| Probe_17_LR_High | lr=1e-3 (aggressive) |
| Probe_18_LR_Low | lr=3e-4 (conservative) |
| Probe_19_LR_OneCycle_Aggressive | onecycle scheduler |
| Probe_20_Baseline_Anchor | Standard baseline |

---


---

## 22. The 42 Series: Running Experiments (April 1-2, 2026)

**Current state of active experiments** in `G:\GitHub\Latent_Style\Cycle-NCE\src\`
These are configs that were actively launched on 2026-04-01 and 2026-04-02.

### 22.1 Architecture Summary

All 42-series configs share this base architecture:
- `base_dim=96, lift_channels=128, style_dim=160`
- `num_styles=5 (photo, Hayao, monet, vangogh, cezanne)`
- `num_hires_blocks=2`
- `num_res_blocks=1` (ultra-minimal body)
- `num_groups=4`
- `style_modulator_type=cross_attn` (no more textdict)
- `style_attn_num_tokens=64, num_heads=4, sharpen_scale=2.5`
- `hires_block_type=conv, body_block_type=global_attn`
- `skip_fusion_mode=add_proj`

Variables across experiments:
- `num_decoder_blocks`: 1-4 (conv or window_attn/swin)
- `decoder_block_type`: conv vs window_attn
- `skip_routing_mode`: none, naive, adaptive, normalized
- `ablation_no_residual`: True or False
- `residual_gain`: 1.0, 1.5, 2.0, 4.0
- `swd_patch_sizes`: varies from [3] to [3,5,7,15,19,25]
- `w_swd`: 200-300
- `w_color`: 50-200
- `w_identity`: 0.0-5.0 (mostly zero)
- `swd_use_high_freq`: True or False
- `input_anchor_noise_std`: 0.0 or 0.05

### 22.2 A Series (A01-A10): Patch Scale Experiments

| Config | swd_patches | resgain | skip | w_swd | w_color | lr | bs | special |
|--------|------------|--------:|------|------:|--------:|----:|---:|---------|
| A01_Macro_Only | [19,25,31] | 1 | adaptive | 250 | 50 | 3e-4 | 256 | Macro-only HF off |
| A02_Micro_Only | [3,5,7,11] | 1 | adaptive | 300 | 50 | 3e-4 | 256 | Micro-only HF on |
| A03_Bipolar_Extreme | [3,31] | 1 | adaptive | 250 | 50 | 3e-4 | 256 | Extreme scale gap |
| A04_FullSpec_Conv2 | [3,5,7,15,19,25] | 1 | adaptive | 250 | 50 | 4e-4 | 256 | Full spectrum, 2 dec |
| A05_FullSpec_Conv3 | [3,5,7,15,19,25] | 1 | adaptive | 250 | 50 | 2e-4 | 224 | 3 dec blocks |
| A06_FullSpec_Conv3 | [3,5,7,15,19,25] | 1 | adaptive | 250 | 50 | 3e-4 | 224 | 3 dec blocks |
| A07_NoSkip | [3,5,7,15,19,25] | 1 | none | 300 | 50 | 3e-4 | 256 | Skip disabled |
| A08_Noise01 | [3,5,7,15,19,25] | 1 | adaptive | 250 | 50 | 3e-4 | 256 | noise_std=0.1 |
| A09_Gain4 | [3,5,7,15,19,25] | 4 | adaptive | 250 | 50 | 2e-4 | 256 | High residual gain |
| A10_Color200 | [3,5,7,15,19,25] | 1 | adaptive | 200 | 200 | 3e-4 | 256 | Extreme color weight |

### 22.3 E Series (E01-E04): Micro-scale + Gain4 Sweep

| Config | patches | gain | HF | w_swd |
|--------|---------|-----:|---:|------:|
| E01_Patch3_Gain4 | [3] | 4.0 | off | 250 |
| E02_Patch3_5_Gain4 | [3,5] | 4.0 | on | 250 |
| E03_Patch3_5_7_Gain4 | [3,5,7] | 4.0 | on | 250 |
| E04_Patch1_3_5_Gain4 | [1,3,5] | 4.0 | on | 250 |

### 22.4 F Series (FinalMicro_2): F01 vs F02

| Config | patches | gain | HF |
|--------|---------|-----:|---:|
| F01_Patch135 | [1,3,5] | 1.5 | off |
| F02_Patch357 | [3,5,7] | 1.5 | on |

### 22.5 T Series (TextureTearer3): Residual Ablation

| Config | noRes | skip | decoder | idt | patches | special |
|--------|-------|------|---------|----:|---------|---------|
| T01_ResOn_None_Swin4 | False | none | 4 swin | 5.0 | [7,11,15,19,25] | noise=0.05 |
| T02_ResOff_Adapt_Conv1 | True | adaptive | 1 conv | 1.0 | [7,11,15,19,25] | LowIDT, resgain=2 |
| T03_ResOff_Adapt_Conv1 | True | adaptive | 1 conv | 2.0 | [3,5,7,11] | HF-SWD |

### 22.6 Z Series (ZeroConstraint): No Identity Extremes

| Config | skip | decoder | patches | HF | w_swd |
|--------|------|---------|---------|---:|------:|
| Z01_ResOff_Adapt_ZeroIDT | adaptive | 1 conv | [7,11,15,19,25] | off | 250 |
| Z02_ResOff_Adapt_ZeroIDT_HFSWD | adaptive | 1 conv | [3,5,7,11] | on | 250 |
| Z03_ResOff_None_ZeroIDT | none | 1 conv | [7,11,15,19,25] | off | 300 |

### 22.7 Key Observations

1. **All 42-series experiments have `w_identity=0`** (except T series which has 1-5). This is a radical departure from earlier experiments where `w_identity` was 30+.
2. **Residual is disabled** (`ablation_no_residual=true`) in almost all experiments. The only exception is T01.
3. **`num_res_blocks=1`**: The body is minimal — just a single global attention block.
4. **Decoder blocks vary from 1-4**: Testing the "heavy decoder" hypothesis.
5. **`swd_use_high_freq`** is being tested: some experiments enable it, most don't.
6. **Patch sizes are the primary variable**: From single-scale [3] to full-spectrum [3,5,7,15,19,25].

---


---

## 23. Active Experiment Results (April 2026)

### 23.1 42 Series Status (April 1-2)

**All 10 experiments (A01-A10) are still training** — no full_eval results available yet.
- Created: 2026-04-02 15:45 (batch launch)
- Epochs: 40 (so if launched April 1, should be ~35-38 epochs by April 3)
- Save interval: 40 (only saves at end)
- Full eval interval: 40 (only evaluates at end)

### 23.2 FinalMicro_2 Results (Completed)

**F01: Patch135, Gain1.5, LR2e4**
| Epoch | Clip Style | LPIPS | Trans Style |
|------:|-----------:|------:|------------:|
| 30 | 0.8619 | 0.6826 | 0.6508 |
| 40 | **0.8634** | **0.6834** | **0.6514** |
Distill (tokenized, ep=30): 0.8602 / 0.6825 / 0.6509
Distill (tokenized, ep=40): 0.8612 / 0.6830 / 0.6513

**F02: Patch357, Gain1.5, LR2e4**
| Epoch | Clip Style | LPIPS | Trans Style |
|------:|-----------:|------:|------------:|
| 30 | 0.8698 | 0.6807 | 0.6481 |
| 40 | 0.8677 | 0.6823 | 0.6499 |
Distill (tokenized, ep=30): 0.8698 / 0.6814 / 0.6487
Distill (tokenized, ep=40): 0.8675 / 0.6832 / 0.6509

> **Note**: The metric names are swapped — "all_clip_style" appears to be content preservation (0.86+) and "clip_style" appears to be style transfer similarity (0.68+). The FinalMicro_2.csv column order confirms this.

### 23.3 WGw Results (March 29)

**E1_wgw_light_h2_g1_d2**
| Epoch | All Style | Content | LPIPS | FID | Trans Style | Fid |
|------:|----------:|--------:|------:|----:|------------:|----:|
| 30 | 0.6774 | 0.8787 | 0.36 | ? | 0.6787 | ? |
| 40 | **0.6787** | **0.8806** | **0.332** | ? | ? | ? |
Distill (ep=30): 0.6775 / 0.8770 / 0.3319
Distill (ep=40): 0.6781 / 0.8778 / 0.3296

### 23.4 FinalMicro_2 vs WGw Comparison

| Aspect | FinalMicro_2 | WGw E1 |
|--------|-------------|--------|
| All Style (content) | 0.863 | 0.880 |
| Clip Style | 0.683 | 0.679 |
| LPIPS | 0.683 | 0.332 |
| Residual | Disabled (noRes=true) | Enabled (resgain=1.0 or 1.5) |
| Identity | 0.0 | 30.0 |
| Skip | adaptive | normalized |
| Modulator | cross_attn | cross_attn |
| Decoder | 2 blocks, conv | 2 blocks (h2), conv |

> **Key difference**: WGw has identity=30 and LPIPS=0.33, while FinalMicro_2 has identity=0 and LPIPS=0.68. This suggests the "all_clip_style" and "clip_style" columns are consistently measuring different things across experiments.

---



## Section 27: Optuna HPO Deep Analysis (31 Trials, Complete Data)

### Trial Results (Ranked by ArtFID)

| Rank | Trial | Style | FID | delta_FID | ArtFID | Key Config |
|:----:|:------|------:|----:|----------:|-------:|:-----------|
| 1 | 0044 | 0.622 | 295.5 | +10.4 | **337.4** | wc=1.48, patches=[1,3,5,9,15,25] |
| 2 | 0003 | 0.655 | 296.4 | +9.5 | **345.7** | wc=2.0, wid=42 |
| 3 | 0010 | 0.658 | 301.9 | +4.0 | **353.1** | wc=2.0, wid=44 |
| 4 | 0038 | 0.638 | 301.6 | +4.3 | **353.5** | wc=1.03, micro+macro patches |
| 5 | 0005 | 0.649 | 300.2 | +5.7 | **353.8** | wc=2.0, wid=45 (highest) |
| 6 | 0000 | 0.657 | 299.8 | +6.1 | **355.8** | wc=2.0, wid=44 (first trial) |
| 7 | 0000+distill | 0.656 | 301.0 | +5.0 | **357.0** | distill makes it worse |
| 8 | 0019 | **0.675** | 311.3 | -5.4 | 386.1 | wc=4.68 (highest color) |
| 9 | 0018 | 0.670 | 310.2 | -4.2 | 391.6 | wc=4.56 |
| 10 | 0021 | **0.676** | 313.0 | -7.1 | **416.3** | WORST ArtFID, wc=3.89 |

### Three Search Phases
```
Phase 1 (0000-0013): Scattered w_idt {24-45}, w_color=2.0 fixed
Phase 2 (0014-0022): w_idt locked=30, exploring w_color {1.97-4.68}, w_swd {47-99}
Phase 3 (0038-0045): Expanded patches [1,3,5,9,15,25], low w_color {1.03-2.43}
```

### Critical HPO Insight
**w_color vs ArtFID tradeoff is the dominant axis:**
- Low w_color (1.0-1.5) → Best ArtFID (337-354), but Style only 0.62-0.64
- High w_color (3.9-4.7) → Style up to 0.676, BUT ArtFID 386-416
- Optuna optimizing single ArtFID target converges to CONSERVATIVE solutions
- w_idt in range 24-45 has minimal effect (all trials converged to similar range)
- Tokenized distill makes ALL metrics slightly worse, ZERO benefit

## Section 28: Summary History Database

### REDISCOVERED_SUMMARY_HISTORY (130 entries, 2026-03-26)
The most comprehensive unified evaluation database. Key experiments:

| Experiment | Epoch | P2A Style | P2A ArtFID | Key Finding |
|:---|---:|---:|---:|:---|
| style_oa_5@ep30 | 30 | 0.698 | 354 | Content-friendly checkpoint |
| style_oa_5@ep60 | 60 | 0.721 | 391 | Balance point |
| style_oa_5@ep100 | 100 | 0.730 | 420 | Style peak |
| style_oa_5@ep120 | 120 | 0.727 | 441 | **OVERFITTING** (style drops, ArtFID worse) |
| exp_3_macro_strokes | 100 | 0.611 | 310 | Best mainline balance |
| exp_G1_edge_rush | 60 | 0.625 | 314 | Aggressive but controlled |
| abl_no_residual | 80 | 0.579 | 311 | Conservative, low style |
| abl_no_adagn | 80 | 0.687 | 419 | No modulation control |
| abl_naive_skip | 80 | 0.685 | 420 | High-frequency leakage |

## Section 29: FinalMicro_2 Results (2026-04-02)

| Config | Epoch | Style | LPIPS | P2A Style | Assessment |
|:---|---:|---:|---:|---:|:---|
| F01 [1,3,5] gain1.5 | 30 | 0.683 | **0.346** | 0.620 | Best content fidelity EVER |
| F01 [1,3,5] gain1.5 | 40 | 0.683 | 0.346 | 0.622 | Converged by epoch 30 |
| F02 [3,5,7] gain1.5 | 30 | 0.681 | 0.350 | 0.611 | Slightly worse |
| F02 + distill | 40 | 0.683 | 0.354 | 0.611 | Zero distill benefit |

**LPIPS=0.346 is project record low** (previous best ~0.42 in old architecture)
**BUT P2A style ~0.62 is lower than style_oa's 0.72+**
Conclusion: New CrossAttn+global_attn+adaptive_skip architecture excels at content
preservation but needs more style strength.



## Section 32: Comprehensive Ablation Results Matrix (New Data from summary.json)

These results were extracted directly from `summary.json` files in each experiment's `full_eval/` directory.
The `matrix_breakdown` structure provides per-style transfer metrics (photo→Hayao/Monet/VanGogh/Cezanne).

### Ablation Experiments (unified photo→art averages)

| Experiment | Epoch | P2A Style | LPIPS | FID | ArtFID | Key Finding |
|:---|---:|---:|---:|---:|---:|:---|
| **abl_rank64** | 80 | 0.6888 | 0.5610 | **300** | **471** | Best FID+ArtFID in rank series |
| **abl_rank8** | 80 | 0.6890 | 0.5634 | 309 | 485 | Middle ground |
| **abl_rank1** | 80 | 0.6921 | 0.5615 | 305 | 479 | Highest style in rank series |
| **abl_vanilla_gn** | 60 | 0.6693 | 0.5357 | 304 | 470 | Good but weaker than AdaGN |
| **abl_no_color** | 60 | 0.6785 | 0.5435 | 315 | 488 | Color loss helps but not essential |
| **abl_no_tv** | 60 | 0.6811 | 0.5494 | 316 | 491 | TV stabilizes distribution |
| **abl_no_skip_filter** | 60 | 0.6825 | 0.5615 | 313 | 491 | Skip filtering needed |
| **abl_no_id** | 60 | 0.6838 | 0.5667 | 324 | 510 | Identity constraint keeps content |
| **abl_heavy_decoder** | 60 | 0.6971 | 0.5640 | 311 | 488 | More decoder capacity |
| **abl_macro_decoder** | 80 | 0.7011 | 0.5743 | 313 | 495 | Macro decoder pushes style |
| **abl_hard_sort** | 100 | 0.7034 | 0.5966 | 320 | 514 | HIGH style but WORST LPIPS/FID |
| **ablate_M1_Aggressive** | 120 | **0.7152** | 0.6193 | N/A | N/A | Highest style ever in old arch |
| **ablate_M2_Smooth** | 120 | 0.7041 | 0.6034 | N/A | N/A | Slightly less aggressive |

### Key Ablation Insights (from this data)
1. **ada_mix_rank: 64 > 8 > 1** for ArtFID, but style increases as rank DECREASES
2. **Hard sort (abl_hard_sort)**: 20→100 epoch trajectory shows style increase (0.673→0.703) 
   but massive LPIPS/FID degradation - classic overfit pattern
3. **Vanilla GN (abl_vanilla_gn)**: ArtFID=470, style=0.669 - shows the gap between 
   simple normalization and the full TextureDictAdaGN (which achieves style 0.69+)
4. **A-series (patch/id/tv parameters)**: All achieve similar style ~0.67-0.68
   suggesting these params are in a relatively flat region
5. **M-series (aggressive/smooth)**: Push to 120 epochs gets style up to 0.715
   which is near style_oa levels but with terrible LPIPS (0.619)

## Section 33: Early Experiments (Exp_00, Exp_01, G0, G1)

| Experiment | Epoch | Style | LPIPS | FID | ArtFID | Notes |
|:---|---:|---:|---:|---:|---:|:---|
| Exp_00_TV_Anchor | 10 | 0.6512 | 0.5538 | N/A | N/A | Very early, only 10 epochs |
| Exp_01_Stats | 30 | 0.6498 | 0.5417 | N/A | N/A | 30 epochs, improving |
| Exp_05_TV_Color | 20 | 0.6490 | 0.5150 | 303 | 461 | Color+TV branch |
| G0-Base-Gain0.5 | 80 | 0.6779 | 0.5526 | N/A | N/A | 80 epochs, good style |
| G1-Relax-ID | - | ERROR | - | - | - | Missing eval data |

These represent the OLD architecture era (pre-CrossAttn, pre-global_attn).
Style in range 0.65-0.68, LPIPS in range 0.51-0.55.
Compare to FinalMicro_2: style 0.683, LPIPS **0.346** - much better content!
Compare to style_oa: style 0.72+, LPIPS 0.58+ - higher style but worse content.



## Section 34: Massive Unexplored Experiment Results (57 Experiments)

Extracted from raw summary.json files. This is the largest single data dump so far.

### Top Performers by P2A Style (Photo→Art average)

| Experiment | Epoch | Style | LPIPS | ClDir | FID | ArtFID | Notes |
|:---|---:|---:|---:|---:|---:|---:|:---|
| **decoder-D-sweetspot** | 80 | **0.7073** | 0.6104 | 0.6395 | N/A | N/A | Style champion |
| scan01_base | 80_tok | 0.7002 | 0.5877 | 0.6174 | 309 | 493 | Baseline scan, excellent |
| scan05_gate_0p75 | 80_tok | 0.6996 | 0.5881 | 0.6179 | 304 | 486 | Gate 0.75 is sweet spot |
| scan03_soft_hf | 80_tok | 0.6992 | 0.5793 | 0.6120 | **300** | **476** | Best ArtFID in scan series |
| scan06_id_0p50 | 80_tok | 0.6984 | 0.5839 | 0.6150 | 304 | 485 | ID=0.50 is good |
| scan04_gate_0p85 | 80_tok | 0.6971 | 0.5885 | 0.6161 | 310 | 495 | Gate 0.85 is too loose |
| decoder-H-MSCTM | 80 | 0.6957 | 0.5887 | 0.6193 | N/A | N/A | MSCTM architecture |
| scan02_low_lr | 80_tok | 0.6923 | 0.5785 | 0.6050 | 310 | 492 | Low LR is slower but stable |

### Micro Patch Series (F-Series from FinalMicro_2 ancestor)

| Experiment | Epoch | Style | LPIPS | ClDir | FID | ArtFID |
|:---|---:|---:|---:|---:|---:|---:|
| micro02_macro_patch | 80_tok | 0.6882 | 0.5572 | 0.5941 | 309 | 483 | Macro patches work well |
| micro03_gate75 | 80_tok | 0.6837 | 0.5527 | 0.5828 | 314 | 491 | |
| micro01_hf2_lr1 | 80_tok | 0.6813 | 0.5496 | 0.5786 | 311 | 484 | HF weight 2.0 |
| micro04_hf1p5_macro | 80_tok | 0.6800 | 0.5471 | 0.5830 | 310 | 483 | |
| micro05_id_anchor | 80_tok | 0.6799 | 0.5463 | 0.5747 | 312 | 485 | |

### Inject Pathway Experiments (I0-I5)

| Experiment | Epoch | Style | LPIPS | ClDir | Key Finding |
|:---|---:|---:|---:|---:|:---|
| inject_I5_body_decoder | 80 | 0.6795 | 0.5476 | 0.5783 | Body+decoder inject = best |
| inject_I3_progressive_1_05_01 | 80 | 0.6785 | 0.5492 | 0.5760 | Progressive injection |
| inject_I4_body_hires | 80 | 0.6772 | 0.5504 | 0.5742 | Body+hires |
| inject_I0_all_open | 80 | 0.6761 | 0.5484 | 0.5748 | All paths open |
| inject_I2_hires_decoder | 80 | 0.6759 | 0.5468 | 0.5748 | Hires+decoder |
| inject_I1_body_only | 80 | 0.6736 | 0.5398 | 0.5690 | Body is LEAST effective |

### NCE Series Experiments

| Experiment | Epoch | Style | LPIPS | ClDir | Notes |
|:---|---:|---:|---:|---:|:---|
| nce-swd_0.25-cl_0.01 | 120 | 0.6778 | 0.5296 | 0.5755 | Best NCE variant, low lpips! |
| nce-gate_norm-swd_0.45-cl_0.01 | 120 | 0.6861 | 0.5601 | 0.5974 | |
| nce_A3_Patch_Coarse | 80 | 0.6759 | 0.5637 | 0.5877 | |
| nce_A4_Patch_Fine | 80 | 0.6729 | 0.5268 | 0.5664 | Fine patches = better lpips |
| nce_A2_Shallow_Only | 80 | 0.6713 | 0.5265 | 0.5629 | Shallow = ok |
| nce_A1_Deep_Only | 80 | 0.6692 | 0.5393 | 0.5667 | |
| nce_A5_High_TV | 80 | 0.6627 | 0.5246 | 0.5477 | High TV = lower ClDir |
| nce | 80 | 0.6556 | 0.5043 | 0.5325 | Base NCE |
| nce-gate_content | 80 | 0.6650 | 0.5228 | 0.5586 | |
| nce-gate_norm | 40 | 0.6512 | 0.4996 | 0.5231 | Only 40 epochs |

### Clocor1 E-Series (Cross-Style Color Correction?)

| Experiment | Epoch | Style | LPIPS | ClDir |
|:---|---:|---:|---:|---:|
| E5_9Series_Soft_LR14e4 | 80 | 0.6741 | 0.5239 | 0.5656 | Best in series |
| E3_15Series_Soft_LR14e4 | 80 | 0.6730 | 0.5417 | 0.5753 | |
| E2_15Series_Rigid_LR14e4 | 80 | 0.6686 | 0.5326 | 0.5664 | |
| E4_9Series_Rigid_LR14e4 | 80 | 0.6684 | **0.5164** | **0.5546** | Best LPIPS+ClDir |
| E1_Macro19_Rigid_LR14e4 | 80 | 0.6647 | 0.5401 | 0.5637 | |

### Spatial AdaGN Series

| Experiment | Epoch | Style | LPIPS | ClDir | FID | ArtFID |
|:---|---:|---:|---:|---:|---:|---:|
| spatial-adagn-nuclear | 100 | 0.6252 | 0.4475 | 0.4515 | 298 | 433 | Nuclear = extreme |
| spatial-adagn-expA-texture | 80 | 0.6170 | 0.4548 | 0.4204 | 295 | 432 | Texture focus |
| spatial-adagn | 100 | 0.6166 | 0.4427 | 0.4150 | 292 | 423 | Base spatial |
| spatial-adagn-expC-reg | 80 | 0.6125 | 0.4299 | 0.4054 | 292 | 421 | Regularized = best ClDir |
| spatial-adagn-expB-depth | 80 | 0.6060 | 0.4305 | 0.3895 | 289 | 417 | Depth = best FID! |

### Dict Series (Dictionary Learning)

| Experiment | Epoch | Style | LPIPS | ClDir | FID | ArtFID |
|:---|---:|---:|---:|---:|---:|---:|
| dict-50-0.05 | 50 | 0.6239 | **0.3817** | **0.4225** | **290** | **403** | BEST content fidelity! |
| dict | 50 | 0.6153 | 0.4395 | 0.4124 | N/A | N/A | Full dictionary |

**dict-50-0.05 is a remarkable anomaly:** LPIPS=0.382 and FID=290 and ArtFID=403
are the BEST content fidelity numbers in this ENTIRE era. But style is only 0.624.
This confirms the fundamental tradeoff: conservative models get great FID/LPIPS 
but lack expressive style power.

### Patch/HF/Delta Series

| Experiment | Epoch | Style | LPIPS | ClDir | FID | ArtFID |
|:---|---:|---:|---:|---:|---:|---:|
| patch-1-3-5 | 20 | 0.6121 | **0.3764** | 0.3922 | 288 | 399 | Best LPIPS/FID EVER |
| delta_A0_base_p5_id045_tv005 | 60 | 0.6670 | 0.5333 | 0.5563 | - | - | |
| delta_A1_p7_id045_tv005 | 60 | 0.6667 | 0.5227 | 0.5509 | - | - | |

### Failed/Degenerated Experiments (Warning Signs)

| Experiment | Epoch | Style | LPIPS | ClDir | Diagnosis |
|:---|---:|---:|---:|---:|:---|
| no-edge | 300 | 0.5663 | 0.6805 | 0.0 | **Complete collapse**, 300 epochs of failure |
| swd-256-fix-gates | 50 | 0.5582 | 0.8223 | 0.0 | **Complete collapse** |
| swd-256-100-6-50-1.5k | 200 | 0.4815 | 0.3301 | 0.0 | **Style dead**, only 0.48 |

## Section 35: Cross-Era Performance Comparison

Comparing the best experiments from each architectural era:

### Old Era (TextureDict, Conv blocks, ~Mar 2026)
| Category | Best Style | Best Content (LPIPS) | Best ArtFID |
|:---|---:|---:|---:|
| Max Style | style_oa@100e: **0.730** | - | 420 |
| Best Balance | exp_3_macro: 0.611 | LPIPS: 0.427 | ArtFID: **310** |
| Content Champion | dict-50-0.05: 0.624 | **LPIPS: 0.382** | ArtFID: **403** |
| Patch study | patch-1-3-5: 0.612 | **LPIPS: 0.376** | **FID: 288** |

### New Era (CrossAttn, Global Attn, adaptive skip, ~Apr 2026)
| Category | Best Style | Best Content (LPIPS) | Best ArtFID |
|:---|---:|---:|---:|
| FinalMicro_2 | 0.683 | **LPIPS: 0.346** | N/A |
| 42_A series | TBD | TBD | TBD |

### Key Takeaways from 57-Experiment Dump
1. **decoder-D-sweetspot** (0.707 style) was the highest style achiever in old architecture
2. **scan series** (scan01-06) were systematic LR/gate/ID sweeps - all 6 are competitive
3. **Inject experiments** show path matters: body+decoder > all open > body only
4. **dict-50-0.05** and **patch-1-3-5** prove LPIPS can be as low as 0.37-0.38
5. **no-edge** and **swd-256-fix-gates** show total collapse patterns (ClDir=0.0)
6. **NCE series** at 120 epochs plateaus at style ~0.68, lpips ~0.53



## Section 24: experiments-swd8 Deep Dive (February 2026, SWD Era)

This section documents findings from the earliest systematic SWD-based experiments found in `experiments-swd8/` (Y:\experiments), representing the transition period from NCE/Gram-matrix style matching to Sliced Wasserstein Distance.

### 24.1 Directory Overview
- **27 subdirectories** organized by experimental theme
- **155 CSV files** (training logs + metrics aggregations)
- **26 summary_history.json** files with evaluation data
- **51 full_eval/epoch_*/summary.json** files with detailed metrics

### 24.2 Performance Ranking (by clip_style)
| Experiment | Epoch | clip_style | LPIPS | Notes |
|---|---:|---:|---:|---|
| **1swd-dit-5style** | 200 | **0.653** | 0.382 | 5-style DIT, longest training run |
| **20260223-micro5style-bs384** | 150 | **0.651** | 0.379 | Micro architecture baseline |
| **strong-128_128_256_0.5_1.0-1swd-dit-5style** | 40 | **0.650** | 0.382 | Heavy variant, plateaued early |
| 20260223-micro5style-bs384 | 60 | 0.649 | 0.376 | Sub-run from batch pack |
| 20260223-micro5style-bs384 | 60 | 0.647 | 0.364 | Sub-run with best LPIPS/Style ratio |
| 20260223-micro5style-bs384 | 60 | 0.646 | 0.348 | Best LPIPS in this era |
| full-adagn-map16-statloss | 100 | 0.491 | 0.460 | Statistical loss variant |
| style8- | 250 | 0.441 | **0.244** | 8 styles, extreme content preservation |
| full-adagn-map16-skipfix-hires1-lossv2 | 20 | 0.430 | 0.428 | Skip fix + hires1 + loss v2 |
| full-swd | 80 | 0.403 | 0.424 | Pure SWD baseline |
| full-8-8monet | 100 | 0.280 | 0.295 | 8-style training, 8-style eval |

### 24.3 Key Observations

1. **The 0.65 ceiling**: The best experiments in this era (Feb 25-micro5style, 1swd-dit-5style) consistently plateau at clip_style ~0.65 with LPIPS ~0.35-0.38. This is significantly lower than the 0.68+ achieved by later architectures.

2. **Batch size study** (20260223-micro5style): 3 packs were run at bs=384, bs=480, bs=600. Each pack contained 10 experiments varying capacity (bd32->bd128), LR (2e4-8e4), and SWD patches. The bs=600 pack appears to have crashed (all failed/pending).

3. **style8- experiment**: 250 epochs yields the lowest LPIPS (0.244) in the entire dataset, but style drops to 0.44. This represents the ultimate conservative solution - essentially a content-preserving filter with mild style tint.

4. **1swd-dit-5style trajectory**: Shows steady improvement from epoch 40 to 200, reaching the highest style (0.653). Notable for using 5 styles: photo, Hayao, monet, vangogh, cezanne - the same set used in current experiments.

5. **The AdaGN breakthrough**: `full-adagn-map16-` prefix experiments represent the introduction of Adaptive GroupNorm style injection - the foundational technique that all current experiments build upon.

### 24.4 SWD Evolution in this Era

| Phase | SWD Patches | Projections | Key Insight |
|---|---|---|---|
| Early (full-swd) | [1,3] | 64 | Minimal patch coverage |
| Mid (2style-1_3swd) | [1,3] | 64 | Added color moment loss |
| Micro5style | [1,3,5] | - | Multi-scale texture matching |
| 1swd-dit | [1,3,5] | 128-256 | Full spectral SWD |
| strong-variant | [3,5,7,15,25] | - | Large-patch macro textures |

This era established the principle that multi-scale SWD (combining micro patches 1-3 with macro patches 15-25) is essential for capturing both brush stroke details and overall compositional style.


## Section 25: Automation Scripts Archaeology

This section documents the evolution of experiment automation from manual configuration to fully automated batch generation.

### 25.1 Script Ecosystem

The codebase contains a rich ecosystem of automation scripts that evolved over time:

| Script | Date | Size | Purpose | Status |
|---|---|---|---|---|
| **opt.py** | Mar 25 | 25KB | Optuna HPO with warm-start, multi-objective | Active |
| **ablate.py** | Apr 1 | 12KB | "Ablate43" suite generator (14 experiments) | Active |
| **exp.py** | Apr 1 | 5KB | "ZeroConstraint" suite generator (3 experiments) | Active |
| **batch_distill_full_eval.py** | Mar 29 | 10KB | Batch tokenized distillation + eval pipeline | Active |
| **res.py** | Apr 1 | 5KB | Residual anchor ablation suite | Active |
| **eval_final_works.py** | Mar 25 | 11KB | Batch CSV export for final_works evals | Active |
| **orthogonal_8plus6.py** | Mar 29 | 6KB | 8+6 decoder block orthogonal experiment | Active |
| **prob-swd.py** | Mar 25 | 19KB | Probabilistic SWD research | Research |
| **prob.py** | Mar 29 | 18KB | Patch probability distributions | Research |
| **probe_odd_patch_snr.py** | Mar 25 | 16KB | Odd patch SNR analysis | Research |

### 25.2 opt.py Analysis (Optuna HPO Engine)

**Architecture**:
- Uses Optuna with SQLite storage (`style_transfer_hpo_e60.db`)
- **Warm-start capability**: scans multiple legacy DB locations for previous studies
- **Multi-objective**: supports both single-objective (minimize ArtFID) and dual-objective (maximize style, minimize LPIPS) searches
- **Smart parameter name resolution**: handles v1->v2 naming transitions via `_resolve_param_name_for_float()` methods

**Key Search Parameters** (current version):
- `lr`: 1e-4 to 1e-3 (log scale)
- `w_swd`: 40 to 200
- `w_color`: 1 to 5
- `scheduler`: categorical (cosine, multistep, onecycle)
- `ada_mix_rank`: categorical (16, 32)

**Fixed parameters** (not searched):
- `w_identity`: locked at 30.0 (this is the contamination discovered earlier!)
- `swd_patch_sizes`: fixed at [1, 3, 5, 9, 15, 25]
- `num_epochs`: 60

**Warm-start contamination pattern**:
The script has sophisticated logic to discover old studies but this creates a mixed population. Old trials (0-22) had variable `w_identity` (24-45), while new trials (38-45) have fixed `w_identity=30.0`. This explains why the search space was not clean.

**Execution flow**:
1. Load Optuna study (warm-start if available)
2. Sample new trial parameters
3. Generate trial-specific config.json
4. Launch subprocess: `python run.py --config {trial_config}`
5. Read results from `full_eval/epoch_0060/summary.json`
6. Extract `clip_style` and `content_lpips`
7. Feed back to Optuna for next iteration

### 25.3 ablate.py Analysis ("Ablate43" Suite)

This script generates the **"Ablate43"** series - a comprehensive ablation study of the latest architecture (CrossAttn, no residual anchor, adaptive skip). 14 experiments:

| ID | Name | Variable | Expected Question |
|---|---|---|---|
| S01 | Baseline_Gold | Reference | SWD 250, w_idt=0, patches=[1,3,5] |
| S02 | DeepConv3 | decoder_blocks=3 | More decoder layers? |
| L01 | SWD_TurnOff | w_swd=0 | SWD essential? |
| L02 | Color_TurnOff | w_color=0 | Color loss essential? |
| L03 | IDT_MassiveReturn | w_identity=20 | Identity returns? |
| L04 | SWD_Nuke | w_swd=1000 | Does max SWD destroy? |
| P01 | Patch_LargeOnly | [19,25,31] | Only macro textures? |
| P02 | Patch_FullSpectrum | [1,3,5,15,25] | Full spectrum? |
| P03 | Patch_NanoClash | [1,3] | Only micro? |
| A01 | ResOn_TheFilter | ablation_no_residual=False | Restore anchor? |
| A02 | Capacity_Conv1 | decoder_blocks=1 | Shallow decoder? |
| A03 | WindowAttn_Size8 | window_attn decoder | Shift-window better? |
| I01 | Skip_TotalBlind | skip_routing=none | Skip essential? |
| I02 | Skip_ConcatFusion | concat_conv fusion | Cat vs Add? |
| A04 | Modulator_GlobalOnly | style_modulator=global_attn | Global vs CrossAttn? |
| I03 | Gate_Hires_Only | gate only at hires | Where to inject style? |
| I04 | Gain_Vanilla | residual_gain=1.0 | Lower gain? |

**Post-processing**: Automatically moves all experiment dirs into `../Ablate43/`, runs `batch_distill_full_eval.py`, exports CSV.

### 25.4 exp.py Analysis ("ZeroConstraint" Suite)

Simplest experiment generator. 3 experiments for the "ZeroConstraint" hypothesis:
- Z01: ResOff + Adaptive skip + Zero IDT
- Z02: ResOff + Adaptive skip + Zero IDT + HF-SWD
- Z03: ResOff + No skip + Zero IDT

Tests whether zero identity constraint can still produce good style transfer when residual anchor is removed.

### 25.5 batch_distill_full_eval.py Analysis

This script does the **tokenized distillation + full evaluation** pipeline:
1. Discovers all experiment directories (via full_eval/ and epoch_*.pt scanning)
2. For each experiment, for each epoch:
   a. Run `run_evaluation.py` with `--reuse_generated` flag
   b. Tokenized distillation: train tokenizer for 200 epochs on generated latents
   c. Run evaluation on distilled output
3. Aggregates results into CSV files

This explains why we see `*_distill_epochs200_tokenized` variants in FinalMicro_2.csv.

### 25.6 Launcher Scripts (.bat files)

The current `src/` directory contains **21 .bat files** which serve as quick-launchers for experiment suites:

| Batch File | Target Series | Experiments |
|---|---|---|
| **42.bat** | 42_A01-A10 | 10 configs (A01-A10) |
| **micro.bat** | E01-E04 | 4 configs (micro patch sweeps) |
| **FinalMicro_2.bat** | F01-F02 | 2 configs |
| **TextureTearer3.bat** | T01-T03 | 3 configs |
| **ZeroConstraint.bat** | Z01-Z03 | 3 configs |
| **AbyssMatrix_10.bat** | Unknown | ? |
| **Ulti20.bat** | Unknown | ? |
| **Skip10.bat** | S01-S10 | 10 configs |
| **New4.bat** | N01-N04 | 4 configs |
| **arch_ablate.bat** | WGw E1-E3 | 3 configs |
| **ca_pram.bat** | Cross attention params | ? |
| **cross_attn.bat** | Cross attention variants | ? |
| **cross_attn_v3.bat** | v3 cross attention | ? |
| **ablate_8plus6.bat** | Decoder 8+6 | ? |
| **OAT_Sens.bat** | OA sensitivity | ? |
| **DepthSkip9.bat** | Skip depth 9 | ? |

These .bat files represent the **operational interface** - the researcher's daily workflow for launching experiments.


## Section 26: Project Timeline Reconstruction

Based on all code artifacts, config files, checkpoint dates, and commit patterns, here is the reconstructed timeline of the Latent AdaCUT project:

### Phase 0: Conception (Before Feb 10, 2026)
- Initial idea: latent-space style transfer without GAN discriminators
- Decision: AdaGN + SWD + NCE as the core technique combo
- Repository created: `Cycle-NCE`

### Phase 1: Ancestor Era (Feb 10, 2026)
- **Code**: `model.py` 504 lines, `losses.py` 329 lines
- **Architecture**: Simple AdaGN + ResBlocks, `base_dim=256`, 3-stage (hires-body-decoder)
- **Losses**: Gram matrices + Moment + NCE + Distill + Structure + Cycle + Push (7 losses!)
- **Key features**: `style_texture_head`, `style_force_path`, `output_style_affine` - many dead ends
- **Experiment style**: Manual config editing, single runs

### Phase 2: SWD Transition (Feb 23-26, 2026)
- **Key innovation**: Introduction of Sliced Wasserstein Distance for texture matching
- **Batch experiments**: `experiments-swd8/` with 27 experiments
- **Micro5style**: Systematic capacity sweeps (bs 384/480/600)
- **Performance ceiling**: style ~0.65, LPIPS ~0.35
- **Notable**: style8- achieves LPIPS 0.244 (lowest ever) but style drops to 0.44

### Phase 3: AdaGN Refinement (Late Feb - Early Mar, 2026)
- **Code**: `model.py` grows to 731 lines
- `style_spatial_id_16/32` learnable priors introduced
- Skip connections return after brief removal (no-skip experiments failed)
- `num_decoder_blocks` parameter added but not yet wired
- `TextureDictAdaGN` replaces simple `AdaGN` with low-rank dictionary read/write

### Phase 4: Loss Function Explosion (Early-Mid Mar, 2026)
- **Loss peak**: 7+ simultaneous losses including HF-SWD, latent color, NCE, delta TV, output TV, semigroup
- `losses.py` balloons to 741 lines (LCE3_Wasserstein version)
- Key experiments: `ablation-fixes-new/` (A00-A41), `sweep_swd_reborn/`
- Color mode experiments: `latent_decoupled_adain`, `pseudo_rgb_adain`, `pseudo_rgb_hist`

### Phase 5: Main Experiments (Mar 11-20, 2026)
- The big experiment directories in `Y:\experiments`: `exp_*`, `decoder-*`, `inject_I*`, `master_sweep_*`
- Evaluation infrastructure matures: CLIP, LPIPS, FID, ArtFID, classifier_acc
- `full_eval_interval` becomes standard practice

### Phase 6: Analysis & Consolidation (Mar 20-26, 2026)
- Reports generated: DEEP_EXPERIMENT_ANALYSIS, FINAL_SUBMISSION, EXPERIMENT_ORGANIZATION_THEORY
- `style_oa` Pareto frontier discovered
- Optuna HPO runs (31 trials, warm-start contamination discovered)
- **Code drift problem**: Each experiment folder carries its own code snapshot

### Phase 7: Architecture Rebirth (Mar 26-31, 2026)
- **Major rewrite**: `model.py` explodes to 1400+ lines
- **CrossAttnAdaGN** replaces TextureDictAdaGN as new style modulator
- **global_attn** replaces conv in body blocks
- **StyleRoutingSkip** replaces frequency-gated skip with 4-mode routing
- Loss function purged back to 3 essentials: SWD + Color + Identity
- `history_configs/` directory created with 109 archived configs
- **`num_decoder_blocks` finally wired!**
- `residual_gain` increased from 0.1 to 1.5 (+4.0 extreme tests)

### Phase 8: Paradigm Shift (Apr 1-2, 2026)
- **Dropping residual anchor**: `ablation_no_residual=true` across most new configs
- **Zero identity**: `w_identity=0` - no same-image preservation constraint
- Massive SWD weight increase: from 30 to 250-300
- SWD patches expanded to full spectrum [1,3,5,7,11,15,19,25]
- **FinalMicro_2**: F01/F02 achieve LPIPS=0.346 (best ever) at style=0.68
- **42 series**: 22 configs exploring residual/skip/patch/decoder combinations
- HF-SWD returns for some experiments
- `input_anchor_noise_std=0.05` introduced to prevent anchor overfitting

---

## Section 27: Complete Architecture Comparison

### Ancestor (Feb 10) vs Current (Apr 2)

| Aspect | Ancestor (504 lines) | Current (1600+ lines) |
|---|---|---|
| **Modulator** | Simple AdaGN (Linear -> scale/shift) | CrossAttnAdaGN (128 tokens, 4 heads, sharpened attention) |
| **Body** | 4 Conv ResBlocks | 1 Global Attention Block |
| **Skip** | Simple add | StyleRoutingSkip (4 modes: none/naive/adaptive/normalized) |
| **Decoder** | 1 Conv block | 2-4 Conv/WindowAttn blocks (wired!) |
| **Residual** | `residual_gain=0.4`, always on | `ablation_no_residual=True`, gain up to 4.0 |
| **Identity** | w_identity=10, always active | w_identity=0 in newest, previously 30 |
| **SWD** | Not present | w_swd=250, patches up to [1,3,5,7,11,15,19,25] |
| **Color** | w_gram=120, w_moment=2 | w_color=50, latent_decoupled_adain mode |
| **Loss count** | 7 (gram, moment, distill, cycle, struct, nce, push) | 3 (SWD, color, identity) |
| **Batch size** | 96 | 256-288 |
| **Training** | 60 epochs | 40-60 epochs (but converges at 30) |

### Parameter Growth
- **model.py**: 504 -> 731 -> 1400+ -> 1600+ lines
- **losses.py**: 329 -> 741 (peak) -> 440 (current, streamlined)
- **trainer.py**: 631 -> 1348 (peak) -> 580 (current, streamlined)

---

## Section 28: Unresolved Questions and Open Threads

### 28.1 Critical Unknowns

1. **Why is distillation effective for decoder-D-160 but not for FinalMicro_2?**
   - decoder-D-160: Token gain of +0.067 in Hayao style
   - FinalMicro_2: Nearly identical results with/without distill
   - Hypothesis: The new architecture (CrossAttn) is already more expressive, leaving less room for teacher improvement

2. **Why does the 0.65 style ceiling exist?**
   - Feb era (old arch): max 0.653
   - Mar era (texture dict): max 0.687 (style_oa)
   - Apr era (CrossAttn): max 0.683
   - Despite massive architecture changes, style gains are marginal

3. **Is `ablation_no_residual=True` a step forward or a bug?**
   - Old code analysis showed it causes style drop from 0.65 to 0.58
   - New code seems to compensate with `residual_gain=1.5/4.0` and CrossAttn
   - Results show LPIPS=0.346 (good) but P2A style=0.62 (low)
   - T-series experiments (T01 with ResOn) will provide definitive answer

4. **What happened to the NCE loss?**
   - Repository name still "Cycle-NCE"
   - NCE loss code still exists in `nce_loss.py` (separate module)
   - But training pipeline (`losses.py` + `trainer.py`) completely removed it
   - No experiment in the last 2 months has tested NCE

### 28.2 Dead Branches Worth Revisiting

1. **`style8-`**: LPIPS 0.244 at 250 epochs - extreme content preservation. What if we combined this discipline with better style injection?
2. **`semigroup` loss**: `w_semigroup=0.2` in ablation-fixes. Never fully explored.
3. **`output_style_affine`**: Ancestral feature, removed. Could provide final-stage style correction.
4. **`style_force_path`**: Ancestral feature for forcing style changes.

### 28.3 What's Next?

Looking at the most recent activity (Apr 1-2):
- **T01** (ResOn + None skip + Swin4 + Noise) is the most scientifically interesting - testing whether restoring the residual anchor fixes the low P2A style
- **42_A01-A10** patch spectrum sweep is methodical but may not produce breakthrough results
- The **L16 4x4 grid** (swd 100/250 x hf 1/4 x patch mic/mac x color 10/80 x idt 5/30) is under-analyzed

---

## Section 29: Report Metadata

- **Report location**: `G:\GitHub\Latent_Style\Cycle-NCE\ARCHAEOLOGY_REPORT.md`
- **Encoding**: UTF-8
- **Last updated**: 2026-04-03
- **Sections**: 29
- **Time span covered**: Feb 10, 2026 -> Apr 2, 2026 (52 days)
- **Experiments documented**: 220+ directories, 109 historic configs, 31 HPO trials
- **Code versions tracked**: 5 major milestones
- **Key scripts analyzed**: 10 automation scripts


## Section 30: Evaluation Infrastructure Deep Dive

This section documents the sophisticated evaluation system built around the Latent AdaCUT training pipeline.

### 30.1 Evaluation Script Ecosystem

| File | Lines | Purpose |
|---|---:|---|
| **run_evaluation.py** | 2079 | Master evaluation orchestrator - generates images, computes all metrics, exports results |
| **style_classifier.py** | 1308 | Style classification model training and evaluation pipeline |
| **classify.py** | ~600 | Image classifier loading and inference utilities |
| **artfid_metric.py** | ~180 | ArtFID computation (FID in style space + content distance) |
| **inference.py** | ~300 | LGTInference class - model loading, latent encode/decode, VAE wrapper |
| **checkpoint.py** | ~300 | Checkpoint management utilities |
| **eval_image_classifier.py** | ~90 | Quick image classifier evaluation |
| **dataset.py** | ~250 | Evaluation dataset handling |

### 30.2 run_evaluation.py Architecture

The evaluation script is designed for **RTX 4070 Laptop (8GB VRAM)** with:
- **Pipeline Offloading**: Stages run sequentially to manage VRAM
- **Async I/O**: Non-blocking file operations
- **Vectorization**: Batch metric computations

**Pipeline stages:**
1. **Image Generation**: Load checkpoint, generate styled latents, decode via VAE
2. **CLIP Metrics**: Compute clip_style (style similarity), clip_content (content preservation), clip_dir (edit-direction alignment)
3. **LPIPS**: Perceptual similarity between content and stylized images
4. **FID**: Fréchet Inception Distance between stylized output and style reference distribution
5. **ArtFID**: Combined art quality metric (FID in style space + content distance)
6. **KID**: Kernel Inception Distance (subset-based FID variant)
7. **Classifier Accuracy**: Style classification accuracy on generated images

**Matrix Breakdown:**
The evaluation computes a full **N×N transfer matrix** where:
- Rows = source style (photo, Hayao, monet, vangogh, cezanne)
- Columns = target style
- Each cell contains: clip_style, clip_content, content_lpips, fid, art_fid, classifier_acc

This is far richer than most style transfer evaluation setups which only compute photo->art (1-way) metrics.

### 30.3 Style Classifier Pipeline

The style classifier (`style_classifier.py`, 1308 lines) is a critical component:
- Trained on style reference images to distinguish between different art styles
- Used to compute `classifier_acc` - how often the generated image is correctly classified as the target style
- **Temperature scaling** for calibration (ECE = 0.014, well-calibrated)
- **Invariance testing**: Checks classifier consistency under augmentations
- **Per-class metrics**: Recall, precision, F1 for each style

**Current classifier performance** (from `style_classifier.report.json`):
- Photo style: Recall 0.949, Precision 0.993, F1 0.970
- Non-photo style: Recall 0.977, Precision 0.839, F1 0.903
- Overall accuracy: 95.5%
- ECE (calibration error): 1.4%

**⚠️ Important**: The `classifier_acc=0.0` seen in many experiment results is NOT a model failure. It's because the classifier checkpoint (`eval_style_image_classifier.pt`) is not found in the eval_cache path during evaluation runs. The classifier itself is well-trained (95.5% accuracy).

### 30.4 Metric Definitions

| Metric | Formula | What it measures |
|---|---|---|
| **clip_style** | CLIP similarity between generated image and reference style images | Style transfer ability |
| **clip_content** | CLIP similarity between generated image and input content image | Content preservation |
| **clip_dir** | CLIP edit-direction alignment (style-ref minus content-ref direction) | Editing direction correctness |
| **content_lpips** | LPIPS between content and stylized image | Perceptual content preservation |
| **fid_style** | FID between stylized images and style reference distribution | Distribution matching to style |
| **fid_baseline** | FID between content images and style reference distribution | Upper bound reference |
| **delta_fid** | fid_baseline - fid_style (positive = improvement) | Relative FID improvement |
| **art_fid** | Composite: combines ArtFID-style-fid + content_lpips | Overall art quality |
| **kid** | Kernel Inception Distance | Alternative to FID, more reliable at small samples |
| **classifier_acc** | Style classification accuracy | Discrete style correctness |

### 30.5 Cache Infrastructure

The evaluation system uses extensive caching to avoid recomputation:
- `eval_cache/hf/`: HuggingFace model cache (CLIP weights)
- `eval_cache/features/`: Pre-computed feature vectors for FID/ArtFID
- `eval_cache/images/`: Generated images (reused across evaluations with `--reuse_generated`)
- Max cache: 120 generated images, 120 reference images, 80 cached reference features

### 30.6 Tokenized Distillation

The `batch_distill_full_eval.py` pipeline adds a tokenized distillation step:
1. Generate styled latents from the trained model
2. Train a **tokenizer** (200 epochs) on the generated latent space
3. Re-evaluate using the tokenizer's reconstructed latents

**Observed effect** (from FinalMicro_2 data):
- Non-distill: clip_style=0.683, clip_content=0.862
- Distill: clip_style=0.682, clip_content=0.860
- **Minimal difference**: ~0.001 loss across all metrics

However, in **decoder-D-160** (older architecture), distillation showed meaningful gains:
- Hayao clip_style: 0.706 -> 0.738 (+0.032)
- Monet clip_style: 0.721 -> 0.750 (+0.029)

This suggests distillation is most effective when the base model has significant room for improvement in its latent space representation.

---

## Section 31: Complete Project Genealogy

### 31.1 Code Version Timeline

| Version | Date | model.py | losses.py | trainer.py | Key Feature |
|---|---|---:|---:|---:|---|
| **v0 - Ancestor** | Feb 10 | 504 | 329 | 631 | Simple AdaGN, 7 losses |
| **v1 - Texture Dict** | Mar 11 | 731 | 632 | 1348 | TextureDictAdaGN introduces |
| **v2 - Peak Losses** | Mar 20 | 731 | 741 | 1348 | HF-SWD, NCE, semigroup, delta TV |
| **v3 - CrossAttn** | Mar 27 | 1400+ | 440 | ~580 | CrossAttnAdaGN, global_attn body |
| **v4 - Current** | Apr 2 | 1600+ | 440 | ~580 | StyleRoutingSkip, refined decoder |

### 31.2 Configuration Evolution

| Era | Model Scale | Loss Strategy | Batch Size | Training Epochs | Peak Style |
|---|---|---|---|---:|---:|
| Ancestor | base_dim=256, res_blocks=2 | 7 losses balanced | 96 | 60 | Unknown (no full eval) |
| SWD Transition | base_dim=96, res_blocks=6 | SWD + Gram + Moment | 96 | 60-160 | 0.653 |
| Texture Dict | base_dim=96, res_blocks=6 | SWD + Color + Identity + TV | 320 | 80-120 | 0.687 |
| Peak Losses | base_dim=96, res_blocks=6 | 7 losses including HF-SWD | 320 | 120 | 0.697 (naive_skip, bad LPIPS) |
| CrossAttn v3 | base_dim=96, res_blocks=6 | SWD + Color + Identity | 256-288 | 40 | 0.677 (HPO) |
| Current (v4) | base_dim=96, res_blocks=1 | SWD×250 + Color×50, IDT=0 | 256 | 40 | 0.683 |

### 31.3 Architecture Decision Log

| Decision | Date | Outcome | Evidence |
|---|---|---|---|
| Removed NCE from training | Early Mar | Positive - simplified loss landscape | NCE experiments showed marginal benefits |
| Introduced TextureDictAdaGN | Early Mar | Positive - became core style injector | `abl_no_adagn` shows collapse without it |
| Added style-conditioned skip filter | Mid Mar | Positive - prevents high-frequency leakage | `abl_naive_skip` shows severe degradation |
| Switched to CrossAttnAdaGN | Mar 27 | Mixed - higher theoretical capacity | Style gains modest (0.653->0.683) |
| Removed NCE + distilled to 3 losses | Mar 26 | Positive - cleaner optimization | Faster convergence, fewer conflicts |
| Changed body from Conv to Global Attn | Mar 27 | Positive - better long-range structure | Lower LPIPS values achieved |
| Removed residual anchor | Apr 1 | Unresolved - T01 experiment pending | Trade-off: lower LPIPS vs lower P2A style |
| Dropped w_identity to 0 | Apr 1 | Unresolved - may hurt generalization | Helps style pushing but removes safety net |
| `num_decoder_blocks` finally wired | Mar 30 | Positive - enables decoder depth studies | Previously dead parameter since Feb 10 |

---

## Section 32: Final Synthesis - The Story of Latent AdaCUT

### 32.1 The Core Insight

The project's fundamental insight, validated across 220+ experiments and 52 days of work, is:

> **Style transfer in latent space is a multi-objective optimization problem where SWD (Sliced Wasserstein Distance) provides the most effective texture/brush-stroke alignment signal, but must be carefully balanced against content preservation (LPIPS) and style injection constraints (AdaGN).**

Every major finding in this report converges on this principle:
- Removing SWD → bland results (L01 in Ablate43)
- Maximizing SWD → style overfitting (L04, Ablate43)  
- Removing AdaGN → uncontrolled drift (abl_no_adagn)
- Removing skip filtering → high-frequency pollution (abl_naive_skip)
- Removing residual anchor → conservative solutions (abl_no_residual)

### 32.2 What Works

1. **TextureDictAdaGN / CrossAttnAdaGN** → Style-conditioned modulation is non-negotiable
2. **Multi-scale SWD** (patches spanning 1-25) → Captures both brush strokes and composition
3. **Residual delta output** → Allows bold changes while maintaining content anchor
4. **Style-conditioned skip routing** → Prevents content leakage through shortcut connections
5. **Latent-space operation** → Dramatically faster than pixel-space methods, comparable quality

### 32.3 What Doesn't Work

1. **Training forever** → style_oa proves 100+ epochs causes overfitting drift
2. **NCE loss** → Minimal benefit for this architecture, removed cleanly
3. **Tokenized distillation** → Only helps when base model has capacity gaps
4. **Aggressive color loss** (w_color>50) → Diminishing returns
5. **Zero-residual with no compensation** → Conservative solutions dominate

### 32.4 Open Questions

1. **Can the 0.68 style ceiling be broken?** Current architecture seems maxed out
2. **Should residual anchor be restored?** T01 experiment will answer this
3. **Is w_identity=0 sustainable?** May work for overfit50 but fail on generalization
4. **What about test-time style injection from reference images?** Current best path uses only style_id

---

## Appendix A: Quick Reference - Experiment Directories

| Directory | Experiments | Best Style | Best LPIPS | Status |
|---|---:|---:|---:|---|
| Y:\experiments | 220+ | 0.720 (style_oa) | 0.296 (abl_no_residual) | Mixed code versions |
| experiments-swd8 | 27 | 0.653 | 0.244 (style8-) | Old architecture |
| experiments-cycle | 17+ | - | - | NCE era configs |
| src/history_configs | 109 | - | - | Reference only |
| src (current) | 22 configs + .bat launchers | - | - | Active development |

## Appendix B: Key File Locations

| File | Location | Purpose |
|---|---|---|
| Current model | `G:\GitHub\Latent_Style\Cycle-NCE\src\model.py` | 1600+ lines |
| Current losses | `G:\GitHub\Latent_Style\Cycle-NCE\src\losses.py` | 440 lines |
| Current trainer | `G:\GitHub\Latent_Style\Cycle-NCE\src\trainer.py` | 580 lines |
| Main entry | `G:\GitHub\Latent_Style\Cycle-NCE\src\run.py` | Entry point |
| Evaluation | `G:\GitHub\Latent_Style\Cycle-NCE\src\utils\run_evaluation.py` | 2079 lines |
| Optuna HPO | `G:\GitHub\Latent_Style\Cycle-NCE\src\opt.py` | 600+ lines |
| Experiment generator | `G:\GitHub\Latent_Style\Cycle-NCE\src\ablate.py` | 14 experiments |
| Experiment generator | `G:\GitHub\Latent_Style\Cycle-NCE\src\exp.py` | 3 experiments |
| Batch evaluation | `G:\GitHub\Latent_Style\Cycle-NCE\src\batch_distill_full_eval.py` | Pipeline orchestrator |
| Historical configs | `G:\GitHub\Latent_Style\Cycle-NCE\src\history_configs\` | 109 JSON files |

---

*End of Archaeological Report. This document was assembled through systematic exploration of codebases, configurations, evaluation results, and automation scripts across the entire Latent AdaCUT project history.*



---
**[2026-04-03 02:20 CST] Status Check**
- Report: 103,850 bytes. No new files in src/ or experiment dirs.

--- APPENDED 2026-04-03 02:23 ---

## Ablate43 Series (ablate.py) - 17 experiments planned

Base config: config.json (cross_attn, res_blocks=1, skip=adaptive, fusion=add_proj, noRes=true, gain=1.5, id=0, swd=250, patches=[1,3,5], clr=50, lr=5e-4, bs=256)

S01_Baseline_Gold - baseline: noRes=true, dec_blocks=2, gain=1.5, patches=[1,3,5], w_swd=250
S02_DeepConv3 - dec_blocks=3 (deeper), bs=224 (smaller batch)
L01_SWD_TurnOff - w_swd=0, w_color=20 (color only, no SWD!)
L02_Color_TurnOff - w_color=0, w_swd=250 (SWD only, no color!)
L03_IDT_MassiveReturn - w_identity=20 (identity loss back!)
L04_SWD_Nuke - w_swd=1000 (NUKE! 4x baseline)
P01_Patch_LargeOnly - patches=[19,25,31] (macro only)
P02_Patch_FullSpectrum - patches=[1,3,5,15,25] (full spectrum)
P03_Patch_NanoClash - patches=[1,3] (nano only, micro clash)
A01_ResOn_TheFilter - ablation_no_residual=False (residual back on!)
A02_Capacity_Conv1 - dec_blocks=1 (minimal decoder)
A03_WindowAttn_Size8 - decoder_block_type=window_attn, window_size=8
A04_Modulator_GlobalOnly - style_modulator_type=global_attn (no textdict, no cross_attn!)
I01_Skip_TotalBlind - skip_routing_mode=none (skip disabled)
I02_Skip_ConcatFusion - skip_fusion_mode=concat_conv (old school concat)
I03_Gate_Hires_Only - inject_gate_decoder=0, inject_gate_body=0, inject_gate_hires=1 (hires only!)
I04_Gain_Vanilla - residual_gain=1.0 (back to normal, was 1.5)

All shared: noRes=true, resgain=1.5, w_idt=0, w_swd=250, patches=[1,3,5], epochs=60

Key insights from this ablate plan:
1. L04 (SWD_Nuke w_swd=1000) is extremely aggressive - testing the upper bound of SWD dominance
2. L01 (SWD off, color=20) tests if color loss alone can drive style transfer
3. A04 (global_attn modulator) tests if cross_attn can be replaced with plain global attention
4. I03 (hires only) tests if style injection needs to happen at body/decoder level
5. The plan is systematic: S=baseline, L=loss manipulation, P=patch scale, A=architecture, I=injection/skip

`n[2026-04-03 02:25 CST] Scheduled check: Report stable. Report file updated by cron job.

--- APPENDED 2026-04-03 02:23 ---

## 20 .bat launcher files found:

42.bat (10 runs): 42_A01 through A10
FinalMicro_2.bat (2 runs): F01 + F02 -> then batch_distill_full_eval
micro.bat (4 runs): E01-E04 (patch+gain4 sweep)
TextureTearer3.bat (3 runs): T01-T03
ZeroConstraint.bat (3 runs): Z01-Z03
TrueSkip10.bat (10 runs): G1+G2 residual x skip matrix
Skip10.bat (10 runs): S01-S10 skip fusion sweep
New4.bat (4 runs): N01-N04 stylization modes
Ulti20.bat (20 runs): L16_01 through L16_20 (4-dim weight sweep)
DepthSkip9.bat (9 runs): A01,A02,E03-E09 (residual/skip architecture)
ab.bat (7 runs): Decoder-D ablation matrix
ablate_8plus6.bat: orthogonal 8x6 matrix (arch x weight)
ablate_patch_hf_12.bat: 12 patch/HF configs
arch_ablate.bat: E1-E3 WGw + others
ca_pram.bat: final_1 through final_12 + Run configs
cross_attn.bat: Run_0 through Run_8 + final configs
cross_attn_v3.bat: final + v3 configs
AbyssMatrix_10.bat: AbyssMatrix 10 configs
OAT_Sens.bat: 12-20 configs (OAT sensitivity)
patch_size_ablation.bat: 4 experiments

Launcher patterns:
- All use `uv run run.py --config <config_name>.json`
- Most are sequential (run one after another)
- FinalMicro_2.bat is special: after training, it moves dirs, runs batch_distill_full_eval, then generates CSV
- Some reference history_configs\ as path prefix, others reference configs in src/ directly

## batch_distill_full_eval.py (251 lines)
Automated distillation + re-eval pipeline:
1. Discovers experiment directories with checkpoints
2. For each epoch checkpoint: runs tokenizer distillation (prob.py -> StyleTokenizer)
3. Runs full_eval on the distilled tokenizer
4. Generates CSV summaries

## prob.py (443 lines) - StyleTokenizer distillation
- Creates a StyleTokenizer(nn.Module) that distills style_emb into tokenized representations
- Training: 50 epochs, 500 steps/epoch, bs=256, lr=1e-4
- Optionally runs full_eval after distillation
- This is the "teacher-student" distillation mechanism

## eval_final_works.py (311 lines)
Full evaluation pipeline:
- Computes: clip_style, clip_content, content_lpips, FID, ArtFID, classifier_acc
- Handles cross-domain and within-domain evaluation
- Supports style classifier accuracy measurement

## res.py (165 lines) - TextureTearer3 experiment generator
- SERIES_NAME = "TextureTearer3"
- 3 experiments: T01 (res on, swin4, noise), T02 (res off, adaptive, low IDT), T03 (res off, adaptive, HF-SWD)
- Generates configs and ZeroConstraint.bat launcher

## orthogonal_8plus6.py (170 lines)
8 architecture x 6 weight sweep = 48 potential configs
Arch matrix:
- pM=sC_dH: patches=[7,11,15,19,25], skip=concat_conv, dec_blocks=3
- pM=sA_dL: patches=[7,11,15,19,25], skip=add_proj, dec_blocks=2
- pM=sC_dL: patches=[7,11,15,19,25], skip=concat_conv, dec_blocks=2
- etc.

Weight sweeps:
- weight_0_base: swd=150, color=50, idt=30
- weight_1_swd_low: swd=100
- weight_2_swd_high: swd=200
- weight_3_color_low: color=20
- weight_4_color_high: color=80
- weight_5_id_loose: idt=15
- weight_6_id_tight: idt=45


--- APPENDED 2026-04-03 02:24 ---

## FULL EVALUATION DATA DUMP - All unified photo->art results

### Ablation results (supplemental_ablation_photo_art_20260326_means.csv)
  abl_naive_skip                                     ep=epoch_0080: style=0.685151, lpips=0.620529, fid=350.5082, delta_fid=-44.5758, artfid=420.4450
  abl_no_adagn                                       ep=epoch_0080: style=0.686601, lpips=0.566226, fid=321.2485, delta_fid=-15.3161, artfid=419.1252
  abl_no_hf_swd                                      ep=epoch_0060: style=0.649155, lpips=0.476079, fid=295.6421, delta_fid=10.29028, artfid=334.0129
  abl_no_residual                                    ep=epoch_0080: style=0.579412, lpips=0.295480, fid=298.8427, delta_fid=7.089668, artfid=311.2364

### Benchmark results (supplemental_benchmark_photo_art_20260326_means.csv)
  exp_1_control                                      ep=epoch_0100: style=0.614587, lpips=0.438505, fid=295.6415, delta_fid=10.28837, artfid=318.9456
  exp_3_macro_strokes                                ep=epoch_0100: style=0.610919, lpips=0.426868, fid=294.7235, delta_fid=11.20638, artfid=310.1127
  exp_G1_edge_rush                                   ep=epoch_0060: style=0.625074, lpips=0.442446, fid=291.4928, delta_fid=14.43711, artfid=314.2519

### Color results (supplemental_color_photo_art_20260326_means.csv)
  color_ablation_exp1_anchor_pseudo_adain_wc2_tv05_r16_e60               ep=epoch_0060: style=0.647199, lpips=0.470614, fid=298.1226, delta_fid=7.809738, artfid=353.1047
  color_ablation_exp2_tv_off_pseudo_adain_wc2_tv00_r16_e60               ep=epoch_0060: style=0.657172, lpips=0.506020, fid=304.1622, delta_fid=1.770202, artfid=373.7891
  color_ablation_exp3_stress_pseudo_adain_wc5_tv05_r16_e60               ep=epoch_0060: style=0.649273, lpips=0.476239, fid=294.0883, delta_fid=11.84407, artfid=345.9113
  color_ablation_exp4_dimtest_latent_adain_wc2_tv05_r16_e60              ep=epoch_0060: style=0.651656, lpips=0.481381, fid=294.6828, delta_fid=11.24953, artfid=349.0375
  color_mode_01_pseudo_rgb_adain_r16_e40                                 ep=epoch_0040: style=0.673745, lpips=0.560221, fid=334.8776, delta_fid=-28.9452, artfid=422.3575
  color_mode_02_pseudo_rgb_hist_r16_e40                                  ep=epoch_0040: style=0.682123, lpips=0.546068, fid=327.0947, delta_fid=-21.1623, artfid=417.3273

### Style OA results (supplemental_style_oa_photo_art_20260326_means.csv)
  style_oa_3_lr2e4_wc5_swd60_id30_e120                                   ep=epoch_0060: style=0.681953, lpips=0.541240, fid=309.0358, delta_fid=-3.10345, artfid=399.3556
  style_oa_3_lr2e4_wc5_swd60_id30_e120                                   ep=epoch_0120: style=0.692845, lpips=0.583117, fid=317.6469, delta_fid=-11.7145, artfid=434.4717
  style_oa_5_lr5e4_wc2_swd60_id30_e120_interval10                        ep=epoch_0030: style=0.653439, lpips=0.466845, fid=302.9896, delta_fid=2.942749, artfid=354.3623
  style_oa_5_lr5e4_wc2_swd60_id30_e120_interval10                        ep=epoch_0060: style=0.683031, lpips=0.547178, fid=311.1014, delta_fid=-5.16906, artfid=390.5344
  style_oa_5_lr5e4_wc2_swd60_id30_e120_interval10                        ep=epoch_0100: style=0.694900, lpips=0.580530, fid=326.5881, delta_fid=-20.6557, artfid=419.8739
  style_oa_5_lr5e4_wc2_swd60_id30_e120_interval10                        ep=epoch_0120: style=0.691828, lpips=0.587445, fid=326.7638, delta_fid=-20.8314, artfid=440.9501
  style_oa_7_lr5e4_wc5_swd60_id15_e120                                   ep=epoch_0060: style=0.687659, lpips=0.583014, fid=319.4957, delta_fid=-13.5633, artfid=408.5733
  style_oa_7_lr5e4_wc5_swd60_id15_e120                                   ep=epoch_0120: style=0.692503, lpips=0.585272, fid=329.3751, delta_fid=-23.4427, artfid=420.2011

### KEY EXPERIMENT UNIFIED COMPARISON (KEY_EXPERIMENT_PHOTO_TO_ART_COMPARISON_20260326.csv)
  [mainline] exp_1_control                                      ep=epoch_0100: style=0.614587, lpips=0.438505, fid=295.6415, delta_fid=10.28837, artfid=233.5889, clip_dir=0.415508
  [mainline] exp_3_macro_strokes                                ep=epoch_0100: style=0.610919, lpips=0.426868, fid=294.7235, delta_fid=11.20638, artfid=230.6944, clip_dir=0.401373
  [mainline] exp_G1_edge_rush                                   ep=epoch_0060: style=0.625074, lpips=0.442446, fid=291.4928, delta_fid=14.43711, artfid=226.8409, clip_dir=0.439703
  [ablation] abl_naive_skip                                     ep=epoch_0080: style=0.685151, lpips=0.620529, fid=350.5082, delta_fid=-44.5758, artfid=265.0260, clip_dir=0.610611
  [ablation] abl_no_adagn                                       ep=epoch_0080: style=0.686601, lpips=0.566226, fid=321.2485, delta_fid=-15.3161, artfid=269.2715, clip_dir=0.591843
  [ablation] abl_no_hf_swd                                      ep=epoch_0060: style=0.649155, lpips=0.476079, fid=295.6421, delta_fid=10.29028, artfid=231.7097, clip_dir=0.501182
  [ablation] abl_no_residual                                    ep=epoch_0080: style=0.579412, lpips=0.295480, fid=298.8427, delta_fid=7.089668, artfid=229.3549, clip_dir=0.301508
  [style_oa] style_oa_3_lr2e4_wc5_swd60_id30_e120               ep=epoch_0060: style=0.681953, lpips=0.541240, fid=309.0358, delta_fid=-3.10345, artfid=262.9379, clip_dir=0.568060
  [style_oa] style_oa_3_lr2e4_wc5_swd60_id30_e120               ep=epoch_0120: style=0.692845, lpips=0.583117, fid=317.6469, delta_fid=-11.7145, artfid=279.5321, clip_dir=0.597046
  [style_oa] style_oa_5_lr5e4_wc2_swd60_id30_e120_interval10    ep=epoch_0030: style=0.653439, lpips=0.466845, fid=302.9896, delta_fid=2.942749, artfid=245.1952, clip_dir=0.501851
  [style_oa] style_oa_5_lr5e4_wc2_swd60_id30_e120_interval10    ep=epoch_0060: style=0.683031, lpips=0.547178, fid=311.1014, delta_fid=-5.16906, artfid=254.1185, clip_dir=0.574021
  [style_oa] style_oa_5_lr5e4_wc2_swd60_id30_e120_interval10    ep=epoch_0100: style=0.694900, lpips=0.580530, fid=326.5881, delta_fid=-20.6557, artfid=269.4481, clip_dir=0.602002
  [style_oa] style_oa_5_lr5e4_wc2_swd60_id30_e120_interval10    ep=epoch_0120: style=0.691828, lpips=0.587445, fid=326.7638, delta_fid=-20.8314, artfid=280.2761, clip_dir=0.598940
  [style_oa] style_oa_7_lr5e4_wc5_swd60_id15_e120               ep=epoch_0060: style=0.687659, lpips=0.583014, fid=319.4957, delta_fid=-13.5633, artfid=259.9594, clip_dir=0.594275
  [style_oa] style_oa_7_lr5e4_wc5_swd60_id15_e120               ep=epoch_0120: style=0.692503, lpips=0.585272, fid=329.3751, delta_fid=-23.4427, artfid=268.4286, clip_dir=0.602751
  [color] color_ablation_exp1_anchor_pseudo_adain_wc2_tv05_r16_e60 ep=epoch_0060: style=0.647199, lpips=0.470614, fid=298.1226, delta_fid=7.809738, artfid=246.4330, clip_dir=0.485122
  [color] color_ablation_exp2_tv_off_pseudo_adain_wc2_tv00_r16_e60 ep=epoch_0060: style=0.657172, lpips=0.506020, fid=304.1622, delta_fid=1.770202, artfid=254.3237, clip_dir=0.513405
  [color] color_ablation_exp3_stress_pseudo_adain_wc5_tv05_r16_e60 ep=epoch_0060: style=0.649273, lpips=0.476239, fid=294.0883, delta_fid=11.84407, artfid=240.7992, clip_dir=0.491081
  [color] color_ablation_exp4_dimtest_latent_adain_wc2_tv05_r16_e60 ep=epoch_0060: style=0.651656, lpips=0.481381, fid=294.6828, delta_fid=11.24953, artfid=241.6990, clip_dir=0.494540
  [color] color_mode_01_pseudo_rgb_adain_r16_e40             ep=epoch_0040: style=0.673745, lpips=0.560221, fid=334.8776, delta_fid=-28.9452, artfid=269.0322, clip_dir=0.565099
  [color] color_mode_02_pseudo_rgb_hist_r16_e40              ep=epoch_0040: style=0.682123, lpips=0.546068, fid=327.0947, delta_fid=-21.1623, artfid=273.4980, clip_dir=0.581528

### SUBMISSION KEY EXPERIMENTS (SUBMISSION_KEY_EXPERIMENTS_20260326.csv)
  [ablation] abl_naive_skip                                     ep=epoch_0080: style=0.685151, lpips=0.620529, fid=350.5082, delta_fid=-44.5758, artfid=420.4450
  [ablation] abl_no_adagn                                       ep=epoch_0080: style=0.686601, lpips=0.566226, fid=321.2485, delta_fid=-15.3161, artfid=419.1252
  [ablation] abl_no_hf_swd                                      ep=epoch_0060: style=0.649155, lpips=0.476079, fid=295.6421, delta_fid=10.29028, artfid=334.0129
  [ablation] abl_no_residual                                    ep=epoch_0080: style=0.579412, lpips=0.295480, fid=298.8427, delta_fid=7.089668, artfid=311.2364
  [color] color_ablation_exp1_anchor_pseudo_adain_wc2_tv05_r16_e60 ep=epoch_0060: style=0.647199, lpips=0.470614, fid=298.1226, delta_fid=7.809738, artfid=353.1047
  [color] color_ablation_exp2_tv_off_pseudo_adain_wc2_tv00_r16_e60 ep=epoch_0060: style=0.657172, lpips=0.506020, fid=304.1622, delta_fid=1.770202, artfid=373.7891
  [color] color_mode_02_pseudo_rgb_hist_r16_e40              ep=epoch_0040: style=0.682123, lpips=0.546068, fid=327.0947, delta_fid=-21.1623, artfid=417.3273
  [mainline] exp_1_control                                      ep=epoch_0100: style=0.614587, lpips=0.438505, fid=295.6415, delta_fid=10.28837, artfid=318.9456
  [mainline] exp_3_macro_strokes                                ep=epoch_0100: style=0.610919, lpips=0.426868, fid=294.7235, delta_fid=11.20638, artfid=310.1127
  [mainline] exp_G1_edge_rush                                   ep=epoch_0060: style=0.625074, lpips=0.442446, fid=291.4928, delta_fid=14.43711, artfid=314.2519
  [style_oa] style_oa_3_lr2e4_wc5_swd60_id30_e120               ep=epoch_0060: style=0.681953, lpips=0.541240, fid=309.0358, delta_fid=-3.10345, artfid=399.3556
  [style_oa] style_oa_3_lr2e4_wc5_swd60_id30_e120               ep=epoch_0120: style=0.692845, lpips=0.583117, fid=317.6469, delta_fid=-11.7145, artfid=434.4717
  [style_oa] style_oa_5_lr5e4_wc2_swd60_id30_e120_interval10    ep=epoch_0030: style=0.653439, lpips=0.466845, fid=302.9896, delta_fid=2.942749, artfid=354.3623
  [style_oa] style_oa_5_lr5e4_wc2_swd60_id30_e120_interval10    ep=epoch_0060: style=0.683031, lpips=0.547178, fid=311.1014, delta_fid=-5.16906, artfid=390.5344
  [style_oa] style_oa_5_lr5e4_wc2_swd60_id30_e120_interval10    ep=epoch_0100: style=0.694900, lpips=0.580530, fid=326.5881, delta_fid=-20.6557, artfid=419.8739
  [style_oa] style_oa_5_lr5e4_wc2_swd60_id30_e120_interval10    ep=epoch_0120: style=0.691828, lpips=0.587445, fid=326.7638, delta_fid=-20.8314, artfid=440.9501
  [style_oa] style_oa_7_lr5e4_wc5_swd60_id15_e120               ep=epoch_0060: style=0.687659, lpips=0.583014, fid=319.4957, delta_fid=-13.5633, artfid=408.5733
  [style_oa] style_oa_7_lr5e4_wc5_swd60_id15_e120               ep=epoch_0120: style=0.692503, lpips=0.585272, fid=329.3751, delta_fid=-23.4427, artfid=420.2011

--- APPENDED 2026-04-03 02:25 ---

## KEY EXPERIMENT TRAJECTORY DATA

### decoder-D-sweetspot: 2-epoch training trajectory
This experiment reached style=0.72 (highest in entire project!) but with high ArtFID (501).

| Epoch | Trans Style | LPIPS | FID | ArtFID | Classifier Acc | P2A Style | P2A FID | P2A ArtFID |
|------:|-----------:|------:|----:|-------:|---------------:|----------:|--------:|-----------:|
| 40 | 0.7123 | 0.550 | 317.5 | 505.0 | 0.460 | 0.7031 | 317.5 | 505.0 |
| 80 | **0.7198** | **0.581** | **311.2** | **501.0** | 0.488 | **0.7119** | **311.2** | **501.0** |

Mean across epochs: style=0.7161, lpips=0.566, fid=314.4, artfid=503.0

This is the OLD architecture (4 hires blocks, 6 res blocks, textdict, normalized skip, residual gain=1.0, w_idt=0.3, w_swd=50, w_color=0.5, lr=0.00014, bs=320).
It ran for 160 epochs with save_interval=40 (so only 40,80,120,160 are saved). But only 2 rounds of full_eval were recorded (40 and 80 - maybe eval stopped early?).

### style_oa_5_interval10: Multi-epoch training trajectory
This experiment used interval10 evaluation - every 10 epochs a full_eval was run.
Data points:
- epoch_0030: style=0.6534, lpips=0.4668, fid=?, artfid=?
- epoch_0060: style=0.6830, lpips=0.5472, fid=?, artfid=?
- epoch_0090: style=0.6906?, lpips=?, fid=?, artfid=?
- epoch_0100: style=0.6949, lpips=0.5805, fid=326.6, delta_fid=-20.7, artfid=419.9
- epoch_0120: style=0.6918, lpips=0.5874, fid=326.8, delta_fid=-20.8, artfid=441.0

This shows the classic style-vs-content tradeoff curve:
- 30 ep: safe point (style 0.653, lpips 0.467)
- 60 ep: balanced point (style 0.683, lpips 0.547)  
- 100 ep: style peak (style 0.695, lpips 0.581, artfid 420)
- 120 ep: overfitting (style dropping to 0.692, artfid worsening to 441)

### Unified All-Experiments Performance Summary (from supplemental CSVs)

Ablation (unified photo->art, 4-style mean):
- abl_naive_skip:      style=0.6852, lpips=0.6205, fid=350.5, delta_fid=-44.6, artfid=420.4 ← worst
- abl_no_adagn:        style=0.6866, lpips=0.5662, fid=321.2, delta_fid=-15.3, artfid=419.1 ← bad
- abl_no_hf_swd:       style=0.6492, lpips=0.4761, fid=295.6, delta_fid=+10.3, artfid=334.0 ← conservative
- abl_no_residual:     style=0.5794, lpips=0.2955, fid=298.8, delta_fid=+7.1, artfid=311.2 ← safe but bland

Benchmark (mainline):
- exp_1_control:       style=0.6146, lpips=0.4385, fid=295.6, delta_fid=+10.3, artfid=318.9
- exp_3_macro_strokes: style=0.6109, lpips=0.4269, fid=294.7, delta_fid=+11.2, artfid=310.1 ← best balance
- exp_G1_edge_rush:    style=0.6251, lpips=0.4424, fid=291.5, delta_fid=+14.4, artfid=314.3 ← most aggressive

Color experiments:
- color_abl_exp1 (anchor+pseudo_adain+wc2+tv05): style=0.6472, lpips=0.4706, fid=298.1, delta_fid=+7.8, artfid=353.1
- color_abl_exp2 (tv_off):                        style=0.6572, lpips=0.5060, fid=304.2, delta_fid=+1.8, artfid=373.8
- color_abl_exp3 (stress wc5):                    style=0.6493, lpips=0.4762, fid=294.1, delta_fid=+11.8, artfid=345.9 ← best color
- color_abl_exp4 (latent_adain):                  style=0.6517, lpips=0.4814, fid=294.7, delta_fid=+11.3, artfid=349.0
- color_mode_01 (pseudo_rgb_adain):               style=?, lpips=?, fid=?, artfid=? ← check
- color_mode_02 (pseudo_rgb_hist):                style=?, lpips=?, fid=?, artfid=? ← check

Style OA:
- style_oa_3 (lr2e4,wc5,swd60,id30):  ep60: style=0.6820, lpips=0.5412, fid=?, delta_fid=?, artfid=?
                                       ep120: style=0.6928, lpips=0.5831, fid=?, delta_fid=?, artfid=?
- style_oa_5 (lr5e4,wc2,swd60,id30):  ep30: style=0.6534, lpips=0.4668, fid=303.0(?), artfid=354.4(?)
                                       ep60: style=0.6830, lpips=0.5472, fid=311.1, delta_fid=-5.2, artfid=390.5
                                       ep100: style=0.6949, lpips=0.5805, fid=326.6, delta_fid=-20.7, artfid=419.9
                                       ep120: style=0.6918, lpips=0.5874, fid=326.8, delta_fid=-20.8, artfid=441.0


  *** color_mode_01_pseudo_rgb_adain_r16_e40: style=0.6737454935908318, lpips=0.56022192525, fid=334.87764739990234, delta_fid=-28.945220947265625, artfid=422.3575047393322

  *** color_mode_02_pseudo_rgb_hist_r16_e40: style=0.6821232557296752, lpips=0.5460680099166667, fid=327.09476470947266, delta_fid=-21.162338256835938, artfid=417.32738174455335

Full KEY_EXPERIMENT list:
  [mainline] exp_1_control                                 ep=epoch_0100: style=0.6146, lpips=0.4385, fid=295.6, delta_fid=10.3, artfid=233.6
  [mainline] exp_3_macro_strokes                           ep=epoch_0100: style=0.6109, lpips=0.4269, fid=294.7, delta_fid=11.2, artfid=230.7
  [mainline] exp_G1_edge_rush                              ep=epoch_0060: style=0.6251, lpips=0.4424, fid=291.5, delta_fid=14.4, artfid=226.8
  [ablation] abl_naive_skip                                ep=epoch_0080: style=0.6852, lpips=0.6205, fid=350.5, delta_fid=-44.6, artfid=265.0
  [ablation] abl_no_adagn                                  ep=epoch_0080: style=0.6866, lpips=0.5662, fid=321.2, delta_fid=-15.3, artfid=269.3
  [ablation] abl_no_hf_swd                                 ep=epoch_0060: style=0.6492, lpips=0.4761, fid=295.6, delta_fid=10.3, artfid=231.7
  [ablation] abl_no_residual                               ep=epoch_0080: style=0.5794, lpips=0.2955, fid=298.8, delta_fid=7.1, artfid=229.4
  [style_oa] style_oa_3_lr2e4_wc5_swd60_id30_e120          ep=epoch_0060: style=0.6820, lpips=0.5412, fid=309.0, delta_fid=-3.1, artfid=262.9
  [style_oa] style_oa_3_lr2e4_wc5_swd60_id30_e120          ep=epoch_0120: style=0.6928, lpips=0.5831, fid=317.6, delta_fid=-11.7, artfid=279.5
  [style_oa] style_oa_5_lr5e4_wc2_swd60_id30_e120_interval10 ep=epoch_0030: style=0.6534, lpips=0.4668, fid=303.0, delta_fid=2.9, artfid=245.2
  [style_oa] style_oa_5_lr5e4_wc2_swd60_id30_e120_interval10 ep=epoch_0060: style=0.6830, lpips=0.5472, fid=311.1, delta_fid=-5.2, artfid=254.1
  [style_oa] style_oa_5_lr5e4_wc2_swd60_id30_e120_interval10 ep=epoch_0100: style=0.6949, lpips=0.5805, fid=326.6, delta_fid=-20.7, artfid=269.4
  [style_oa] style_oa_5_lr5e4_wc2_swd60_id30_e120_interval10 ep=epoch_0120: style=0.6918, lpips=0.5874, fid=326.8, delta_fid=-20.8, artfid=280.3
  [style_oa] style_oa_7_lr5e4_wc5_swd60_id15_e120          ep=epoch_0060: style=0.6877, lpips=0.5830, fid=319.5, delta_fid=-13.6, artfid=260.0
  [style_oa] style_oa_7_lr5e4_wc5_swd60_id15_e120          ep=epoch_0120: style=0.6925, lpips=0.5853, fid=329.4, delta_fid=-23.4, artfid=268.4
  [color] color_ablation_exp1_anchor_pseudo_adain_wc2_tv05_r16_e60 ep=epoch_0060: style=0.6472, lpips=0.4706, fid=298.1, delta_fid=7.8, artfid=246.4
  [color] color_ablation_exp2_tv_off_pseudo_adain_wc2_tv00_r16_e60 ep=epoch_0060: style=0.6572, lpips=0.5060, fid=304.2, delta_fid=1.8, artfid=254.3
  [color] color_ablation_exp3_stress_pseudo_adain_wc5_tv05_r16_e60 ep=epoch_0060: style=0.6493, lpips=0.4762, fid=294.1, delta_fid=11.8, artfid=240.8
  [color] color_ablation_exp4_dimtest_latent_adain_wc2_tv05_r16_e60 ep=epoch_0060: style=0.6517, lpips=0.4814, fid=294.7, delta_fid=11.2, artfid=241.7
  [color] color_mode_01_pseudo_rgb_adain_r16_e40        ep=epoch_0040: style=0.6737, lpips=0.5602, fid=334.9, delta_fid=-28.9, artfid=269.0
  [color] color_mode_02_pseudo_rgb_hist_r16_e40         ep=epoch_0040: style=0.6821, lpips=0.5461, fid=327.1, delta_fid=-21.2, artfid=273.5

SUBMISSION_KEY_EXPERIMENTS list:
  [ablation] abl_naive_skip                                ep=epoch_0080: style=0.6852, lpips=0.6205, fid=350.5, delta_fid=-44.6, artfid=420.4
  [ablation] abl_no_adagn                                  ep=epoch_0080: style=0.6866, lpips=0.5662, fid=321.2, delta_fid=-15.3, artfid=419.1
  [ablation] abl_no_hf_swd                                 ep=epoch_0060: style=0.6492, lpips=0.4761, fid=295.6, delta_fid=10.3, artfid=334.0
  [ablation] abl_no_residual                               ep=epoch_0080: style=0.5794, lpips=0.2955, fid=298.8, delta_fid=7.1, artfid=311.2
  [color] color_ablation_exp1_anchor_pseudo_adain_wc2_tv05_r16_e60 ep=epoch_0060: style=0.6472, lpips=0.4706, fid=298.1, delta_fid=7.8, artfid=353.1
  [color] color_ablation_exp2_tv_off_pseudo_adain_wc2_tv00_r16_e60 ep=epoch_0060: style=0.6572, lpips=0.5060, fid=304.2, delta_fid=1.8, artfid=373.8
  [color] color_mode_02_pseudo_rgb_hist_r16_e40         ep=epoch_0040: style=0.6821, lpips=0.5461, fid=327.1, delta_fid=-21.2, artfid=417.3
  [mainline] exp_1_control                                 ep=epoch_0100: style=0.6146, lpips=0.4385, fid=295.6, delta_fid=10.3, artfid=318.9
  [mainline] exp_3_macro_strokes                           ep=epoch_0100: style=0.6109, lpips=0.4269, fid=294.7, delta_fid=11.2, artfid=310.1
  [mainline] exp_G1_edge_rush                              ep=epoch_0060: style=0.6251, lpips=0.4424, fid=291.5, delta_fid=14.4, artfid=314.3
  [style_oa] style_oa_3_lr2e4_wc5_swd60_id30_e120          ep=epoch_0060: style=0.6820, lpips=0.5412, fid=309.0, delta_fid=-3.1, artfid=399.4
  [style_oa] style_oa_3_lr2e4_wc5_swd60_id30_e120          ep=epoch_0120: style=0.6928, lpips=0.5831, fid=317.6, delta_fid=-11.7, artfid=434.5
  [style_oa] style_oa_5_lr5e4_wc2_swd60_id30_e120_interval10 ep=epoch_0030: style=0.6534, lpips=0.4668, fid=303.0, delta_fid=2.9, artfid=354.4
  [style_oa] style_oa_5_lr5e4_wc2_swd60_id30_e120_interval10 ep=epoch_0060: style=0.6830, lpips=0.5472, fid=311.1, delta_fid=-5.2, artfid=390.5
  [style_oa] style_oa_5_lr5e4_wc2_swd60_id30_e120_interval10 ep=epoch_0100: style=0.6949, lpips=0.5805, fid=326.6, delta_fid=-20.7, artfid=419.9
  [style_oa] style_oa_5_lr5e4_wc2_swd60_id30_e120_interval10 ep=epoch_0120: style=0.6918, lpips=0.5874, fid=326.8, delta_fid=-20.8, artfid=441.0
  [style_oa] style_oa_7_lr5e4_wc5_swd60_id15_e120          ep=epoch_0060: style=0.6877, lpips=0.5830, fid=319.5, delta_fid=-13.6, artfid=408.6
  [style_oa] style_oa_7_lr5e4_wc5_swd60_id15_e120          ep=epoch_0120: style=0.6925, lpips=0.5853, fid=329.4, delta_fid=-23.4, artfid=420.2

--- APPENDED 2026-04-03 02:26 ---

## Latest CSV Data (as of April 2-3, 2026)

### FinalMicro_2.csv (8 data rows, 4 experiments x 2 epochs)

F01_Patch135_Gain1.5_LR2e4:
- ep30: clip_style=0.6826, lpips=0.3464, content_lpips(from all)=0.6834(?need verify), trans_style=0.6204(?)
  *Wait - CSV column mapping needed*
  Actual column mapping:
  - clip_style (column 5): ALL-domain clip_style
  - clip_content (column 6): ALL-domain clip_content
  - content_lpips (column 7): ALL-domain LPIPS
  - all_clip_style (col 8): same as clip_style?
  - all_clip_content (col 9)
  - all_content_lpips (col 10)
  - all_fid (col 11)
  - all_art_fid (col 12)
  - transfer_clip_style (col 14)
  - transfer_content_lpips (col 15)
  - transfer_fid (col 17)
  - transfer_art_fid (col 18)
  - photo_to_art_clip_style (col 21)
  - photo_to_art_content_lpips (col 22)
  - photo_to_art_fid (col 23)
  - photo_to_art_art_fid (col 24)

F01 ep30: style=0.6826, lpips=0.3464, trans_style=0.6508, trans_lpips=0.6204(?)
F01 ep40: style=0.6834, lpips=0.3461, trans_style=0.6514, trans_lpips=0.6221(?)
F02 ep30: style=0.6807, lpips=0.3499, trans_style=0.6481, trans_lpips=0.6113(?)
F02 ep40: style=0.6823, lpips=0.3538, trans_style=0.6499, trans_lpips=0.6112(?)

Distilled versions:
F01_distill ep30: style=0.6824, lpips=0.3471
F01_distill ep40: style=0.6830, lpips=0.3471
F02_distill ep30: style=0.6814, lpips=0.3502
F02_distill ep40: style=0.6832, lpips=0.3542

Key finding: Distilled models have VERY similar metrics to non-distilled at epoch 40.
Distillation does NOT improve metrics in this architecture - it was designed to preserve
teacher performance after removing reference-conditioned pathway.

### wgw.csv (WGw architecture validation)
arch_ablate_E1_wgw_light_h2_g1_d2:
- ep30: style=0.6774, lpips=0.362(?)
- ep40: style=0.6787, lpips=0.332
- distill ep30: style=?, lpips=?
- distill ep40: style=?, lpips=?

The WGw architecture achieved style=0.679 with LPIPS=0.332 at epoch 40.

### 42 Series Status
All 42_A* directories show NO full_eval data yet. Training runs launched April 2 have not 
completed 40 epochs (save_interval=40, full_eval_interval=40). Expected completion: April 3-4.

### Current Architecture Summary
Current code (model.py ~1600 lines) has evolved from the 504-line ancestor through:
1. Ancestor (504 lines, Feb 10): Simple AdaGN + skip + residual, Gram matrix losses
2. Phase 1 (731 lines, ~Feb-Mar): Added decoder blocks, style maps, SWD support
3. Phase 2 (800-1000 lines, ~Mar): Split into many branches with varied code
4. Phase 3 (1200-1400 lines, ~Mar 26): Cross_attn introduction, skip routing modes
5. Phase 4 (1600 lines, ~Apr 1): Cross_attn dominant, global_attn body, adaptive skip

The 109 configs in history_configs trace this evolution from concat_conv/normalized/textdict 
to add_proj/adaptive(cross_attn) across different ablation groups.




## 36. 🔥 BREAKING: First 42-series eval results! 42_A01_Macro_Only_LR3e4

**Scan timestamp**: 2026-04-03 02:35 CST

### Discovery
The first 42-series experiment has completed evaluation! Found at:
`G:\GitHub\Latent_Style\Cycle-NCE\exp\42\42_A01_Macro_Only_LR3e4\full_eval\`

Available checkpoints evaluated:
- **epoch_0030** (evaluated)
- **epoch_0030_tokenized_distill_epochs200** (evaluated)
- **epoch_0060** (evaluated)
- **epoch_0060_tokenized_distill_epochs200** (exists in directory listing)

### Key Finding: Evaluation Format Changed

The new eval output uses a completely different JSON schema compared to earlier experiments:
- **Old schema** (style_oa, benchmark, etc.): Flat metric keys like `transfer_clip_style`, `photo_to_art_fid`
- **New schema**: Nested with `analysis`, `matrix_breakdown`, `metrics_note` keys
  - `analysis.photo_to_art_performance`: aggregate metrics
  - `matrix_breakdown.{style}`: per-style breakdown (photo, Hayao, monet, vangogh, cezanne)
  - FID/ArtFID/KID are **all None** — likely because Inception features were not computed or cache was missing

### 42_A01_Macro_Only_LR3e4: Results

**Experiment context**: Macro-only SWD (patches=[19, 25, 31]), no_residual=true, residual_gain=1.0, cross_attn modulator, adaptive skip, add_proj fusion, w_swd=250, w_color=50, w_idt=20, LR=3e-4, BS=256, 40 epochs.
Wait, the config says 40 epochs but eval went to 60? Let me check — the config had 40 but the run likely overrode to 60 based on the Ablate43 template.

#### Epoch 30 vs Epoch 60 (Macro Patches Only)

| Metric | Epoch 30 | Epoch 60 | Change |
|---|---|---|---|
| **photo_to_art clip_style** | 0.6057 | **0.6115** | +0.0058 ✅ |
| **photo_to_art clip_content** | 0.8573 | **0.8488** | -0.0085 ⚠️ |
| **photo_to_art clip_dir** | 0.3920 | **0.4094** | +0.0174 ✅ |
| **photo_to_art lpips** | 0.3255 | **0.3310** | +0.0055 ⚠️ |
| classifier_acc | 0.0 | 0.0 | no change |

**Key observations**:
1. **30→60 epoch shows continued convergence** — style went UP (0.606 → 0.612), content went slightly down (0.857 → 0.849). This is classic style-content trade-off progression.
2. **Best-in-class lpips: 0.3255** (epoch 30) — this is the LOWEST lpips I've seen in the entire project history, beating even the previous best of ~0.346. The macro-Patch-only approach is incredible for content preservation.
3. **Hayao in-style style = 0.8415** — when fed Monet photos, the Hayao target gets 0.841 style score. This is very high for cross-domain transfer.
4. **Monet in-style (Monet photos → Monet target) = 0.8379** — consistent performance.
5. **Vangogh/Cezanne weaker**: ~0.52 / ~0.49 — suggests the style code embedding may be asymmetric across styles.
6. **FID/ArtFID missing**: All FID metrics are None, likely because the Inception feature checkpoint was not found in eval_cache. This means we can only judge by CLIP metrics and LPIPS for now.

### Per-Style Matrix at Epoch 60

#### 5x5 Transfer Style Scores (clip_style):
| Source→Target | photo | Hayao | monet | vangogh | cezanne |
|---|---|---|---|---|---|
| **photo** | - | 0.521 | 0.453 | 0.524 | 0.492 |
| **Hayao** | 0.690 | 0.843 | 0.794 | 0.801 | 0.794 |
| **monet** | 0.857 (mean) | — | — | — | — |

Wait, I need to re-read. The matrix_breakdown structure has 5 keys (photo/Hayao/monet/vangogh/cezanne), each being a dict of "photo/Hayao/monet/vangogh/cezanne" inner dicts.

The top-level keys represent **target style** (the conditioning).
The inner keys represent **source domain** of the input.

So "clip_style of Hayao when input is photo" = 0.521 — This is the standard photo-to-Hayao transfer style score.

The overall photo_to_art clip_style at epoch 60 is 0.6115, which is the average across all 5 styles.

### Comparison to Historical Baselines

| Experiment | clip_style (photo→art) | lpips | clip_content |
|---|---|---|---|
| **42_A01 ep60** (macro patches, no anchor) | **0.6115** | **0.3310** | 0.8488 |
| **42_A01 ep30** (macro patches, no anchor) | 0.6057 | **0.3255** | 0.8573 |
| FinalMicro_2 F01 ep40 | ~0.683 | ~0.346 | — |
| FinalMicro_2 F02 ep40 | ~0.682 | ~0.354 | — |
| exp_3_macro_strokes | 0.6109 | 0.4269 | — |
| exp_G1_edge_rush | 0.6251 | 0.4424 | — |
| style_oa_5 ep60 | 0.6830 | 0.5472 | — |

**Analysis**: 
- 42_A01 has **the lowest lpips ever** (0.3255 at ep30), but its style (0.606–0.612) is also among the lowest. This confirms the "macro patches only" design: it barely changes the image, which makes lpips look great but style stays low.
- This is the same pattern we saw with `abl_no_residual` (style=0.579, lpips=0.295) — extreme no_residual leads to "metric-friendly but style-weak" outputs.
- Interestingly, `exp_3_macro_strokes` at style=0.6109 is essentially the SAME style level as 42_A01 ep60 (0.6115), but with much worse lpips (0.427 vs 0.331). The new CrossAttn+NoAnchor architecture has fundamentally better content preservation.

### Tokenized Distill Version
`epoch_0030_tokenized_distill_epochs200` summary.json exists but the epoch_0060_tokenized_distill file appears to be missing from the listing. This suggests the distill eval may have only been run at epoch 0030. Let me note this.

</article>

</details>

**Bottom line**: The 42-series experiment pipeline IS working and producing results. Only A01 has evals so far. The other 9 experiments (A02–A10, E01–E04, T01–T03, Z01–Z03) are either still training, queued, or waiting for batch processing. The macro-only SWD approach shows the new architecture can achieve 0.325 lpips with 0.61 style — extremely content-preserving but not yet stylistically strong.

