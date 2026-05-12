# Baseline Reproduction And Evaluation Requirements

## Goal

This document turns the current baseline reproduction work into a paper-grade comparison track.

The immediate objective is not to claim SOTA yet. The immediate objective is to align the repository with the three evaluation lines that matter most for 2024-2025 style transfer papers:

1. `CAST / StyleID / SRCA-SM` style 800-output protocol
2. `SaMST` multi-style protocol
3. `ArtBank` artist/domain protocol

At the moment, this repo already has:

- internal `CLIP-style`
- internal `CLIP-content`
- internal `content LPIPS`
- `ArtFID/FID` support in `SchrodingerBridge/src/utils/run_evaluation.py`
- post-hoc modern metrics support in `SchrodingerBridge/src/utils/modern_metrics.py`
  - `cmmd`
  - `dino_structure`
  - `gram_micro`
  - `gram_macro`

But the baseline pipeline is still centered on the old lightweight protocol, so the main task is to upgrade the comparison stack without losing the work already completed.

Current execution note on `2026-05-11`:

- the current engineering protocol is frozen to the Ours reference folder at `SchrodingerBridge/exp/pareto_probe_4/S-add__K-3_C-2_W-10_Col-15/full_eval/epoch_0001/images`
- this gives `5 source styles x 5 target styles x 30 source images = 750 outputs`
- styles are `photo / monet / vangogh / cezanne / Hayao`
- `ukiyoe` is not used in this protocol
- the paper-exact `20 content x 40 style = 800 outputs` protocol remains a separate future alignment step for direct CAST/StyleID comparisons

## Current Repository Reality

### What already exists

- Baseline launch scripts exist in `Related_Works/baseline_pipeline/scripts/`
  - `copy_cut_results.py`
  - `run_s2wat.py`
  - `run_samst.py`
  - `run_styleid.py`
  - `run_style_aligned.py`
- Old baseline evaluation exists in `Related_Works/baseline_pipeline/evaluation/`
  - `eval_all_baselines.py`
  - `eval_with_sb.py`
  - `run_sb_eval_all.py`
  - `run_sb_eval_v2.py`
- Main-model evaluation stack already supports stronger metrics in `SchrodingerBridge`
  - `ArtFID/FID` hooks in `SchrodingerBridge/src/utils/run_evaluation.py`
  - post-hoc modern metrics in `SchrodingerBridge/src/utils/modern_metrics.py`
  - batch refresh helper in `SchrodingerBridge/append_modern_metrics.py`

### What is already reproduced or partially reproduced

- `CUT`
  - current-protocol outputs were manually migrated from reusable `Related_Works/runs/cut_5x5/infer_5x5/images`
  - `results/cut/protocol_a_800/images`: `750` matched images
  - current protocol metrics are recorded
- `StyleID`
  - inference script exists and has been repaired for the current diffusers img2img call path
  - full current-protocol inference completed for `photo / monet / vangogh / cezanne / Hayao`
  - `results/styleid/protocol_a_800/images`: `750` matched images
  - current protocol metrics and per-target inference timing are recorded
- `SaMST`
  - training and inference scripts exist
  - checkpoints exist for `monet / vangogh / cezanne / ukiyoe / Hayao`
  - `photo` checkpoint/output is still missing
  - current-protocol outputs were manually migrated from reusable external full-eval results
  - `results/samst/protocol_a_800/images`: `750` matched images
  - current protocol metrics are recorded
- `S2WAT`
  - training and inference scripts exist
  - checkpoints exist for `photo / monet / vangogh / cezanne / Hayao`
  - `ukiyoe` is not complete
  - current-protocol inference completed for `photo / monet / vangogh / cezanne / Hayao`
  - `results/s2wat/protocol_a_800/images`: `750` matched images
  - current protocol metrics are recorded
- `StyleAligned`
  - script exists
  - no reliable recorded evaluation table found yet

### What the current baseline tables actually cover

`Related_Works/baseline_pipeline/results/protocol_eval_table_protocol_a_800.csv` is the best current snapshot of protocol-aligned baseline comparisons already executed. It covers:

- `ours_pareto_probe_4_epoch_0001`
- `cut`
- `samst`
- `s2wat`
- `styleid`
- `sdturbo`
- `sdedit_str_0p10`
- `sdedit_str_0p20`
- `sdedit_str_0p35`
- `sdedit_str_0p40`

with only:

- `content_lpips`
- `clip_style`
- `clip_content`
- `eval_sec`

This is useful as the current internal screening table, but it is not the final main-paper protocol because `ArtFID/FID/CFSD` and paper-exact `800` outputs still need to land.

## AAAI Baseline Strategy

The paper story should not rely on only `CycleGAN/CUT`, and it should not rely on only `AdaIN/StyTr2`.

Current target positioning:

```text
fast latent-space multi-style / domain-level artistic style transfer
```

This requires three comparison lines:

1. multi-style / efficiency line
2. arbitrary style transfer line
3. artist-domain transfer line

### Main-paper minimum baseline set

```text
AdaIN
StyTr2
AesPA-Net
AesFA
CAST
StyleID
SaMST
CycleGAN
CUT / FastCUT
Ours
```

### Stronger AAAI baseline set

```text
AdaIN
AdaAttN
StyTr2
AesPA-Net
AesFA
CAST
StyleID
SaMST
CycleGAN
CUT / FastCUT
ArtBank
ACID-Style
Ours
```

### Main-paper table plan

Table 1: COCO/WikiArt standard AST protocol.

```text
20 content x 40 style = 800 outputs
Methods: AdaIN, AdaAttN, StyTr2, AesPA-Net, AesFA, CAST, StyleID, Ours
Metrics: ArtFID, FID, LPIPS, CFSD, CLIP-content, CLIP-style, Time, Params
```

Table 2: SaMST-style multi-style efficiency protocol.

```text
500 content x 100 style = 50k outputs
Methods: StyleBank, S2WAT/MicroAST/ATK, SaMST, Ours
Metrics: ArtFID, CF, GE+LP, FLOPs, Time, Params, OIP, Style Capacity
```

Table 3: artist-domain / CycleGAN-CUT protocol.

```text
Domains: Monet, Van Gogh, Cezanne, Ukiyo-e, Photo
Methods: CycleGAN, CUT/FastCUT, ArtBank, Ours
Metrics: CLIP artist prompt score, LPIPS, DINO/CLIP-content, KID/FID pooled by target, train time, inference time
```

Table 4: user study.

```text
Content preservation preference
Style fidelity preference
Overall preference / aesthetics
```

### Execution priority

First batch:

```text
AdaIN
StyTr2
AesPA-Net
AesFA
CAST
StyleID
SaMST
CycleGAN
CUT / FastCUT
Ours
```

Second batch:

```text
ArtBank
AdaAttN
EFDM
ACID-Style
```

Optional:

```text
QuantArt
InST
DiffuseIT
DiffStyle
CSGO
StyleSSP
Attention Distillation
```

### Claim boundary

Until all relevant protocol-aligned tables exist, do not claim:

```text
Ours outperforms SaMST / CAST / StyleID / AesFA.
```

Current defensible framing is:

```text
content-preserving + very fast adaptation
```

not yet:

```text
style fidelity SOTA
```

## Main Protocols To Align

## Protocol A: 800-output paper protocol

This is the first paper protocol to land.

Important distinction:

- current engineering protocol: `750` outputs from the frozen Ours reference folder, no `ukiyoe`
- paper target protocol: `800` outputs following the standard `20 content x 40 style` COCO/WikiArt-style setup

- dataset: `COCO-like content + WikiArt-like style`
- size: `20 content x 40 style = 800 outputs`
- resolution: `512 x 512`
- primary metrics:
  - `ArtFID`
  - `FID`
  - `LPIPS`
  - `CFSD` or `CSFD`
  - `CLIP-style`
  - `Time`
  - `Params`

Target comparison set:

- `AdaIN`
- `StyTr2`
- `AesPA-Net`
- `CAST`
- `StyleID`
- `SaMST`
- `Ours`

Optional additions if capacity allows:

- `AesFA`
- `SRCA-SM`
- `AdaAttN`
- `EFDM`

## Protocol B: SaMST multi-style protocol

This is the second protocol to land.

- size: `500 content x 100 style = 50,000 outputs`
- primary metrics:
  - `ArtFID`
  - `CF`
  - `GE+LP`
  - `FLOPs`
  - `Time`
  - `Params`
  - `OIP`
  - `Style capacity`

This protocol is expensive and should be treated as a final-stage run, not a daily dev loop.

## Protocol C: ArtBank artist/domain protocol

This is the third protocol to land if the paper story remains artist/domain oriented.

- dataset: artist/domain transfer set
- primary metrics:
  - `CLIP score to artist prompt`
  - `user preference`
  - `Time`

Target comparison set:

- `CycleGAN`
- `CUT`
- `ArtBank`
- `Ours`

## Metric Status In This Repo

### Already implemented or close to usable

- `CLIP-style`
- `CLIP-content`
- `content LPIPS`
- `ArtFID`
- `FID`
- `cmmd`
- `dino_structure`
- `gram_micro`
- `gram_macro`

### Not found as working repo-level implementations yet

- `CFSD / CSFD`
- `CF`
- `GE+LP`
- `AesPA style loss`
- `AesPA pattern difference`
- `SSIM`
- `user preference`
- unified `Time / Params / FLOPs / OIP / style capacity` collection

This means the repo is no longer at zero, but it is also not yet aligned with any single strong paper protocol end-to-end.

## Baseline Priority

## Tier 0: must land

- `SaMST`
- `CAST`
- `StyleID`
- `AesPA-Net`
- `StyTr2`
- `AdaIN`

## Tier 1: strongly recommended

- `AesFA`
- `ArtBank`
- `CUT`
- `CycleGAN`

## Tier 2: stretch

- `SRCA-SM`
- `QuantArt`
- `EFDM`
- `AdaAttN`
- `InST`
- `DiffuseIT`
- `DiffStyle`

## Execution Plan

## Phase 1: stop protocol drift

Deliverables:

- use this file as the baseline evaluation contract
- maintain live status in `SchrodingerBridge/docs/experiments/2026-05-11-baseline-reproduction-progress.md`
- maintain run history in `SchrodingerBridge/docs/experiments/2026-05-11-baseline-reproduction-lab-notes.md`

## Phase 2: unify baseline outputs

Requirement:

- each baseline must produce a clean evaluation-ready image set
- naming should remain compatible with `SchrodingerBridge` reuse flow
- image set should be explicitly versioned by protocol

Minimum naming rule:

- preserve filenames with source style, source stem, and target style
- avoid mixing old exploratory outputs and protocol outputs in the same folder

Recommended output roots:

- `Related_Works/baseline_pipeline/results/<baseline>/protocol_a_800/`
- `Related_Works/baseline_pipeline/results/<baseline>/protocol_b_samst/`
- `Related_Works/baseline_pipeline/results/<baseline>/protocol_c_artist/`

## Phase 3: baseline evaluation upgrade

Replace the old assumption that baseline evaluation only means:

- `clip_style`
- `clip_content`
- `content_lpips`

New requirement:

- baseline outputs must be runnable through the same `SchrodingerBridge` evaluation stack used by the main model
- for protocol A, baseline tables should at minimum include:
  - `ArtFID`
  - `FID`
  - `LPIPS`
  - `CLIP-style`
  - `CLIP-content`
  - `cmmd`
  - `dino_structure`
  - `gram_micro`
  - `gram_macro`

Note:

- `cmmd / dino_structure / gram_*` are not one-to-one replacements for paper metrics like `CFSD`, but they are useful as internal bridge metrics before the official missing metrics are implemented.

## Phase 4: missing metric implementation

Required additions:

1. `CFSD / CSFD`
2. `Time / Params / FLOPs`
3. `CF / GE+LP`
4. optional `SSIM`
5. optional user-study tooling

## Immediate Next Runs

Recommended order:

1. freeze the current protocol-A file layout and image naming
2. finish baseline completeness checks for `CUT / StyleID / SaMST / S2WAT / StyleAligned`
3. patch baseline evaluation so baseline outputs can reuse `SchrodingerBridge` `ArtFID/FID + modern metrics`
4. generate the first current-protocol comparison subset for:
   - `Ours`
   - `StyleID`
   - `SaMST`
   - `CUT`
5. add `AdaIN` and `StyTr2`
6. only after protocol A is stable, move to `SaMST` 50k and `ArtBank` artist/domain evaluation

Current state:

- steps 1, 2, and the CLIP/LPIPS part of step 4 are done for the `750`-image engineering protocol
- `StyleID` inference timing is recorded in `results/runtime_summary_protocol_a_800.csv`
- next execution focus is official-checkpoint setup or fallback reproduction for `AdaIN / StyTr2 / CAST / AesPA-Net / AesFA / CycleGAN`

## Final Main-Paper Scope

Do not place a dozen baselines into the main table.

Main paper should be scoped to:

- Table 1 quality comparison: `AdaIN / StyTr2 / AesPA-Net / AesFA / CAST / StyleID / SaMST / Ours`
- Table 2 efficiency and scalability: `SaMST / CAST / StyleID / Ours`
- Figure 1 time-to-quality: `CycleGAN / FastCUT / SaMST / Ours`
- Table 3 ablation: six key variants only
- Table 4 user study: `Ours` versus `CAST / StyleID / SaMST / StyTr2`

Execution consequence:

- `CycleGAN` and `FastCUT` are not part of the AST main quality table
- `CycleGAN` is trained locally only for the time-to-quality figure
- `FastCUT/CUT` already has local checkpoints and migrated outputs, so it should not be retrained now
- `AdaIN / StyTr2 / AesPA-Net / AesFA / CAST` should use official pretrained inference paths, not local scratch training, unless official weights are unavailable

## Current Honest Claim Boundary

Until protocol A is landed, the honest claim boundary is:

- the repo shows promising internal content-preservation behavior
- the repo already has partial baseline reproductions
- the repo already has code support for stronger metrics than the baseline tables currently show
- the repo does not yet support a fair headline claim against `SaMST / CAST / StyleID / AesPA / AesFA`

Do not claim:

- better than `SaMST`
- better than `CAST`
- better than `StyleID`
- better than `AesFA`

until the corresponding protocol-aligned table is actually produced.
