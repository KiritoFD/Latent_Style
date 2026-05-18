# SchrodingerBridge (LBM) — Codebase Guide

## Quick Start

### Training
```bash
cd SchrodingerBridge
PYTHONPATH=src python src/run.py --config config.json
```

For experiment configs (inherit from base):
```bash
PYTHONPATH=src python src/run.py --config configs/exp_sanity.json
```

### Evaluation
```bash
python run_evaluation.py <checkpoint_dir>
```
This runs inference on all `.pt` files in the directory, computes CLIP style/content + LPIPS, and saves results to `<checkpoint_dir>/full_eval/batch_summary.csv`.

### Config Inheritance
Configs support `_base` for inheritance:
```json
{
  "_base": "../config.json",
  "bridge": { "terminal_swd_weight": 2.0 }
}
```

---

## Architecture

### Core Files
| File | Lines | Purpose |
|------|-------|---------|
| `src/losses.py` | 436 | OTFlowMatchingObjective — flow matching + kinetic + terminal SWD |
| `src/ot_cost.py` | 76 | SWDTransportCost — full-band random projection SWD |
| `src/trainer.py` | 614 | SBTrainer — training loop, CSV logging, checkpoints |
| `src/model.py` | ~350 | TimeConditionedLANCETBridge — bridge wrapper over backbone |
| `src/lancet_backbone.py` | ~1580 | LatentAdaCUT — U-Net backbone with style modulation |
| `src/run.py` | ~170 | Entry point |
| `src/config_loader.py` | ~44 | Config loading with `_base` inheritance |

### Loss Components (post-cleanup)
The training loss is:
```
L = L_flow + w_kinetic * L_kinetic + w_curvature * L_curvature + terminal_swd_weight * L_terminal_swd
```

- **L_flow**: Flow matching MSE between predicted and target velocity
- **L_kinetic**: Isotropic kinetic energy regularizer (velocity^2)
- **L_curvature**: Velocity smoothness at t vs t+dt (default off: w_curvature=0)
- **L_terminal_swd**: SWD between integrated endpoint and OT-matched target

### Key Config Parameters

#### `model.*`
- `style_strength_default` (1.0): Style intensity at inference [0, 1]
- `skip_routing_mode` ("none"): Skip connection mode (none/naive/adaptive/normalized)
- `output_moment_match` (false): AdaIN-style output alignment

#### `bridge.*`
- `w_kinetic` (1.5): Kinetic energy weight
- `w_curvature` (0.0): Curvature regularization weight
- `terminal_swd_weight` (0.15): Terminal SWD loss weight
- `terminal_num_steps` (4): Integration steps for terminal SWD

#### `training.*`
- `batch_size` (64): Training batch size
- `num_epochs` (80): Training epochs
- `save_interval` (10): Checkpoint save frequency

---

## Phase 1 Cleanup (Completed)

Removed all experimentally-verified dead code. **4 commits, ~650 lines deleted.**

### What Was Removed

1. **Heuristic content losses** (`losses.py` -499 lines):
   - `calc_latent_patch_nce_loss` — PatchNCE contrastive loss
   - `calc_low_freq_structure_loss` — dead code, never called
   - `_calc_local_contextual_color_loss` — local color alignment
   - `_compute_omf_details` / `_compute_omf` — OMF discrete mode (253 lines)
   - `_freq_split` — high/low frequency decomposition
   - `_cosine_lock_loss` — cycle consistency
   - `_collect_repulsive_components` — repulsive diversity loss
   - Config params: `w_color`, `w_repulsive`, `w_nce`, `w_cycle`, `w_low_freq`, `objective_mode`

2. **Frequency split in OT cost** (`ot_cost.py` -153 lines):
   - `_get_sobel_kernels` — Sobel edge detection
   - `_compute_fused_hf_feature` — gradient magnitude
   - `_prepare_micro_features` / `_prepare_macro_features` — HF/LF decomposition
   - Config params: `swd_use_high_freq`, `swd_hf_weight_ratio`, `swd_micro/macro_weight`

3. **Log bloat** (`trainer.py` -28 lines):
   - `_TRAIN_LOG_COLUMNS`: 33 → 12 columns
   - tqdm: 12 → 6 items (loss, flow, kin, ot, tswd, t)

4. **Config cleanup** (`config.json`):
   - Removed: `objective_mode`, `w_low_freq`, `w_cycle`, `w_color`, `w_repulsive`, `w_nce`, `low_freq_kernel_size`, `swd_use_high_freq`

### Why These Were Safe to Remove
- `w_color=0.0`, `w_repulsive=0.0`, `w_nce=0.0` — already disabled in config
- `w_low_freq=1.0`, `w_cycle=1.0` — only used inside `_compute_omf_details` which was gated by `objective_mode="omf"`. After removing OMF mode, these became dead code
- `calc_low_freq_structure_loss` — defined but never called anywhere
- Frequency split — replaced by full-band SWD which is faster and avoids manifold tearing

---

## Phase 2 A/B Tests (Planned)

Three switches to test with isolated config changes:

### Switch 1: skip_routing_mode
- `"none"` (current) → `"normalized"` (style-modulated skip connections)
- Hypothesis: Skip connections may improve style transfer by injecting multi-scale encoder features
- **Result**: Tested, no measurable difference (clip_style 0.635 vs 0.635)

### Switch 2: w_curvature
- `0.0` (current) → `1.0`
- Hypothesis: Explicit curvature regularization may straighten flow field
- Not yet tested

### Switch 3: output_moment_match
- `false` (current) → `true`
- Hypothesis: AdaIN output alignment may fix color drift
- Not yet tested

---

## Known Issues

### clip_style Ceiling (~0.67)
All experiments produce clip_style_all ≈ 0.67, clip_style_transfer ≈ 0.635. Aggressive parameters (terminal_swd_weight=2.0, style_strength=1.5) did not improve this.

**Root cause identified**: The inference pipeline in `run_evaluation.py` uses `style_strength=1.0` from `src/inference_config.json`, overriding the model's `style_strength_default`. To fix:
1. Add `"full_eval_style_strength": 1.5` to `training` section in config, OR
2. Modify `src/inference_config.json` to change the default

The `inference_config.json` mapping chain:
```
config.training.full_eval_style_strength
  → inference_config.py resolve_full_eval_section()
    → full_eval.style_strength
      → LGTInference(style_strength=...)
        → model._resolve_style_strength()
```

### Epoch 4-8 Eval Failures
In the sanity experiment (terminal_swd_weight=2.0), epochs 4-8 failed during eval. Likely OOM during inference with aggressive training parameters producing larger activations.

---

## Remote Server

Server: `100.115.18.62:2222`, user `administrator`

### Long-running tasks
Use `schtasks` (NOT `Start-Process` — it ties to SSH session):
```
ssh administrator@100.115.18.62 "schtasks /create /tn TaskName /tr \"path\to\script.bat\" /sc once /st 00:00 /f && schtasks /run /tn TaskName"
```

### GameViewer restart
```
ssh administrator@100.115.18.62 "sc stop GameViewerService & timeout /t 5 /nobreak >nul & sc start GameViewerService"
```
Or run: `I:\Github\Latent_Style\exp\highres\restart_gameviewer.bat`

### SaMST training (in progress)
Pipeline: `I:\Github\Latent_Style\exp\highres\samst_full_pipeline.py`
Status: Baroque epoch 30/50, other styles waiting
Log: `I:\Github\Latent_Style\exp\highres\samst_pipeline.log`

---

## Experiment Results

| Exp | skip_routing | w_curv | tswd_w | style_str | clip_style_all | clip_style_t | lpips_t | Notes |
|-----|-------------|--------|--------|-----------|---------------|-------------|---------|-------|
| baseline | none | 0.0 | 0.15 | 1.0 | 0.671 | 0.635 | 0.304 | Phase 1 cleaned |
| exp1 | normalized | 0.0 | 0.15 | 1.0 | 0.672 | 0.635 | 0.303 | No difference |
| sanity | none | 0.0 | 2.0 | 1.5 | 0.671 | 0.635 | 0.304 | No improvement, epochs 4-8 eval failed |

**Conclusion**: clip_style is bottlenecked by inference style_strength (hardcoded to 1.0 in inference_config.json), not by training parameters.
