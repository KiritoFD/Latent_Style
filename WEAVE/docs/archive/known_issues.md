# Known Issues

## 1. clip_style Ceiling (~0.67)

**Symptom:** All experiments produce `clip_style_all` ≈ 0.67, `clip_style_transfer` ≈ 0.635. Aggressive training parameters (terminal_swd_weight=2.0, style_strength=1.5) do not improve this.

**Root cause:** The inference pipeline in `run_evaluation.py` uses `style_strength=1.0` from `src/inference_config.json`, overriding the model's `style_strength_default`.

The inference config resolution chain:
```
inference_config.json → inference section → style_strength: 1.0
  ↓
resolve_full_eval_section()
  ↓ maps training.full_eval_style_strength → full_eval.style_strength
  ↓
LGTInference(style_strength=1.0)  ← hardcoded default
  ↓
model._resolve_style_strength(1.0)
  ↓ returns 1.0 (not style_strength_default)
```

**Fix options:**
1. Add `"full_eval_style_strength": 1.5` to `training` section in config.json
2. Modify `src/inference_config.json` to change the default
3. Modify `run_evaluation.py` to pass `style_strength=None` (let model use its default)

**Status:** Unresolved. Requires code change.

## 2. Epoch 4-8 Eval Failures

**Symptom:** In the sanity experiment (terminal_swd_weight=2.0), epochs 4-8 failed during eval while epochs 1-3 succeeded.

**Likely cause:** OOM during inference. Aggressive training parameters (terminal_swd_weight=2.0) may produce larger activations that overflow GPU memory during the eval inference pass.

**Status:** Not investigated. May resolve with smaller eval batch size.

## 3. model.py vs lancet_backbone.py Default Mismatch

**Symptom:** Two `_MODEL_CONFIG_DEFAULTS` dicts exist with partially different defaults.

Example: `model.py` defaults `skip_routing_mode` to `"none"` while `lancet_backbone.py` defaults to `"normalized"`.

**Impact:** The `model.py` version is what `trainer.py` calls, so its defaults take precedence. But this can cause confusion when reading the backbone code.

**Status:** Low priority. Documented here for reference.

## 4. Deleted File Noise in Git

**Symptom:** `git status` shows hundreds of deleted files from Cycle-NCE and other directories outside SchrodingerBridge/.

**Cause:** The git repo root is at `G:/GitHub/Latent_Style/`, not `SchrodingerBridge/`. Changes in parent directories show up.

**Workaround:** When committing, use explicit file paths: `git add SchrodingerBridge/config.json SchrodingerBridge/src/...`

**Status:** Informational.
