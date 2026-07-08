# Semantic SWD Notes

This directory tracks the semantic/cross-attention-guided SWD cleanup for the AAAI 2027 LBM/WEAVE experiments.

## Current Takeaway

The correct direction is to treat cross-attention guidance as a local empirical mass for SWD sampling, not as a multiplicative feature amplitude. The current active implementation now does that, and a real-batch probe verifies:

- `style_latent_conditioning_active = 1.0`
- `swd_guidance_active = 1.0`
- `last_cross_attn_guidance` has shape `[B, 1, 64, 64]` with nonzero spatial variance

The important model-side fix is that spectral training now feeds the sampled target reference latent into the style-token pathway. Before this, `SpectralODEObjective620` looked for `conditioning["target_style_latent"]`, but the packed dataset only returned `target_style`; therefore the model silently fell back to class-level `style_id` memory.

## Active Code Path

- Training entry: `src/run.py`
- Trainer: `src/trainer.py`
- Model factory: `src/model.py`
- Active spectral model: `src/spectral_bridge620.py`
- Active spectral objective: `src/spectral_losses620.py`
- SWD probe: `scripts/probe_semantic_swd_batch.py`

`src/trainer.py` now refuses to train if active model/loss modules are loaded from `src/exp/*` historical snapshots. This keeps future mechanism experiments from accidentally running stale code.

## Key Commands

Probe a real batch:

```powershell
python scripts\probe_semantic_swd_batch.py --config configs\semantic_swd_musiq\semantic_swd_ref_guided_cons5.json --batch-size 4
```

Train the current reference-latent guided config:

```powershell
python src\run.py --config configs\semantic_swd_musiq\semantic_swd_ref_guided_cons5.json
```

Compile check:

```powershell
python -m py_compile src\trainer.py src\spectral_losses620.py src\spectral_bridge620.py src\utils\training.py scripts\probe_semantic_swd_batch.py
```

## Current Results Snapshot

Reference-latent guided SWD, 5 epochs:

- Run: `exp/semantic_swd_ref_guided_cons5`
- Full eval: `exp/semantic_swd_ref_guided_cons5/full_eval/epoch_0005/summary.json`
- All-pairs CLIP-S: `0.7203`
- All-pairs LPIPS: `0.4015`
- Transfer CLIP-S: `0.6923`
- Transfer LPIPS: `0.4140`
- MUSIQ: pending for this new reference-latent run

Old pre-fix MUSIQ values, not directly claimable for the new mechanism:

- `old_dwt`: `41.1092`
- `semantic_swd_global_clean5`: `40.3275`
- `semantic_swd_guided5`: `40.7904`
- `semantic_swd_guided_cons5`: `40.9104`

## Open Items

- Recompute MUSIQ for `semantic_swd_ref_guided_cons5`.
- Run `semantic_swd_ref_global_clean5` as a fair control: same target latent tokens, no guided SWD.
- Only update the paper claim after the new MUSIQ and CLIP/LPIPS tradeoff are compared against the v4 table.
