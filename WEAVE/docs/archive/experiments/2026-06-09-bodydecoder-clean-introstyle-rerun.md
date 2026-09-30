# BodyDecoder Clean IntroStyle Rerun

Date: 2026-06-09

Problem found:

- the remote `3060` had drifted into an invalid multi-lane state
- active GPU consumers included:
  - `aaai2027_inmortal_hold4mid_e8_spatial_carriergate_bodydecoder_seed42_b8a2` post-train full-eval
  - `aaai2027_inmortal_xpred_kmanifold_pattn_edgegated_anisostokes_queue_from_e13_seed42_b8a2` training
  - extra watcher / eval helper processes
- this violated the single-lane machine contract

Why the earlier `bodydecoder` eval is not paper-safe:

- it was running while another training lane was live
- it was still on the older `CLIP-only` fast-eval contract
- log evidence showed `checkpoint/model key mismatch` fallback during eval model load
- therefore the existing `full_eval/epoch_0004..0006` outputs should be treated as provisional only

Immediate correction:

1. kill the conflicting GPU consumers
2. return the remote card to idle
3. sync the current local `src/` packet
4. rerun clean eval for the `bodydecoder` packet under the reviewed contract

Root cause found during rerun:

- the remote machine still had a stale shadow file:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/src/utils/config_schema.py`
- when `src/utils/run_evaluation.py` was executed directly, Python imported that stale `config_schema` first
- the stale schema did not contain the newer mechanism fields such as:
  - `tokenizer_projection_mode`
  - `style_spatial_mode`
  - `style_injection_mode`
  - `proximal_mode`
- that silently forced eval-time model reconstruction back to the older factorized tokenizer path and caused the checkpoint/model mismatch

Correction:

- delete the stale remote shadow file
- ensure eval-time imports resolve from:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/src/config_schema.py`
- `src/utils/inference.py` and `src/utils/run_evaluation.py` now push `/src` to the front of `sys.path` before local imports

Chosen clean closure for this pass:

- packet:
  - `aaai2027_inmortal_hold4mid_e8_spatial_carriergate_bodydecoder_seed42_b8a2`
- checkpoints:
  - `epoch_0008`
  - `epoch_0012`
- metrics:
  - `LPIPS`
  - optional `CLIP-S` triage
  - `IntroStyle` sidecar
- no `ArtFID` in this inner-loop repair pass

Expected outputs:

- `summary.json`
- `metrics.csv`
- `introstyle_metrics.csv`
- `introstyle_summary.json`

Observed clean `CLIP/LPIPS` result:

- `epoch_0008`
  - transfer `clip_style = 0.6881147`
  - transfer `content_lpips = 0.5177287`
- `epoch_0012`
  - transfer `clip_style = 0.6881476`
  - transfer `content_lpips = 0.5171345`

Read:

- this confirms the `body+decoder spatial_carrier_gate` line remains on the same weak plateau as the earlier dirty run
- relative to the current mainline anchors, it does not reopen a useful frontier:
  - still below `LBM-Knee` on style
  - far worse than `Hold4Mid` on structure

Current `IntroStyle` blocker after schema repair:

- the remote eval process now reaches the `IntroStyle` sidecar call correctly
- but the sidecar fails inside the remote `samam312` env with:
  - `cannot import name 'EncoderDecoderCache' from 'transformers'`
- observed remote package versions:
  - `transformers = 4.41.2`
  - `diffusers = 0.29.2`

Interpretation:

- the schema/import bug is fixed
- the remaining `IntroStyle` failure is now a pure environment-compatibility issue
- until that env gap is repaired, this packet has only clean `LPIPS + CLIP` closure, not `IntroStyle` closure

Decision rule after rerun:

- if `e12` does not reopen style relative to the existing `Hold4Mid` / `LBM-Knee` anchors under `IntroStyle`, this family should be closed as a negative or near-negative rescue branch
- if it does reopen target-specific style without collapsing structure, then it earns a fuller retained-point sweep
