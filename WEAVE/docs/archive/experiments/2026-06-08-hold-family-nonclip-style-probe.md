# Hold-Family Non-CLIP Style Probe Attempt

Date: 2026-06-08

Scope:

- use the existing Distinct5 non-CLIP style classifier to diagnose the new hold-family points
- compare:
  - `LBM-Knee e13`
  - `LBM-PS-v2 e13`
  - `Hold4Mid e8`
  - `Hold4SlowMid e12`
  - `Seedream-4.5`

Manifest:

- [2026-06-08-hold-family-nonclip-style-probe-manifest.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-08-hold-family-nonclip-style-probe-manifest.csv)

Probe assets:

- classifier:
  - [distinct5_convnext_style_classifier.pt](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/distinct5_convnext_style_classifier.pt)
- script:
  - [eval_nonclip_style_probe.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/eval_nonclip_style_probe.py)

Observed blocker:

- the probe runs successfully on existing paper-facing points whose eval bundles saved generated images
- it fails on `Hold4Mid e8` and, by the same logic, would also fail on `Hold4SlowMid e12`
- concrete reason:
  - the current fast eval path keeps `metrics.csv` and `summary.json`
  - but it was run with `--no-save_generated_images`
  - therefore the synced `epoch_0008/metrics.csv` exists, while the corresponding generated-image files needed by the non-CLIP classifier do not

Why this matters:

- for the current project, `CLIP-S + LPIPS` is no longer enough for surprising points
- any unusually strong geometry/content point should now satisfy an additional closure rule:
  - rerun the selected checkpoint with generated images saved
  - then run the non-CLIP style probe
  - and, when relevant, include it in a direct visual comparison against `Seedream`

Operational conclusion:

- from now on, every retained point that is likely to influence theory or paper writing should be classified into one of two buckets:
  - `fast-screen only`
  - `paper-facing audit point`
- `paper-facing audit points` must keep images
- the hold-family geometry anchors are now clearly in the second bucket

Next action:

- rerun the selected hold-family checkpoints with `save_generated_images=true`
- then rerun this manifest and compare:
  - target-style accuracy
  - target probability
  - target-source margin
  - visual failure modes relative to `Seedream`
