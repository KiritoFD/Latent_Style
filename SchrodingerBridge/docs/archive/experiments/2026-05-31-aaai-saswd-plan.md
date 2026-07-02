# AAAI Draft and SA-SWD Experiment Plan

## Decision

The review suggestion is directionally reasonable, but it must be constrained to claims supported by code and experiments.

Adopt now:
- Rename the terminal semantic projection loss as **Semantic-Aligned Sliced Wasserstein (SA-SWD)**.
- Frame LBM as controlled latent transport: OT anchors endpoints, flow matching learns the path, kinetic regularization limits displacement, and SA-SWD corrects terminal target-style statistics along semantic routing axes.
- Add an explicit Random-SWD vs SA-SWD ablation path in code, so the paper can defend that the novelty is projection-axis selection rather than SWD itself.
- Tighten the training-time claim by tying extrapolated numbers to same-hardware single-epoch profiling.

Do not adopt without evidence:
- Do not claim "first", "perfectly eliminates artifacts", or "90% of deployments".
- Do not insert placeholder MUSIQ/HF-KID/LPIPS values into the paper.
- Do not claim user-study preference until an actual pairwise study is run.

## Code Change

`BridgeConfig` now exposes:

```json
"terminal_swd_axis_source": "semantic"
```

Allowed values:
- `semantic`: main SA-SWD path. Projection axes are selected from semantic cross-attention key responses.
- `random`: matched ablation. The same endpoint and terminal weight are used, but the terminal loss falls back to ordinary aligned SWD projections.

The helper script below generates paired configs from any current base config:

```powershell
py -3 tools\make_saswd_ablation_configs.py `
  --base configs\<base>.json `
  --out-dir configs `
  --prefix <base>
```

This produces:
- `<base>_saswd_semantic.json`
- `<base>_saswd_random.json`

## Remote Experiment Plan

Run only on the remote RTX 3060. Keep VRAM around 9.0G-10.8G for formal runs.

1. Select the current best reproducible base config:
   - preferred: the true-gradient tokenbudget branch if quick/full eval improves target direction without LPIPS regression;
   - fallback: the current best true-integrate EMA base.

2. Generate paired ablation configs:
   - `terminal_swd_axis_source=semantic`
   - `terminal_swd_axis_source=random`

3. Train both from the same initialization:
   - same checkpoint or from-scratch seed,
   - same batch size,
   - same terminal weight,
   - same number of epochs,
   - same evaluation checkpoints.

4. Evaluation:
   - quick n=6 eval after the first completed checkpoint;
   - full strict 750 if quick eval is not clearly bad;
   - record CLIP-style, CLIP-content, LPIPS, CLIP-dir, ArtFID, and artifact diagnostics when available.

5. Acceptance criterion for the paper claim:
   - SA-SWD must improve the style/content trade-off over Random-SWD, or show lower artifact diagnostics at comparable CLIP-style and LPIPS.
   - If Random-SWD matches or beats SA-SWD, the paper should not center the novelty on semantic projection-axis selection.

## Figure Plan

Add one supplementary diagnostic figure after the ablation is confirmed:
- content image;
- semantic cross-attention response heatmap;
- target patches selected as top SA-SWD projection axes;
- generated comparison for Random-SWD vs SA-SWD.

This figure is explanatory only; it should not replace the quantitative ablation.
