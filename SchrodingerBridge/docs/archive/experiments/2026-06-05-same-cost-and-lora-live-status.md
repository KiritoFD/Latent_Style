# Same-Cost And LoRA Live Status

Date: 2026-06-05

Scope: status snapshot for the two still-open branches of the current AAAI
evidence push:

- `K-longer` same-cost closure on the remote `3060`
- `img2img-turbo` / LoRA-side local smoke on the `4070`

This note is a live-state snapshot, not a final result table.

## 1. SDXL-Latent Branch Is Closed

The `Distinct5-512` local `SDXL-fix` latent branch is no longer open.

Closed evidence:

- [2026-06-05-distinct5-sdxl-fix-eval-repair.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-05-distinct5-sdxl-fix-eval-repair.md)
- [distinct5_eval_curve_comparison.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/local_distinct5_sdxl_fix_vs_ema_20260605/distinct5_eval_curve_comparison.md)

Current read:

- no `SDXL-fix` epoch beats the retained `LBM-K e1` EMA-latent baseline on
  transfer or full `clip_style`
- `epoch_0004` briefly improves LPIPS at the cost of weaker style
- `epoch_0008` is worse than EMA on both style and LPIPS

Working conclusion:

- do not replace the current `EMA` latent training path with the present
  `SDXL-fix` latent path for the paper

## 2. Remote Same-Cost: `K-longer`

Authoritative remote run root:

- `I:\Github\Latent_Style\SchrodingerBridge\exp\aaai2027_longer_train_k_seed42_b44_e8\full_eval_artfid`

As of this snapshot:

- `epoch_0005`
  - `images/`
  - `metrics.csv`
  - `summary.json`
  - `summary_grid.png`
  - still missing standalone `aggregate_targetwise_artfid.json`
- `epoch_0006`
  - `images/`
  - `metrics.csv`
  - `summary.json`
  - `summary_grid.png`
  - still missing standalone `aggregate_targetwise_artfid.json`
- `epoch_0007`
  - active in the remote holder according to
    [remote_k_longer_eval_5_8_artfid.log](/G:/GitHub/Latent_Style/SchrodingerBridge/_codex_tmp/remote_k_longer_eval_5_8_artfid.log)
- `epoch_0008`
  - not closed yet at snapshot time

Important closure rule:

- `summary.json` alone is not a skip-safe complete packet
- paper-facing closure requires:
  - `images/`
  - `metrics.csv`
  - `summary.json`
  - `aggregate_targetwise_artfid.json`

Local closure watcher now active:

- [watch_remote_k_longer_then_close_artfid.ps1](/G:/GitHub/Latent_Style/SchrodingerBridge/_codex_tmp/watch_remote_k_longer_then_close_artfid.ps1)
- log:
  - [watch_remote_k_longer_then_close_artfid.log](/G:/GitHub/Latent_Style/SchrodingerBridge/_codex_tmp/watch_remote_k_longer_then_close_artfid.log)

Current behavior:

- it does not interrupt the live remote tmux holder
- it waits for the holder to stop reporting `LIVE`
- it then back-fills standalone `aggregate_targetwise_artfid.json` for
  `epoch_0005 .. epoch_0008`

## 3. Local LoRA-Side Smoke: `img2img-turbo`

Launcher:

- [run_img2img_turbo_distinct5_smoke.py](/G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/scripts/run_img2img_turbo_distinct5_smoke.py)

Current smoke target:

- `Early_Renaissance`

Dataset root:

- `F:\wikiart_distinct5_img2img_turbo_datasets\to_Early_Renaissance`

Resolved output root used by the training entry:

- `Y:\Latent_Style\Related_Works\runs\img2img_turbo_distinct5_smoke_auto\Early_Renaissance`

Launch artifacts:

- `launch_manifest.json`
- `launch_command.txt`
- `train.log`

Observed runtime state from this snapshot:

- launcher process:
  - wrapper `python` still alive
- child training process:
  - `src/train_cyclegan_turbo.py`
- GPU state:
  - about `7.8 / 8.2 GB`
  - `100%` utilization on the local `RTX 4070 Laptop GPU`

What is already proven:

- the run is no longer a preflight-only stub
- model downloads and startup passed
- validation/reference preparation passed
- the actual training loop has been entered (`Steps: 0/20` observed)

What is not yet proven at this snapshot:

- first checkpoint write
- first validation image bundle after training starts
- final smoke exit code

## 4. Timing Sidecar

Timing-sidecar work is already landed as repo state, not just a note:

- [timing_tables_leibniz.tex](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai_submission/snippets/timing_tables_leibniz.tex)
- [timing_sidecar_appendix.tex](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai_submission/snippets/timing_sidecar_appendix.tex)
- [paper_aaai2026.tex](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai_submission/paper_aaai2026.tex)

This means the timing sidecar is no longer blocked on finding the earlier
sub-agent output; the paper-facing TeX hooks are now in the repo and compile.

## 5. Current Gate

The global objective is still open until both of the following become true:

1. `K-longer` reaches a skip-safe same-cost closure for `epoch_0005 .. 0008`
   including standalone targetwise ArtFID packets.
2. `img2img-turbo` smoke yields concrete post-start evidence
   (`checkpoint` and/or validation bundle), so the LoRA-side comparison can be
   promoted from launch-readiness to actual experimental evidence.
