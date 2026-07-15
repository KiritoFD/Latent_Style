# Submission Baseline Reproduction

## Result

The clean 15-epoch reproduction passes. Epoch 6 is the selected checkpoint because it has the highest DINO-S while retaining the paper baseline's content quality. No mixed score is used.

| Run | Epoch | DINO-S (higher) | CLIP-S (higher) | LPIPS (lower) | DINO-C (higher) |
|---|---:|---:|---:|---:|---:|
| Paper row | 10 | 0.4859 | 0.7075 | 0.2583 | 0.8287 |
| Fresh reproduction | 6 | **0.4867** | 0.7074 | **0.2508** | 0.8280 |
| Difference | | +0.0008 | -0.0001 | -0.0075 | -0.0007 |

The small DINO-C difference is not a content-collapse signal: LPIPS improves, DINO-C remains within 0.001 of the paper row, and epochs 6-8 form a stable content plateau. The DINO-S peak occurs at epoch 6 and then declines, so evaluating every epoch is necessary even though training itself is short.

Full-precision results and checkpoint hashes are in `baseline_epoch_metrics.csv`.

## Fixed Protocol

- Training preset: `brk_a`, seed 42, 15 epochs, one checkpoint per epoch.
- Training start: fresh initialization; the log states `No checkpoint found, start from scratch`.
- Training data: 5 styles with 1,000 packed latent samples each (5,000 active samples).
- Evaluation board: 150 D5 source images crossed with 5 target styles, for 750 outputs per checkpoint.
- Inference: 8 Euler steps and endpoint AdaIN scale 2.0.
- Metrics: DINO-S is primary; CLIP-S is secondary; DINO-C and LPIPS reject content-collapse gains.
- Selection: maximize DINO-S subject to stable content metrics. No custom combined score.

The model has 873,680 parameters. Remote training ran from 2026-07-15 00:49:23 to 00:54:05 (Asia/Shanghai), saved all 15 checkpoints, and reached a final loss of 1.0364. Peak allocated/reserved VRAM was 5.58/6.33 GB. Full evaluation of all checkpoints took 2,979 seconds after the cache contract was fixed.

## Portable Paths

Both machines use the same paths relative to `WEAVE/`:

```text
data/train/                  # packed training latents and pairing cache
data/test/                   # 150-image D5 board
runs/cache/hf/hub/           # untracked CLIP, DINOv2, and VAE snapshots
runs/submission/...          # untracked checkpoints, images, metrics, and logs
```

The local and remote datasets are physically stored at `data/train` and `data/test`; they are not junctions or machine-specific links. Dataset and model-cache contents are deliberately untracked.

The remote model snapshots initially existed only under the user-level Hugging Face cache. They were copied into `runs/cache/hf/hub` before the successful strict-offline evaluation. This keeps all tracked configuration paths relative without forcing model downloads during evaluation.

## Commands

From `WEAVE/` on either machine:

```powershell
# Fresh training followed by evaluation of every epoch
powershell -ExecutionPolicy Bypass -File scripts/run_submission_repro.ps1

# Resume only the evaluation stage; completed summaries are reused
powershell -ExecutionPolicy Bypass -File scripts/run_submission_repro.ps1 -EvalOnly
```

Do not pass `-AllowNetwork` for a release reproduction. Missing snapshots should fail immediately rather than mutate the evaluation environment.

## Provenance

- Branch: `submission`.
- Training launch state: commit `4171d5f6b` (source/config content is unchanged by the later fail-fast evaluator fix).
- Evaluation guardrail fix: commit `dd044d1c6`.
- Training config SHA-256: `573a8030bf478bf3830e4a12d92c626c2aa392ce5ad6f575fe7cfa79f07d9759`.
- Inference override SHA-256: `7a0712bb9c3bfce425d7aa5789428236a94969d0afa585d767089d1cc56f0321`.
- Selected checkpoint SHA-256: `67ca62f377f1606f2369904ebd9535f250d5b98caf254cb50ec834a788aec621`.

The remote working directory is a synchronized execution copy rather than a Git checkout, so file hashes, the committed launcher state, checkpoint hashes, and captured logs are the reproducibility anchors.

## Gate Decision

The pre-refactor baseline was reproduced, fixed-checkpoint and full-board root-layout equivalence passed, and the legacy `src/` entry points were retired. The project now uses only root modules, project-relative data paths, `config.json`, and `inference.json` for submission runs.
