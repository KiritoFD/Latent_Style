# Submission Repository Completion Audit

Date: 2026-07-15
Branch: `submission`

## Requirement Evidence

| Requirement | Evidence | Status |
|---|---|---|
| Commit before cleanup and work on `submission` | Cleanup/reproduction history is preserved as frequent commits; current branch is `submission`. | Pass |
| Reproduce before refactoring | The clean 15-epoch baseline and all epoch metrics were committed before root-layout retirement and project rename. | Pass |
| Promote `src/` to the project root | `run.py`, `model.py`, `trainer.py`, and `utils/` are root modules; no active local or remote `src/` exists. Fixed-checkpoint latent output is exactly equal to the archived implementation. | Pass |
| Use project-relative paths | `config.json` and `inference.json` contain no drive, UNC, `/mnt/`, or `$index:` paths. The machine-specific dataset index was retired. | Pass |
| Align local and remote datasets | Both machines use physical `data/train` and `data/test` directories under WEAVE. Remote audit found 5,021 train files including caches/manifests and 150 test files. | Pass |
| Consolidate active configuration | `config.json` is the canonical model/training/data config; `inference.json` is the canonical 8-step inference/evaluation config. Historical configs are archived. | Pass |
| Train from scratch without post-processing | Canonical config has no resume checkpoint, `freeze_mode=none`, image/latent post-processing disabled, and no output appearance alignment. Tests enforce these invariants. | Pass |
| Evaluate every baseline epoch | `baseline_epoch_metrics.csv` contains exactly epochs 1 through 15 with DINO-S, CLIP-S, DINO-C, and LPIPS. The launcher trains first, then `batch_eval_all.py` evaluates every saved checkpoint. | Pass |
| Select the best point without a mixed score | Epoch 6 is selected directly by primary DINO-S with stable DINO-C/LPIPS; no custom combined metric is used. | Pass |
| Evaluate every parameter point | `endpoint_adain_axis.csv` contains complete 750-image evaluations for scales 1.5, 2.0, and 2.5. Scale 2.0 is retained; 2.5 collapses content. | Pass |
| Attempt architecture improvement from scratch | The oriented target-HF route was trained for 15 epochs from scratch and every epoch was evaluated. It raises the style ceiling but is not promoted because of content cost. | Pass |
| Rename the project to WEAVE | Commit `68884b684` records 5,010 exact `R100` path renames. Local active path is `G:\GitHub\Latent_Style\WEAVE`; remote active path is `I:\Github\Latent_Style\WEAVE`. | Pass |
| Verify the final state | Local: 42 tests pass, plus both CLI entry points and config loading. Remote: both CLI entry points and config loading pass; config/inference SHA-256 values exactly match local. | Pass |

## Canonical Results

| Point | DINO-S | CLIP-S | LPIPS | DINO-C |
|---|---:|---:|---:|---:|
| Fresh baseline epoch 6 | 0.4867438668 | 0.7073850305 | 0.2507627213 | 0.8280021926 |
| Oriented HF epoch 4 | 0.4915434669 | 0.7125622596 | 0.2595971213 | 0.8103006359 |

The baseline remains canonical because the architecture candidate does not
improve style without material content loss.

## Hash Contract

| File | SHA-256 |
|---|---|
| `config.json` | `FAEBA748463055B26C4C6527953964442CAD2359C1DE41E5DBC55552DC44C4E7` |
| `inference.json` | `C488272F0AC7EC6407C5881E261E9054C415593C9B89173CBDC939E2742423B7` |

These hashes match on local and remote WEAVE trees.

## Windows Workspace Note

The current Codex task was opened with the former directory as its process
working directory. Windows therefore retains an empty, untracked
`G:\GitHub\Latent_Style\SchrodingerBridge` shell until that handle is released.
It contains zero files and is not part of the Git tree; all project, data, run,
and archive content is under `WEAVE/`.
