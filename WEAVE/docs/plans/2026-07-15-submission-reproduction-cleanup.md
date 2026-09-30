# Submission Reproduction and Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Reproduce the paper baseline under one fixed protocol, then reorganize the project into a portable submission layout without changing model behavior.

**Architecture:** The existing `src/` implementation remains untouched until a fresh 15-epoch reproduction has produced per-epoch DINO-S, CLIP-S, DINO-C, and LPIPS results. After that gate passes, the active Python package is copied to the project root, paths are made project-relative, and training plus inference configuration is reduced to one canonical training JSON and one inference JSON. Compatibility is checked by loading the same checkpoint and comparing outputs before removing legacy entry points.

**Tech Stack:** Python 3.12, PyTorch, diffusers VAE, DINOv2-small, CLIP/LPIPS evaluation, PowerShell, Windows OpenSSH, Git.

---

## Invariants

- Branch: `submission`.
- Reproduction starts from the committed pre-refactor code at `f5bf0cc7e`.
- Training configuration: `brk_a`, seed 42, 15 epochs, checkpoint every epoch.
- Paper inference configuration: endpoint AdaIN scale 2.0, 8 Euler steps.
- Primary style metric: DINO-S. CLIP-S is secondary.
- DINO-C and LPIPS must be reported for every point; style gains caused by content collapse are rejected.
- No custom mixed metric is used to rank checkpoints.
- Downloaded models, datasets, generated images, and caches are not committed.

## Task 1: Freeze the pre-refactor workspace

**Files:**
- Modify: `.gitignore`
- Modify nested baseline repositories only to preserve their local evaluation scripts and ignore downloaded weights.

**Steps:**

1. Inspect untracked file sizes and exclude generated images/model caches.
2. Commit each dirty nested Git repository.
3. Stage the complete main-repository code, documentation, configs, metrics, and deletions.
4. Commit the workspace snapshot.
5. Verify `git status --short` is empty.

**Expected commits:**

- `bc12a1d01 chore: snapshot full research workspace before submission cleanup`
- `f5bf0cc7e chore: record top-level StyleShot dependency state`

## Task 2: Create an isolated pre-refactor reproduction

**Files:**
- Read: `src/default_config.json`
- Read: `configs/eval_adain_20.json`
- Read: `scripts/batch_eval_all.py`
- Create remotely: `I:/Github/Latent_Style/submission_repro_pre_refactor/`

**Steps:**

1. Run local import/config tests on the committed source.
2. Copy the committed `src/`, `dataset_index.json`, canonical inference override, and batch evaluator into the isolated remote directory.
3. Keep all output under `exp/submission/repro_brk_a_15ep/` relative to that directory.
4. Train from scratch for 15 epochs with `save_interval=1`.
5. Do not reuse checkpoints from the earlier `brk_a_ll03_15ep` run.
6. Commit the exact reproduction launcher before starting the run.

**Verification:**

```powershell
python src/run.py --config src/default_config.json
```

Expected: 15 distinct checkpoints, no resume message, finite losses, and no missing-path warning.

## Task 3: Evaluate every epoch under the paper protocol

**Files:**
- Use: `scripts/batch_eval_all.py`
- Use: `configs/eval_adain_20.json`
- Create: `docs/reproduction/baseline_epoch_metrics.csv`
- Create: `docs/reproduction/baseline_reproduction.md`

**Steps:**

1. Generate all 750 D5 outputs for every epoch using endpoint AdaIN 2.0.
2. Compute CLIP-S and LPIPS from each epoch's generated board.
3. Load DINOv2-small once and compute DINO-S/DINO-C for every epoch.
4. Export one CSV row per epoch with checkpoint hash/config hash/protocol.
5. Select the DINO-S peak among points that do not show material content collapse.
6. Compare the best reproduced point with the paper row: DINO-S 0.4859, CLIP-S 0.7075, LPIPS 0.2583, DINO-C 0.8287.
7. Commit metrics and the reproduction conclusion.

## Task 4: Establish one portable path contract

**Files:**
- Create: `data/README.md`
- Create: `scripts/setup_data_links.ps1`
- Modify: `dataset_index.json` or retire it after compatibility verification.
- Modify: canonical configuration files.

**Path contract:**

```text
SchrodingerBridge/
  data/train/       # junction to cached training latents
  data/test/        # junction to the D5 test images
  data/hf_cache/    # junction to Hugging Face model cache
  runs/             # all checkpoints and evaluation outputs
```

Both local and remote commands use these same project-relative paths. The setup script accepts machine-specific absolute targets only when creating junctions; no absolute target is stored in a tracked JSON file.

**Verification:**

1. Resolve every configured path from the repository root.
2. Run a one-batch dataset smoke test locally and remotely.
3. Search tracked Python/JSON/PowerShell files for drive-letter paths.
4. Commit the path contract independently.

## Task 5: Promote the active source package to the project root

**Files:**
- Copy initially: `src/*.py` to the `SchrodingerBridge/` root.
- Copy initially: `src/utils/` to `SchrodingerBridge/utils/`.
- Modify: imports, entry points, tests, and evaluation scripts.
- Keep temporarily: `src/` compatibility wrappers.

**Steps:**

1. Add failing import/entry-point tests for the root layout.
2. Copy active modules without deleting `src/`.
3. Update imports to work from the repository root.
4. Load the reproduced best checkpoint in old and new layouts.
5. Compare fixed-input outputs with tight numerical tolerance.
6. Run training and inference smoke tests.
7. Only after equivalence passes, replace old modules with thin compatibility wrappers or remove them in a separate commit.

## Task 6: Converge configuration

**Files:**
- Create: `config.json` as the only canonical training/model/data configuration.
- Create: `inference.json` as the complete paper inference configuration.
- Archive or remove active duplicate configs after extracting useful provenance.

**Steps:**

1. Generate a resolved-config dump from the reproduced run.
2. Remove fields not consumed by active code.
3. Preserve explicit values for every active model, bridge, training, data, checkpoint, and evaluation field.
4. Put all paper inference values in `inference.json`; do not rely on undocumented CLI defaults.
5. Add schema tests rejecting unknown keys and absolute tracked paths.
6. Verify the canonical files reproduce the same model parameter count and fixed-input output.
7. Commit configuration convergence separately.

## Task 7: Parameter experiments with complete per-epoch evaluation

**Files:**
- Create experiment manifests under `experiments/` from `config.json` without duplicating the full config.
- Create one result CSV and one conclusion document per axis.

**Order:**

1. Inference-only endpoint AdaIN points around the reproduced best checkpoint.
2. Training length/early stopping confirmed from all 15 checkpoints.
3. One training parameter axis at a time; no coupled sweep until a single-axis gain is reproduced.
4. Architecture changes only after the clean baseline remains numerically equivalent.

Every trained point saves and evaluates every epoch. Every inference-only point evaluates the identical checkpoint and image board. Each axis receives its own Git commit with config, metrics, and conclusion.

## Task 8: Final documentation and release check

**Files:**
- Rewrite: `README.md`
- Create/update: `docs/REPRODUCTION.md`
- Create/update: `docs/REMOTE_WORKFLOW.md`
- Update: `docs/713/HANDOFF_2026-07-13.md` or replace it with a clearly current handoff.

**Verification:**

1. Fresh-shell local smoke using only documented commands.
2. Fresh remote-directory smoke using the same relative commands.
3. Unit tests and checkpoint compatibility tests.
4. Confirm no tracked drive-letter paths, generated images, caches, or model weights.
5. Confirm `git status --short` is empty.

