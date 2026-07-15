# Submission Handoff: 2026-07-15

## 1. Source of Truth

- Branch: `submission`.
- Active code: project-root Python modules and `utils/`.
- Training config: `config.json`.
- Inference/evaluation config: `inference.json`.
- Active launchers: `scripts/run_submission_repro.ps1` and `scripts/batch_eval_all.py`.
- Historical code, configs, launchers, tests, and rejected post-processing are under `archives/`.

The method must be trained from a fresh initialization. Do not promote results
from frozen-checkpoint adapters, checkpoint surgery, spectral image blending,
latent editing, or metric-time post-processing. Canonical inference explicitly
uses `postprocess_mode: none`, `latent_postprocess_mode: none`, and no output
appearance alignment.

## 2. Portable Local/Remote Contract

Local project:

```text
G:\GitHub\Latent_Style\SchrodingerBridge
```

Remote project:

```text
I:\Github\Latent_Style\SchrodingerBridge
```

Remote access:

```powershell
ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62
```

Both machines must contain physical data directories at identical paths
relative to the project root:

```text
data/train
data/test
```

`data/train` contains five styles with 1,000 packed latents per style and the
pairing cache. `data/test` contains the same 150-image D5 board on both
machines. The old machine-specific `dataset_index.json` mechanism is retired;
tracked configs contain only relative paths.

The remote directory is a synchronized execution copy, not a Git checkout.
Sync only active code/config/scripts and never overwrite `data/`, `runs/`, or
remote model caches. Verify file hashes when reproducing a paper number.

Remote audit on 2026-07-15 found 5,021 files below `data/train` (including
caches/manifests) and 150 files below `data/test`. The remote config and
inference JSON hashes matched local exactly. The remote training interpreter
loads both entry points and the canonical config, but does not currently have
`pytest`; run the full test suite locally unless the remote environment is
explicitly provisioned for development.

## 3. Canonical From-Scratch Run

From the project root on either machine:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_submission_repro.ps1
```

This performs:

1. fresh 15-epoch training from `config.json`;
2. one checkpoint per epoch;
3. full evaluation of every checkpoint using `inference.json`;
4. DINO-S, CLIP-S, DINO-C, and LPIPS reporting without a mixed score.

To resume evaluation without retraining:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_submission_repro.ps1 -EvalOnly
```

For release reproduction, do not pass `-AllowNetwork`. Required CLIP, DINOv2,
VAE, and LPIPS artifacts should already exist below `runs/cache/hf` or the
declared local cache. A missing model should fail instead of silently changing
the environment.

## 4. Reproduced Baseline

The clean baseline has 873,680 trainable parameters and was trained from
scratch on the remote RTX 3060 in about five minutes. All 15 epochs were fully
evaluated. Epoch 6 is selected because DINO-S peaks there while content remains
stable.

| Point | DINO-S | CLIP-S | LPIPS | DINO-C |
|---|---:|---:|---:|---:|
| Paper row | 0.4859 | 0.7075 | 0.2583 | 0.8287 |
| Fresh epoch 6 | **0.4867** | 0.7074 | **0.2508** | 0.8280 |

Full precision, hashes, and timing are in
`docs/reproduction/baseline_reproduction.md` and
`docs/reproduction/baseline_epoch_metrics.csv`.

## 5. Method and Information Path

The current baseline is best understood as:

```text
source/style latent pair
  -> one-level Haar coordinates
  -> content-protected LL target + target-style HF supervision
  -> compact rectified-flow transport with style-memory conditioning
  -> 8-step solver
  -> declared endpoint HF-statistics alignment
  -> VAE decode
```

The training target is not source reconstruction. LL is a content-heavy blend
with limited style-stat alignment; LH/HL/HH are taken from the target-style
latent. This frequency-weighted target prevents the easiest low-frequency
identity shortcut while retaining a direct high-frequency style objective.

Probe evidence gives a narrower diagnosis than the old claim of a hard style
ceiling:

- The supervised target-HF signal is strong, but target-image HF is weak as a
  conditioning input to the learned HF velocity heads.
- Raw spatial target-HF injection raises style metrics but leaks target layout
  and destroys content.
- Coordinate-free per-orientation HF codes can raise DINO-S without collapse,
  proving the architecture has usable style capacity.
- Increasing residual magnitude or adding more generic statistics does not fix
  the direction of the conditional correction.
- HH is difficult to improve because it is low-energy, phase-sensitive, and
  weakly aligned with perceptual style gradients; an explicit HH head has so
  far spent capacity on unstable detail rather than robust style.

The detailed gradient and information-flow evidence remains in
`GRADIENT_INFORMATION_FLOW_DEBUG_2026-07-14.md` and
`HF_ARCHITECTURE_PROBE_2026-07-13.md`.

## 6. Latest Architecture Result

The coordinate-free oriented HF route was trained from scratch for 15 epochs
and evaluated at every epoch.

| Point | DINO-S | CLIP-S | LPIPS | DINO-C |
|---|---:|---:|---:|---:|
| Baseline epoch 6 | 0.4867 | 0.7075 | **0.2508** | **0.8281** |
| Oriented route epoch 4 | **0.4915** | **0.7126** | 0.2596 | 0.8103 |
| Oriented route epoch 6 | 0.4878 | 0.7101 | 0.2563 | 0.8215 |

This route raises the style ceiling but does not dominate the baseline because
the content loss is measurable. It remains an architecture probe, not the
canonical config. The endpoint AdaIN scale axis is closed at 2.0; scale 2.5
causes severe collapse and should not be revisited as a tuning direction.

## 7. Repository Cleanup Completed

- Root implementation verified against the old `src/` implementation with
  exact fixed-checkpoint latent equality.
- Full-board root-layout drift is within ordinary repeated GPU/VAE evaluation
  noise.
- Legacy `src/`, 834 configs, 445 launchers, and obsolete test suites are
  archived.
- One-off root diagnostics and rejected spectral/content image blending scripts
  are archived.
- Machine-specific dataset path indexing is removed.
- Current tests: 42 passed with one existing test-only tensor conversion warning.

## 8. Next Architecture Gate

The next justified experiment is gradient ownership for the oriented target-HF
residual route:

1. initialize the full baseline and residual route together from scratch;
2. train the original backbone and velocity heads normally;
3. feed detached shared features into the target-HF residual heads so that the
   new branch cannot rewrite the shared content transport through its gradient;
4. keep LL and HH behavior unchanged;
5. use no image/latent post-processing;
6. evaluate every epoch and accept only a point that improves DINO-S and
   CLIP-S without material DINO-C/LPIPS damage.

Do not tune learning rate, loss weights, epochs, or endpoint scales before this
architecture gate is complete. Do not select by a custom mixed metric.
