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
G:\GitHub\Latent_Style\WEAVE
```

Remote project:

```text
I:\Github\Latent_Style\WEAVE
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
  -> compact endpoint-velocity transport with style-memory conditioning
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

This route raises the style ceiling with a measurable but bounded content cost.
Epoch 4 is the current D5 main-table checkpoint and is selected by the internal
dynamics rule below. The endpoint AdaIN scale axis is closed at 2.0; scale 2.5
causes severe collapse and should not be revisited as a tuning direction.

## 7. Repository Cleanup Completed

- The tracked project directory was renamed from `SchrodingerBridge/` to
  `WEAVE/` in commit `68884b684`; Git verified all 5,010 tracked paths as
  exact `R100` renames.
- The remote execution tree was renamed to `I:\Github\Latent_Style\WEAVE`.
  Both entry points and canonical config loading pass from the new path, and
  local/remote config hashes remain identical.
- Root implementation verified against the old `src/` implementation with
  exact fixed-checkpoint latent equality.
- Full-board root-layout drift is within ordinary repeated GPU/VAE evaluation
  noise.
- Legacy `src/`, 834 configs, 445 launchers, and obsolete test suites are
  archived.
- One-off root diagnostics and rejected spectral/content image blending scripts
  are archived.
- Machine-specific dataset path indexing is removed.
- Current tests: 46 passed with one existing test-only tensor conversion warning.
  `pytest.ini` restricts collection to `tests/`, so archived diagnostics are not
  mistaken for the active suite.

Windows note: the current Codex task was opened with the former directory as
its process working directory, so an empty, untracked `SchrodingerBridge/`
shell may remain until that workspace handle is released. It contains zero
files; all tracked and runtime content is under `WEAVE/`.

## 8. Internal-Dynamics Stop

External-metric early stopping is no longer required for the oriented-HF run.
The original 15-epoch curve showed that FM-target residual alignment continues
to improve after the DINO-S peak, so loss convergence is not an adequate
proxy. The useful event is the first epoch where:

1. the mean LH/HL target-HF gate changes from contraction to expansion; and
2. the shared-trunk LL/HF gradient-norm ratio contracts by at least 35% from the preceding epoch.

The retrospective curve and a fresh online run both select epoch 4. Seed/probe
stress tests exposed that the first absolute crossing-at-one implementation was
scale-sensitive, so the implemented event uses
`rho_epoch / rho_previous <= 0.65`. It selects epochs 3--4 across seeds
7/42/123 and probe batches 2/4/8. The fresh
run used the canonical 15-epoch cosine schedule, decoded no images, consulted
no DINO/CLIP/LPIPS metric, and stopped after saving `epoch_0004.pt`. The probe
uses a fixed latent batch, two extra backwards per epoch, and preserves the
CPU/CUDA RNG state. See `docs/reproduction/internal_dynamics_early_stop.md`.

Active configs:

- record only: `experiments/architecture/hf_oriented_internal_probe.json`;
- automatic stop: `experiments/architecture/hf_oriented_internal_early_stop.json`.

## 9. Metric Audits

The current oriented-HF epoch-4 packet was regenerated from its checkpoint and
matched all 750 requests. Its targetwise ArtFID is 295.27, versus SaMam 297.32,
Seedream 310.97 (720 successful API requests), Z-STAR 332.91, and StyleAligned
368.63. IDT remains much lower at 216.51 because the multiplicative ArtFID
content term is exactly zero for a no-op. See
`docs/reproduction/artfid_d5_audit.md`.

The previous D5 TGT values mixed individual/off-diagonal statistics with the
full board. The corrected full-750 reference is DINO-S 1.000, CLIP-S 0.863,
LPIPS 0.776, and DINO-C 0.215. Across the first five deterministic target
exemplars, TGT LPIPS ranges from 0.752 to 0.783 and DINO-C from 0.189 to 0.246;
StyleAligned LPIPS 0.869 remains outside every reference choice. Raw results are
in `docs/reproduction/tgt_reference_sensitivity.{csv,json}`.

## 10. Next Architecture Gate

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
