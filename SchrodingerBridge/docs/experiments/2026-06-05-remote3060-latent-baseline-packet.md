# Remote 3060 WSL Latent Baseline Packet

Date: 2026-06-05

Scope:

- formal remote latent baseline line for `SaMam` and `SaMST`
- datasets:
  - `legacy256_overfit50`
  - `distinct5_512`
- execution policy:
  - no local GPU
  - no broad hyperparameter sweep
  - run each lane with one fixed convergence budget unless it fails structurally

## Fixed dataset contracts

`legacy256_overfit50`

- latent root:
  - `G:\GitHub\Latent_Style\latent-256`
- eval root:
  - `G:\GitHub\Latent_Style\style_data\overfit50`

`distinct5_512`

- remote latent train root:
  - `/mnt/i/wikiart_distinct5_samam_512_latents_ema/train`
- remote latent test root:
  - `/mnt/i/wikiart_distinct5_samam_512_latents_ema/test`
- remote eval root:
  - `/mnt/i/wikiart_distinct5_samam_512_classview/test`
  - or `/mnt/i/wikiart_distinct5_samam_512_classview_real/test` when the `_real` alias exists

## Packet sync helper

Use:

- [push_remote_latent_baseline_packet.py](/G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/scripts/push_remote_latent_baseline_packet.py)

It pushes:

- reviewed latent baseline scripts
- only the reviewed `SaMam` files required by the latent path
- the latent `SaMST` wrapper files
- evaluator-side `run_evaluation.py` dependencies

The sync source of truth is the local workspace, not the current remote git branch.

Important boundary:

- `Related_Works/repos/SaMam` is a nested git repo with unrelated local dirt
- the packet is intentionally narrow so remote sync does not overwrite unrelated
  nested-repo changes

## Fixed formal training budgets

These are the first non-sweep convergence budgets.

`SaMam`

- `legacy256_overfit50`
  - `25000` steps
- `distinct5_512`
  - `20000` steps

`SaMST`

- `legacy256_overfit50`
  - `25000` steps
- `distinct5_512`
  - `20000` steps

These are intentionally fixed launch budgets for the first pass. If a lane fails, close it as a structural failure instead of drifting into tuning.

## Remote launch commands

Push packet:

```bash
cd /mnt/i/Github/Latent_Style
python3 Related_Works/baseline_pipeline/scripts/push_remote_latent_baseline_packet.py
```

`SaMam` legacy256:

```bash
cd /mnt/i/Github/Latent_Style
python3 Related_Works/baseline_pipeline/scripts/run_samam_latent_baseline.py \
  --dataset legacy256_overfit50 \
  --out-root /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_latent_legacy256_remote \
  --iterations 25000 \
  --batch-size 2 \
  --precision 32-true \
  --checkpoint-every-n-steps 5000
```

`SaMST` legacy256:

```bash
cd /mnt/i/Github/Latent_Style
python3 Related_Works/baseline_pipeline/scripts/run_samst_latent_baseline.py \
  --dataset legacy256_overfit50 \
  --out-root /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samst_latent_legacy256_remote \
  --epochs 200 \
  --max-steps 25000 \
  --batch-size 2
```

`SaMam` distinct5:

```bash
cd /mnt/i/Github/Latent_Style
python3 Related_Works/baseline_pipeline/scripts/run_samam_latent_baseline.py \
  --dataset distinct5_512 \
  --out-root /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_latent_distinct5_remote \
  --iterations 20000 \
  --batch-size 1 \
  --precision 32-true \
  --checkpoint-every-n-steps 2500
```

`SaMST` distinct5:

```bash
cd /mnt/i/Github/Latent_Style
python3 Related_Works/baseline_pipeline/scripts/run_samst_latent_baseline.py \
  --dataset distinct5_512 \
  --out-root /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samst_latent_distinct5_remote \
  --epochs 200 \
  --max-steps 20000 \
  --batch-size 1
```

## Formal eval closure

After a retained checkpoint exists, close with:

`SaMam`

- [run_samam_latent_eval_bundle.py](/G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/scripts/run_samam_latent_eval_bundle.py)

`SaMST`

- [run_samst_latent_eval_bundle.py](/G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/scripts/run_samst_latent_eval_bundle.py)

Required paper-safe artifacts:

- `summary.json`
- `metrics.csv`
- `aggregate_targetwise_artfid.json`

## Stop rule

Stop the latent baseline expansion and return to `LBM A1/A2` when:

1. both `distinct5_512` lanes are closed
2. both methods fail structurally on `legacy256_overfit50`
3. only one method is viable and has already produced the first full evaluator packet

## 2026-06-05 live status

Verified repair sequence for remote `SaMam` latent on `legacy256_overfit50`:

- local packet fix:
  - [inference.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/utils/inference.py) now prefers a narrower VAE import path before falling back to package-level `diffusers`
- remote env repair on `/home/xy/venvs/samam312`:
  - `transformers==4.41.2`
  - `diffusers==0.29.2`
  - `modelscope==1.37.1`

Reason:

- `transformers 5.x` broke `mamba_ssm`
- `diffusers 0.38.0` pulled newer autoencoder families that expected newer `transformers`
- `huggingface.co` was unreachable from remote WSL, so the VAE download needed `modelscope`

Current active probe:

- output root:
  - `/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_latent_legacy256_probe4`
- start time in log:
  - `2026-06-05T23:34:58`
- verified observations:
  - the run no longer dies in the first `10s`
  - `modelscope` successfully downloaded `stabilityai/sd-vae-ft-ema`
  - `16-mixed` was rejected by `mamba_ssm` selective scan, so `SaMam` latent now runs with `32-true`
  - the latent `StyleEmbedder` path needed a small-shape compatibility fix; `4x4` token maps now skip over-aggressive pool blocks
  - after the fix and precision change, sanity check completed and real training steps started
  - at around `23:36-23:37`, the log had already advanced through dozens of training steps in `Epoch 0`
  - observed GPU usage during active training reached about `7.6 / 12.3 GiB`

Interpretation:

- this closes the earlier structural import/download blockers
- this also closes the latent shape mismatch and AMP incompatibility blockers for `SaMam`
- the next check should confirm retained checkpoint creation and full convergence progress under the fixed `25000`-step budget

## 2026-06-05 SaMST live status

Current active probe:

- output root:
  - `/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samst_latent_legacy256_probe3`
- start time in log:
  - `2026-06-05T23:44:29`

Verified repair sequence:

- fixed latent style discovery in:
  - [train_latent.py](/G:/GitHub/Latent_Style/Related_Works/repos/SaMST-main/train_model/train2/train_latent.py)
  - latent wrapper now enumerates style subdirectories instead of only flat files
- fixed workspace root derivation for `SchrodingerBridge/src` imports
- fixed latent VAE decode output dtype to `float32` before VGG features

Observed status:

- the first probe died on `ModuleNotFoundError: No module named 'utils'`
- the second probe died at VGG input dtype mismatch (`Half` vs `float`)
- the third probe remained alive beyond the `90s` front-run window
- at about `23:46`, both wrapper and training processes were alive in WSL:
  - `run_samst_latent_baseline.py`
  - `train_latent.py`
- log had progressed past startup into repeated cuDNN forward/backward warnings, with no immediate Python exception
- while `SaMam` and `SaMST` were both active, observed GPU usage reached about `12.0 / 12.3 GiB`

Interpretation:

- `SaMST` has crossed the early import/config/dtype blockers and is now in the "watch for first real train log or next structural failure" stage
- if the run keeps advancing, this becomes the second formal latent baseline lane on `legacy256_overfit50`

Cap adjustment:

- user later tightened the remote VRAM ceiling to `11.2 GiB`
- because concurrent `SaMam` + `SaMST` exceeded that cap, `samst_latent_legacy256_probe3` was stopped
- current active formal latent lane is only:
  - `/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_latent_legacy256_probe4`
- `SaMST` should resume only as a single-run probe or after an explicit lower-VRAM calibration that stays below the cap
