# Remote 3060 WSL SaMST Step Packet

Date: 2026-06-05

Scope:

- unify the next SaMST Distinct5 baseline packet onto the remote `RTX 3060` WSL surface
- keep the packet tied to the local reviewed step-checkpoint toolchain
- avoid relying on the current remote repo branch as the source of truth for baseline scripts

## Why this packet is needed

The remote owner workspace currently diverges from the local branch in a way that matters for SaMST:

- remote git branch:
  - `Tokenizer`
- remote HEAD at inspection time:
  - `83209205b97e55d8cf7dd545c304fc6d2b0ee7f2`
- remote `git ls-files` does not track the current local SaMST helper scripts:
  - `Related_Works/baseline_pipeline/scripts/run_samst_distinct5_local.py`
  - `Related_Works/baseline_pipeline/scripts/run_samst_distinct5_eval_bundle.py`
  - `Related_Works/baseline_pipeline/scripts/generate_samst_distinct5_eval.py`
- remote `train_model/train2/train.py` exists, but it is not currently tracked there either

Therefore the remote workspace should not be treated as a self-sufficient formal baseline surface for step-aligned SaMST until the reviewed local packet is pushed explicitly.

## Minimal reviewed packet

The smallest file set needed for step-aligned SaMST is:

- `Related_Works/baseline_pipeline/scripts/run_samst_distinct5_local.py`
- `Related_Works/baseline_pipeline/scripts/generate_samst_distinct5_eval.py`
- `Related_Works/baseline_pipeline/scripts/run_samst_distinct5_eval_bundle.py`
- `Related_Works/repos/SaMST-main/train_model/train2/train.py`

Important boundary:

- do **not** sync `train.yml` or `test.yml`
- those are runtime-overwritten environment files and are currently dirty / machine-specific

## New sync helper

Use:

- [push_remote_samst_step_packet.py](/G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/scripts/push_remote_samst_step_packet.py)

It pushes the four files above to:

- remote host: `100.115.18.62:2222`
- remote WSL root: `/mnt/i/Github/Latent_Style`

Transport:

- local Python tarball
- `tar` streamed over `ssh`
- extracted directly inside the remote WSL repo root
- followed by remote `python3 -m py_compile` verification

## Step-aligned run contract

Target use case:

- same-cost / short-budget SaMST on Distinct5-512
- save by optimizer step instead of epoch
- close the packet with the same evaluator contract used by current paper evidence

Training side:

```bash
cd /mnt/i/Github/Latent_Style
python3 Related_Works/baseline_pipeline/scripts/run_samst_distinct5_local.py \
  --data-root /mnt/i/datasets/wikiart_distinct5_samam_512 \
  --out-root /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samst_distinct5_512_wsl_stepalign40_remote_20260605 \
  --styles Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e \
  --epochs 30 \
  --max-steps 40 \
  --batch-size 6 \
  --image-size 256 \
  --style-size 512
```

Evaluation side:

```bash
cd /mnt/i/Github/Latent_Style
python3 Related_Works/baseline_pipeline/scripts/run_samst_distinct5_eval_bundle.py \
  --run-root /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samst_distinct5_512_wsl_stepalign40_remote_20260605 \
  --epochs 40 \
  --ckpt-name step_000040.model \
  --label step_000040 \
  --full-eval \
  --test-root /mnt/i/wikiart_distinct5_samam_512_classview/test \
  --style-names Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e \
  --max-src-per-style 30 \
  --resize-content 512 \
  --eval-batch-size 8 \
  --eval-target-chunk-size 1 \
  --eval-image-save-workers 4
```

Why `--epochs 40` above:

- the bundle script still needs one integer handle for logging and packet identity
- `--ckpt-name step_000040.model` is the authoritative checkpoint selector
- `--label step_000040` prevents the evaluator output from pretending this is an epoch packet
- `--full-eval` is required for targetwise ArtFID and paper-safe same-cost closure

## Expected durable outputs

Run root:

- `I:\Github\Latent_Style\Related_Works\baseline_pipeline\results\samst_distinct5_512_wsl_stepalign40_remote_20260605`

Key packet outputs:

- `checkpoints\<style>\step_000040.model`
- `eval_bundle\eval_step_000040\step_000040\summary.json`
- `eval_bundle\eval_step_000040\step_000040\metrics.csv`
- `eval_bundle\eval_step_000040\step_000040\aggregate_targetwise_artfid.json`
- `eval_bundle\bundle_summary.json`

## Landed remote packet

The first formal remote packet landed under:

- `I:\Github\Latent_Style\Related_Works\baseline_pipeline\results\samst_distinct5_512_wsl_stepalign40_remote_20260605_r1`

Observed runtime closure:

- training wall time:
  - `1069.20 s` (`17.8 m`) from `run.log`
- pure 750-image generation wall time:
  - `382.182 s`
  - `509.58 ms / image`
- full evaluator wall time after reuse-generated:
  - `1171.989 s` from `eval.log`

Observed full-packet metrics from `step_000040_full`:

- transfer-only `CLIP-S`:
  - `0.653563`
- transfer-only `LPIPS`:
  - `0.744621`
- all-pairs `CLIP-S`:
  - `0.661342`
- all-pairs `LPIPS`:
  - `0.743015`
- targetwise transfer ArtFID derived from `summary.json` matrix:
  - `543.867`

Operational reading:

- this is a valid remote `3060` reproduction packet
- it is **not** the same as the earlier local `2.0m` same-cost row
- on the unified remote surface, the same `40-step/style` SaMST packet is materially slower and remains in the high-damage regime

Current caveat:

- this evaluator branch did not emit a standalone `aggregate_targetwise_artfid.json`
- the targetwise transfer ArtFID above was therefore derived from the full `summary.json` matrix by averaging transfer-only per-target means
- the packet is still usable as evidence because the per-direction `art_fid` entries are present in the retained summary

## Immediate next gate

Before launching the remote formal run:

1. push the reviewed packet with `push_remote_samst_step_packet.py`
2. verify remote `py_compile` passes
3. verify that the remote data root is the flat `samam_512` layout or a classview layout accepted by the helper
4. only then start the remote training + eval packet

This keeps the remote 3060 baseline surface explicit and reviewable instead of relying on ad hoc remote leftovers.
