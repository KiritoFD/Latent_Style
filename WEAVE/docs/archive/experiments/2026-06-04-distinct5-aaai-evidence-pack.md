# Distinct5 AAAI Evidence Pack

Date: 2026-06-04

Scope: Distinct5-512 only. This memo does not edit the paper. It defines the current AAAI-facing evidence packet for same-cost, longer-train, more-capacity, and the allowed LoRA audit path.

## Working definition

The clarified `same-cost` question is:

- same training set: `Distinct5-512`
- compare methods by matched training wall time first
- keep eval wall separate
- record VRAM as part of cost where available
- if a true matched-budget rerun is started later for `SaMAM` / `SaMST`, keep it on the same remote `RTX 3060` surface; `SaMAM` stays in `WSL`
- prioritize transfer-only `CLIP-S`, `delta_idt_transfer`, `LPIPS`, and `targetwise ArtFID`
- keep `full` metrics as companion evidence, not the headline axis

Current reused timing rows are in:

- [2026-06-04-distinct5_same_cost_inventory.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-04-distinct5_same_cost_inventory.csv)

## P0 Same-Cost Inventory

This packet is closed as an inventory, not yet as a matched-budget conclusion.

Authoritative reused evidence:

- `LBM` reviewed operating points:
  - `F e1`, `H e1`, `H e2`, `K e1`
  - configs from [resolved_headline_config.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/distinct5_512_20260602/resolved_headline_config.md)
  - full/transfer timings from [clip_style_vs_1lpips_full_transfer_points.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/distinct5_512_20260602/tables/clip_style_vs_1lpips_full_transfer_points.csv)
  - `F/K` targetwise ArtFID from [distinct5_aggregate_artfid_keypoints.remote.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/distinct5_512_20260602/artfid_metric_hacking/distinct5_aggregate_artfid_keypoints.remote.csv)
  - `H` transfer deltas from [paired_idt_transfer_bootstrap.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/distinct5_512_20260602/bootstrap/paired_idt_transfer_bootstrap.csv)
- `SaMAM`:
  - manuscript-safe `2250` and audit-only `3000` from [baseline_packet_status_20260604/README.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/distinct5_512_20260602/baseline_packet_status_20260604/README.md)
  - closed packet details for `3000` from [2026-06-04-distinct5-samam-samst-packet-status.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-04-distinct5-samam-samst-packet-status.md)
- `SaMST`:
  - `e5` and `e15` from the same two packet-status notes above

Current cost band:

- `LBM` closed points are still only in the `1.2m -> 2.3m` range.
- `K-longer` has landed `8` checkpoints but only partial paper-facing eval; its `epoch_0004` point is about `4.5m`.
- `SaMST e5` is about `116.0m`.
- `SaMST e15` is about `347.3m`.
- `SaMAM 2250` is about `458.6m`.
- `SaMAM 3000` is about `612.6m`.

Decision:

- current evidence is enough to say `minute-scale compact LBM points already work`
- current evidence is not enough to claim `same-cost closure` against `SaMST` or `SaMAM`, because the LBM family has not yet been run anywhere near the baseline hour-scale budgets

## P1 K-Longer

Priority arm: `K-longer`, not `F-longer`.

Why:

- `F-longer` already has negative reviewer-side evidence and should not be rerun
- `K` is the current style anchor and the only same-family arm that can still plausibly move the transfer frontier upward without first inventing a new model

Authoritative config and output root:

- config: [longer_train_k_seed42_b44_e8.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/longer_train_k_seed42_b44_e8.json)
- output root on the remote owner surface:
  - `I:\Github\Latent_Style\SchrodingerBridge\exp\aaai2027_longer_train_k_seed42_b44_e8`

Remote verification already performed:

- GPU idle check:
  - `ssh -p 2222 -T -o LogLevel=ERROR administrator@100.115.18.62 "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader"`
  - observed at verification time: `844 MiB / 12288 MiB`, `0% util`, `11.97 W`
- output-root check:
  - `dir /b I:\Github\Latent_Style\SchrodingerBridge\exp\aaai2027_longer_train_k_seed42_b44_e8`
  - verified `epoch_0001.pt .. epoch_0008.pt`, `full_eval`, `logs`, `src`
- `full_eval` check:
  - `epoch_0001 .. epoch_0004` each have `summary.json`
  - `epoch_0005` currently has only `images/` and `metrics.csv`
  - no paper-facing `targetwise ArtFID` has been computed for this run
- actual launch-path verification on `2026-06-04`:
  - owner shell + repo root check via `ssh` and `dir`
  - WSL foreground eval path works:
    - `wsl -d Ubuntu-26.04 -- bash /mnt/i/Github/Latent_Style/SchrodingerBridge/_codex_tmp/remote_k_longer_eval_5_8_artfid.sh`
  - tmux launcher works up to image generation start:
    - `wsl -d Ubuntu-26.04 -- bash /mnt/i/Github/Latent_Style/SchrodingerBridge/_codex_tmp/run_k_longer_eval_5_8_artfid_tmux.sh`
  - `schtasks` and `Start-Process` were not reliable holders for this eval path on the current owner surface

Updated paper-entry reading after full `e5 .. e8` closure:

- `K-longer` remains negative relative to base `K e1`
- best retained balance point is now `K-longer e5`
- `K e1` transfer:
  - `CLIP-S = 0.671167`
  - `delta_idt_transfer = +0.031244`
  - `LPIPS = 0.372281`
  - `targetwise ArtFID = 406.151`
- `K-longer e5` transfer:
  - `CLIP-S = 0.667010`
  - `delta_idt_transfer = +0.027088`
  - `LPIPS = 0.358785`
  - `targetwise ArtFID = 408.309`
- later epochs `e6 .. e8` recover style toward `K e1`, but LPIPS and targetwise
  `ArtFID` both worsen

Conservative gate for paper entry:

- require at least `+0.006` on transfer `CLIP-S` or `delta_idt_transfer`
- `LPIPS` must not worsen materially
- `targetwise ArtFID` must not clearly worsen

Current result:

- `K-longer` does not pass the gate even after the full `epoch_0005 .. epoch_0008`
  closure
- this arm is now a closed negative result, not an open recovery task

## Highest-Priority GPU Action

This packet is now closed. The next GPU action should **not** be more
same-family longer training.

Authoritative launch manifests created for this closure step:

- [remote_k_longer_eval_5_8_artfid.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/_codex_tmp/remote_k_longer_eval_5_8_artfid.sh)
- [run_k_longer_eval_5_8_artfid_tmux.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/_codex_tmp/run_k_longer_eval_5_8_artfid_tmux.sh)
- [remote_k_longer_reuse_e5_artfid.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/_codex_tmp/remote_k_longer_reuse_e5_artfid.sh)

Use this as the owner-surface command in WSL:

```bash
cd /mnt/i/Github/Latent_Style/SchrodingerBridge
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

RUN_DIR="exp/aaai2027_longer_train_k_seed42_b44_e8"
TEST_DIR="/mnt/i/wikiart_distinct5_samam_512_classview/test"
CACHE_DIR="/mnt/i/Github/Latent_Style/eval_cache"
LOG="_codex_tmp/remote_k_longer_eval_5_8_artfid.log"

: > "$LOG"
for epoch in 5 6 7 8; do
  ep="$(printf '%04d' "$epoch")"
  out="$RUN_DIR/full_eval_artfid/epoch_${ep}"
  echo "[eval] $RUN_DIR/epoch_${ep}.pt -> $out" | tee -a "$LOG"
  python3 src/utils/run_evaluation.py \
    --checkpoint "$RUN_DIR/epoch_${ep}.pt" \
    --output "$out" \
    --test_dir "$TEST_DIR" \
    --cache_dir "$CACHE_DIR" \
    --profile_timing \
    --eval_enable_art_fid \
    --no-eval_only_lpips_clip_style \
    2>&1 | tee -a "$LOG"
done
```

Durable outputs to expect:

- `I:\Github\Latent_Style\SchrodingerBridge\exp\aaai2027_longer_train_k_seed42_b44_e8\full_eval_artfid\epoch_0005..0008\summary.json`
- `...metrics.csv`
- `...aggregate_targetwise_artfid.json`
- log:
  - `I:\Github\Latent_Style\SchrodingerBridge\_codex_tmp\remote_k_longer_eval_5_8_artfid.log`

Closure status:

- `full_eval_artfid/epoch_0005 .. epoch_0008` now each retain:
  - `summary.json`
  - `metrics.csv`
  - `aggregate_targetwise_artfid.json`
- the recovery path is complete and should now be treated as a landed negative
  evidence packet
- do not reopen this lane unless the paper objective changes

## P2 Paper Entry Rule

Use the conservative gate above.

Interpretation rule:

- `K-longer` did not clear the gate after `e5 .. e8`
- safe paper reading:
  - same-family longer training on `K` did not improve the retained Distinct5
    transfer frontier within the current compact family
- do not package that as improvement

## P3 More-Capacity

Do not start a new more-capacity run now.

Reason:

- the current repo already contains several capacity-leaning representation variants (`A/B/M`), but the retained evidence does not show a low-risk capacity-only win
- the current Distinct5 notes explicitly point away from `bigger tokenizer` as the next main experiment
- there is no already-indexed `more-capacity but still formal-3060-safe` config that is clearly better than the current `F/H/K` family

Current decision:

- `more-capacity` is `next optional run`, not `start now`
- blocker:
  - no low-risk, already-authoritative config with both expected `3060` fit and a strong reason to believe it can beat `K-longer` before `K-longer` itself is closed

## P4 LoRA Feasibility

Allowed audit target:

- `Related_Works/repos/cyclegan_turbo/img2img-turbo`

Explicitly excluded:

- `Related_Works/repos/blora/B-LoRA`
- reason:
  - `SDXL` base
  - heavier `3060` cost
  - interface drifts away from the paper's domain-style transfer setting

Verdict on `img2img-turbo`:

- not recommended for the current round
- it is a plausible future `large-prior adaptation cost anchor`, but it is not a closed runnable paper packet under the current repo state

What is already closed:

- the repo has an unpaired `CycleGAN-Turbo` training entry:
  - [training_cyclegan_turbo.md](/G:/GitHub/Latent_Style/Related_Works/repos/cyclegan_turbo/img2img-turbo/docs/training_cyclegan_turbo.md)
  - [train_cyclegan_turbo.py](/G:/GitHub/Latent_Style/Related_Works/repos/cyclegan_turbo/img2img-turbo/src/train_cyclegan_turbo.py)
- custom checkpoint inference exists:
  - [inference_unpaired.py](/G:/GitHub/Latent_Style/Related_Works/repos/cyclegan_turbo/img2img-turbo/src/inference_unpaired.py)
- the unpaired dataset format already matches the required `train_A / train_B / test_A / test_B` layout with fixed prompts:
  - [training_utils.py](/G:/GitHub/Latent_Style/Related_Works/repos/cyclegan_turbo/img2img-turbo/src/my_utils/training_utils.py)

What is still open, and why it blocks a current-round run:

- evaluation integration is missing:
  - `baseline_pipeline` names `cyclegan_turbo` in [main.py](/G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/main.py), but there is no adapter script or launch wrapper for it in the pipeline
  - [BASELINE_CKPT_STATUS.md](/G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/BASELINE_CKPT_STATUS.md) still marks it as `Partial repo present | No checkpoints found | No until adapter/repo is validated`
- target-specific 5-adapter data builder is missing:
  - the repo does not yet materialize
    - `train_A = union(other 4 domains)`
    - `train_B = target domain`
    - for each of `photo`, `Hayao`, `monet`, `vangogh`, `cezanne`
- full Distinct5 evaluator bridging is missing:
  - there is no current script that loops over the `5` target adapters and emits one unified `full 5x5 / 750` packet into the same evaluator contract used by LBM / SaMAM / SaMST
- `512` on remote `3060` is not calibrated for this repo:
  - the official example trains at `256` crop with `batch_size=1`
  - there is no retained `3060` VRAM probe or smoke note proving `512` Distinct5 training is stable for this codebase
- protocol caveat for `full` companion metrics:
  - the required target-specific setup trains `A = other four domains`
  - therefore `T -> T` cells in the full `5x5` matrix are out-of-distribution for the `T` adapter
  - this does not kill the transfer-only headline, but it weakens the meaning of the full companion rows unless they are clearly caveated

Minimal executable version if this lane is revived later:

1. Build `5` target-specific unpaired datasets under one new root, one folder per target domain.
2. Train `5` separate `CycleGAN-Turbo` adapters in WSL against `stabilityai/sd-turbo`.
3. Write one inference wrapper that chooses adapter by target column and exports one merged Distinct5 packet.
4. Run the same full/transfer evaluator and record:
   - per-target adapter average train wall
   - total `5`-adapter train wall
   - `ms/img`
   - base backbone + trainable LoRA weight count

Current call:

- do not spend the current round on `img2img-turbo`
- the most defensible statement is:
  - feasible in principle
  - not closed enough today for a paper-safe LoRA anchor

## Start / Plan Split

Use now:

- `K-longer` eval completion on remote `3060`

Plan only:

- matched-budget `LBM` continuation toward the first real same-cost waypoint (`SaMST e5` at about `116m`)
- any new more-capacity `LBM` arm
- any `img2img-turbo` LoRA run
