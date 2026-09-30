# Remote Stage Summary Backfill

This note records the first real remote application of:

- `tools/experiments/backfill_phase618_stage_summary.py`

The point is not just that the tooling works locally, but that current remote stage roots can now be re-read through the same close-result diagnosis contract.

## 1. Remote commands

Remote host:

```bash
ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62
```

Backfill commands run on the remote WSL repo:

```bash
python3 /mnt/i/Github/Latent_Style/SchrodingerBridge/tools/experiments/backfill_phase618_stage_summary.py \
  --stage-root /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/20250618_ot_rerun_lowrank_auto

python3 /mnt/i/Github/Latent_Style/SchrodingerBridge/tools/experiments/backfill_phase618_stage_summary.py \
  --stage-root /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/20250618_lite_ot_vertical_auto
```

## 2. Current remote artifact status

### A. `exp/20250618_ot_rerun_lowrank_auto`

Backfill succeeded.

Observed `stage_summary.json` after backfill:

- `close_result_diagnosis.status = "separated"`
- best run:
  - `h0_vertical_fm`
  - best epoch `epoch_0002`
  - style `0.6620`
  - LPIPS `0.3341`
- completed peer:
  - `h1_linear_fm`
  - latest style `0.6495`
  - latest LPIPS `0.4627`

Interpretation:

- this is **not** currently a close-result cluster
- style delta is about `0.0126`
- LPIPS delta is about `0.1286`
- both are far outside the current close-cluster thresholds:
  - `style_eps = 0.005`
  - `lpips_eps = 0.018`

Backfilled validity reads:

- `h0_vertical_fm`
  - `artifact_status = valid`
  - `effect_contract = training_real_eval_inert`
  - suite = `stage1_h0_h6_repaired_lowrank`
- `h1_linear_fm`
  - `artifact_status = valid`
  - `effect_contract = unknown`
  - suite = `stage1_h0_h6_repaired_lowrank`

The key takeaway is:

> the currently completed remote rerun evidence does **not** support the claim that all old-OT family members are numerically indistinguishable.

What it does support is narrower:

- at least one real remote run (`h0`) is now explicitly tagged as
  `training_real_eval_inert`
- so if later OT-family runs land very close to it, the first interpretation should still be
  "plain no-reference eval path is weak / inert", not "the code never changed the model"

### B. `exp/20250618_lite_ot_vertical_auto`

Backfill also succeeded, but:

- `run_count = 0`
- `close_result_diagnosis.status = "insufficient_runs"`

Interpretation:

- this stage root has no scored runs left to diagnose
- it is not current evidence for or against the close-result hypothesis

### C. `exp/20250618_stage3_style_auto`

At the time of inspection, this directory did not exist on the remote host.

Interpretation:

- there is no live remote repaired-base style-sweep artifact yet to backfill
- all repaired-base style-sweep close-result conclusions still come from the local probe/matrix evidence, not from a current remote stage root

## 3. What changed in our reading

Before this backfill, it was still easy to talk about "remote close results" in a vague way.

After backfill, the remote state is more precise:

1. current remote OT rerun evidence is **partially complete**
2. the completed `h0/h1` pair is **not close**
3. the completed best run is already tagged as `training_real_eval_inert`

So the present remote evidence does **not** yet contradict the main 618 diagnosis.
It simply says the live rerun has not yet reached the stage where a true multi-run close cluster can be read off the remote stage root itself.
