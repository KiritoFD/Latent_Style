# 616 Tools

This file records the reusable tooling, remote entrypoints, logs, and recovery paths
for the phase-616 three-stage experiment loop.

## 1. Remote machine

Remote Windows host:

```bash
ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62
```

Remote repo:

```text
Windows: I:\Github\Latent_Style\SchrodingerBridge
WSL:     /mnt/i/Github/Latent_Style/SchrodingerBridge
```

Current stable WSL distro:

```text
Ubuntu-26.04
```

## 2. Core automatic runner

Main controller:

```bash
tools/experiments/phase616_auto.py
```

Per-stage local/WSL entrypoints:

```bash
tools/experiments/run_phase616_stage1_auto.sh
tools/experiments/run_phase616_stage2_auto.sh
tools/experiments/run_phase616_stage3_auto.sh
tools/experiments/run_phase616_auto_tree.sh
tools/experiments/run_phase618_style_sweep.sh
tools/experiments/run_phase618_ot_rerun.sh
tools/experiments/run_phase618_plain_path_distill.sh
```

Default behavior:

- Each run probes `20 steps` first.
- Probe timeout defaults to `40s`.
- Batch selection targets `9.0-10.8 GB` VRAM.
- `> 11.3 GB` is treated as OOM.
- Prefer batch sizes divisible by 8 or 16.
- After launch, perform a `1 minute` health check.
- At `10 minutes`, estimate ETA.
- Then sleep until roughly `ETA - 5 minutes`.
- Run full eval every epoch.
- Use `CLIP-S + LPIPS` convergence signals for early stop.

Default stage roots:

```text
stage1: /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/20250618_lite_ot_vertical_auto
stage2: /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/20250618_lite_ot_ablation_auto
stage3: /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/20250618_lite_ot_best_auto
stage3_style: /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/20250618_stage3_style_auto
ot_rerun: /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/20250618_ot_rerun_lowrank_auto
plain_path_distill: /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/20250618_plain_path_distill_auto
```

## 3. Remote scheduled-task entry

Remote launchers:

```bash
tools/experiments/launch_phase616_stage1_auto_remote.sh
tools/experiments/launch_phase616_stage2_auto_remote.sh
tools/experiments/launch_phase616_stage3_auto_remote.sh
tools/experiments/launch_phase616_auto_tree_remote.sh
tools/experiments/launch_phase618_style_sweep_remote.sh
tools/experiments/launch_phase618_ot_rerun_remote.sh
tools/experiments/launch_phase618_plain_path_distill_remote.sh
tools/experiments/launch_phase618_plain_path_distill_remote_when_idle.sh
```

These use:

```bash
tools/experiments/launch_remote_wsl_command.py
```

Current sync contract:

- syncs the requested `--sync-path` files/directories into remote WSL
- verifies the synced `--verify-python-file` set by `sha256` before launch
- then runs remote `py_compile` on the same verified file list

So a scheduled-task launch is no longer just “syntax-valid”; it is also checked
against stale-code drift on the critical Python entrypoints.

Current stable scheduled-task action:

```text
wsl.exe -d Ubuntu-26.04 --exec /bin/bash -lc "cd /mnt/i/Github/Latent_Style/SchrodingerBridge && bash <script>.sh"
```

Constraints:

- Use Windows scheduled tasks to enter WSL directly.
- Do not wrap the 616 stage entry in `.ps1` as the main task action.
- Keep single-lane GPU protection:
  - if a WSL training job is already active
  - or the GPU is above the idle ceiling
  - the launcher must refuse to open another training lane
- Runtime VRAM ceiling is `11570 MiB` (`11.3 GB` safety line).
- Phase-618 remote launchers now hash-verify:
  - `src/run.py`
  - `src/losses.py`
  - `src/trainer.py`
  - `src/utils/training.py`
  - `tools/audit_phase618_run_validity.py`
  - the active probe / auto-launch helpers

## 4. Common launch commands

Run directly inside remote WSL:

```bash
cd /mnt/i/Github/Latent_Style/SchrodingerBridge
bash tools/experiments/run_phase616_auto_tree.sh
```

Run the repaired phase-618 style sweep:

```bash
cd /mnt/i/Github/Latent_Style/SchrodingerBridge
bash tools/experiments/run_phase618_style_sweep.sh
```

Run the repaired phase-618 plain-path distill family:

```bash
cd /mnt/i/Github/Latent_Style/SchrodingerBridge
bash tools/experiments/run_phase618_plain_path_distill.sh
```

Queue the repaired phase-618 plain-path distill family to start after the current remote lane goes idle:

```bash
cd /mnt/i/Github/Latent_Style/SchrodingerBridge
bash tools/experiments/run_phase618_plain_path_distill_when_idle.sh
```

Current stable phase-618 batch defaults:

- old OT rerun: `B16`
  - repaired low-rank carrier pushed `B20` over the `11.3 GB` line on the remote 3060 lane
- plain-path distill: `B20`
- style sweep: `B20`

Current stable phase-618 convergence packet contract:

- `tools/experiments/report_round2_convergence.py` now emits both:
  - Pareto-style convergence: `converged`
  - objective-gap patience stop: `objective_patience_converged`, `stop_ready`, `stop_reason`
- `src/run.py` and `tools/experiments/phase616_auto.py` now honor the same `stop_ready` signal
- this fixes the old split where a run could advance to the next variant while
  `round2_convergence.json` still said `converged=false`

Repo-backed authoritative bases:

```text
OT rerun base:
  /mnt/i/Github/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-18-stage1-lowrank-rerun-audit/remote_base_phase618_ot_rerun_lowrank.json

Style-sweep base:
  /mnt/i/Github/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-18-stage1-lowrank-rerun-audit/baseline_h1_lowrank_config.json

Plain-path distill base:
  /mnt/i/Github/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-18-stage1-lowrank-rerun-audit/baseline_h1_lowrank_config.json
```

Style-sweep default behavior now includes a config-effect preflight:

- it runs `tools/probe_config_effectiveness.py` on the requested style-sweep specs
- it labels each spec as:
  - `plain_eval_change`
  - `train_graph_only`
  - `no_effect`
- unless a name is explicitly requested, `train_graph_only` and `no_effect` specs are filtered out by default

Useful flags:

```bash
python3 tools/experiments/phase616_auto.py style-sweep --include-train-graph-only
python3 tools/experiments/phase616_auto.py style-sweep --skip-config-effect-preflight
```

Interpretation:

- this prevents default style-sweep runs from spending GPU time on candidates that only change the matched-target training graph while leaving the benchmarked no-reference eval graph unchanged
- explicit `--include-name ...` still forces a candidate through even if it is classified as `train_graph_only`
- `run_phase618_style_sweep.sh` force-cleans the stage root, generates a repaired low-rank base config, and passes it through `--base-cfg`
- `phase616_auto.py style-sweep` now rejects legacy bases that do not already have:
  - `matched_target_conditioning_mode=both`
  - `matched_target_style_encoder_mode=residual`
  - `style_code_spatial_mode=lowrank`

Why this matters:

- the old style-sweep launcher could accidentally run on the pre-repair base
- on that old base, `r7/r8/r10` looked strong mainly because they were repairing the dead no-reference carrier
- on the repaired low-rank base, those same variants become `no_effect`, while the actual bold lever is the blend sweep itself

Launch the remote scheduled task from local:

```bash
cd /mnt/g/GitHub/Latent_Style/SchrodingerBridge
bash tools/experiments/launch_phase616_auto_tree_remote.sh
```

Rerun the old OT family after enabling the low-rank no-reference style carrier:

```bash
cd /mnt/i/Github/Latent_Style/SchrodingerBridge
bash tools/experiments/run_phase618_ot_rerun.sh
```

This wrapper:

- starts from the repo-backed repaired low-rank rerun base
- force-cleans the dedicated rerun stage root before launch
- skips the config-effect / training-effect preflights because this rerun is for full old-OT evidence, not screening
- pins the rerun at `batch_size=16` after the low-rank carrier pushed `B20` above the `11.3 GB` safety line
- reruns the old `h0`-`h6` OT / bridge family from scratch under a no-reference-eval-live base
- keeps the normal stage1 early-stop / per-epoch eval behavior

Run the current highest-priority contract-gap family:

```bash
cd /mnt/i/Github/Latent_Style/SchrodingerBridge
bash tools/experiments/run_phase618_plain_path_distill.sh
```

This wrapper:

- starts from the same repo-backed repaired low-rank base as the repaired style sweep
- force-cleans a dedicated `plain_path_distill` stage root
- runs the full `h0/h1/h2/h3/h4/h5/h6 + w_plain_path_distill=0.5` family from scratch
- keeps validity / config-effect / training-effect preflights visible in `stage_summary.json`
- does **not** filter the family by plain-eval liveness, because this suite is intentionally `training_only_by_design`
- skips the VRAM probe by default and pins `batch_size=16` to stay on the safe side of the repaired-lowrank memory envelope

If the remote GPU is still occupied by an older lane and we want the corrected family to start automatically next:

```bash
cd /mnt/g/GitHub/Latent_Style/SchrodingerBridge
bash tools/experiments/launch_phase618_plain_path_distill_remote_when_idle.sh
```

This launches a lightweight remote WSL watcher that:

- polls `src/run.py` process count
- polls GPU residency
- waits for `active_train_count == 0` and `gpu_used_mib <= 1500`
- then starts `run_phase618_plain_path_distill.sh`

Dashboard comparison anchors:

```text
exp/phase616_live_dashboard/external_baselines.csv
```

Current mirrored external rows:

- `StyleGallery, 750, 0.697547, 0.710688`
- `StyleShot, 750, 0.806562, 0.698320`
- `CSGO low-VRAM, 750, 0.654125, 0.820927`

For the paired diagnosis proving why style-sweep must already start from the repaired base:

- `docs/experiments/2026-06-18-style-sweep-base-audit/README.md`

Run stage 2 only:

```bash
bash tools/experiments/run_phase616_stage2_auto.sh \
  --stage1-root /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/20250618_lite_ot_vertical_auto
```

## 5. Logs and artifacts

Key per-run artifacts:

```text
<run>/logs/training_*.csv
<run>/full_eval_transfer/clip_lpips_curve.csv
<run>/full_eval_transfer/round2_convergence.json
<run>/auto_run_summary.json
<run>/_probe/probe_summary.json
```

Remote launcher-related outputs:

```text
/mnt/i/Github/Latent_Style/SchrodingerBridge/docs/616/logs/*.log
/mnt/i/Github/Latent_Style/SchrodingerBridge/SchrodingerBridge/_codex_rt/*.sh
/mnt/i/Github/Latent_Style/SchrodingerBridge/SchrodingerBridge/_codex_rt/*.pid
```

Path caution:

- Some older configs wrote `checkpoint.save_dir` as a relative path.
- That can create nested paths like:

```text
SchrodingerBridge/mnt/i/...
```

- Treat those as valid evidence locations before deciding to rerun.

## 5.1 Local forensic probes

Conditioning / topology path-liveness probe:

```bash
py -3.12 tools/probe_conditioning_sensitivity.py \
  --config G:/GitHub/Latent_Style/SchrodingerBridge/exp/local_wsl_distinct5_512_ema_k_b16_step2min/config.json \
  --output-dir G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-18-conditioning-sensitivity-probe
```

Outputs:

```text
conditioning_sensitivity.csv
topology_sensitivity.csv
topology_pairwise.csv
path_anatomy.csv
effective_config.json
summary.json
```

Use this before trusting any new OT / tokenizer / TopoGate ablation.

## 5.2 Phase-618 family validity auditor

Primary auditor for the "metrics are nearly tied, did the model actually change?" question:

```bash
py -3.12 tools/audit_phase618_run_validity.py --run-dir <run_dir>
```

`--run-dir` expects a locally present run directory with `config.json` and, if available,
`logs/training_*.csv`. If the run only exists on the remote machine or only as a recorded
base/spec pair, use `--config + --variant-spec + --variant-name` instead.

It can also audit a base config plus a variant spec before launch:

```bash
py -3.12 tools/audit_phase618_run_validity.py \
  --config docs/experiments/2026-06-18-stage1-lowrank-rerun-audit/baseline_h1_lowrank_config.json \
  --variant-spec docs/experiments/2026-06-18-bold-eval-graph-preflight/variant_spec.json \
  --variant-name r11_linear_blend_0p00
```

Authoritative matrix backing the verdicts:

```text
docs/experiments/2026-06-18-family-validity-matrix/summary.json
```

What it returns:

- `artifact_status`
  - `valid`: no known implementation trap detected
  - `suspect`: config contradicts the intended mechanism
  - `stale`: artifact predates a logging / probe contract we now require
  - `confounded`: experiment mixes multiple causal changes, so the result is not scientifically clean
- `effect_contract`
  - `runtime_and_training_real`: plain eval graph changes and training graph changes
  - `training_real_eval_inert`: training path changes but plain no-reference eval path does not
  - `training_only_by_design`: intended training-only contract, such as plain-path distill
  - `runtime_real`: runtime graph changes were confirmed but training evidence was not part of the suite summary

Current high-value reads:

- old-base style sweep like `r8_linear_code_map_lowrank_both`
  - `artifact_status=confounded`
  - reason: base repair and bold direction are mixed
- repaired lowrank plain-path distill like `h1_plain_path_distill_0p50`
  - `artifact_status=valid`
  - `effect_contract=training_only_by_design`
- repaired bold blend sweep like `r11_linear_blend_0p00`
  - `artifact_status=valid`
  - `effect_contract=runtime_and_training_real`
  - but the matrix still reads it as a weak lever, not a style rescue

Current remote reality-check helper:

```bash
py -3.12 tools/experiments/build_phase618_remote_run_audit.py
```

It:

- SSHes to the remote 3060 host
- runs `backfill_phase618_stage_summary.py` in-place on current phase-618 stage roots when possible
- reuses the same `validity_audit + close_result_diagnosis` contract as local analysis
- writes a local snapshot under:

```text
docs/experiments/2026-06-18-remote-real-run-audit/
```

Use this when the question is no longer "what should this family mean in theory?",
but "what do the actual remote stage roots currently prove?"

When results are extremely close, the default order should be:

1. Run `audit_phase618_run_validity.py`
2. Check whether the family is `confounded`, `stale`, or `training_real_eval_inert`
3. Only then decide whether a near-tie is theoretical negative evidence or just an implementation / evidence problem

Runner integration:

- `tools/experiments/phase616_auto.py stage1`
- `tools/experiments/phase616_auto.py style-sweep`
- `tools/experiments/phase616_auto.py plain-path-distill`

now also write:

```text
<stage_root>/_preflight_validity_audit/preflight_summary.json
```

and attach a compact `validity_audit` block onto each stage-manifest run entry.

For `style-sweep`, `stage_summary.json::skipped_by_preflight` also preserves each skipped
candidate's `validity_audit`, so "didn't run" and "shouldn't be trusted" stay visible in
the same artifact.

The saved `stage_summary.json` now also carries:

```text
close_result_diagnosis
```

This is a compact automatic read on whether a near-tied cluster looks like:

- implementation / evidence risk
- train/eval contract gap
- or a runtime-real but weak lever family

For older stage roots produced before this field existed, backfill with:

```bash
py -3.12 tools/experiments/backfill_phase618_stage_summary.py --stage-root <stage_root>
```

This rewrites both:

- `<stage_root>/stage_manifest.json`
- `<stage_root>/stage_summary.json`

to inject missing `validity_audit` blocks and regenerate `close_result_diagnosis`.

That means the experiment directory itself now records whether a family member is:

- `confounded`
- `stale`
- `suspect`
- or `valid`

before you start reading close metric clusters as theory.

What it proves:

- whether `matched_target_conditioning_mode=code` actually changes `forward/predict_transport_base/integrate`
- whether explicit `style_code_override` is being bypassed or overwritten by the content router
- whether `semantic_self_topology_blend` is a real lever or a no-op because `semantic_self_topology_gate=false`
- where the first nonzero style-conditioned delta appears in the executed path
- whether training-time spatial matched-target style reaches `h_body` while no-reference code-only style stays decoder-only
- whether cached output-appearance style context actually contains the resolved spatial map (`cached_output_style_map_abs`)

Useful anatomy interpretation:

- `path_anatomy.csv::code_only_no_reference`
  - if deltas stay zero until `h_dec_post_mod`, then the no-reference style path is effectively just `dec_mod`
- `path_anatomy.csv::spatial_matched_target`
  - if `style_map_a_vs_b_mean_abs` and `h_body_a_vs_b_mean_abs` are large, then the training graph is relying on a strong spatial matched-target body path
- if those two patterns coexist, the run has a train/eval style mismatch even when the OT implementation itself is live

New phase-618 repair lever:

- `model.style_code_spatial_mode=lowrank`
- with:
  - `style_code_spatial_hidden_dim`
  - `style_code_spatial_rank`
  - `style_code_spatial_base_hw`
  - `style_code_spatial_scale`

Purpose:

- synthesize a no-reference `style_code -> map_16` spatial carrier
- feed a body-level style map even when eval has no matched target latent
- optionally ride alongside matched-target spatial conditioning during training so the no-reference path gets gradient

Direct low-rank probe example:

```bash
py -3.12 tools/probe_conditioning_sensitivity.py \
  --config G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-18-remote-h1-e18-diagnosis/remote_config.json \
  --output-dir G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-18-remote-h1-e18-diagnosis/probe_random_init_lowrank_cli \
  --override model.style_code_spatial_mode='"lowrank"' \
  --override model.style_code_spatial_hidden_dim=64 \
  --override model.style_code_spatial_rank=8 \
  --override model.style_code_spatial_base_hw=16 \
  --override model.style_code_spatial_scale=0.35
```

This writes the exact effective config used for the probe to `effective_config.json`.

Config-diff differential probe:

```bash
py -3.12 tools/probe_config_effectiveness.py \
  --config G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-18-remote-h1-e18-diagnosis/remote_config.json \
  --variant-spec G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-18-remote-h1-e18-diagnosis/config_effect_variants.json \
  --output-dir G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-18-remote-h1-e18-diagnosis/config_effect_probe \
  --device cpu
```

Outputs:

```text
variant_effects.csv
summary.json
baseline_effective_config.json
variant_spec.expanded.json
```

Use this when a sweep changes only config levers and you need to know whether it changed:

- the plain no-reference eval graph
- the matched-target training graph
- both
- or neither

Contexts:

- `plain`
- `configured`
- `spatial`
- `code`

Most important interpretation rules:

- if `plain == 0` and `configured > 0`, the experiment is train-graph real but eval-path inert
- if `plain > 0`, the no-reference benchmarked path itself has changed
- if blend variants change `configured/spatial` but not `plain`, TopoGate is live in training but no-reference eval still has no spatial carrier
- if low-rank code-map variants change `plain` and flip `anatomy_code_body_dead_spatial_body_live` to `false`, the repair has finally reached body-level no-reference style actuation

This probe is the fastest guardrail against wasting a full rerun on a config change that never touches the graph we actually evaluate.

Training-path differential probe:

```bash
py -3.12 tools/probe_training_variant_effect.py \
  --config G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-18-remote-h1-e18-diagnosis/remote_config.json \
  --variant-spec G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-18-stage1-training-effect-probe/stage1_variant_spec.json \
  --output-dir G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-18-stage1-training-effect-probe/probe_random_init \
  --device cpu
```

Outputs:

```text
variant_training_effects.csv
summary.json
baseline_effective_config.json
variant_spec.expanded.json
```

Use this when variants differ mainly in OT, coupling, projection, or bridge settings and you need to know whether the training path itself changed.

What it compares against baseline:

- `matched_target`
- `objective_target`
- `x_t`
- `target_velocity`
- `pred_velocity`
- selected OT / bridge metrics

Most important interpretation rules:

- if `matched_target_vs_base_mean_abs == 0` but `x_t_vs_base_mean_abs > 0`, the variant is a bridge-only training change
- if `matched_target_vs_base_mean_abs > 0`, the OT hypothesis really changed the training target
- if both the training probe and the eval-graph probe stay near zero, the experiment is extremely likely to be a practical no-op

Checkpoint-vs-init response probe:

```bash
py -3.12 tools/probe_checkpoint_style_response.py \
  --config G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-18-remote-h1-e18-diagnosis/remote_config.json \
  --checkpoint G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-18-remote-h1-e18-diagnosis/epoch_0018.pt \
  --output-dir G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-19-checkpoint-style-response-audit/remote_h1_epoch18 \
  --device cpu \
  --batch-size 2 \
  --latent-size 32 \
  --style-id 0 --style-id 1 --style-id 2 --style-id 3 --style-id 4
```

Outputs:

```text
summary.json
comparison_metrics.csv
init_conditioning_sensitivity.csv
checkpoint_conditioning_sensitivity.csv
init_path_anatomy.csv
checkpoint_path_anatomy.csv
init_styleid_pairwise.csv
checkpoint_styleid_pairwise.csv
```

Use this when:

- random-init probes are live
- trained checkpoints are still clustered
- you need to know whether training suppressed one lever, amplified another, or left the path unchanged

Most important interpretation rules:

- `trained_style_suppression`:
  - training shrank a lever that was live at init
- `trained_style_amplification`:
  - training strengthened that lever
- `matched_target_suppressed_styleid_amplified_body_dead`:
  - the model is **not** a no-op
  - matched-target / topology response was suppressed
  - plain no-reference `style_id -> decoder` response was amplified
  - `h_body` still stayed dead

Reference artifact:

- `docs/experiments/2026-06-19-checkpoint-style-response-audit/remote_h1_epoch18/README.md`

Important paired-example:

- `docs/experiments/2026-06-18-stage1-lowrank-distill-contract-probe/README.md`
- this is the reference case for a **training-live but runtime-inert by design** lever:
  - `probe_training_variant_effect.py` shows `plain_path_distill_active = 1`
  - `probe_config_effectiveness.py` shows `plain/configured/spatial/code` all stay `0.0`

Config-family audit wrapper:

```bash
py -3.12 tools/audit_config_family.py \
  --baseline-config <baseline_config.json> \
  --variant-dir <run_root_or_config_dir> \
  --output-dir <audit_output_dir> \
  --device cpu
```

Purpose:

- diff real generated `config.json` files into a reusable `variant_spec.json`
- ignore run-local noise like `checkpoint.save_dir`, `ablation.*`, and `training.resume_*`
- then feed that exact family through:
  - `probe_config_effectiveness.py`
  - `probe_training_variant_effect.py`

Best use case:

- when named runs are close and you no longer trust the historical meaning of `h0/h1/r5/...`
- after base repairs changed what a family member actually inherits

Smoke example:

- `docs/experiments/2026-06-18-config-family-audit-smoke/README.md`

Family validity matrix generator:

```bash
py -3.12 tools/experiments/build_phase618_family_validity_matrix.py
```

Outputs:

```text
docs/experiments/2026-06-18-family-validity-matrix/
  README.md
  global_invalidators.csv
  suite_validity_matrix.csv
  family_validity_matrix.csv
  summary.json
```

Use this after refreshing the underlying probes when you want one place that answers:

- which experiment families were invalidated by implementation bugs
- which families are training-real but eval-inert
- which bold directions are real runtime levers but still weak
- which next-stage levers are most aligned with fixing the train/eval contract gap

Stage1 no-reference eval-effect probe:

```bash
py -3.12 tools/probe_config_effectiveness.py \
  --config G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-18-remote-h1-e18-diagnosis/remote_config.json \
  --variant-spec G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-18-stage1-training-effect-probe/stage1_variant_spec.json \
  --output-dir G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-18-stage1-config-effect-probe/probe_random_init \
  --device cpu
```

Observed on 2026-06-18 relative to `h1_linear_fm`:

- `h0`, `h2`, `h3`, `h4`, `h5`, `h6` all classify as `no_effect`
- `plain/configured/spatial/code` deltas all stay `0.0`

Interpretation:

- stage1 variants are real in the training graph
- but the current no-reference eval graph is unchanged across that family
- if those runs come out nearly tied, that is expected from train/eval contract mismatch rather than proof that OT / bridge code never executed

Stage1 low-rank rerun audit:

- `docs/experiments/2026-06-18-stage1-lowrank-rerun-audit/README.md`

What it adds on top of the earlier stage1 probes:

- the repaired `h1_lowrank` baseline flips `anatomy_code_body_dead_spatial_body_live` to `false`
- the old `h0` / `h2` / `h3` / `h4` / `h5` / `h6` family still does not change the plain forward graph pairwise
- but the training-effect probe still shows large OT / target differences, especially for `h2`, `h5`, and `h6`

Interpretation:

- after the low-rank repair, a near-tie across the old OT family is much less likely to be an implementation-dead artifact
- it becomes much stronger evidence that the old OT hypotheses are simply not moving the benchmark enough
- this is the right handoff point to the bold architecture directions in `docs/618/bold_directions.md`

## 6. Metrics and stop policy

Primary metrics:

- `transfer_clip_style`
- `transfer_content_lpips`

Training probes:

- `ot_target_gini`
- `gpu_vram_used_gb_peak`
- `gpu_power_w_peak`
- `ot_topogate_diag_mean`
- `ot_topogate_entropy_mean`
- `ot_topogate_cost_mean`
- `ot_structure_transport_cost_var`
- `ot_total_cost_matrix_var`
- `ot_topogate_complexity_term_var`
- `ot_latent_affinity_term_var`

Targets:

- `style >= 0.74`
- `lpips <= 0.30`

Resource targets:

- VRAM `9.0-10.8 GB`
- Power close to `135W+`

Safety thresholds:

- `VRAM > 11.3 GB` => OOM
- `LPIPS > 0.45` => dangerous
- `ot_target_gini > 0.6` => dangerous

## 7. OT implementation focus

Current 616 OT priorities:

- `bridge_path_mode="vertical"` remains the main baseline line.
- `topogate_attention_gw` replaces the old tokenizer-entropy route.
- OT structural cost should come from model-internal features, not style-map tokenizer outputs.
- For `topogate_attention_gw`, do not read `ot_structure_cost_mean == 1.0` as
  "constant matrix":
  - that mean is fixed by internal normalization in the blended TopoGate path
  - inspect the variance-bearing fields instead:
    - `ot_topogate_descriptor_blocks`
    - `ot_structure_transport_cost_var`
    - `ot_total_cost_matrix_var`
    - `ot_topogate_complexity_term_var`
    - `ot_latent_affinity_term_var`
  - on multiblock semantic bodies, `ot_topogate_descriptor_blocks` should usually be
    `> 1`; if it drops back to `1` unexpectedly, audit
    `tools/probe_topogate_descriptor_coverage.py` before trusting h5/h6 results

Main code locations:

```text
src/losses.py
src/run.py
src/trainer.py
src/utils/training.py
```

## 8. Practical rules

- If stage 1 is already running, do not inject another tree into the same target root.
- Prefer the automatic stage runners for stage 2 and stage 3 because they unify:
  - batch probe
  - 1 minute health check
  - 10 minute ETA check
  - sleep-to-ETA
  - full-eval convergence stop
- Judge experiments from full-eval outputs, not only the probe.
- Every stage trains from scratch unless the task is explicitly to resume the same run.

## 9. Best-image materialization recovery

When a run was evaluated with `--no-save_generated_images`, its eval directory can still
have metrics and convergence data but no reusable PNGs for CLIP-T or VLM follow-up.

Recovery entry:

```bash
python3 tools/experiments/phase616_auto.py materialize-best-images --run-dir <run_dir>
```

This recovery path:

- reads the current best epoch from `full_eval_transfer/clip_lpips_curve.csv`
- re-runs transfer-only eval for that best checkpoint
- forces `--save_generated_images`
- writes a record to `<run_dir>/best_eval_materialization.json`

Current remote waiting helper:

```bash
exp/20250618_lite_ot_vertical_auto/phase616_materialize_best_images.sh
```

Current helper behavior:

- waits until no active `phase616_auto.py`, `src/run.py`, or `src/utils/run_evaluation.py`
- materializes best images for `h0_vertical_fm`
- materializes best images for `h1_linear_fm`
- appends progress to:

```text
/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/20250618_lite_ot_vertical_auto/phase616_materialize_best_images.log
```
