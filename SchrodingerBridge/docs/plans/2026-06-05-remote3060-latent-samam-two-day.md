# Remote 3060 Latent SaMam Two-Day Experiment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Land a paper-usable Distinct5-512 remote `RTX 3060` experiment packet for `latent SaMam`, or close the line negatively with enough evidence that we can state why it should not appear in the paper.

**Architecture:** Keep the experiment on the existing remote `WSL Ubuntu-26.04` surface and reuse the current Distinct5 latent/eval contract. The minimal path is not a full SaMam rewrite: adapt SaMam to `4`-channel latent input/output, decode only for perceptual loss and final evaluation, and keep all outputs synchronized into a reviewable remote packet with timing, summaries, and per-image rows.

**Tech Stack:** Python, PyTorch Lightning, SaMam upstream code, Stable Diffusion VAE decode, SchrodingerBridge evaluator, remote `RTX 3060` WSL, git with small commits.

---

## Fixed Constraints

- Formal experiments run on remote `3060` only. No local GPU continuation.
- Dataset contract:
  - latent train root: `/mnt/i/wikiart_distinct5_samam_512_latents_ema/train`
  - pairing cache reference: `/mnt/i/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/prototype_pairing_top8.pt`
  - RGB eval/test root: `/mnt/i/wikiart_distinct5_samam_512_classview/test`
- Remote repo root: `/mnt/i/Github/Latent_Style`
- Do not disturb `Mencius`.
- Do not spend day 2 on LoRA or SDXL. This packet is `latent SaMam` first.

## Definition Of Done

One of the following must be landed by the end of day 2:

1. **Positive closure**
   - remote latent SaMam training is stable past the smoke gate
   - at least one retained checkpoint has full eval on Distinct5-512
   - packet contains `summary.json`, `metrics.csv`, targetwise `ArtFID`, train wall, eval wall, and a same-cost plot point
   - outputs are synced under an authoritative `I:\...` result root

2. **Negative closure**
   - two explicit remote latent SaMam attempts are documented
   - failure mode is pinned to one of: NaN instability, decode-loss mismatch, throughput collapse, or clearly inferior frontier
   - the note includes exact commands, logs, wall time spent, and why the line should be excluded from the paper

## Day 1

### Task 1: Freeze the run contract

**Files:**
- Create: `Related_Works/baseline_pipeline/scripts/push_remote_samam_latent_packet.py`
- Create: `SchrodingerBridge/docs/experiments/2026-06-05-remote3060-latent-samam-launch.md`
- Modify: `SchrodingerBridge/docs/timing/` packet index if a new timing CSV is needed

**Steps:**
1. Record the authoritative remote paths, dataset roots, and output root naming convention in the launch note.
2. Mirror the SaMST packet pattern: local reviewed files are the source of truth, then tar-sync to remote, then `py_compile`.
3. Create the sync helper for the smallest latent SaMam packet only.
4. Commit and push the packet skeleton before the first remote launch.

**Pass gate:**
- a single command can sync the latent SaMam packet to `/mnt/i/Github/Latent_Style`
- remote `py_compile` passes on all packet files

**Suggested commit:**
- `git commit -m "Add remote latent SaMam packet scaffold"`

### Task 2: Build the minimal latent SaMam path

**Files:**
- Modify: `Related_Works/repos/SaMam/MODEL/SaMam_model.py`
- Modify: `Related_Works/repos/SaMam/ARCHI/Decoder.py`
- Create: `Related_Works/repos/SaMam/TRAIN/lightning_module/latent_datamodule.py`
- Create: `Related_Works/repos/SaMam/TRAIN/lightning_module/latent_lightningmodel.py`
- Create: `Related_Works/baseline_pipeline/scripts/run_samam_latent_distinct5_remote.py`

**Implementation target:**
- SaMam encoder takes latent tensors with `in_chans=4`
- decoder emits latent tensors with `out_chans=4`
- training loss decodes predicted/content/style latents back to RGB only for VGG-based losses
- training data reads cached latents directly from the Distinct5 latent root instead of JPG folders

**Steps:**
1. Add configurable `in_chans` and `out_chans` to SaMam model/decoder without breaking the existing RGB path.
2. Create a latent Lightning module that wraps VAE decode before loss computation.
3. Create a latent datamodule that reads the Distinct5 latent cache layout directly.
4. Add a remote runner that saves step checkpoints and logs wall time.
5. Run local `py_compile` only. Do not run local training.
6. Commit and push after compile is clean.

**Pass gate:**
- the latent codepath imports cleanly
- the RGB SaMam path still compiles
- no hard-coded `3`-channel assumption remains in the latent path

**Suggested commit:**
- `git commit -m "Add minimal latent SaMam training path"`

### Task 3: Remote smoke run and stability verdict

**Files:**
- Create: `Related_Works/baseline_pipeline/scripts/run_samam_latent_eval_bundle.py`
- Update: `SchrodingerBridge/docs/experiments/2026-06-05-remote3060-latent-samam-launch.md`
- Update: `SchrodingerBridge/docs/timing/<new timing csv>`

**Remote launch target:**
- batch size starts conservatively
- save checkpoints every `100` or `200` steps
- hard wall budget for the first smoke run: about `45-60 min`

**Steps:**
1. Sync the packet to remote and verify imports in WSL.
2. Launch smoke run `A` with conservative batch/precision.
3. Watch at sub-30s cadence until the run is clearly healthy or clearly bad.
4. If NaNs or OOM appear, adjust only one stability variable for smoke run `B`.
5. Stop after two smoke attempts. Do not drift into endless tuning on day 1.
6. If a checkpoint is retained, run a tiny generation sanity check before sleeping.
7. Commit the launch note and timing evidence.

**Pass gate:**
- training remains finite through the first checkpoint boundary
- one checkpoint can generate non-degenerate outputs

**Fail-fast rule:**
- if both smoke attempts die before the first retained checkpoint, close the line as `negative feasibility` and spend day 2 on the fallback packet

**Suggested commit:**
- `git commit -m "Log remote latent SaMam smoke results"`

## Day 2

### Task 4: Formal short-budget latent SaMam packet

**Files:**
- Update: `Related_Works/baseline_pipeline/scripts/run_samam_latent_distinct5_remote.py`
- Update: `Related_Works/baseline_pipeline/scripts/run_samam_latent_eval_bundle.py`
- Create: `SchrodingerBridge/docs/experiments/2026-06-06-remote3060-latent-samam-packet.md`

**Steps:**
1. Promote the best stable smoke setting into one formal short-budget run.
2. Keep the packet step-aligned, not open-ended.
3. Retain at least two checkpoints so we can pick a same-cost point instead of a blind final point.
4. Run Distinct5 full eval using the same bundle contract already used for SaMST/SaMAM closure.
5. Record train wall, generation wall, eval wall, and exact checkpoint identity.
6. Commit and push once the packet is closed.

**Pass gate:**
- one checkpoint has full `CLIP-S`, `LPIPS`, targetwise `ArtFID`, and per-image `metrics.csv`
- timing is recorded in a root timing artifact, not only in ad hoc logs

**Suggested commit:**
- `git commit -m "Close remote latent SaMam evaluation packet"`

### Task 5: Same-cost comparison artifact for the paper

**Files:**
- Modify: `SchrodingerBridge/docs/experiments/2026-06-04-distinct5_same_cost_inventory.csv`
- Modify: `SchrodingerBridge/tools/build_local_experiment_inventory.py` only if the new packet must be indexed there
- Create: `SchrodingerBridge/docs/experiments/2026-06-06-latent-samam-same-cost-note.md`

**Steps:**
1. Add the retained latent SaMam point as a new row, clearly labeled experimental and remote.
2. Compare against:
   - LBM `F e1` / `K e1`
   - SaMAM `step_2250`
   - SaMST short-budget packet if the time scale is close enough
3. Plot exactly one paper-relevant frontier:
   - `1 - LPIPS` on x/y orientation consistent with the current paper preference
   - `CLIP-S` or IDT-adjusted `CLIP-S` on the other axis
   - include the IDT reference line
4. Write one paragraph that states whether latent SaMam strengthens the fairness story or should be excluded.

**Pass gate:**
- we can answer "does latent SaMam help the paper?" from a single artifact without reopening logs

**Suggested commit:**
- `git commit -m "Add latent SaMam same-cost comparison artifact"`

### Task 6: Fallback if latent SaMam fails

**Trigger:** execute only if Task 3 fails fast or Task 4 collapses.

**Files:**
- Create: `SchrodingerBridge/docs/experiments/2026-06-06-latent-samam-negative-closure.md`
- Optionally create: one remote same-cost baseline note under `SchrodingerBridge/docs/experiments/`

**Steps:**
1. Write the negative closure with exact remote logs and failure signatures.
2. State whether failure came from architecture mismatch, loss/decode instability, or unacceptable speed/quality.
3. Use the remaining remote time for the highest-value paper-safe fallback only:
   - strengthen a matched short-budget baseline packet on the same remote surface
   - do not branch into LoRA, SDXL, or unrelated ablations
4. Commit and push the negative closure.

**Pass gate:**
- the paper can safely say we tested the latent SaMam direction and chose not to include it for a documented reason

**Suggested commit:**
- `git commit -m "Document latent SaMam negative closure"`

## Recommended Remote Commands

These are the only command families that should exist by the time execution starts:

```bash
cd /mnt/i/Github/Latent_Style
python3 Related_Works/baseline_pipeline/scripts/push_remote_samam_latent_packet.py
python3 Related_Works/baseline_pipeline/scripts/run_samam_latent_distinct5_remote.py ...
python3 Related_Works/baseline_pipeline/scripts/run_samam_latent_eval_bundle.py ...
```

Avoid ad hoc one-liners outside the packet unless they are only for process inspection.

## Commit Rhythm

Minimum commit points:

1. packet scaffold synced and compile-verified
2. latent codepath compiles
3. smoke verdict landed
4. formal packet landed or negative closure landed
5. same-cost artifact landed

Every commit should be pushed immediately so the remote experiment packet always has a recoverable local source of truth.

## What Not To Do

- do not resume local WSL or local GPU experiment paths
- do not spend the two-day budget on writing polish first
- do not bury timing only inside terminal logs
- do not let a local-only `/home/...` packet become the authoritative result
- do not over-tune more than one variable at a time during smoke stabilization

## Success Readout For The Paper

At the end of day 2, the decision should be binary:

- **Keep latent SaMam in the paper** only if it gives a closed remote packet and a useful same-cost comparison point.
- **Exclude latent SaMam from the paper** if the line is unstable, too slow, or clearly uncompetitive under the matched remote budget, and keep the negative closure note as reviewer-defense evidence.
