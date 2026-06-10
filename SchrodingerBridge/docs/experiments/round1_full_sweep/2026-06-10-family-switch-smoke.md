# Round 1 Family Switch Smoke

Date: 2026-06-10

Purpose:

- verify that every round-1 tokenizer / backbone / solver family can be selected through config switches
- verify that the DINO sidecar path is not a dead stub:
  - `content_dino_patches`
  - `content_dino_hw`
  - `target_style_dino_bank_patches`
- verify one minimal forward + integration + loss + backward pass per family before more remote lane time is spent

Entrypoint:

- script:
  - [smoke_round1_family_switches.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/smoke_round1_family_switches.py)
- command used:
  - `py -3 SchrodingerBridge\tools\experiments\smoke_round1_family_switches.py --device cpu`
- machine-readable result:
  - [round1_family_switch_smoke_20260610.json](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_family_switch_smoke_20260610.json)

Scope:

- batch:
  - `1`
- synthetic latent size:
  - `4 x 32 x 32`
- checks per family:
  - direct model forward
  - `predict_transport_base`
  - `integrate_transport`
  - `OTFlowMatchingObjective.compute(...)`
  - `loss.backward()`

Result:

- selected family count:
  - `11`
- failure count:
  - `0`
- all families completed:
  - `legacy-factorized + attention variants`
  - `solver variants`
  - `tok_a_dino_dict`
  - `tok_b_cross_image`
  - `tok_c_residual_adapter`
  - `tok_d_vlm_prompt`
- launcher integration:
  - `launch_remote_round1_family_train.py` now runs this smoke gate by default before any formal remote launch
  - if smoke fails:
    - the remote lane is not touched
  - if smoke passes but another family is already `running`:
    - launch is still refused after smoke, preserving the single-lane rule
  - `run_round1_family_queue.py` now consumes the recorded smoke evidence too:
    - prefer `switch_smoke_status=ok`
    - skip `switch_smoke_status=failed` by default
  - `launch_remote_round1_family_train.py` now also persists the smoke result to the manifest immediately
  - after a successful direct remote launch it also marks the family `running` immediately and refreshes round-1 status docs
  - after a successful direct remote launch it also arms the detached runtime watcher automatically
  - after a successful direct remote launch it also arms the remote fast-eval watcher automatically by default
  - shared family followups now also arm the local remote-fast-eval sync watcher automatically
  - shared family followups now also arm a queue-idle watcher that waits for zero `running` families and then invokes the existing round1 queue once
  - queue-driven launches explicitly disable that default and continue to launch fast-eval through the queue-owned path

Gate validation:

- representative controlled test:
  - `solver_pc`
- observed behavior:
  - local prelaunch smoke passed and wrote:
    - [round1_solver_pc_switch_smoke_latest.json](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_solver_pc_switch_smoke_latest.json)
  - remote formal launch was then correctly refused because:
    - `solver_tangent_rk` is still the active running family

Interpretation:

- all round-1 family switches are now covered by one reusable smoke harness instead of ad hoc manual spot checks
- the DINO tokenizer families and `dino_masked_swd` both executed through the real runtime-conditioning path, not a mocked bypass
- this smoke proves:
  - config parsing
  - model construction
  - family dispatch
  - minimal forward/backward viability
  - launch-path precheck viability
- this smoke does not prove:
  - convergence
  - remote VRAM compliance
  - external-board quality

Notable read:

- all returned tensors kept the expected latent shape:
  - `1 x 4 x 32 x 32`
- all losses were finite
- all families produced at least one non-empty gradient read after backward
