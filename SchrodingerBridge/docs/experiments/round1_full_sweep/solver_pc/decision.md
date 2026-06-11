# solver_pc Decision

- Decision date:
  - `2026-06-11`
- Current status:
  - `running`
- Current round read:
  - `solver_pc` is still an active formal lane, not a closure candidate
  - the authority remains the remote all-ckpt `CLIP-S + LPIPS` curve
  - local deep review stays blocked until a bestfew handoff exists

## Formal Setting

- Canonical config:
  - [aaai2027_round1_solver_pc_seed42_b8a2.json](G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/round1_full_sweep/aaai2027_round1_solver_pc_seed42_b8a2.json)
- Formal batch setting:
  - `batch=16`
  - `accumulation_steps=2`
- VRAM calibration:
  - `batch=8 -> 5216 MiB`
  - `batch=14 -> 8226 MiB`
  - `batch=16 -> 9334 MiB` at first formal health check
- Current live band check:
  - `10344 MiB / 12288 MiB`
  - read: safely inside the requested formal band

## Curve Read

- Best transfer `CLIP-S` remains:
  - `epoch_0001`
  - transfer `0.7074 / 0.5621`
- Best transfer `LPIPS` remains:
  - `epoch_0009`
  - transfer `0.6911 / 0.4548`
- Strong late tradeoff frontier points after the `0009` LPIPS knee:
  - `epoch_0013`
    - transfer `0.6968 / 0.5101`
    - full `0.7142 / 0.4996`
  - `epoch_0015`
    - transfer `0.6962 / 0.4854`
    - full `0.7165 / 0.4746`
  - `epoch_0017`
    - transfer `0.6982 / 0.5075`
    - full `0.7159 / 0.4964`
- Latest locally pulled point:
  - `epoch_0030`
  - transfer `0.6860 / 0.4851`
  - full `0.7067 / 0.4758`
  - read: `epoch_0029` was a minor repair, but `epoch_0030` still remained a non-frontier tail point

## Decision

- Keep running.
- Rationale:
  - `patience=6` for solver families
  - `since_last_pareto=13`, so the line is now very deep into the post-patience tail
  - `tail_flat=false`, so the family still does not satisfy true closure
  - `epoch_0021-0030` reads more like noisy tail drift than renewed frontier search
  - best style and best LPIPS are still split across different checkpoints, so the family is still exploring the tradeoff surface
- Promotion rule:
  - do not promote this family on internal oscillation alone
  - require a full family closure packet plus deep review before any keep/reject decision

## Next Action

- Let the remote lane continue.
- Keep syncing every retained checkpoint into the local fast-curve packet.
- Open deep review only after the training lane truly closes and a bestfew shortlist is frozen.
