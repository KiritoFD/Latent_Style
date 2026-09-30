# solver_pc Decision

- Decision date:
  - `2026-06-11`
- Current status:
  - `reviewing`
- Current round read:
  - remote training is closed
  - the authority remains the all-ckpt remote `CLIP-S + LPIPS` curve
  - the family now moves to bestfew review rather than more open-ended training

## Formal Setting

- Canonical config:
  - [aaai2027_round1_solver_pc_seed42_b8a2.json](G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/round1_full_sweep/aaai2027_round1_solver_pc_seed42_b8a2.json)
- Final formal batch setting:
  - `batch=16`
  - `accumulation_steps=2`
- Final remote segment reached:
  - settled through `epoch_0036`

## Curve Read

- Best transfer `CLIP-S` remained:
  - `epoch_0001`
  - transfer `0.7074 / 0.5621`
- Best transfer `LPIPS` remained:
  - `epoch_0009`
  - transfer `0.6911 / 0.4548`
- Strongest late tradeoff frontier remained:
  - `epoch_0015`
  - transfer `0.6962 / 0.4854`
  - full `0.7165 / 0.4746`
- Final settled point:
  - `epoch_0036`
  - transfer `0.6834 / 0.4964`
  - full `0.7028 / 0.4875`
  - read: non-frontier tail point

## Decision

- Stop training and move to review.
- Rationale:
  - `since_last_pareto` is far beyond solver-family patience
  - bounded continuation through `epoch_0036` still failed to create a new Pareto point
  - the tail remains noisy rather than frontier-seeking
  - further continuation is no longer justified by the observed metric trajectory

## Promotion Rule

- Do not promote on internal oscillation alone.
- Require the same deep-review package as every other round-1 family before any keep/reject conclusion.

## Next Action

- Freeze the current training packet.
- Build the bestfew handoff.
- Open local `IntroStyle + DINO + frozen VLM` review.
- Hand the remote formal lane to `solver_unsb_cycle`.
