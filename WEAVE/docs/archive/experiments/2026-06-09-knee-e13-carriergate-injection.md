# `LBM-Knee e13 + Carrier-Gate Injection` Next-Line Packet

Date: 2026-06-09

Why this packet exists:

- `LBM-Knee e13` is now the strongest current internal balanced point under:
  - full750 `IntroStyle`
  - full750 `DINO`
  - and the current local VLM read
- `LBM-PS-v2` is no longer a trustworthy internal style frontier
- `Hold4Mid`-anchored reopen packets have so far remained too weak on style

Therefore the next coherent mechanism question is:

- can style be reopened from the current best internal tradeoff basin
- without reopening the `LBM-PS-v2` generic fog failure mode

Mechanism:

- resume from:
  - `aaai2027_inmortal_xpred_kmanifold_pattn_anisostokes_queue_from_pattn_seed42_b8a2/epoch_0013.pt`
- keep:
  - the `LBM-Knee` transport / proximal family
- train only:
  - `carrier_gate` style injection modules

Training contract:

- `freeze_mode = injection_only`
- `resume_model_strict = false`
- `resume_optimizer = false`
- `resume_training_state = false`

Success condition:

- beat `LBM-Knee e13` on `IntroStyle`
- stay clearly left of `LBM-PS-v2` on the DINO axis
- avoid the VLM failure pattern that currently punishes `LBM-PS-v2`

Failure condition:

- style does not rise materially above `LBM-Knee`
- or structure/artifact behavior drifts toward `LBM-PS-v2`

Why this is higher-value than more hold-family schedule tweaks:

- it directly targets the actual current burden:
  - reopen style from the best current internal balanced point
- it no longer assumes the strongest geometry anchor is also the best style-recovery anchor
