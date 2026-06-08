# `Hold4Mid e8 + Carrier-Gate Injection` Remote Packet

Date: 2026-06-08

Scope:

- dataset: `Distinct5-512`
- surface: remote `3060 WSL`
- config:
  - [inmortal_hold4mid_e8_carriergate_injection_seed42_b8a2.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_hold4mid_e8_carriergate_injection_seed42_b8a2.json)

Intent:

- stop asking the clamp schedule alone to solve style reopening
- treat `Hold4Mid e8` as a stable geometry anchor
- add an explicit late style carrier on top of that anchor using the already-implemented `carrier_gate` injection path

Why this candidate exists:

- `Hold4Mid e8` is the strongest ultra-low-LPIPS point
- non-CLIP audit shows the hold family is not style-dead
- `Hold4SlowMid` and likely `Hold4TwoStage` show schedule-only continuation is not enough
- the next coherent mechanism is therefore:
  - freeze the transport/geometry backbone
  - train only a style-recovery branch

Mechanism:

- resume from:
  - `Hold4Mid e8`
- training mode:
  - `freeze_mode = injection_only`
- resume mode:
  - `resume_model_strict = false`
  - reason:
    - the new carrier-gate modules do not exist in the parent checkpoint, so strict checkpoint loading would fail before training starts
- active modules:
  - `style_injection_mode = body_decoder`
  - `style_injection_form = carrier_gate`

Success condition:

- raise style over `Hold4Mid e8` while staying near its LPIPS band
- preserve the geometry-anchor behavior
- improve visual style specificity under non-CLIP and visual audit

Failure condition:

- style injection only is too weak to lift style
- or it raises style only by reopening geometry drift and artifacts

Early health:

- remote launch passed first health under the paper-facing machine contract
- the packet is materially lighter than the full-family continuation runs:
  - `epoch 1 loss = 8.7436`
  - `terminal_swd = 5.0938`
  - `epoch_time = 144.90s`
  - `samples_per_sec = 34.51`
- interpretation:
  - `freeze_mode=injection_only` is working as intended
  - the carrier-gate modules are trainable and stable after non-strict resume
  - this is the first late style-recovery line that truly changes the active mechanism family rather than only changing a clamp schedule
