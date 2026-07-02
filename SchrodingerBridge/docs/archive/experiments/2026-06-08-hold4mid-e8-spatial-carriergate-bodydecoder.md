# `Hold4Mid e8 + Spatial Carrier-Gate Body+Decoder Injection`

Date: 2026-06-08

Config:

- [inmortal_hold4mid_e8_spatial_carriergate_bodydecoder_seed42_b8a2.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_hold4mid_e8_spatial_carriergate_bodydecoder_seed42_b8a2.json)

Status:

Status update:

- the packet has now been formally launched on the remote `3060 WSL`
- launcher task:
  - `SB-AAAI2027_aaai2027_inmortal_hold4mid_e8_spatial_carriergate_bodydecoder_seed42_b8a2`
- the earlier queue watcher did reach the handoff point and attempted launch
  - that first attempt failed only because the new config file had not been synced yet
  - the reviewed formal launcher then relaunched the packet successfully

Current queue state:

- historical note:
  - the queue watcher reached the handoff point after the decoder-only lane released the GPU
  - the first auto-launch attempt failed because the new config file was not yet present on the remote checkout
  - this was an orchestration / sync failure, not a mechanism failure
  - the formal launcher has since corrected this and the packet is now live

First health:

- observed GPU band after launch:
  - about `4154 MiB / 12288 MiB`
- model params:
  - `6,253,087`
- freeze mode:
  - `injection_only`
- trainable branch count:
  - `28`
- active trainable families now include both:
  - `body_*spatial*`
  - `decoder_*spatial*`
- interpretation:
  - this is the intended stronger follow-up to the decoder-only negative packet
  - it remains comfortably under the remote paper-facing memory contract

Current live status:

- the packet is actively training
- observed checkpoints already landed:
  - `epoch_0001.pt`
  - `epoch_0002.pt`
  - `epoch_0003.pt`
  - `epoch_0004.pt`
  - `epoch_0005.pt`
  - `epoch_0006.pt`
  - `epoch_0007.pt`
- current runtime evidence:
  - `remote_train.log` is growing
  - `numeric_debug.jsonl` is growing
  - runtime memory remains around `4.1 GiB`
- early train read:
  - the packet has passed the first-health gate cleanly
  - it has not yet reached an eval-backed retained conclusion
  - current epoch-level CSV read:
    - `epoch 1: loss 8.9013, terminal_swd 5.3438`
    - `epoch 2: loss 9.0052, terminal_swd 5.3438`
    - `epoch 3: loss 8.8924, terminal_swd 5.5000`
    - `epoch 4: loss 8.8143, terminal_swd 5.3125`
    - `epoch 5: loss 9.0223, terminal_swd 5.3125`
    - `epoch 6: loss 9.0498, terminal_swd 5.5000`
  - interpretation:
    - the stronger body+decoder packet is stably running
    - and its first six epochs still look broadly similar to the early decoder-only packet rather than clearly stronger

Post-train watcher:

- task:
  - `inmortal-bodydecoder-posteval-watch`
- behavior:
  - wait for `remote_train.log` to close
  - then run `full_eval_fast_snapshot`
  - then refresh:
    - stage summary
    - epoch eval table
- current state:
  - watcher is armed and waiting
  - eval has not started yet because training is still active

Why this queued variant exists:

- the decoder-only spatial carrier packet is currently:
  - numerically stable
  - low-VRAM
  - but flat-to-slightly-negative on early `terminal_swd`
- that suggests a plausible failure mode:
  - decoder-only reinjection may be too late and too weak
  - style-specific spatial texture may not survive strongly enough to the endpoint

Mechanism difference from the current live packet:

- current live packet:
  - `style_injection_mode = decoder`
  - `style_injection_form = spatial_carrier_gate`
- queued stronger packet:
  - `style_injection_mode = body_decoder`
  - `style_injection_form = spatial_carrier_gate`

Rationale:

- let the same source-aware spatial carrier act at:
  - the body stage
  - and the decoder stage
- this is the smallest coherent escalation after decoder-only failure
- it stays faithful to the same theory:
  - recover target-specific spatial texture
  - without switching to a global fog-style solution
  - and without reopening the full transport field
