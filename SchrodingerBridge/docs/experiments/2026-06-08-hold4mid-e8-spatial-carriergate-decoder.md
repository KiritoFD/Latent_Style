# `Hold4Mid e8 + Spatial Carrier-Gate Decoder Injection`

Date: 2026-06-08

Config:

- [inmortal_hold4mid_e8_spatial_carriergate_decoder_seed42_b8a2.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_hold4mid_e8_spatial_carriergate_decoder_seed42_b8a2.json)

Intent:

- move beyond the current global/channel-only `carrier_gate` path
- reopen style from the `Hold4Mid e8` geometry basin
- inject target-specific spatial texture, not just more global painterly energy

Why this packet exists:

- the multi-source visual diagnosis now says:
  - `LBM-Knee` is geometry-strong but pale and under-committed
  - `LBM-PS-v2` is stronger on painterly energy but drifts into generic fog
  - `SaMST` gets visible style mostly by sacrificing structure
- therefore the next coherent mechanism is:
  - keep the `Hold4Mid e8` transport field frozen
  - add a source-aware spatial residual branch only at the decoder stage

Mechanism:

- resume from:
  - `Hold4Mid e8`
- training mode:
  - `freeze_mode = injection_only`
- resume mode:
  - `resume_model_strict = false`
  - reason:
    - the new spatial injection modules do not exist in the parent checkpoint
- active modules:
  - `style_injection_mode = decoder`
  - `style_injection_form = spatial_carrier_gate`

What is new relative to the earlier `carrier_gate` packet:

- the old packet injected only a global channel bias
- the new packet injects:
  - a style-spatial carrier derived from the learned target spatial prior
  - a content-conditioned channel gate
  - a source-aware local gate from the current latent
  - an enforced high-pass residual on the injected spatial field

Expected upside:

- stronger target-style texture hierarchy than `Hold4Mid e8`
- less risk of a global fog/filter failure than `LBM-PS-v2`
- less risk of full geometry drift than `SaMST`

Primary question:

- can a decoder-only spatial residual reopen visible style
- while preserving the `Hold4Mid e8` geometry anchor
- across the multiple source-family cases now tracked in the visual diagnosis packet

Early health:

- remote launch passed the first health window
- observed first-health GPU band:
  - about `3944 MiB / 12288 MiB`
- observed trainable branch under `freeze_mode=injection_only`:
  - `decoder_content_gate.*`
  - `decoder_style_spatial_proj.*`
  - `decoder_structure_gate.*`
- interpretation:
  - the new packet is materially lighter than full-family continuation
  - the spatial carrier path is actually the only trainable branch
  - this is the first direct test of `target-specific spatial reinjection from the Hold4Mid geometry basin`

Post-train watcher:

- task:
  - `inmortal-spatial-carrier-posteval-watch`
- behavior:
  - wait for `remote_train.log` to close
  - then run `full_eval_fast_snapshot`
  - then refresh:
    - stage summary
    - epoch eval table
  - note:
    - the main training process is also running its own deferred `full_eval/epoch_xxxx` packet first
    - so early paper-facing reads may appear under `full_eval/` before `full_eval_fast_snapshot/` exists

Current live status:

- training is still active
- latest direct read:
  - run is already inside `Epoch 4/12`
  - `epoch_0001.pt` is present under the run root
  - runtime memory remains around `3.9 GiB`
- interpretation:
  - the packet is stable enough to continue unattended
  - the first fast-eval packet has not started yet only because training has not ended

Early epoch-level read:

- latest per-epoch CSV snapshot:
  - `epoch 1: loss 8.9028, terminal_swd 5.3438`
  - `epoch 2: loss 9.0098, terminal_swd 5.3438`
  - `epoch 3: loss 8.9008, terminal_swd 5.5625`
  - `epoch 4: loss 8.8242, terminal_swd 5.3125`
  - `epoch 5: loss 9.0359, terminal_swd 5.3125`
  - `epoch 6: loss 9.0626, terminal_swd 5.5312`
- later per-epoch CSV read:
  - `epoch 7: loss 8.9708, terminal_swd 5.2188`
  - `epoch 8: loss 8.8262, terminal_swd 5.1562`
  - `epoch 9: loss 8.9697, terminal_swd 5.4375`
  - `epoch 10: loss 9.1089, terminal_swd 5.5938`
  - `epoch 11: loss 8.9452, terminal_swd 5.3125`
- late live read from `epoch 12` tail:
  - rolling `terminal_swd` re-enters the `5.28 ~ 5.75` band
  - final visible tail point in the train log is about:
    - `loss 8.7991`
    - `flow 0.8276`
    - `kin 0.0836`
    - `tswd 5.2812`
- interpretation:
  - the packet is numerically stable
  - the epoch-level style signal is not opening clearly
  - but it also is not collapsing monotonically
  - the later tail read is better than the mid-run pessimistic read
  - current training-side evidence is better described as `flat / oscillatory`
- current evidence is therefore:
  - `stable`
  - and still `unproven as a positive mechanism until fast-eval`

First full-eval read:

- currently landed:
  - `full_eval/epoch_0001/summary.json`
  - `full_eval/epoch_0002/summary.json`
  - `full_eval/epoch_0003/summary.json`
- transfer read:
  - `epoch_0001 = 0.6676 / 0.2884`
  - `epoch_0002 = 0.6672 / 0.2888`
  - `epoch_0003 = 0.6674 / 0.2910`

Comparison to current anchors:

- `Hold4Mid e8`
  - `0.6679 / 0.2877`
- interpretation:
  - the new packet is not opening a stronger style regime
  - it is behaving like a near-tie around the `Hold4Mid` geometry band
  - at the current read, it is slightly worse than `Hold4Mid e8`
- relative to `LBM-Knee`
  - it remains far below the stronger style frontier

Current closure tendency:

- unless a later eval checkpoint improves materially, this packet should be treated as a likely `near-tie negative`
- the queued next escalation remains:
  - [2026-06-08-hold4mid-e8-spatial-carriergate-bodydecoder.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-08-hold4mid-e8-spatial-carriergate-bodydecoder.md)

Fast-snapshot read:

- currently landed:
  - `full_eval_fast_snapshot/epoch_0001 ~ epoch_0008`
- transfer read:
  - `epoch_0001 = 0.6889 / 0.5149`
  - `epoch_0002 = 0.6888 / 0.5153`
  - `epoch_0003 = 0.6889 / 0.5157`
  - `epoch_0004 = 0.6892 / 0.5169`
  - `epoch_0005 = 0.6889 / 0.5173`
  - `epoch_0006 = 0.6890 / 0.5172`
  - `epoch_0007 = 0.6888 / 0.5166`
  - `epoch_0008 = 0.6888 / 0.5166`

Interpretation of fast-snapshot:

- this packet is not a `Hold4Mid` near-tie on the actual paper-facing transfer read
- instead it reopens style only by paying a much larger LPIPS cost
- relative to anchors:
  - `Hold4Mid e8 = 0.6679 / 0.2877`
  - `LBM-Knee e13 = 0.7102 / 0.4603`
- so the decoder-only spatial carrier currently lands in an unattractive middle zone:
  - style below `LBM-Knee`
  - LPIPS worse than `LBM-Knee`
  - and much worse geometry than `Hold4Mid`

Updated closure tendency:

- this is now a strong `negative closure` candidate
- if the remaining checkpoints do not reverse this picture materially, the packet should be closed as:
  - `style reopening without a useful frontier gain`
