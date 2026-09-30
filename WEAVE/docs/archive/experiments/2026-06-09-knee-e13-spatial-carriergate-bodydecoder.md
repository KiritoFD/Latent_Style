# `LBM-Knee e13 + Spatial Carrier-Gate Body+Decoder` Next-Line Packet

Date: 2026-06-09

Why this packet exists:

- plain `Knee e13 + CarrierGate` improved absolute `IntroStyle target` only slightly
- but still stayed negative on `delta-IDT`
- and still moved rightward on the DINO axis relative to `LBM-Knee`

That suggests:

- the current `Knee` anchor is not style-dead
- but a plain channel carrier is still too weak or too generic

So the next coherent step is:

- keep the same `Knee e13` anchor
- switch from:
  - `carrier_gate`
- to:
  - `spatial_carrier_gate`

Mechanism:

- resume from:
  - `aaai2027_inmortal_xpred_kmanifold_pattn_anisostokes_queue_from_pattn_seed42_b8a2/epoch_0013.pt`
- freeze mode:
  - `injection_only`
- active injection family:
  - `style_injection_mode = body_decoder`
  - `style_injection_form = spatial_carrier_gate`

Hypothesis:

- the stronger spatial carrier may add:
  - more target-specific spatial style statistics
  - more explicit texture hierarchy
- without requiring the `Hold4Mid` geometry basin that already proved too style-weak

Success condition:

- beat `LBM-Knee` on `IntroStyle`
- avoid the negative `delta-IDT` pattern of the plain `carrier_gate` line
- stay left of `LBM-PS-v2` on `DINO`

Failure condition:

- style still does not rise in a target-directional way
- or structure drifts without opening a meaningful style advantage

## Live runtime update

Latest checked runtime state:

- the training phase is now finished
- retained checkpoints landed through `epoch_0012.pt`
- `full_eval_fresh_localreview` has now emitted `summary.json` through `epoch_0012`
- a remote `IntroStyle` probe over the saved fresh-localreview images is now running

Current fast-eval artifact:

- [knee_spatial_carriergate_bodydecoder_fast_eval_curve_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/knee_spatial_carriergate_bodydecoder_fast_eval_curve_20260609.csv)
- [knee_spatial_carriergate_bodydecoder_fresh_localreview_curve_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/knee_spatial_carriergate_bodydecoder_fresh_localreview_curve_20260609.csv)

Early fast-eval read:

- `epoch_0001`
  - transfer `CLIP-style / LPIPS = 0.7039 / 0.4396`
- `epoch_0002`
  - transfer `CLIP-style / LPIPS = 0.7038 / 0.4397`
- `epoch_0003`
  - transfer `CLIP-style / LPIPS = 0.7037 / 0.4389`
- `epoch_0004`
  - transfer `CLIP-style / LPIPS = 0.7036 / 0.4393`
- `epoch_0005`
  - transfer `CLIP-style / LPIPS = 0.7037 / 0.4397`
- `epoch_0006`
  - transfer `CLIP-style / LPIPS = 0.7034 / 0.4398`
- `epoch_0007`
  - transfer `CLIP-style / LPIPS = 0.7034 / 0.4391`
- `epoch_0008`
  - transfer `CLIP-style / LPIPS = 0.7038 / 0.4398`
- `epoch_0012`
  - transfer `CLIP-style / LPIPS = 0.7035 / 0.4396`

Interpretation:

- the line is stable
- but the full `epoch_0001..0012` fresh-localreview curve is still essentially flat
- best transfer LPIPS is only:
  - `epoch_0003 = 0.7036 / 0.4387`
- best transfer CLIP-style is only:
  - `epoch_0008 = 0.7038 / 0.4396`
- so this family now depends almost entirely on the non-CLIP read to justify staying alive
