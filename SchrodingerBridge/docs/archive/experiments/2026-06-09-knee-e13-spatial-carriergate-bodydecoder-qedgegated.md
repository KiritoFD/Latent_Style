# `LBM-Knee e13 + Spatial Carrier-Gate Body+Decoder + Quantile Edge-Gated Structure Leash`

Date: 2026-06-09

Why this packet exists:

- the plain spatial-carrier line showed:
  - slightly better `IntroStyle`
  - but much worse `DINO`
- the first edge-gated follow-up showed:
  - slightly better `DINO`
  - but gave back part of the style gain

That suggests the current edge gate is still too broad:

- it helps structure
- but it also suppresses style too aggressively

Mechanism:

- keep:
  - `Knee e13`
  - `spatial_carrier_gate`
  - `body_decoder`
  - weak `Stokes`
- replace the current soft edge gate with:
  - `quantile_edge_gated_anisotropic_plus_stokes`
- core idea:
  - only the strongest content edges should receive high normal-pressure penalty
  - flatter regions should be released more aggressively for style texture

Config:

- [inmortal_knee_e13_spatial_carriergate_bodydecoder_qedgegated_seed42_b8a2.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_knee_e13_spatial_carriergate_bodydecoder_qedgegated_seed42_b8a2.json)

Initial intended read:

- if this works:
  - `DINO` should stay closer to `LBM-Knee`
  - while `IntroStyle` should recover more of the style gain lost by the first edge-gated line
- if it fails:
  - that means the current carrier family likely needs a stronger style branch, not just a better leash

## Runtime update

Latest checked runtime state:

- the packet is live on the remote `3060`
- the actual run directory is:
  - `/mnt/i/Github/Latent_Style/exp/inmortal-exp/aaai2027_inmortal_knee_e13_spatial_carriergate_bodydecoder_qedgegated_seed42_b8a2`
- first-health passed under the machine contract
- early retained checkpoints have already landed:
  - `epoch_0001.pt`
  - `epoch_0002.pt`
  - `epoch_0003.pt`
  - `epoch_0004.pt`
  - `epoch_0005.pt`
  - `epoch_0006.pt`
  - `epoch_0007.pt`
  - `epoch_0008.pt`
  - `epoch_0009.pt`
  - `epoch_0010.pt`
  - `epoch_0011.pt`
- the post-train fresh-eval watcher is already armed for this packet

Current execution read:

- the line is machine-safe
- the train process is still alive
- the first retained `full_eval` summaries have now started landing:
  - `epoch_0001`
  - `epoch_0002`
  - `epoch_0003`
  - `epoch_0004`
- current early read:
  - `epoch_0001 = 0.7048 / 0.4521`
  - `epoch_0002 = 0.7057 / 0.4699`
  - `epoch_0003 = 0.7060 / 0.4791`
  - `epoch_0004 = 0.7074 / 0.4878`
  - `epoch_0005 = 0.7084 / 0.4944`
  - `epoch_0006 = 0.7094 / 0.4991`
  - `epoch_0007 = 0.7102 / 0.5024`
  - `epoch_0008 = 0.7102 / 0.5036`
  - `epoch_0009 = 0.7104 / 0.5055`
  - `epoch_0010 = 0.7106 / 0.5068`
- interpretation:
  - this is a clearly stronger early `CLIP-style` signal than the first edge-gated line
  - but it is also a clearly worsening `LPIPS` trajectory
  - so at this stage it is not a neutral early signal:
    - it is a `style-up / structure-worse` trajectory
  - and unlike the first edge-gated line, this pattern is now visible across multiple retained points, not just a single first snapshot
- the next handoff condition is:
  - wait for more retained eval points before triggering local non-CLIP review
