# `XPred_Barycenter` Remote Packet

Date: 2026-06-07

Scope:

- dataset: `Distinct5-512`
- surface: `H-family` remote `3060 WSL`
- config:
  - [inmortal_xpred_bary_seed42_b16.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_xpred_bary_seed42_b16.json)

Intent:

- test the user-proposed `x-prediction / endpoint prediction` direction
- stop asking the model to regress raw residuals as the primary target
- replace sampled OT target pressure with:
  - structure-aware OT cost
  - barycentric target projection
  - weak low-frequency EMA teacher

Expected upside:

- higher style ceiling than plain residual/velocity prediction
- lower target variance than single sampled OT endpoints
- less drift toward trivial mean residuals

Expected failure mode:

- endpoint prediction may become too coarse and oversmooth style structure
- barycentric target may wash out high-frequency modes if the target teacher or top-k projection dominates too hard

Reflection template:

- does endpoint prediction raise transfer `CLIP-style` faster than the velocity baseline?
- is `base_transfer_clip_style` already useful, or is all quality deferred to later terminal correction?
- does barycentric target smoothing reduce instability without flattening style?
- does the EMA teacher help or over-average the target manifold?
