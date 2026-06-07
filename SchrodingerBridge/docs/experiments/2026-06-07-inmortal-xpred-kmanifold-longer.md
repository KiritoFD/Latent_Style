# `XPred + K_manifold` Longer-Training Continuation

Date: 2026-06-07

Purpose:

- revisit the strongest pre-proximal `x-pred` family under the corrected rule that `8 epochs` is not proof of convergence

Why this continuation exists:

- the original `8-epoch` `XPred + K_manifold` packet kept improving LPIPS through the late epochs
- it should therefore be treated as an incomplete convergence study rather than a closed family
- this continuation is the clean backfill for that earlier under-budget packet

Protocol:

- resume from:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/aaai2027_inmortal_xpred_kmanifold_seed42_b32/epoch_0008.pt`
- extend the horizon from `8` to `12` epochs
- keep the same mechanism family and batch
- evaluate with the same snapshot-matched fast `clip+lpips` contract

Success condition:

- style stays near the original promoted band
- LPIPS continues improving beyond the `epoch_0007 / epoch_0008` region

Failure condition:

- later epochs plateau or trade back LPIPS without compensating style gains
