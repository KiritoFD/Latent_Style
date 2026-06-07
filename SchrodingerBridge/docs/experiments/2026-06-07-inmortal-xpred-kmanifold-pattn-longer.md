# `XPred + K_manifold + P_attn` Longer-Training Continuation

Date: 2026-06-07

Purpose:

- take the first promoted proximal frontier
  - `XPred + K_manifold + P_attn`
- and test whether the remaining LPIPS gap is still shrinking with more training budget

Why this continuation is justified:

- the `8-epoch` packet already beats `XPred + Kmanifold` on both primary transfer metrics
- LPIPS continues improving into the late epochs
- this is now the strongest evidence-backed family, so increasing its budget is more justified than opening another new side branch first

Protocol:

- resume from:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/aaai2027_inmortal_xpred_kmanifold_pattn_seed42_b16/epoch_0008.pt`
- extend the training horizon from `8` to `12` epochs
- keep the same batch and mechanism family
- evaluate with the same snapshot-matched fast `clip+lpips` contract

Success condition:

- transfer `clip_style` stays in the promoted band
- transfer `content_lpips` keeps improving beyond the `epoch_0006` frontier

Failure condition:

- later epochs recover only negligible LPIPS gains while style degrades
- or the curve plateaus so clearly that additional budget no longer buys frontier movement
