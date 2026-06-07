# `XPred + K_manifold + P_attn + S_stokes` Longer-Training Continuation

Date: 2026-06-07

Purpose:

- revisit the `P_attn + Stokes` tradeoff family under the corrected rule that `8 epochs` is not proof of convergence

Why this continuation exists:

- the `8-epoch` Stokes packet keeps improving LPIPS deep into the later epochs
- it underperforms the current promoted `P_attn` frontier on style, but it is the strongest structure-side LPIPS tradeoff family so far
- that makes it a valid continuation candidate rather than a closed dead packet

Protocol:

- resume from:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/aaai2027_inmortal_xpred_kmanifold_pattn_stokes_seed42_b16/epoch_0008.pt`
- extend the horizon from `8` to `12` epochs
- keep the same family and batch
- evaluate with the same snapshot-matched fast `clip+lpips` contract

Success condition:

- style holds near the late `Stokes` band
- LPIPS continues improving enough to justify the style tradeoff

Failure condition:

- the later epochs flatten without material LPIPS gain
- or style collapses faster than LPIPS improves
