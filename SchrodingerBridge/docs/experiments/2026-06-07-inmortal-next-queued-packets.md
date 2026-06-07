# inmortal Next Queued Packets

Date: 2026-06-07

Purpose:

- prepare the next unrun `inmortal` packets while the remote `3060` is blocked by host-side GUI VRAM load
- keep the next experiment surface reproducible and ready to launch as soon as the GPU returns to a safe idle band
- standardize all new packet outputs under:
  - `./exp/inmortal-exp/<run_name>`

Queued configs:

- [inmortal_k_spectral_seed42_b16.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_k_spectral_seed42_b16.json)
  - fills the missing single-family `K_spectral` control
- [inmortal_xpred_structot_seed42_b16.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_xpred_structot_seed42_b16.json)
  - isolates structure-aware OT without barycentric smoothing or teacher support
- [inmortal_xpred_teacher_endpoint_seed42_b16.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_xpred_teacher_endpoint_seed42_b16.json)
  - isolates the endpoint EMA teacher family
- [inmortal_xpred_queue_seed42_b16.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_xpred_queue_seed42_b16.json)
  - isolates the fixed queue-smoothing bundle
- [inmortal_xpred_kmanifold_pattn_queue_seed42_b16.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_xpred_kmanifold_pattn_queue_seed42_b16.json)
  - strongest-family queue escalation
- [inmortal_xpred_kmanifold_pattn_anisostokes_queue_from_pattn_seed42_b8a2.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_xpred_kmanifold_pattn_anisostokes_queue_from_pattn_seed42_b8a2.json)
  - strongest-family combo finetune aligned with the corrected `C6` spirit

Launch priority once the current backfill is complete and the remote host frees VRAM:

1. `K_spectral`
2. `XPred_StructOT`
3. `XPred_EndpointTeacher`
4. `XPred_QueueSmoothing`
5. `XPred_Kmanifold_Pattn_Queue`
6. `XPred_Kmanifold_Pattn_AnisoStokesQueue_from_pattn`

Interpretation rule:

- the first four packets close single-family explanatory gaps
- the last two packets are the most plausible direct ceiling-push candidates on top of the current strongest family

Current execution status:

- `K_spectral` is no longer just queued.
  - the over-cap `b16` launch was invalidated on machine-contract grounds
  - the safety-corrected rerun now in flight is:
    - [inmortal_k_spectral_seed42_b12.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_k_spectral_seed42_b12.json)
- remaining not-yet-started queued packets:
  1. `XPred_StructOT`
  2. `XPred_EndpointTeacher`
  3. `XPred_QueueSmoothing`
  4. `XPred_Kmanifold_Pattn_Queue`
  5. `XPred_Kmanifold_Pattn_AnisoStokesQueue_from_pattn`
