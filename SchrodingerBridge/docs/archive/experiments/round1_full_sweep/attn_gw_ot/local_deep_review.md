# attn_gw_ot Local Deep Review

- Expected: `IntroStyle + DINO + frozen VLM`
- Relaunch prep:
  - [relaunch_prep.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/round1_full_sweep/attn_gw_ot/relaunch_prep.md)
- Current deferred stage-close chain:
  - [round1_attn_gw_ot_stageclose_deferred_20260610.stdout.log](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_gw_ot_stageclose_deferred_20260610.stdout.log)
  - waits for:
    - local fast convergence
  - then runs:
    - bestfew image-backed rerun
    - local `IntroStyle + DINO`
    - external-baseline `VLM`

<!-- ROUND1_AUTO_STATUS:START -->
## Auto Status

- Fast shortlist root: [round1_attn_gw_ot_fast_local](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_gw_ot_fast_local)
- Local review root: [round1_attn_gw_ot_localreview](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_gw_ot_localreview)
- No fast bestfew handoff CSV found yet.
- No localreview bestfew handoff CSV found yet.
<!-- ROUND1_AUTO_STATUS:END -->
