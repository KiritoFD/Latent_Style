# attn_sa_mod Local Deep Review

- Expected: `IntroStyle + DINO + frozen VLM`
- Current refreshed shortlist now in flight:
  - `best_transfer_clip_style = epoch_0001`
  - `best_transfer_lpips | best_structure_preserving = epoch_0008`
  - `best_allpairs_clip_style = epoch_0003`
  - `latest = epoch_0024`
- Current image-backed reruns available:
  - old image-backed summaries:
    - `epoch_0001`
    - `epoch_0002`
    - `epoch_0003`
  - refreshed image-backed reruns added on `2026-06-10`:
    - `epoch_0008`
    - `epoch_0024`
- Local execution note:
  - the first detached `IntroStyle` launch was started before the Windows child-pid lock fix
  - that pre-fix detached launch has since been cleared from the machine
  - treat any partial output from that pre-fix launch as diagnostic only
  - do not finalize `IntroStyle`, `DINO`, or `VLM` decisions from it
  - the refreshed deep-review pipeline was relaunched on `2026-06-10`
  - `IntroStyle` is currently running on the refreshed shortlist; `DINO` is queued behind it
- Stage-closure intent:
  - fast-eval convergence is already satisfied through `epoch_0024`
  - stage-closure gate is now considered satisfied:
    - `IntroStyle`
    - `DINO`
    - frozen external-baseline `VLM`
  - current decision tendency is tracked in:
    - [decision.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/round1_full_sweep/attn_sa_mod/decision.md)
  - frozen decision evidence is pinned in:
    - [vlm_stageclose_snapshot.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/round1_full_sweep/attn_sa_mod/vlm_stageclose_snapshot.md)
- Frozen `VLM` status:
  - snapshot205 compare chain launched on `2026-06-10`
  - manifests:
    - [round1_attn_sa_mod_vlm_manifest_e08_vs_seedream_samam_20260610.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_sa_mod_vlm_manifest_e08_vs_seedream_samam_20260610.csv)
    - [round1_attn_sa_mod_vlm_manifest_e24_vs_seedream_samam_20260610.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_sa_mod_vlm_manifest_e24_vs_seedream_samam_20260610.csv)
  - live outputs:
    - [round1_attn_sa_mod_vlm_snapshot205_e08_vs_seedream_samam_20260610.jsonl](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_sa_mod_vlm_snapshot205_e08_vs_seedream_samam_20260610.jsonl)
    - [round1_attn_sa_mod_vlm_snapshot205_e08_vs_seedream_samam_20260610.method_summary.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_sa_mod_vlm_snapshot205_e08_vs_seedream_samam_20260610.method_summary.csv)
    - [round1_attn_sa_mod_vlm_snapshot205_20260610.stdout.log](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_sa_mod_vlm_snapshot205_20260610.stdout.log)
  - target frozen board outputs:
    - `round1_attn_sa_mod_vlm_snapshot205_e08_vs_seedream_samam_20260610.method_summary.csv`
    - `round1_attn_sa_mod_vlm_snapshot205_e24_vs_seedream_samam_20260610.method_summary.csv`
    - `round1_attn_sa_mod_vlm_snapshot205_board_20260610.md`
  - current launch topology:
    - `e08` is still running
    - `e24` was launched in a second detached chain on `2026-06-10` so the two snapshots no longer wait on one serial queue
  - first partial read:
    - frozen stageclose `e08` summary:
      - `Seedream = 94 / 200`
      - `SaMAM = 104 / 200`
      - `AttnSA_e08 = 2 / 200`
    - early `e08` rows already prefer `Seedream` on style and often `SaMAM` on structure/artifact
    - `AttnSA_e08` is still reading as the weakest arm of the triple
    - this is directionally consistent with the negative `IntroStyle` margins
    - frozen stageclose `e24` summary:
      - `Seedream = 72 / 169`
      - `SaMAM = 97 / 169`
      - `AttnSA_e24 = 0 / 169`
  - closure implication:
    - the family is already decisively below `SaMAM / Seedream`
    - additional frozen `VLM` rows are no longer required before rejecting the family for promotion

<!-- ROUND1_AUTO_STATUS:START -->
## Auto Status

- Fast shortlist root: [round1_attn_sa_mod_fast_local](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_sa_mod_fast_local)
- Local review root: [round1_attn_sa_mod_localreview](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_sa_mod_localreview)
- Current canonical fast bestfew handoff:
  - `best_transfer_clip_style = epoch_0001`
  - `best_transfer_lpips | best_structure_preserving = epoch_0008`
  - `best_allpairs_clip_style = epoch_0003`
  - `latest = epoch_0024`
- Current localreview handoff:
  - `best_transfer_clip_style = epoch_0001`
  - `best_transfer_lpips | best_structure_preserving = epoch_0008`
  - `best_allpairs_clip_style = epoch_0003`
  - `latest = epoch_0024`
- Deep review artifacts:
  - `IntroStyle csv exists = True`
  - `DINO csv exists = True`
  - `Merged csv exists = True`
<!-- ROUND1_AUTO_STATUS:END -->

































