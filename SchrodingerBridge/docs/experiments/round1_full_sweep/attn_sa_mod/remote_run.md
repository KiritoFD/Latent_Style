# attn_sa_mod Remote Run Log

- Run dir: `./exp/inmortal-exp/aaai2027_round1_attn_sa_mod_seed42_b8a2`
- Launch status:
  - remote training launched on `2026-06-10`
  - launcher task:
    - `round1-aaai2027_round1_attn_sa_mod_seed42_b8a2-train`
  - fast-eval watcher task:
    - `round1-aaai2027_round1_attn_sa_mod_seed42_b8a2-fast-eval`
- Remote logs:
  - train:
    - `/mnt/i/Github/Latent_Style/exp/inmortal-exp/aaai2027_round1_attn_sa_mod_seed42_b8a2_train.log`
  - fast eval:
    - `/mnt/i/Github/Latent_Style/exp/inmortal-exp/aaai2027_round1_attn_sa_mod_seed42_b8a2_fast_eval.log`
- First health:
  - first bootstrap launch surfaced an incorrect inherited `freeze_mode=injection_only`
  - backbone families were then moved to `freeze_mode=attention_only`
  - the next attempt still sat below the requested VRAM band
  - backbone family batch size was then raised to `14`
  - the current authoritative formal relaunch remote GPU sample is `9698 MiB / 12288 MiB`
  - first watcher attempt failed before the first retained checkpoint existed
  - watcher was updated to tolerate empty-checkpoint cycles and relaunched
  - the generic remote launcher now also supports minimum-runtime-memory health checks for later families
- Under-band formal attempt:
  - the intermediate `batch_size=12` formal attempt was cleaned before the current authoritative relaunch
- Archived bootstrap root:
  - `/mnt/i/Github/Latent_Style/exp/inmortal-exp/aaai2027_round1_attn_sa_mod_seed42_b8a2_bootstrap4e_20260610`
- Current formal checkpoint pull observed locally:
  - through `epoch_0013.pt`
- Fast eval execution surface:
  - remote fast watcher has been retired for the current lane to avoid sharing the `3060` with training
  - the authoritative fast all-ckpt watcher now runs locally under:
    - [round1_attn_sa_mod_fast_local](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_sa_mod_fast_local)
  - after `epoch_0012.pt` was pulled, the Windows local GPU lock path was hardened so stale parent locks no longer crash the watcher
  - the machine has since been cleared and the local fast watcher has been relaunched cleanly
- Remote training completion:
  - remote checkpoint list reaches `epoch_0024.pt`
  - train log contains:
    - `Epoch 24/24`
    - `Saved checkpoint: ... epoch_0024.pt`
    - `Training completed.`
  - formal remote training for `attn_sa_mod` is complete

<!-- ROUND1_AUTO_STATUS:START -->
## Auto Status

- Family id: `attn_sa_mod`
- Run name: `aaai2027_round1_attn_sa_mod_seed42_b8a2`
- Remote run dir: `./exp/inmortal-exp/aaai2027_round1_attn_sa_mod_seed42_b8a2`
- Config: [aaai2027_round1_attn_sa_mod_seed42_b8a2.json](G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/round1_full_sweep/aaai2027_round1_attn_sa_mod_seed42_b8a2.json)
- Manifest status: `rejected`
- Local fast root: [round1_attn_sa_mod_fast_local](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_sa_mod_fast_local)
- Local review root: [round1_attn_sa_mod_localreview](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_sa_mod_localreview)
<!-- ROUND1_AUTO_STATUS:END -->



























