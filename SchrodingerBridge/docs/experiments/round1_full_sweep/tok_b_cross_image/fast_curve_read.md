# tok_b_cross_image Fast Curve Read

- Curve CSV: `clip_lpips_curve.csv`
- reconstruction-pretrain blockage on `2026-06-12`:
  - retained checkpoints exist on remote:
    - `epoch_0001.pt`
    - `epoch_0002.pt`
  - recovery after freeing remote disk:
    - removed `34G` `eval_cache/modelscope/stabilityai/stable-diffusion-2-1-base`
    - restored the remote workspace manifest from the intact copy on remote `C:`
  - recovered remote fast-eval state:
    - `epoch_0001` now has a valid fast-eval packet
    - local authority packet path:
      - [clip_lpips_curve.csv](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_tok_b_cross_image_remote_full_eval_pull/clip_lpips_curve.csv)
      - [epoch_0001.summary.json](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_tok_b_cross_image_remote_full_eval_pull/epoch_0001.summary.json)
    - current recovered point:
      - `transfer_clip_style = 0.6771`
      - `transfer_lpips = 0.7927`
      - `all_pairs_clip_style = 0.6785`
      - `all_pairs_lpips = 0.7922`
  - remaining loss:
    - `epoch_0002.pt` is corrupted (`64` bytes) and cannot be evaluated
  - direct cause:
    - remote workspace drive `I:` had `0` free bytes
    - the remote workspace manifest file was observed as zero-length at the same time
  - implication:
    - the reconstruction-pretrain line now has a one-point authoritative fast curve
    - but it still cannot be judged as converged or promotable because the newest retained checkpoint is corrupted and the run died before a clean continuation point

<!-- ROUND1_AUTO_STATUS:START -->
## Auto Status

- Authority root:
  - [round1_tok_b_cross_image_remote_full_eval_pull](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_tok_b_cross_image_remote_full_eval_pull)
- Remote fast-eval status:
  - `manual_recovery_partial_fast_eval`
- Run name:
  - `aaai2027_round1_tok_b_cross_image_reconpretrain_seed42_b8a2`
- Remote run dir:
  - `/mnt/i/Github/Latent_Style/exp/inmortal-exp/aaai2027_round1_tok_b_cross_image_reconpretrain_seed42_b8a2`
- Expected eval subdir:
  - `full_eval_fast_snapshot`
- Latest settled epoch:
  - `epoch_0001`
- Pending retained checkpoint:
  - `epoch_0002` (`corrupted`)
- Remote train pid count:
  - `0`
- Remote fast-eval pid count:
  - `0`
- Sync summary:
  - [sync_summary.json](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_tok_b_cross_image_remote_full_eval_pull/sync_summary.json)
<!-- ROUND1_AUTO_STATUS:END -->
