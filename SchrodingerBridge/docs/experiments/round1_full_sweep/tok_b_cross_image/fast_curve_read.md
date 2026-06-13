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
- current live continuation:
  - clean restart `aaai2027_round1_tok_b_cross_image_reconpretrain_seed42_b8a2_r2` produced one valid retained/eval point
  - first retained `r2` checkpoint now exists and has a valid fast-eval packet
  - current `r2` first point:
    - `transfer_clip_style = 0.6768`
    - `transfer_lpips = 0.7927`
    - `all_pairs_clip_style = 0.6782`
    - `all_pairs_lpips = 0.7922`
  - local authority packet path:
    - [r2_clip_lpips_curve.csv](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_tok_b_cross_image_remote_full_eval_pull/r2_clip_lpips_curve.csv)
    - [r2_epoch_0001.summary.json](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_tok_b_cross_image_remote_full_eval_pull/r2_epoch_0001.summary.json)
    - [r2_round1_convergence.json](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_tok_b_cross_image_remote_full_eval_pull/r2_round1_convergence.json)
  - the first `r2` point was recovered manually once the checkpoint landed
  - this DINO reconstruction-pretrain line has now been intentionally stopped because the main objective pivoted to pure-latent tokenizer plus true I2SB

<!-- ROUND1_AUTO_STATUS:START -->
## Auto Status

- Authority root:
  - [round1_tok_b_cross_image_remote_full_eval_pull](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_tok_b_cross_image_remote_full_eval_pull)
- Remote fast-eval status:
  - `r2_first_fast_eval_recovered`
- Run name:
  - `aaai2027_round1_tok_b_cross_image_reconpretrain_seed42_b8a2_r2`
- Remote run dir:
  - `/mnt/i/Github/Latent_Style/exp/inmortal-exp/aaai2027_round1_tok_b_cross_image_reconpretrain_seed42_b8a2_r2`
- Expected eval subdir:
  - `full_eval_fast_snapshot`
- Latest settled epoch:
  - `epoch_0001`
- Remote train pid count:
  - `0`
- Remote fast-eval pid count:
  - `0`
- Sync summary:
  - [sync_summary.json](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_tok_b_cross_image_remote_full_eval_pull/sync_summary.json)
<!-- ROUND1_AUTO_STATUS:END -->
