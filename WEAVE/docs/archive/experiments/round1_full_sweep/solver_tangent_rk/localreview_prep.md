# Round1 Localreview Prep

Purpose:

- record the current image-backed deep-review prep state for this family
- avoid re-deriving which checkpoints must be rerun with images before IntroStyle / DINO / VLM

## Current Fast Bestfew

- Handoff CSV: `G:\GitHub\Latent_Style\SchrodingerBridge\aaai2027\round1_solver_tangent_rk_fast_local\full_eval_fast_snapshot_bestfew_handoff.csv`
- Canonical epochs: `epoch_0001, epoch_0019, epoch_0007, epoch_0032`
- Reasons:
  - `best_transfer_clip_style -> epoch_0001`
  - `best_transfer_lpips | best_structure_preserving -> epoch_0019`
  - `best_allpairs_clip_style -> epoch_0007`
  - `latest -> epoch_0032`

## Next Commands

- Build image-backed rerun packet:
  - `python SchrodingerBridge\tools\experiments\run_round1_family_bestfew_pipeline.py --config SchrodingerBridge\configs\aaai2027\round1_full_sweep\aaai2027_round1_solver_tangent_rk_seed42_b8a2.json --fast-local-root G:\GitHub\Latent_Style\SchrodingerBridge\aaai2027\round1_solver_tangent_rk_fast_local --fast-eval-subdir full_eval_fast_snapshot --review-local-root G:\GitHub\Latent_Style\SchrodingerBridge\aaai2027\round1_solver_tangent_rk_localreview --review-eval-subdir full_eval_fresh_localreview --use-remote-rerun --skip-introstyle --skip-dino`
- Run local IntroStyle / DINO after image-backed rerun exists:
  - `python SchrodingerBridge\tools\experiments\run_local_round1_family_review.py --config SchrodingerBridge\configs\aaai2027\round1_full_sweep\aaai2027_round1_solver_tangent_rk_seed42_b8a2.json --eval-subdir full_eval_fresh_localreview --local-root G:\GitHub\Latent_Style\SchrodingerBridge\aaai2027\round1_solver_tangent_rk_localreview`

## Notes

- Review output root: `G:\GitHub\Latent_Style\SchrodingerBridge\aaai2027\round1_solver_tangent_rk_localreview`
- Review eval subdir: `full_eval_fresh_localreview`
- This note is intentionally command-oriented so the next deep-review handoff is fast and repeatable.
