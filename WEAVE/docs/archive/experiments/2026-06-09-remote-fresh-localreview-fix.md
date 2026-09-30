# Remote Fresh-LocalReview Fix

Date: 2026-06-09

Problem:

- the original post-train fresh-eval watcher for the active `Knee` line was only set up to rerun:
  - `CLIP + LPIPS`
  - without generated image saving
- that is not enough for the current paper-facing review stack, because the follow-up review needs:
  - local or remote `IntroStyle`
  - `DINO`
  - VLM panel inspection
  - all of which require generated images

Fix landed:

- [run_inmortal_posttrain_eval_latest_epochs_when_done.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_inmortal_posttrain_eval_latest_epochs_when_done.py)
  now forwards:
  - `save_generated_images`
  - `save_summary_grid`
  - optional `IntroStyle` arguments
- a dedicated launcher for the active packet was added:
  - [launch_remote_knee_spatial_carriergate_bodydecoder_fresh_eval_watcher.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_knee_spatial_carriergate_bodydecoder_fresh_eval_watcher.py)
- a more reliable remote pull path was also added:
  - [pull_remote_eval_dir.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/pull_remote_eval_dir.py)
  - this uses:
    - remote temp tar on `C:\Users\administrator`
    - then `scp`
  - instead of the fragile direct stdout tar stream
- the remote `IntroStyle` launcher was corrected to prefer the already-cached local model path:
  - `/mnt/i/Github/Latent_Style/eval_cache/modelscope/stabilityai/stable-diffusion-2-1-base`
  - rather than re-triggering network download on every new packet

Operational consequence:

- the old watcher was stopped
- the corrected watcher was relaunched
- the active packet now successfully produced:
  - `full_eval_fresh_localreview/epoch_0001..0012`
  - with generated images

Next use:

- when a new remote packet needs post-train local review, do not reuse the old no-image watcher contract
- use the corrected watcher path or the same arguments directly
