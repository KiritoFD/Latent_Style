# Color Mode Experiment Plan (Rank16, Epoch40)

## Goal
Compare three spatial-agnostic color loss modes under the same training setup:
- pseudo_rgb_adain
- pseudo_rgb_hist
- latent_decoupled_adain

## Fixed Settings
- model.ada_mix_rank: 16
- training.num_epochs: 40
- training.full_eval_interval: 40
- training.save_interval: 40
- loss.w_color: 6.0
- loss.w_latent_color: 0.0
- Keep all other settings identical to config_decoder-D-sweetspot.json

## Generated Configs
- config_color_mode_01_pseudo_rgb_adain_r16_e40.json
- config_color_mode_02_pseudo_rgb_hist_r16_e40.json
- config_color_mode_03_latent_decoupled_adain_r16_e40.json

## Generated Runner
- run_color_mode_r16_e40.bat

## Run Procedure
1. Open terminal and go to src directory.
2. Run all three experiments in sequence:
   - run_color_mode_r16_e40.bat
3. Optional: run each experiment manually:
   - uv run run.py --config config_color_mode_01_pseudo_rgb_adain_r16_e40.json
   - uv run run.py --config config_color_mode_02_pseudo_rgb_hist_r16_e40.json
   - uv run run.py --config config_color_mode_03_latent_decoupled_adain_r16_e40.json

## Output Directories
- ../color_mode_01_pseudo_rgb_adain_r16_e40
- ../color_mode_02_pseudo_rgb_hist_r16_e40
- ../color_mode_03_latent_decoupled_adain_r16_e40

## What To Compare
- Training stability:
  - loss curve smoothness
  - color metric trend in logs
- Full eval at epoch 40:
  - style classifier score
  - CLIP-based metrics
  - ArtFID (if enabled)
- Visual review:
  - global palette match
  - structural preservation
  - oversaturation or color shift artifacts

## Expected Behavior
- pseudo_rgb_adain: most stable baseline, good global tone alignment
- pseudo_rgb_hist: strongest palette matching, may be more aggressive
- latent_decoupled_adain: best structure/lightness retention while shifting chroma

## Decision Rule
Pick winner by weighted decision:
- 50% quantitative full-eval metrics
- 30% visual palette fidelity
- 20% structural consistency

## Notes
- If color effect is too weak, increase loss.w_color from 6.0 to 7.0 or 8.0.
- If structure starts degrading, first try latent_decoupled_adain or lower w_color.
