# Remote IntroStyle Page-1 Probe

Date: 2026-06-08

Purpose:

- provide one stable entrypoint for the remote `Distinct5` page-1 `IntroStyle` shortlist
- avoid re-deriving the `ModelScope` cache path and probe command again

Launcher:

- [launch_remote_introstyle_page1_probe.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_introstyle_page1_probe.py)

Remote backbone path:

- `/mnt/i/Github/Latent_Style/eval_cache/modelscope/stabilityai/stable-diffusion-2-1-base`

Remote resolution policy:

- prefer `ModelScope repo id = stabilityai/stable-diffusion-2-1-base`
- cache root:
  - `/mnt/i/Github/Latent_Style/eval_cache/modelscope`
- local snapshot path remains the expected resolved target after download

Default remote style bank:

- `/mnt/i/wikiart_distinct5_samam_512_classview/test`

Default output root:

- `/mnt/i/Github/Latent_Style/SchrodingerBridge/aaai2027/introstyle_page1`

Default protocol:

- `sample_rows = 20`
- `bank_limit_per_style = 4`
- `batch_size = 4`
- `ensemble_size = 1`
- `t = 25`
- `up_ft_index = 1`

Operational note:

- the launcher is safe to queue while another formal GPU lane is still active
- actual start still follows the remote single-lane gate in `launch_remote_wsl_command.py`
- preferred use is:
  - after the active formal training/eval lane fully releases the GPU
  - or in a window where the user explicitly approves spending that lane on `IntroStyle`
