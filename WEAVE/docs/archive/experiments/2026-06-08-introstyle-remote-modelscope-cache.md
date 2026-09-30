# IntroStyle Remote ModelScope Cache

Date: 2026-06-08

Decision:

- the remote `3060 WSL` surface should use `ModelScope`, not `huggingface.co`, to bootstrap the `IntroStyle` backbone

What was verified:

- remote env:
  - `/home/xy/venvs/samam312/bin/python`
- `modelscope` package is available inside that env
- remote `snapshot_download` succeeded for:
  - `stabilityai/stable-diffusion-2-1-base`

Resolved cache root:

- `/mnt/i/Github/Latent_Style/eval_cache/modelscope/stabilityai/stable-diffusion-2-1-base`

Implication:

- the earlier blocker:
  - `WSL cannot reach huggingface.co`
- no longer blocks `IntroStyle`
- future remote `IntroStyle` runs should point directly at the local `ModelScope` snapshot path instead of relying on HF download fallback

Operational note:

- this removes the model-access blocker only
- remote `IntroStyle` still needs:
  - an explicit scheduled slot on the single formal GPU lane
  - and a concrete shortlist / held-out-bank manifest
