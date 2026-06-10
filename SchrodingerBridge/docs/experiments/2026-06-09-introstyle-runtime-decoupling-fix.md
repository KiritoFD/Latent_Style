# IntroStyle Runtime Decoupling Fix

Date: 2026-06-09

Goal:

- make `IntroStyle` runnable on the current remote `samam312` env
- avoid forcing a remote env upgrade just to evaluate style

Observed failure:

- the original `IntroStyle` path imported `StableDiffusionPipeline`
- on the remote env this triggered:
  - `peft -> transformers.EncoderDecoderCache`
- remote package versions at the time:
  - `transformers = 4.41.2`
  - `diffusers = 0.29.2`

What changed:

- rewrote [introstyle_eval.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/utils/introstyle_eval.py) to avoid `StableDiffusionPipeline`
- the extractor now directly loads only:
  - `AutoencoderKL`
  - `DDIMScheduler`
  - custom `_IntroStyleUNet`
  - `CLIPTokenizer`
  - `CLIPTextModel`
- prompt encoding is handled manually
- one-step latent noising / UNet feature extraction is handled manually

Why this matters:

- this removes the `peft` / `EncoderDecoderCache` dependency path
- it keeps the remote `ModelScope` snapshot usable without changing the formal training env
- it is the correct direction for paper-facing evaluation stability:
  - smaller dependency surface
  - less hidden pipeline behavior
  - easier to audit

Validation completed:

- remote CPU single-image sanity check succeeded after syncing the new file:
  - source image:
    - `Impressionism__alfred-sisley_riverbank-at-veneux-1881.jpg`
  - output feature shape:
    - `(1, 1280, 32, 32)`

Remaining step:

- rerun the formal remote `IntroStyle` sidecar on actual paper-facing packets once the current active GPU lane releases
- first target should remain:
  - `aaai2027_inmortal_hold4mid_e8_spatial_carriergate_bodydecoder_seed42_b8a2`
  - `epoch_0008`
  - `epoch_0012`

Queued follow-up:

- remote watcher task launched:
  - `bodydecoder-introstyle-after-trust`
- it waits for:
  - `aaai2027_inmortal_xpred_kmanifold_pattn_anisostokes_queue_trust_from_e13_seed42_b8a2`
  - to leave the GPU
- then it reruns:
  - `bodydecoder e8`
  - `bodydecoder e12`
  - under `full_eval_introstyle_clean_v3`
  - with the new `IntroStyle` runtime path

Watcher bug found and corrected:

- the first watcher version misfired while the trust lane was still alive
- root cause:
  - the old `train-pattern` matcher assumed the run name would appear as a single underscore-joined token
  - the real training process exposed:
    - `src/run.py --config configs/aaai2027/inmortal_...json`
  - so `train_alive` incorrectly became false
- correction:
  - process scanning now reads `/proc/*/cmdline`
  - matcher accepts several real command-line variants, including:
    - raw pattern
    - `*.json`
    - `configs/aaai2027/<tail>.json`
- after the fix, the relaunched watcher reports:
  - `train_alive=True`
  - while `trust_from_e13` is still active
