# Local Dataset / Latent / Feature Cache Manual Policy - 2026-06-05

Scope:

- `G:\GitHub\Latent_Style\Dataset`
- `G:\GitHub\Latent_Style\style_data`
- `G:\GitHub\Latent_Style\latent-256`
- `G:\GitHub\Latent_Style\clip-feats-vitb32`
- `G:\GitHub\Latent_Style\SchrodingerBridge\scale`
- `G:\GitHub\Latent_Style\SchrodingerBridge\datasets\horse2zebra`
- `G:\GitHub\Latent_Style\wikiart_fewshot`

This pass checks the large local data/cache roots one by one. The main purpose is to avoid deleting per-image latent tensors and feature caches as if they were model checkpoints.

## Summary

The large local data/cache roots are mostly valid dataset material:

- `Dataset`: 12.605GB, active Distinct5/WikiArt512/stress split images and VAE latents.
- `SchrodingerBridge/scale`: 18.958GB before cleanup, high-resolution WikiArt images/latents plus a VAE dependency.
- `clip-feats-vitb32`: 1.304GB, legacy per-image CLIP features.
- `latent-256`: 177.702MB, legacy per-image VAE latents.
- `SchrodingerBridge/datasets/horse2zebra`: 672.418MB, raw images plus latents.
- `style_data`: 385.598MB, legacy 5-style source images.
- `wikiart_fewshot`: 3.015MB, few-shot ukiyo-e data.

Cleanup action:

- Delete `SchrodingerBridge/scale/datasets/wikiart_81k`, which is only a failed HuggingFace cache shell with a 63.947MB `.incomplete` download and metadata. It has no actual dataset payload and is not git tracked.

## Manually Opened Evidence

### `Dataset`

Opened:

- top-level subdirectories: `distinct5_512`, `eval`, `legacy256_overfit50`, `wikiart_stress_splits_512`, `wikiart512_5style`;
- `Dataset/distinct5_512/test_manifest.json`;
- `Dataset/wikiart512_5style/test_manifest.json`;
- `Dataset/wikiart_stress_splits_512/logs/wikiart_stress1_encode_train.log`;
- representative latent tensor under `wikiart_stress_splits_512`.

The manifest files explicitly define train/test dirs, style lists, and per-style counts. A representative `latents_ema` file loads as a tensor of shape `(1, 4, 64, 64)`, which is a VAE latent, not a checkpoint.

Decision: keep.

### `latent-256`

Opened:

- style directories: `cezanne`, `Hayao`, `monet`, `photo`, `vangogh`;
- representative tensor: `latent-256/cezanne/00001.pt`.

The sample loads as tensor shape `(1, 4, 32, 32)`.

Decision: keep. This is legacy per-image latent data.

### `clip-feats-vitb32`

Opened:

- style directories: `cezanne`, `Hayao`, `monet`, `photo`, `vangogh`;
- representative tensor: `clip-feats-vitb32/cezanne/00001.feat.pt`.

The sample loads as tensor shape `(512,)`.

Decision: keep. This is per-image CLIP feature cache.

### `SchrodingerBridge/scale`

Opened:

- `scale/datasets` subdirectory sizes;
- `scale/vae_sdxl/diffusion_pytorch_model.bin`;
- representative latent tensor under `scale/datasets/wikiart_latent_128`;
- `SchrodingerBridge/docs/remote_server.md` references to `wikiart_1024_matched`, `wikiart_1024_27test`, and `wikiart_1024_27support`;
- `scale/datasets/wikiart_81k`.

The representative `wikiart_latent_128` sample loads as tensor shape `(4, 128, 128)`.

`scale/datasets/wikiart_81k` contains only:

- `.cache/huggingface/.gitignore`
- `CACHEDIR.TAG`
- small `.metadata` files
- `dataset.tar.gz.lock`
- one 63.947MB `.incomplete` file

It has no actual image or latent dataset payload. Decision: delete `wikiart_81k` as failed download residue.

Valid `scale` roots are retained. They are high-resolution dataset/cache material, not training checkpoints.

### `SchrodingerBridge/datasets/horse2zebra`

Opened:

- `raw`
- `train_images`
- `test_images`
- `latents_train`
- `latents_test`
- representative latent tensor `latents_test/horse/n02381460_1000.pt`

The sample loads as tensor shape `(4, 32, 32)`.

Decision: keep.

### `style_data` and `wikiart_fewshot`

Opened top-level data directories. These are source/few-shot image datasets and are retained.

## Cleanup Boundary

Deleted:

- failed HF dataset cache shell `SchrodingerBridge/scale/datasets/wikiart_81k`.

Not deleted:

- any real images;
- any VAE latent tensors;
- any CLIP feature tensors;
- any dataset manifests/logs;
- `scale/vae_sdxl/diffusion_pytorch_model.bin`, because it is a VAE dependency.

## Follow-Up

Disk recovery from this class requires a dataset-retention decision, not checkpoint cleanup. The next meaningful cleanup policies are remote epoch thinning and remote SaMAM checkpoint thinning.
