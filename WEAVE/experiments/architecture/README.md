# Submission Architecture Experiments

## `hf_oriented_nohh`

The baseline target already contains target-style LH/HL/HH bands, but the predictor mainly receives a style-memory identity. This experiment adds a coordinate-free condition route:

```text
target latent -> Haar DWT -> independent pooled LH/HL codes
              -> matching residual velocity heads
```

The route does not expose target spatial coordinates and does not modify LL. The HH velocity head remains disabled because previous HH-head experiments were negative: at the latent resolution, HH is dominated by unstable residual detail, while the VAE decoder can synthesize image-space fine texture from LL/LH/HL context.

The comparison changes architecture only. It keeps the canonical optimizer, objective, dataset, 15-epoch schedule, and inference protocol. Every epoch must receive the full 750-image D5 evaluation.

## Result

The route raises the style ceiling but does not dominate the baseline. Epoch 4 reaches DINO-S 0.4915 with a measurable content cost. Epoch 6 is the conservative candidate at DINO-S 0.4878, CLIP-S 0.7101, LPIPS 0.2563, and DINO-C 0.8215.

Gate probes show that the route remains active and grows after the style peak, so additional gate strength or training length is not justified. The next experiment remains fully trainable from scratch but detaches shared backbone features at the target-HF residual branch, preventing that branch from rewriting the content transport through its private gradient path. No post-processing is introduced. See `docs/reproduction/hf_oriented_nohh_result.md`.
