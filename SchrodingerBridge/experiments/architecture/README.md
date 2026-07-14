# Submission Architecture Experiments

## `hf_oriented_nohh`

The baseline target already contains target-style LH/HL/HH bands, but the predictor mainly receives a style-memory identity. This experiment adds a coordinate-free condition route:

```text
target latent -> Haar DWT -> independent pooled LH/HL codes
              -> matching residual velocity heads
```

The route does not expose target spatial coordinates and does not modify LL. The HH velocity head remains disabled because previous HH-head experiments were negative: at the latent resolution, HH is dominated by unstable residual detail, while the VAE decoder can synthesize image-space fine texture from LL/LH/HL context.

The comparison changes architecture only. It keeps the canonical optimizer, objective, dataset, 15-epoch schedule, and inference protocol. Every epoch must receive the full 750-image D5 evaluation.
