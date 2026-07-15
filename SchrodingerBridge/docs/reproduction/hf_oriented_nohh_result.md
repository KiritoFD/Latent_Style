# Oriented Target-HF Route Result

## Setup

This architecture adds independent, coordinate-free target-style codes for the LH and HL velocity residuals. LL remains disconnected from the target image and the HH velocity head remains disabled. The model was trained from scratch for 15 epochs with the canonical optimizer, objective, data, and inference protocol.

- Manifest: `experiments/architecture/hf_oriented_nohh.json`.
- Manifest SHA-256: `08912e97b3662e2b6ade9b9bb289e37afbab256e492ec50bc6f6589397fd0bb9`.
- Parameters: 1,037,087 versus 873,680 for the baseline.
- Peak allocated/reserved VRAM: 5.70/6.46 GB versus 5.58/6.33 GB.
- Training: fresh initialization, 15 checkpoints, about five minutes on the remote RTX 3060.
- Evaluation: every epoch on the complete 750-image D5 board.

Full-precision metrics are in `hf_oriented_nohh_epoch_metrics.csv`.

## Result

| Point | DINO-S | CLIP-S | LPIPS | DINO-C | Reading |
|---|---:|---:|---:|---:|---|
| Fresh baseline epoch 6 | 0.4867 | 0.7075 | **0.2508** | **0.8281** | Canonical balanced point |
| Oriented route epoch 1 | 0.4856 | 0.7117 | **0.2411** | 0.8230 | Better LPIPS/CLIP, lower DINO-S |
| Oriented route epoch 4 | **0.4915** | 0.7126 | 0.2596 | 0.8103 | Strongest DINO-S, visible content cost |
| Oriented route epoch 6 | 0.4878 | **0.7101** | 0.2563 | 0.8215 | Conservative style-improved candidate |

Epoch 6 improves both style metrics over the fresh baseline:

- DINO-S: `+0.0011`.
- CLIP-S: `+0.0027`.
- LPIPS: `+0.0056` (worse).
- DINO-C: `-0.0066`.

This is a new Pareto candidate, not a strict all-metric improvement. Content does not collapse, but the fresh baseline remains the canonical configuration because the user requirement is to raise style without materially damaging content.

Checkpoint hashes:

- Epoch 4: `b0a53b883f1a62fdb923c1ed3b5aa00026aeea8b4d7b88230e591d1430aa8298`.
- Epoch 6: `f0b2c60c6c722cd000030d6ccd2ed67dd078daec8627d6bc3747a58b0e15ac4e`.

## Information-Flow Diagnosis

The route is effective: DINO-S reaches 0.4915 without raw target spatial maps, proving that coordinate-free target-HF information can raise the style ceiling.

The later decline is not caused by the route gate closing. LH/HL gate values increase from about `0.173/0.173` at epoch 1 to `0.188/0.192` at epoch 15, while DINO-S peaks at epoch 4 and then declines. The constant HH gate (`0.1781` at every epoch) confirms that the disabled HH branch receives no gradient and has no effect.

The likely failure mode is joint co-adaptation: target-HF residual gradients also update the shared backbone and velocity field. Early epochs buy style, then continued end-to-end training reorganizes the transport toward content recovery and erases the style gain even as the residual gate grows. Therefore a larger gate or a longer run is not the next lever.

## Next Experiment

Keep the method trainable from scratch and isolate the new route inside the architecture:

1. Initialize the complete baseline and target-HF route together from scratch.
2. Train the baseline backbone and velocity heads normally through the original transport path.
3. Feed detached backbone features to the target-HF residual heads, so residual-branch gradients train the new route but cannot rewrite the shared content transport through that branch.
4. Keep LL and HH behavior unchanged and use no image or latent post-processing.
5. Evaluate every epoch under the same four metrics.

This tests whether the missing target-HF condition can improve style while reducing harmful route-to-backbone co-adaptation. It remains a single end-to-end method trained from scratch; the detach operation only defines gradient ownership between architectural branches.
