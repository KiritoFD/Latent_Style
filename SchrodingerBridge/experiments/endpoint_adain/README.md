# Endpoint AdaIN Axis

This inference-only axis uses the reproduced epoch 6 checkpoint and the complete canonical protocol inherited from `inference.json`.

Points: `1.5`, `2.0`, and `2.5`. Every point generates and evaluates the same 750-image board with CLIP-S, LPIPS, DINO-S, and DINO-C. DINO-S is the primary metric; DINO-C and LPIPS are content guardrails. No combined score is used.

## Result

| Scale | DINO-S | CLIP-S | LPIPS | DINO-C | Decision |
|---:|---:|---:|---:|---:|---|
| 1.5 | 0.4846 | **0.7175** | 0.2871 | 0.8039 | Reject: weaker DINO style and content |
| 2.0 | **0.4867** | 0.7075 | **0.2508** | **0.8281** | Keep |
| 2.5 | 0.3060 | 0.6941 | 0.5814 | 0.2586 | Reject: collapsed |

The response is not a smooth style-strength tradeoff. Scale 1.5 raises CLIP-S but lowers the primary DINO-S and both content metrics. Scale 2.5 collapses all four metrics. The reproduced scale 2.0 remains the only acceptable point, so this axis is closed without changing the canonical configuration.

Full-precision values are recorded in `docs/reproduction/endpoint_adain_axis.csv`.
