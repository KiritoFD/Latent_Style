# Endpoint AdaIN Axis

This inference-only axis uses the reproduced epoch 6 checkpoint and the complete canonical protocol inherited from `inference.json`.

Points: `1.5`, `2.0`, and `2.5`. Every point generates and evaluates the same 750-image board with CLIP-S, LPIPS, DINO-S, and DINO-C. DINO-S is the primary metric; DINO-C and LPIPS are content guardrails. No combined score is used.
