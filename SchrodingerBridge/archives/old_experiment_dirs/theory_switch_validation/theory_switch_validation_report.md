# Theory Switch Validation

## Design

- Goal: verify whether the new optional switches improve the style/content Pareto point, not to replace the main model blindly.
- Base: original `S-add__K-1_C-0_W-20_Col-0/config.json`, with `K=2`, `terminal_swd_weight=20`, `w_cycle=0`, 3 epochs.
- Evaluation: every epoch on the strict 750-image protocol.
- Primary score: `EC = CLIP-style * (1 - LPIPS)`.

## Best Epoch Per Variant

| rank | variant | epoch | style | content | LPIPS | EC | photo_style | photo_LPIPS | note |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | T3_entropy_gate_5p0 | epoch_0001 | 0.6916 | 0.8804 | 0.3684 | 0.4368 | 0.6327 | 0.3921 | Kinetic penalty gated by semantic attention entropy, strong strength. |
| 2 | T1_sinkhorn_routing | epoch_0001 | 0.6889 | 0.8817 | 0.3667 | 0.4363 | 0.6237 | 0.3799 | Semantic attention uses Sinkhorn-style doubly normalized routing. |
| 3 | T2_entropy_gate_2p5 | epoch_0002 | 0.6939 | 0.8791 | 0.3714 | 0.4362 | 0.6330 | 0.3936 | Kinetic penalty gated by semantic attention entropy, moderate strength. |
| 4 | T4_sinkhorn_entropy | epoch_0001 | 0.6916 | 0.8853 | 0.3694 | 0.4361 | 0.6300 | 0.3993 | Sinkhorn routing plus moderate entropy-gated kinetic penalty. |
| 5 | T0_k2_baseline | epoch_0002 | 0.6971 | 0.8721 | 0.3813 | 0.4313 | 0.6394 | 0.4066 | K2 baseline from the same base config; no new switch enabled. |
| 6 | T7_all_switches_mild | epoch_0001 | 0.6911 | 0.8755 | 0.3766 | 0.4308 | 0.6285 | 0.4040 | Combined mild package: Sinkhorn routing, entropy gate, and Gumbel color transport. |
| 7 | T6_color_gumbel_w2 | epoch_0002 | 0.7006 | 0.8469 | 0.4067 | 0.4156 | 0.6452 | 0.4414 | Mild contextual color loss with hard Gumbel transport. |
| 8 | T5_color_soft_w2 | epoch_0001 | 0.7017 | 0.8491 | 0.4131 | 0.4118 | 0.6517 | 0.4720 | Mild contextual color loss with regular softmax transport. |

## Delta Against T0 Baseline

| variant | Delta style | Delta content | Delta LPIPS | Delta EC | reading |
|---|---:|---:|---:|---:|---|
| T3_entropy_gate_5p0 | -0.0056 | +0.0083 | -0.0129 | +0.0055 | promising |
| T1_sinkhorn_routing | -0.0082 | +0.0096 | -0.0146 | +0.0050 | promising |
| T2_entropy_gate_2p5 | -0.0033 | +0.0070 | -0.0099 | +0.0049 | promising |
| T4_sinkhorn_entropy | -0.0055 | +0.0132 | -0.0119 | +0.0048 | promising |
| T7_all_switches_mild | -0.0061 | +0.0034 | -0.0047 | -0.0005 | mixed |
| T6_color_gumbel_w2 | +0.0035 | -0.0252 | +0.0254 | -0.0157 | negative |
| T5_color_soft_w2 | +0.0046 | -0.0230 | +0.0318 | -0.0195 | negative |

## SaMST Reference

- SaMST strict: style `0.7194`, content `0.8193`, LPIPS `0.4664`, EC `0.3839`.
- Best validation row vs SaMST: Delta style `-0.0278`, Delta LPIPS `-0.0980`, Delta EC `+0.0529`.
