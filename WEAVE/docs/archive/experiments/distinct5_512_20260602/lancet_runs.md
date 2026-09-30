# LANCET Distinct5-512 Runs

更新时间：2026-06-02

## 评估口径

所有正式结果使用 Distinct5-512 all 5x5 / 750 images：

```text
5 source styles x 30 test images per source x 5 target styles = 750
```

主指标：

- `clip_style`：越高越好。
- `content_lpips`：越低越好。

训练使用 EMA VAE latent、packed latent cache、远程 3060，正式 b44 显存约 9.6-9.7GB。

## 远程 run 目录

远程根目录：

```text
/mnt/i/Github/Latent_Style/SchrodingerBridge/exp
```

主要 run：

| 变体 | 远程目录 |
|---|---|
| baseline | `distinct5_512_ema_baseline_direct_atom_residual_b44_remote` |
| A | `distinct5_512_ema_variant_a_class_prototypes_b44_remote` |
| B | `distinct5_512_ema_variant_b_global_vq_b44_remote` |
| C | `distinct5_512_ema_variant_c_content_guided_spatial_b44_remote` |
| D | `distinct5_512_ema_variant_d_vq_content_guided_b44_remote` |
| E | `distinct5_512_ema_variant_e_latent_prototype_ot_queue_b44_remote` |
| F | `distinct5_512_ema_variant_f_annealed_prototype_ot_queue_e3_b44_remote` |
| G | `distinct5_512_ema_variant_g_stratified_prototype_ot_queue_e3_b44_remote` |
| H | `distinct5_512_ema_variant_h_hard_explore_queue_e3_b44_remote` |
| I | `distinct5_512_ema_variant_i_dual_target_mix_queue_e3_b44_remote` |
| J | `distinct5_512_ema_variant_j_aux_hard_swd_queue_e3_b44_remote` |
| K | `distinct5_512_ema_variant_k_content_adaptive_vq_queue_e3_b44_remote` |
| L | `distinct5_512_ema_variant_l_content_adaptive_annealed_queue_e3_b44_remote` |
| M | `distinct5_512_ema_variant_m_style_gated_content_router_e3_b44_remote` |

## 结果总表

| 模型 | 最优 epoch | clip_style | content_lpips | 决策 |
|---|---:|---:|---:|---|
| Baseline direct atom residual | 8 / 1 | 0.687649 | 0.446756 | 弱 baseline |
| Variant A class prototypes | 8 / 1 | 0.684946 | 0.446381 | 拒绝 |
| Variant B global VQ | 8 | 0.687321 | 0.444600 | 弱保留，仅 LPIPS 小幅好 |
| Variant C content-guided spatial | 2 | 0.690659 | 0.422593 | 保留 |
| Variant D VQ + content-guided | 1 | 0.689761 | 0.415599 | 保留 |
| Variant E latent prototype OT queue | 1 / 3 | 0.697347 | 0.333086 | 强保留 |
| Variant F annealed prototype OT queue | 1 | 0.696915 | 0.318645 | 当前 LPIPS 最优 |
| Variant G stratified prototype OT queue | 2 / 3 | 0.697271 | 0.332391 | 拒绝 |
| Variant H hard-explore prototype OT queue | 2 / 1 | 0.699383 | 0.321333 | 当前均衡点 |
| Variant I dual-target latent mix queue | 2 / 1 | 0.696633 | 0.347966 | 拒绝 |
| Variant J auxiliary hard-target SWD queue | 1 | 0.697653 | 0.332274 | 拒绝 |
| Variant K content-adaptive VQ atom routing | 1 | 0.700995 | 0.362294 | 当前 style 最优，style-only 保留 |
| Variant L content-adaptive annealed queue | 1 | 0.697777 | 0.339710 | 拒绝 |
| Variant M style-gated content router | 1 / 2 | 0.698726 | 0.345800 | 拒绝，部分结果 |

## 当前保留基线

| 用途 | 选择 | 理由 |
|---|---|---|
| 内容保持压力基线 | F epoch 1 | `content_lpips=0.318645` 最低 |
| 综合基线 | H epoch 1/2 | `clip_style=0.699383` 且 `content_lpips=0.321333` 接近 F |
| style-only 上限 | K epoch 1 | `clip_style=0.700995`，但 LPIPS 代价明显 |

## 表征结论

1. 单纯加 tokenizer 容量没有用：A/B 都没有打过 baseline 的 Pareto。
2. content-guided spatial routing 有效：C/D 明确降低 LPIPS。
3. prototype-aware latent target queue 是最大提升来源：E/F/H 把 LPIPS 从 0.41-0.44 区间压到 0.32-0.33 区间。
4. hard target pressure 需要稀疏、随机、离散：H 优于 deterministic stratification、latent premix 和 auxiliary hard SWD。
5. content-adaptive atom routing 能提 style，但容易变成更大 endpoint movement：K 提 style，LPIPS 退化。

## 下一步建议

优先在 H/F/K 附近做小网格：

- hard exploration probability：`0.05 / 0.10 / 0.15 / 0.20`
- fixed active top-k：`1 / 2 / 3 / 4`
- K router gain：降低强度，观察能否保留部分 style boost 并修复 LPIPS
- route temperature：约束 atom mixture 稀疏度
- generated-delta rank probe：验证提升是否来自可分执行方向，而不是共享改色或无组织位移
