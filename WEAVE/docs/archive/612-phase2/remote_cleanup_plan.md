# 远程清理计划: inmortal-exp 目录

Generated: 2026-06-13
Remote: I:\GitHub\Latent_Style\SchrodingerBridge\exp\inmortal-exp\

## 删除清单 (可以安全删除的目录及其 checkpoint)

### 已确认 drop/negative 的完整实验 (删除全部 .pt)

| 目录 | 大小 | 原因 |
|------|------|------|
| aaai2027_inmortal_xpred_phighpass_seed42_b28 | 599M | drop: style远低于kmanifold |
| aaai2027_inmortal_xpred_kmanifold_phighpass_seed42_b32 | 458M | drop: 双重退化 |
| aaai2027_inmortal_xpred_bary_seed42_b40 | 1.1G | drop: 大batch LPIPS更差 |
| aaai2027_inmortal_xpred_kmanifold_pattn_aniso_seed42_b16 | 471M | drop: 各向异性太激进 |
| aaai2027_inmortal_xpred_kmanifold_pmod_seed42_b32 | 461M | hold(obsolete): 已超越 |
| aaai2027_inmortal_xpred_bary_seed42_b16 | 637M | hold(obsolete): 已被pattn超越 |
| aaai2027_inmortal_xpred_queue_seed42_b16 | 453M | hold(obsolete) |
| aaai2027_inmortal_xpred_structot_seed42_b16 | 453M | hold(obsolete) |
| aaai2027_inmortal_xpred_teacher_endpoint_seed42_b16 | 455M | hold(obsolete) |
| aaai2027_inmortal_xpred_kmanifold_pattn_edgegated_anisostokes_queue_from_e13_seed42_b8a2 | 411M | drop: edge gate坏LPIPS |
| aaai2027_inmortal_k_spatial_seed42_b32 | ? | drop: 同b16但更大 |
| aaai2027_inmortal_k_spatial_seed42_b44 | ? | drop: 同b16但更大 |
| aaai2027_inmortal_k_spectral_seed42_b16 | ? | drop(obsolete) |
| aaai2027_inmortal_p_highpass_seed42_b16 | ? | drop |
| aaai2027_inmortal_p_highpass_seed42_b32 | ? | drop |

**回收: ~6GB**

### 清理 checkpoint 仅保留 best + 最后 + eval 数据

| 目录 | 大小 | 保留策略 |
|------|------|----------|
| aaai2027_inmortal_k_spatial_seed42_b16 | 1.1G | 保留 e6+summary, 删 e1-5,e7-8 ckpt |
| aaai2027_inmortal_k_manifold_seed42_b16 | 1.1G | 保留 e6+summary, 删 e1-5,e7-8 ckpt |
| aaai2027_inmortal_xpred_kmanifold_seed42_b32 | 977M | 保留 e7+summary, 删 e1-6,e8 ckpt |
| aaai2027_inmortal_k_spectral_seed42_b12 | 455M | 保留 e2+summary, 删 e1,e3-6 ckpt |
| aaai2027_inmortal_xpred_kmanifold_pattn_seed42_b16 | 471M | 保留 e6+summary, 删 e1-5,e7-8 ckpt |
| aaai2027_inmortal_xpred_kmanifold_pattn_stokes_seed42_b16 | 471M | 保留 e3+summary, 删 e1-2,e4-8 ckpt |
| aaai2027_inmortal_xpred_kmanifold_pattn_queue_seed42_b16 | 469M | 保留 e6+summary, 删 e1-5,e7-8 ckpt |
| aaai2027_inmortal_xpred_kmanifold_pattn_stokes002/se_from_pattn | ? | 保留 e13+summary, 删 e12,e14-16 ckpt |

**回收: ~3GB**

### from_e13 续跑实验 (drop/hold, 大量冗余 ckpt)

这些实验都是从 e13 checkpoint 续跑的，每个都有 8-16 个 epoch 的 .pt 文件。
绝大多数被标记为 drop 或 hold，不被引用。

| 目录 | 大小 | 决策 |
|------|------|------|
| clamp_reseed_from_e13 | 937M | keep eval, 删所有 ckpt |
| clamprelease_reseed_from_e13 | 937M | keep eval, 删所有 ckpt |
| clampreleasewide_reseed_from_e13 | 939M | drop, 全部删除 |
| clampreleaselatewide_reseed_from_e13 | 939M | hold, keep eval+csv, 删 ckpt |
| clamphold4wide_reseed_from_e13 | 882M | keep eval+summary, 删 ckpt |
| clamphold4mid_reseed_from_e13 | 1.0G | keep eval+summary, 删 ckpt |
| clamphold4slowmid_reseed_from_e13 | 1.1G | keep eval+summary, 删 ckpt |
| clamphold4twostage_reseed_from_e13 | 1.5G | drop, 全部删除 |
| trust_from_e13 | 645M | drop, 全部删除 |
| trust_reseed_from_e13 | 939M | drop, 全部删除 |
| anisostokes_queue_from_pattn | 388M | keep (活跃引用) |

**回收: ~8GB**

### 其他可清理的无关目录

| 目录 | 大小 | 决策 |
|------|------|------|
| hold4mid_e8_carriergate_injection | ? | drop: carriergate注入实验 |
| hold4mid_e8_spatial_carriergate_bodydecoder | 851M | keep eval, 删 ckpt |
| hold4mid_e8_spatial_carriergate_decoder | ? | keep eval, 删 ckpt |
| knee_e13_carriergate_injection | ? | drop |
| knee_e13_spatial_carriergate_bodydecoder | 1.3G | keep eval, 删 ckpt |

**回收: ~2GB**

## 总计可回收: ~19GB

## 保留清单 (不删除)

- aaai2027_inmortal_xpred_kmanifold_pattn_stokes_from_pattn_seed42_b16 (keep: 最佳LPIPS前线)
- aaai2027_inmortal_xpred_kmanifold_pattn_stokes002_from_pattn_seed42_b16 (keep: 最佳raw style)
- aaai2027_inmortal_xpred_kmanifold_pattn_seed42_b16_e12_continue (keep: 延长训练证据)
- aaai2027_inmortal_xpred_kmanifold_pattn_stokes_seed42_b16_e12_continue (keep: LPIPS改善证据)
- aaai2027_inmortal_k_manifold_seed42_b16 (keep: best kinetic-only)
- aaai2027_inmortal_xpred_kmanifold_seed42_b32 (keep: strongest raw style)
- aaai2027_inmortal_xpred_kmanifold_pattn_seed42_b32 (keep: b32对照)
- aaai2027_inmortal_xpred_kmanifold_pattn_seed42_b24 (keep: b24对照)
- aaai2027_inmortal_xpred_kmanifold_pattn_stokes003_from_pattn_seed42_b4a2 (keep: batch4对照)
- aaai2027_inmortal_xpred_kmanifold_pattn_anisostokes_queue_from_pattn_seed42_b8a2 (keep: 活跃引用)
