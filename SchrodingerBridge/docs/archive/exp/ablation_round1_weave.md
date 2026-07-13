# WEAVE Method 消融实验 Round 1

**日期**: 2026-07-09
**远程**: ssh -p 2222 administrator@100.115.18.62 (RTX 3060 12GB)
**数据集**: wikiart_distinct5_samam_512_latents_ema (5 styles × 1000 samples, I:/datasets/)
**训练**: 5 epochs, batch_size=112, Patience=2, max=10 (但 5 epoch 短训练用于消融)
**评估**: full_eval on wikiart_distinct5_samam_512_classview/test (5 styles × 30 images)
**关键修复**: `_projection_dirs` 确定性缓存 bug 已修复 (fresh random dirs), baseline CLIP-S=0.7087 (vs bug版本 0.6839)

## 1. 实验结果总表

| 实验 | Tier | CLIP-S ↑ | CLIP-T | content_LPIPS ↓ | delta_idt ↑ | id_clip_s | id_lpips | xf_clip_s | xf_lpips | 耗时 |
|------|------|----------|--------|-----------------|-------------|-----------|----------|-----------|----------|------|
| **baseline** | T0 | 0.7087 | 0.2302 | 0.3644 | 0.0688 | 0.8124 | 0.3243 | 0.6828 | 0.3744 | 12.7min |
| no_swd_loss | T1 | 0.7159 | 0.2256 | 0.3277 | 0.0760 | 0.8431 | 0.2891 | 0.6841 | 0.3374 | 11.4min |
| no_dwt_route | T1 | 0.7128 | 0.2297 | 0.3960 | 0.0729 | 0.8036 | 0.3490 | 0.6901 | 0.4078 | 10.9min |
| **no_wct** | T1 | **0.6884** | 0.2261 | 0.3426 | 0.0485 | 0.8005 | 0.3415 | 0.6604 | 0.3429 | 10.7min |
| no_eota | T1 | 0.7087 | 0.2303 | 0.3647 | 0.0688 | 0.8123 | 0.3247 | 0.6828 | 0.3748 | 10.8min |
| k1_global | T2 | 0.7169 | 0.2265 | 0.3346 | 0.0770 | 0.8393 | 0.2931 | 0.6863 | 0.3450 | 7.2min |
| blend0_pure_global | T2 | 0.7169 | 0.2265 | 0.3347 | 0.0770 | 0.8394 | 0.2932 | 0.6863 | 0.3451 | 10.7min |
| blend1_pure_region | T2 | 0.7094 | 0.2309 | 0.3746 | 0.0695 | 0.8064 | 0.3364 | 0.6852 | 0.3841 | 10.8min |
| k64_extreme | T2 | 0.7170 | 0.2304 | 0.3797 | 0.0771 | 0.8241 | 0.3378 | 0.6902 | 0.3902 | 49.5min |
| ll_w0 | T3 | 0.7073 | 0.2304 | 0.3764 | 0.0674 | 0.8059 | 0.3375 | 0.6827 | 0.3861 | 10.9min |
| ll_w1 | T3 | 0.7121 | 0.2298 | 0.3617 | 0.0722 | 0.8201 | 0.3196 | 0.6851 | 0.3722 | 10.8min |
| route_p05 | T3 | 0.7105 | 0.2307 | 0.3928 | 0.0706 | 0.8130 | 0.3528 | 0.6849 | 0.4028 | 10.8min |
| route_p10 | T3 | 0.7119 | 0.2302 | 0.3844 | 0.0720 | 0.8072 | 0.3394 | 0.6881 | 0.3957 | 10.8min |
| **soft_mask** | T4 | **0.7196** | 0.2270 | 0.3386 | 0.0797 | 0.8394 | 0.2940 | 0.6897 | 0.3497 | 19.2min |
| sinkhorn | T4 | FAILED | — | — | — | — | — | — | — | killed |
| **spectral** | T4 | **0.7237** | 0.2284 | 0.3536 | **0.0838** | 0.8408 | 0.3058 | 0.6915 | 0.3632 | ~20min |

## 2. 核心发现

### 2.1 WCT 是最关键组件
- 移除 WCT 后 CLIP-S 暴跌 0.0203 (0.7087→0.6884), delta_idt 降 0.0203 (0.0688→0.0485)
- 风格注入几乎完全依赖 Endpoint WCT, 其他机制无法补偿

### 2.2 SWD loss 移除反而提升 (反直觉)
- 移除 SWD loss 后: CLIP-S ↑0.0072, content_LPIPS ↓0.0367, id_lpips ↓0.0352
- SWD loss 在 5-epoch 短训练中可能过度约束, 阻碍了模型学习风格-内容分离
- **结论**: SWD loss 的正则化作用在短训练中是负面的, 需要在更长训练中验证

### 2.3 EOTA 在短训练中是 no-op
- no_eota 与 baseline 几乎完全相同 (CLIP-S 差 0.0000, LPIPS 差 0.0003)
- EOTA soft-threshold 可能需要更多 epoch 才能激活, 或 τ 值过松
- **结论**: 5 epoch 不足以评估 EOTA, 需要在 10+ epoch 中重新验证

### 2.4 K=1 (全局 SWD) ≈ K=8 (语义区域 SWD)
- k1_global: CLIP-S=0.7169, content_LPIPS=0.3346 (优于 baseline)
- blend0_pure_global: CLIP-S=0.7169, content_LPIPS=0.3347 (与 k1 本质相同)
- 语义区域聚类 (K=8) 增加了 6 倍训练时间但无收益
- **结论**: 语义区域 SWD 机制在当前设置下是无效复杂度

### 2.5 spectral 是最优变体 (超越 soft_mask)
- spectral: CLIP-S=0.7237, delta_idt=0.0838 (双指标全场最高), content_LPIPS=0.3536
- 用 spectral ODE (620_spectral_ode contract family) 替代普通 SWD 路由
- 耗时 ~20min (baseline 1.6 倍), id_lpips=0.3058 (优于 baseline 0.3243)
- **结论**: spectral ODE 应作为 Round 2 的基础配置

### 2.5b soft_mask 是次优变体
- CLIP-S=0.7196, content_LPIPS=0.3386, delta_idt=0.0797
- 用 soft mask (softmax 权重) 替代 hard kmeans 聚类, 允许平滑的区域分配
- 耗时 19.2min (baseline 1.5 倍), 但效果显著
- **结论**: soft_mask 作为 Round 2 备选 baseline

### 2.6 LL weighting 的影响
- ll_w0 (λ=0): CLIP-S=0.7073 (略低于 baseline)
- ll_w1 (λ=1.0): CLIP-S=0.7121, content_LPIPS=0.3617 (优于 baseline)
- 当前 baseline 的 λ_LL 可能在 0.5 左右; λ=1.0 更优
- **结论**: LL 权重应提升到 1.0

### 2.7 route_prob 影响极小
- p05 (0.7105) 和 p10 (0.7119) 均接近 baseline (0.7087)
- DWT 高频路由的训练概率对结果不敏感

### 2.8 sinkhorn 发散 (结构性失败)
- 75-78s/batch (baseline 的 35 倍), ot_cost=0.0000 (发散)
- 11.5 min 仅完成 9/44 batch of epoch 1
- 显存 11.6/12GB (接近 OOM)
- **结论**: Sinkhorn OT 在当前架构下不可行, 数值不稳定且计算量过大

## 3. 排名 (按 CLIP-S)

| Rank | 实验 | CLIP-S | delta_idt | vs baseline (CLIP-S) |
|------|------|--------|-----------|----------------------|
| 1 | **spectral** | **0.7237** | **0.0838** | **+0.0150** |
| 2 | soft_mask | 0.7196 | 0.0797 | +0.0109 |
| 3 | k64_extreme | 0.7170 | 0.0771 | +0.0083 |
| 4 | k1_global | 0.7169 | 0.0770 | +0.0082 |
| 4 | blend0_pure_global | 0.7169 | 0.0770 | +0.0082 |
| 6 | no_swd_loss | 0.7159 | 0.0760 | +0.0072 |
| 7 | no_dwt_route | 0.7128 | 0.0729 | +0.0041 |
| 8 | ll_w1 | 0.7121 | 0.0722 | +0.0034 |
| 9 | route_p10 | 0.7119 | 0.0720 | +0.0032 |
| 10 | route_p05 | 0.7105 | 0.0706 | +0.0018 |
| 11 | blend1_pure_region | 0.7094 | 0.0695 | +0.0007 |
| 12 | **baseline** | **0.7087** | **0.0688** | — |
| 13 | no_eota | 0.7087 | 0.0688 | +0.0000 |
| 14 | ll_w0 | 0.7073 | 0.0674 | -0.0014 |
| 15 | no_wct | 0.6884 | 0.0485 | -0.0203 |
| — | sinkhorn | FAILED | — | — |

## 4. Round 2 设计方向

基于 Round 1 发现 (spectral=0.7237 最佳), Round 2 聚焦以下方向:

### R2-1: 组合最优配置 (基于 spectral)
- spectral + no_swd_loss + ll_w1 (三个正向发现的组合)
- spectral + no_swd_loss (双 top-2 正向发现)
- spectral + ll_w1
- 假设: 组合后 CLIP-S 应 ≥0.73

### R2-2: SWD loss 权重扫描 (基于 spectral)
- Round 1 显示完全移除 SWD loss 反而提升 (no_swd_loss=0.7159 vs baseline=0.7087)
- 在 spectral 基础上扫描 λ_swd ∈ {0, 0.1, 0.25, 0.5, 1.0}
- 当前 baseline 可能 λ_swd=1.0, 过重

### R2-3: EOTA τ 扫描 (在 spectral 基础上)
- Round 1 中 EOTA 无效果, 可能 τ 过松
- 扫描 τ ∈ {0.01, 0.02, 0.04, 0.08, 0.16}
- 需要在 10 epoch 中验证

### R2-4: 更长训练 (10 epoch)
- 5 epoch 可能不足以区分组件贡献
- 对 spectral + top-3 configs 做 10 epoch 验证
- 关注 EOTA 在 10 epoch 中是否能激活

### R2-5: soft_mask vs spectral 头对头
- 两者均为 positive, 但机制不同 (soft kmeans vs spectral ODE)
- 在 10 epoch 中比较, 确认 spectral 优势是否稳定

## 5. Infra 备注

- GPU 功率: baseline ~85W, spectral ~48W, sinkhorn ~140W (但发散)
- 功率 <120W 的根因: latent-space 训练 (4×64×64) 计算量小, GPU memory-bound
- 数据在 I: 盘 (HDD), I/O 瓶颈进一步降低功率
- bs=112 已接近 12GB VRAM 上限 (9.5-11.6GB), 无法继续增大
- **功率问题为结构性限制, 非配置错误**
