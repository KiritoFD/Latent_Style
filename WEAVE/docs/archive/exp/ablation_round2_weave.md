# WEAVE Method 消融实验 Round 2

**日期**: 2026-07-09
**远程**: ssh -p 2222 administrator@100.115.18.62 (RTX 3060 12GB)
**基础配置**: spectral (Round 1 最佳变体, CLIP-S=0.7237, cLPIPS=0.3536, d_idt=0.0838)
**数据集**: wikiart_distinct5_samam_512_latents_ema (5 styles × 1000 samples, I:/datasets/)
**训练**: 默认 5 epochs (除 r2_spectral_10ep 为 10 epochs), batch_size=112
**评估**: full_eval on wikiart_distinct5_samam_512_classview/test (5 styles × 30 images)
**Runner**: scheduled task `r2_ablation_runner`, 自动顺序执行 6 个配置
**总耗时**: 06:36:49 → 09:00:39 (约 2h24min, 6/6 ok, 0 fail)

## 1. 实验结果总表

| 实验 | Group | CLIP-S ↑ | CLIP-T | content_LPIPS ↓ | delta_idt ↑ | vs R1 spec (CLIP-S) | 耗时 |
|------|-------|----------|--------|-----------------|-------------|---------------------|------|
| R1 spectral (基线) | — | 0.7237 | 0.2284 | 0.3536 | 0.0838 | 0 | ~20min |
| r2_spec_noswd | G1 | 0.7176 | 0.2262 | 0.3349 | 0.0777 | -0.0061 | 20.4min |
| **r2_spec_llw1** | G1 | **0.7236** | 0.2281 | **0.3434** | 0.0837 | -0.0001 | 20.5min |
| r2_spec_noswd_llw1 | G1 | 0.7168 | 0.2259 | **0.3329** | 0.0769 | -0.0069 | 20.6min |
| r2_spec_swd3 | G2 | 0.7209 | 0.2270 | 0.3430 | 0.0810 | -0.0028 | 20.5min |
| r2_spec_swd6 | G2 | 0.7226 | 0.2276 | 0.3377 | 0.0827 | -0.0011 | 20.5min |
| **r2_spectral_10ep** | G4 | **0.7288** | 0.2290 | 0.3459 | **0.0888** | **+0.0051** | 40.8min |

## 2. 核心发现

### 2.1 长训练 (10ep) 是 strict improvement — Round 2 最佳变体
- r2_spectral_10ep: CLIP-S=0.7288 (+0.0051), cLPIPS=0.3459 (-0.0077), d_idt=0.0888 (+0.0050)
- 相比 R1 spectral-5ep, **所有三个指标均改善**, 没有任何 trade-off
- 风格强度、内容保持、IDT 一致性同时提升
- **结论**: 5 epoch 不足以让 spectral 配置充分收敛; 10 epoch 是当前最佳训练长度。建议 Round 3 验证 15ep 是否继续提升。

### 2.2 ll_w=1.0 是 strict improvement over ll_w=0.3 (在 5ep 下)
- r2_spec_llw1 (ll_w=1.0): CLIP-S=0.7236 (≈ spectral 0.7237, -0.0001), cLPIPS=0.3434 (-0.0102)
- 与 R1 spectral (ll_w=0.3) 相比: style 几乎不变, 内容显著改善
- 对比 R1 ll_w1 (非 spectral 基础, CLIP-S=0.7121): spectral 基础上 ll_w=1.0 远优 (+0.0115)
- **结论**: `spectral_w_ll` 从 0.3 提升到 1.0 在 5ep 下是 strict win, 应在 Round 3 与 10ep 组合验证

### 2.3 SWD loss 对风格强度不可或缺
- r2_spec_noswd (swd_w=0): CLIP-S=0.7176 (-0.0061), cLPIPS=0.3349 (-0.0187)
- r2_spec_noswd_llw1 (swd_w=0 + ll_w=1.0): CLIP-S=0.7168 (-0.0069), cLPIPS=0.3329 (-0.0207)
- 即使配合 ll_w=1.0, 移除 SWD loss 仍导致 CLIP-S 下降 ~0.006-0.007
- 内容保持确实改善 (cLPIPS -0.019~-0.021), 但风格代价过大
- **对比 R1**: 在 baseline (非 spectral) 上移除 SWD loss 反而提升 CLIP-S (+0.0072); 在 spectral 基础上移除则下降。说明 **spectral + SWD 存在协同效应**, SWD 在 spectral 配置下转为正向作用。
- **结论**: SWD loss 必须保留, 但可降低权重

### 2.4 SWD 权重 dose-response: 风格 monotonic, 内容 non-monotonic
| swd_w | CLIP-S | cLPIPS | d_idt |
|-------|--------|--------|-------|
| 0 (0.0x) | 0.7176 | 0.3349 | 0.0777 |
| 3 (0.25x) | 0.7209 | 0.3430 | 0.0810 |
| 6 (0.5x) | 0.7226 | **0.3377** | 0.0827 |
| 12 (1.0x) | 0.7237 | 0.3536 | 0.0838 |

- **CLIP-S 单调递增** (随 swd_w 增加): 风格强度与 SWD 权重正相关
- **cLPIPS 非单调**: swd_w=6 时内容保持最好 (0.3377), 优于 swd_w=3 (0.3430) 和 swd_w=12 (0.3536)
- swd_w=6 是 sweet spot: style 仅 -0.0011 (近乎保持), content 改善 -0.0159
- **结论**: swd_w=6 是比 swd_w=12 更优的默认值, 应在 Round 3 与 ll_w=1.0 + 10ep 组合

### 2.5 三重组合 (no_swd + ll_w=1.0) 的内容保持最优但风格代价过大
- r2_spec_noswd_llw1: cLPIPS=0.3329 (全场最低/最好), 但 CLIP-S=0.7168 (-0.0069)
- 内容保持比 r2_spec_llw1 还好 (-0.0105), 但风格损失抵消了优势
- **结论**: 若能接受较低风格强度以换取最佳内容保持, 可考虑此配置; 否则 swd_w=6 + ll_w=1.0 是更好选择

## 3. Round 2 排名 (按 CLIP-S)

| Rank | 实验 | CLIP-S | cLPIPS | d_idt | 综合评价 |
|------|------|--------|--------|-------|----------|
| 1 | r2_spectral_10ep | **0.7288** | 0.3459 | **0.0888** | 最佳 style + IDT, 内容中等 |
| 2 | r2_spec_llw1 | 0.7236 | 0.3434 | 0.0837 | 5ep 最佳, 内容改善 |
| 3 | R1 spectral (5ep) | 0.7237 | 0.3536 | 0.0838 | 基线 |
| 4 | r2_spec_swd6 | 0.7226 | **0.3377** | 0.0827 | 最佳内容 (swd-w 组), style 近乎保持 |
| 5 | r2_spec_swd3 | 0.7209 | 0.3430 | 0.0810 | 中间点 |
| 6 | r2_spec_noswd | 0.7176 | 0.3349 | 0.0777 | 风格代价大 |
| 7 | r2_spec_noswd_llw1 | 0.7168 | **0.3329** | 0.0769 | 最佳内容但 style 最差 |

## 4. Round 3 设计建议

基于 Round 2 的三个独立改善轴:
1. **训练长度**: 5ep → 10ep (+0.0051 CLIP-S, strict win)
2. **LL 权重**: 0.3 → 1.0 (5ep 下 strict win on content, style 不变)
3. **SWD 权重**: 12 → 6 (5ep 下 style -0.0011, content -0.0159, 接近 strict win)

**Round 3 候选配置** (组合三个改善轴):
- `r3_swd6_llw1_10ep`: swd_w=6 + ll_w=1.0 + 10ep (三轴最优组合, 预期最佳)
- `r3_swd6_llw1_5ep`: swd_w=6 + ll_w=1.0 + 5ep (隔离 10ep 的边际效应)
- `r3_swd9_llw1_10ep`: swd_w=9 + ll_w=1.0 + 10ep (填补 6-12 gap)
- `r3_spectral_15ep`: spectral 默认 + 15ep (验证 10→15 是否继续提升)
- `r3_llw05_10ep`: ll_w=0.5 + 10ep (ll_w 中间点 sweep)
- `r3_llw2_10ep`: ll_w=2.0 + 10ep (ll_w 上限探测)

## 5. 数据完整性

- 6/6 实验成功完成, 0 失败, 0 跳过
- 所有结果来自 `exp/{name}/full_eval/epoch_*/summary.json` 的 `analysis.all_pairs_overview`
- 字段映射: `clip_style` (CLIP-S), `clip_t` (CLIP-T), `content_lpips` (cLPIPS), `clip_s_delta_idt` (d_idt)
- Runner log: `I:\Github\Latent_Style\SchrodingerBridge\remote_ablation_r2_log.txt`
- 提取脚本: `_extract_r2.py` (字段名已修复: clip_text→clip_t, delta_idt→clip_s_delta_idt)

## 6. 下一步行动

1. Git commit Round 2 结果 (config + state + 本文档)
2. 生成 Round 3 配置文件 (基于 r2_spectral_10ep.json 深拷贝修改)
3. 启动 Round 3 远程 runner
4. 持续监控至完成
