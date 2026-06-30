# Phase 2: The Blindfolded Tokenizer — Style Masking 实验 (2026-06-30)

## 理论依据 (docs/630/mask.md)
引入 Masking 制造信息瓶颈, 强制 Tokenizer 丢弃全局内容拓扑, 只保留局部平稳的风格统计量:
- **内容 (Content)**: 全局拓扑相关, 被 high-ratio dropout + shuffle 摧毁
- **风格 (Style)**: 局部平稳遍历, 在 mask 下存活
- **目标**: 打破 Gate Collapse, 减少 content leakage, 提升 zero-shot 泛化

## 实现 (TDD)
### 接入点
`StyleConditioner620.forward()`: 在 `patch_proj` 投影后、返回 `img_tokens` 前应用 masking.

### 配置字段 (config_schema.py)
- `style_mask_ratio: float = 0.0` (drop 比例, 0.75 = 保留 25%)
- `style_mask_mode: str = "none"` (none | random | shuffle)

### Masking 模式
1. **random**: 随机丢弃 mask_ratio 比例的 token, 保留 (1-ratio). 每样本独立随机.
2. **shuffle**: 打乱 token 顺序 (保留数量, 破坏空间位置). 每样本独立打乱.
3. **none**: 直通 (默认, 向后兼容)

### TDD 测试 (tests/test_630_masking.py, 9 tests)
- config 字段存在性
- conditioner 属性
- random 减少 token 数量
- shuffle 保留 token 数量
- none 直通
- 配置传播
- 随机性 (两次调用不同)
- 完整 bridge forward

## 消融实验 (3 epochs, allpairs n=750)

| 实验 | ratio | mode | clip_style | content_lpips | clip 判定 | lpips 判定 |
|------|-------|------|-----------|---------------|----------|-----------|
| baseline (Phase 1D) | 0.0 | none | 0.7251 | 0.3373 | PASS | PASS |
| **random_50 (最佳)** | 0.5 | random | **0.7261** | 0.3296 | **PASS** | **PASS** |
| random_75 | 0.75 | random | 0.7250 | 0.3278 | PASS | PASS |
| shuffle_50 | 0.5 | shuffle | 0.7234 | 0.3205 | FAIL | PASS |
| shuffle_75 | 0.75 | shuffle | 0.7232 | 0.3177 | FAIL | PASS |

阈值: clip_style >= 0.7243, content_lpips <= 0.3453

## 关键发现

### 1. random mode 优于 shuffle mode
- **random_50**: clip=0.7261 (最佳), lpips=0.3296 (比 baseline 改善 -0.0077)
- **shuffle mode**: clip 降到阈值以下 (0.7234, 0.7232 均 FAIL)
- **原因**: shuffle 破坏了 DINO patches 的空间统计顺序, 虽然 cross-attn 无 PE, 但 patches 本身携带的空间相关性被扰乱, 损害风格提取

### 2. random mode 两项指标都优于无 mask baseline
- clip_style: 0.7261 vs 0.7251 (+0.0010)
- content_lpips: 0.3296 vs 0.3373 (-0.0077)
- **验证 mask.md 理论**: masking 减少 content leakage, 同时保留 style 信息

### 3. mask_ratio 的权衡
- **ratio=0.5 (random)**: clip 最佳, lpips 改善
- **ratio=0.75 (random)**: clip 略降但仍 PASS, lpips 进一步改善
- **趋势**: mask 越激进, lpips 越好 (内容保留越好), 但 clip_style 下降 (风格信息减少)
- **最优**: ratio=0.5 在两项指标上平衡最佳

### 4. shuffle mode 的 lpips 改善最大但 clip 代价过高
- shuffle_75: lpips=0.3177 (最佳总体, 比 baseline -0.0196)
- 但 clip=0.7232 (FAIL, 差阈值 0.0011)
- **不推荐**: 风格损失过大

## 结论

### 最佳配置: `style_mask_ratio=0.5, style_mask_mode="random"`
- 两项指标均 PASS
- 比无 mask baseline 在 clip 和 lpips 上都有改善
- 符合 mask.md 的 "信息瓶颈" 理论
- 计算开销几乎为零 (只是 token 子集选择)

### 推荐默认值
建议将 `style_mask_ratio=0.5, style_mask_mode="random"` 作为新的 default 配置 (后续完整 10 epochs 训练确认).

## 修改文件清单
1. `src/config_schema.py` - 添加 style_mask_ratio, style_mask_mode 字段
2. `src/style_encoder620.py` - 实现 _apply_mask() 方法
3. `src/spectral_bridge620.py` - 传递 mask 配置给 StyleConditioner620
4. `tests/test_630_masking.py` - 9 个 TDD 测试
5. `configs/630_phase2b_mask_random_50.json` - 最佳配置
6. `configs/630_phase2c_mask_random_75.json` - 消融配置
7. `configs/630_phase2c_mask_shuffle_50.json` - 消融配置
8. `configs/630_phase2c_mask_shuffle_75.json` - 消融配置
9. `docs/630/phase2_masking.md` - 本文档

## Next Steps
- Phase 3: 基于理论探索其他提升方法 (频率掩码, 显著性反向掩码等)
