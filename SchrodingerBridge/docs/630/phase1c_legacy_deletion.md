# Phase 1C: Legacy 文件批量删除 (2026-06-30)

## 概述
删除 TimeConditionedLANCETBridge 相关的所有 legacy 代码, 共 ~11346 行 dead code.

## 删除清单

### model.py 精简 (2133 → 93 行, -2040 行)
- **删除**: `TimeConditionedLANCETBridge` 类 (~2070 行)
- **删除**: `sinusoidal_time_embedding` 函数 (仅被 TimeConditionedLANCETBridge 使用)
- **删除**: legacy imports (`lancet_blocks`, `lancet_backbone`, `style_families`, `utils.diffeomorphic`)
- **迁移**: `count_parameters` 函数 (从 `lancet_backbone.py` 迁移到 `model.py`)
- **保留**: `build_model_from_config` (精简, 支持 620_spectral_ode + 620_spatial_bridge, legacy 报错)
- **保留**: `_normalize_skip_routing_mode`, `_attach_bridge_runtime_fields` (config 兼容性)

### trainer.py 修改
- **移除**: 顶部 `from losses import OTFlowMatchingObjective`
- **移除**: 顶部 `from losses620 import SpatialBridgeObjective620`
- **改为**: 在 if/elif/else 分支内 lazy import (仅在对应 contract_family 时导入)

### src/__init__.py 修改
- **移除**: `from model import TimeConditionedLANCETBridge`
- **保留**: `from model import build_model_from_config`

### Legacy 文件删除 (9 个文件, 9306 行)
| 文件 | 行数 | 用途 |
|------|------|------|
| src/losses.py | 3275 | OTFlowMatchingObjective (legacy loss) |
| src/ot_cost.py | 376 | SWDTransportCost (legacy OT cost) |
| src/lancet_runtime.py | 1099 | LANCET 运行时 (legacy) |
| src/lancet_blocks.py | 838 | LANCET blocks (legacy) |
| src/lancet_backbone.py | 596 | LatentAdaCUT backbone (legacy) |
| src/style_tokenizer.py | 458 | 风格 tokenizer (legacy) |
| src/semantic_tokenizer.py | 991 | 语义 tokenizer (legacy) |
| src/round1_registry.py | 139 | Round1 注册表 (legacy) |
| src/round2_registry.py | 198 | Round2 注册表 (legacy) |
| tests/test_infra_guardrails.py | 1336 | Legacy 基础设施测试 |
| **TOTAL** | **9306** | |

### 保留的文件
- `src/model620.py` (1013 行) - 620_spatial_bridge 契约, 仍有配置文件引用
- `src/losses620.py` (793 行) - 620_spatial_bridge 契约 loss
- `src/style_families.py` - 被 trainer.py 和 utils/inference.py 使用
- `src/exp/` 目录 - 不可变历史训练快照

## 验证
- Smoke test: **PASS** (loss=4.594, GPU 33.8MB, 903,248 params)
- 与 Phase 1B baseline 一致 (loss=4.594 vs 4.595)

## 删除策略说明
1. **TimeConditionedLANCETBridge 已被 620_spectral_ode 取代**: active 配置使用 620_spectral_ode 契约, legacy 契约不再需要
2. **620_spatial_bridge 保留**: 虽然不是 active 契约, 但仍有 ~70 个配置文件引用, 保留以支持历史实验复现
3. **src/exp/ 不修改**: 历史训练快照是不可变的, 包含自己的代码副本
4. **tools/ legacy 脚本不删除**: 历史工具脚本, 不影响 active 路径

## 修改文件清单
1. `src/model.py` - 精简至 93 行 (从 2133 行)
2. `src/trainer.py` - legacy import 改为 lazy import
3. `src/__init__.py` - 移除 TimeConditionedLANCETBridge 导出
4. 删除 9 个 legacy src 文件
5. 删除 tests/test_infra_guardrails.py
6. `docs/630/phase1c_legacy_deletion.md` - 本文档

## Next Steps
- Phase 1D: 短训练验证最终最简 codebase 性能不下降 (2-3 epochs)
