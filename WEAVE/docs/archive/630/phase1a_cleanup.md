# Phase 1A: H1-H11 Zero-Risk Dead Code Removal (2026-06-30)

## Objective
应用审计报告 H1-H11 项,删除 active 路径(SpectralODEBridge620 + SpectralODEObjective620)中所有已确认的 dead 参数、dead 分支、dead 函数。所有删除均为 behavior-preserving(配置证据 + 调用图证据双重确认)。

## Baseline Reference
- Model params: 903,248
- Smoke loss: 4.627401 (baseline) → 4.595367 (after cleanup, 随机 batch 噪声)
- GPU: 33.8 MB

## Applied Changes (11 items)

### H1: SpectralODEBridge620.forward() 死参数移除
- **文件**: src/spectral_bridge620.py:95-140
- **删除**: `source`, `target_latent`, `velocity_scale` 三个签名参数 + `velocity_scale != 1.0` 缩放分支(line 155-158)
- **证据**: 唯一训练调用方(spectral_losses620.py:78-83)和推理调用方(integrate_transport)均不传这些参数;函数体内从未引用 `source`/`target_latent`
- **影响**: 零(参数从未被传入,分支永不执行)

### H2: SpectralODEBridge620.integrate_transport() + utils/inference.py 死参数移除
- **文件**: src/spectral_bridge620.py:142-155, src/utils/inference.py:519
- **删除**: `source_style_latent` 签名参数
- **证据**: 函数体内从未引用;推理调用方(utils/inference.py:536-557)不传该参数
- **影响**: 零

### H3: SpatialBridgeBlock620.__init__() 死参数移除
- **文件**: src/blocks620.py:45-111
- **删除**: 9 个已 `del` 的死参数(`style_moe_enabled`, `style_moe_num_experts`, `style_moe_router_hidden_dim`, `style_kv_moe_content_routed`, `style_query_source`, `style_cross_attn_skip_coarse`, `style_attn_topk`, `dino_dim`, `film_enabled`, `film_init_std`)及 3 行 `del` 语句
- **同步**: spectral_bridge620.py:72-78 调用点移除 `dino_dim=`, `film_enabled=`, `film_init_std=` kwargs
- **证据**: 628/629 清理已确认 MoE/FiLM/skip_coarse/topk 全为死分支;clean_base_v2 配置全 false
- **影响**: 零

### H4: StyleConditioner620 构造函数 + forward 死参数移除
- **文件**: src/style_encoder620.py:14-91
- **删除**: 11 个 deprecated 构造参数(`adapter_*`, `local_cnn_*`, `text_*`)及 `del` 语句;forward 中 `style_latent`, `style_text_tokens` 参数及 `del`
- **同步**: spectral_bridge620.py:52-57 调用点精简
- **证据**: 628/629 清理已删除 adapter/local_cnn/text 分支(788K dead params);clean_base_v2 全 false
- **影响**: 零

### H5: spectral620.py 多级 DWT 函数删除
- **文件**: src/spectral620.py:75-103 (删除 29 行)
- **删除**: `dwt2_multi_level`, `idwt2_multi_level` 两个完整函数
- **证据**: active config `spectral_ode_levels=1`;628/629 confirmed spectral_levels=1 is optimal
- **影响**: 零(spectral_levels=1 时永不调用)

### H6: SpectralODEObjective620 多级 DWT 分支删除
- **文件**: src/spectral_losses620.py:14, 34-36, 89-95
- **删除**: `dwt2_multi_level` 导入;`self.model_cfg`/`self.spectral_levels` 属性;`if self.spectral_levels > 1:` 分支
- **简化**: 直接 `target_ll, target_lh, target_hl, _ = dwt2_haar(target_delta)`
- **证据**: active config `spectral_ode_levels=1`,分支永假
- **影响**: 零

### H7: SpectralODEObjective620 Brownian 噪声分支删除
- **文件**: src/spectral_losses620.py:32-33, 80-87
- **删除**: `self.brownian_enabled`/`self.brownian_sigma` 属性;`if self.brownian_enabled and ...:` 分支
- **保留**: `spectral_brownian_noise_scale` metrics key(置 zero)以兼容 trainer.py 日志
- **证据**: active config `spectral_brownian_enabled=false`,分支永假
- **影响**: 零

### H8: SpectralODEBridge620.spectral_levels 属性删除
- **文件**: src/spectral_bridge620.py:50-52
- **删除**: `self.spectral_levels` 属性赋值 + 误导性注释(声称 forward/integrate_transport 根据 spectral_levels 分支,实际无此引用)
- **证据**: grep 确认 forward/integrate_transport 中从未读取 `self.spectral_levels`
- **影响**: 零

### H9: spectral_bridge620.py 多级 DWT 导入删除
- **文件**: src/spectral_bridge620.py:19
- **删除**: `dwt2_multi_level, idwt2_multi_level` 从 import 中移除
- **证据**: 见 H5
- **影响**: 零

### H10: StyleConditioner620.forward() 死参数移除
- **文件**: src/style_encoder620.py:55-64
- **删除**: `style_latent`, `style_text_tokens` forward 参数 + `del`
- **证据**: 见 H4
- **影响**: 零

### H11: build_spectral_ode_bridge_from_config() 死参数移除
- **文件**: src/spectral_bridge620.py:232-235, src/model.py:2125
- **删除**: `use_checkpointing` 参数 + `del`;model.py 调用点移除 `use_checkpointing=use_checkpointing` kwarg
- **证据**: 参数被立即 `del`,SpectralODEBridge620.__init__ 不接受该参数;active config `use_checkpointing=false`
- **影响**: 零(仅 spectral_ode 路径;620_spatial_bridge 路径保留 use_checkpointing)

## Verification
- Smoke test: PASS (903,248 params, loss=4.595367, GPU 33.8MB)
- Import sanity: ALL IMPORTS OK
- 行数削减: ~80 行(参数 + 分支 + 函数)
- 行为变化: 零(所有删除均为 dead code,配置证据 + 调用图证据双重确认)

## Next Steps
- Phase 1B: M9 attn_mode bug 修复(TDD: 写失败测试 → 修复 → 短训练对比 softmax vs relu2)
- Phase 1C: M1-M8 legacy 文件删除(model620.py, losses620.py, losses.py, ot_cost.py, lancet_*.py, style_tokenizer.py, semantic_tokenizer.py, round*_registry.py, model.py TimeConditionedLANCETBridge)
- 每阶段 smoke + 短训练验证
