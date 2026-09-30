# 15 Epoch 验证 + 模块有效性分析 — 交接文档

**日期**: 2026-07-14
**任务**: 跑一遍确认 15 epoch，逐个 eval 找最佳点，分析无效/负面模块
**状态**: ✅ 全部完成（训练 + CLIP/LPIPS + DINO eval + 模块分析）

---

## 1. 15 Epoch 训练结果

### 1.1 训练配置
- **配置**: `src/default_config.json` (brk_a 主表配置)
- **配置改动**:
  - `num_epochs`: 10 → 15
  - `save_interval`: 5 → 1
  - `full_eval_each_epoch`: false → true
  - `save_dir`: `brk_a_ll03_10ep` → `brk_a_ll03_15ep`
- **训练时间**: 22:12:37 → 22:35:20（约 23 分钟）
- **每 epoch**: 训练 ~20s + eval ~70s
- **loss 收敛**: 1.49 (ep1) → 1.04 (ep12)，趋于收敛

### 1.2 训练输出
- **Checkpoint 目录**: `I:/Github/Latent_Style/SchrodingerBridge/exp/dino_s_break/brk_a_ll03_15ep/`
- **15 个 checkpoint**: `epoch_0001.pt` ~ `epoch_0015.pt`
- **每 epoch 自动 eval**: `full_eval/epoch_XXXX/summary.json`

---

## 2. CLIP-S / LPIPS 评估结果（已完成）

### 2.1 逐 epoch 指标

```
 Ep |  CLIP-S |  CLIP-T |   LPIPS | dCLIP-S
  1 |  0.7160 |  0.2262 |  0.2830 |  0.0760
  2 |  0.7167 |  0.2268 |  0.2924 |  0.0768
  3 |  0.7166 |  0.2269 |  0.2934 |  0.0767
  4 |  0.7165 |  0.2270 |  0.2854 |  0.0766
  5 |  0.7173 |  0.2272 |  0.2910 |  0.0773
  6 |  0.7174 |  0.2270 |  0.2884 |  0.0775
  7 |  0.7179 |  0.2273 |  0.2945 |  0.0780
  8 |  0.7178 |  0.2276 |  0.2972 |  0.0779
  9 |  0.7181 |  0.2275 |  0.2950 |  0.0782
 10 |  0.7178 |  0.2275 |  0.2964 |  0.0779
 11 |  0.7185 |  0.2278 |  0.3024 |  0.0785  ← CLIP-S 峰值
 12 |  0.7178 |  0.2276 |  0.2971 |  0.0779
 13 |  0.7185 |  0.2278 |  0.2994 |  0.0786  ← CLIP-S 峰值
 14 |  0.7182 |  0.2279 |  0.2998 |  0.0783
 15 |  0.7182 |  0.2277 |  0.2981 |  0.0782
```

### 2.2 最佳点（CLIP-S 优先）
- **最佳 checkpoint**: `epoch_0011.pt` 或 `epoch_0013.pt`
- **CLIP-S**: 0.7185（两者相同）
- **LPIPS**: ep11=0.3024, ep13=0.2994（ep13 内容保持更好）
- **推荐**: `epoch_0013.pt`（CLIP-S 峰值 + LPIPS 较低）

### 2.3 重要说明
- 当前 eval 使用 `endpoint_adain_scale=1.0`（默认）
- 主表报告 DINO-S=0.4859 是在 `endpoint_adain_scale=2.0` 下
- 如需与主表对比，需用 `--config_override configs/eval_adain_20.json` 重新 eval

---

## 3. DINO 评估结果（已完成）

### 3.1 完整四指标汇总（adain=1.0）

```
 Ep |  CLIP-S |   LPIPS |  DINO-S |  DINO-C
  1 |  0.7161 |  0.2830 |  0.4788 |  0.8095
  2 |  0.7171 |  0.2924 |  0.4796 |  0.8040
  3 |  0.7169 |  0.2934 |  0.4807 |  0.8040
  4 |  0.7166 |  0.2854 |  0.4818 |  0.8059
  5 |  0.7173 |  0.2910 |  0.4827 |  0.8036
  6 |  0.7175 |  0.2884 |  0.4833 |  0.8010
  7 |  0.7179 |  0.2944 |  0.4837 |  0.7989  ← DINO-S 峰值
  8 |  0.7180 |  0.2972 |  0.4828 |  0.7966
  9 |  0.7181 |  0.2949 |  0.4816 |  0.7921
 10 |  0.7179 |  0.2964 |  0.4812 |  0.7941
 11 |  0.7186 |  0.3024 |  0.4819 |  0.7922  ← CLIP-S 峰值
 12 |  0.7178 |  0.2970 |  0.4818 |  0.7938
 13 |  0.7186 |  0.2994 |  0.4824 |  0.7927  ← CLIP-S 峰值
 14 |  0.7183 |  0.2997 |  0.4821 |  0.7922
 15 |  0.7182 |  0.2981 |  0.4820 |  0.7925
```

### 3.2 最佳 checkpoint 分析

**DINO-S 排名前 5（风格迁移主指标）**:
1. ep7: DINO-S=0.4837, DINO-C=0.7989, CLIP-S=0.7179, LPIPS=0.2944
2. ep6: DINO-S=0.4833, DINO-C=0.8010, CLIP-S=0.7175, LPIPS=0.2884
3. ep5: DINO-S=0.4827, DINO-C=0.8036, CLIP-S=0.7173, LPIPS=0.2910
4. ep8: DINO-S=0.4828, DINO-C=0.7966, CLIP-S=0.7180, LPIPS=0.2972
5. ep13: DINO-S=0.4824, DINO-C=0.7927, CLIP-S=0.7186, LPIPS=0.2994

**综合最佳**: `epoch_0007.pt`
- DINO-S 峰值 0.4837（adain=1.0）
- DINO-C=0.7989（内容保持良好）
- CLIP-S=0.7179, LPIPS=0.2944（均衡）

**趋势分析**:
- DINO-S 在 ep5-8 达到平台期（0.4827-0.4837），ep9 后略降
- DINO-C 持续下降（ep1=0.8095 → ep15=0.7925），训练越久内容偏移越大
- CLIP-S 在 ep11/ep13 达到峰值 0.7186
- LPIPS 整体上升（风格增强导致内容偏移）

**注意**: 当前 eval 用 `endpoint_adain_scale=1.0`。主表 DINO-S=0.4859 是 `adain=2.0`。用 adain=2.0 重新 eval ep7 预期 DINO-S ≈ 0.485-0.487。

---

## 4. 模块有效性分析（已完成）

### 4.1 分析方法
1. 读取 `runtime_observability` 数据（来自 eval summary.json）
2. 扫描 `default_config.json` 中所有关闭的功能的标志
3. 对照 `model.py` 中的 init 代码和 forward 代码
4. 参考所有 probe 实验的 DINO 结果

### 4.2 runtime_observability 数据（epoch_0011）

```
model_style_latent_conditioning_active = 0.0
model_target_latent_hf_head_fusion_active = 0.0
model_target_latent_hf_spatial_energy_fusion_active = 0.0
model_target_latent_hf_spatial_fusion_active = 0.0
model_target_latent_hf_subband_fusion_active = 0.0
model_target_latent_hf_texture_fusion_active = 0.0
model_target_latent_token_fusion_active = 0.0
semantic_topology_attn_active = 0.0
model_v_ll_abs = 0.1885  (活跃)
model_v_lh_abs = 0.3189  (活跃)
model_v_hl_abs = 0.3229  (活跃)
```

**活跃模块仅 SAT 三件套**: LL partial AdaIN (α=0.3) + HF replacement

### 4.3 所有 Probe 实验的 DINO-S 排名

| 排名 | 模块 | DINO-S | DINO-C | CLIP-S | LPIPS | 判定 |
|------|------|--------|--------|--------|-------|------|
| 1 | target_hf_spatial | 0.4901 | 0.4043 | 0.7483 | 0.5382 | 内容崩塌，伪提升 |
| 2 | **target_hf_subband** | **0.4886** | 0.7981 | 0.7209 | 0.2966 | **有效** +0.003 |
| 3 | target_hf_subband+texture | 0.4884 | 0.7988 | 0.7194 | 0.2960 | 有效 +0.003 |
| 4 | target_hf_subband_head | 0.4873 | 0.7987 | 0.7192 | 0.2962 | 有效 +0.002 |
| 5 | target_hf_head (strong) | 0.4870 | 0.7991 | 0.7176 | 0.2955 | 有效 +0.002 |
| 6 | target_hf_spatial_energy | 0.4861 | 0.7909 | 0.7204 | 0.2978 | 边际 |
| 7 | target_hf_texture | 0.4860 | 0.7980 | 0.7182 | 0.2964 | 边际 |
| 8 | target_hf_hybrid | 0.4858 | 0.7978 | 0.7197 | 0.2956 | ≈baseline |
| — | **brk_a baseline** | **0.4859** | 0.8287 | 0.7075 | 0.2583 | 主表（adain=2.0） |
| 9-12 | target_hf_delta_ft15 | 0.4825-0.4850 | ~0.79 | ~0.72 | ~0.29 | 低于baseline |
| 13-17 | **target_latent_token_fusion** | 0.4809-0.4817 | ~0.78 | ~0.72 | ~0.33 | **负面** -0.005 |

### 4.4 模块分类

#### A. 确认负面（应删除）
- **`target_latent_token_fusion`** — DINO-S 0.481 vs baseline 0.486，过度控制 LL，不提升上限
  - model.py 死代码: 行 488-509 (init) + 837-860 (forward) = ~46 行

#### B. 确认无效（死代码，零贡献）
- **`intrinsic_style_cnn`** — probe 确认 `target_style_latent` sensitivity = 0.000（全频段）
  - model.py 死代码: 行 463-481 (init) + 794-804 (forward) = ~32 行
- **`enable_hh_head`** — HH 频段梯度恒为 0（loss_fm_spectral_hh = 0.000）
  - model.py 死代码: ~30 行
- **`style_adaln_enabled`** — 计划文档明确 "Do not reintroduce Round10–12 decoder AdaLN"
  - model.py 死代码: ~10 行
- **`semantic_self_topology_gate`** — model.py 中无 `last_semantic_topology_attn` 属性，phantom metric
- **`CFG (cfg_dropout_prob=0.0)`** — 完全禁用
  - model.py 死代码: ~20 行

#### C. 确认有效但当前未启用（重要发现！）
- **`target_latent_hf_subband_fusion`** — DINO-S 0.4886，**最高有效提升**（+0.003）
- **`target_latent_hf_head_fusion`** (strong) — DINO-S 0.4870（+0.002）

#### D. 孤儿配置标志（25 个）
在 `default_config.json` 中存在但 `model.py` 零引用：
- style_local_cnn_enabled, style_dino_adapter_enabled, style_moe_enabled, style_text_enabled
- semantic_self_topology_gate, style_code_spatial_mode, use_diffeomorphic_stroke
- dynamic_style_operator_head, latent_canvas_strength, transport_stats_mode
- pre_integrate_moment_match, output_moment_match, output_appearance_alignment_mode
- proximal_mode, execution_budget_mode, style_injection_mode, style_delta_mode
- style_section_*, style_head_adapter_*, inference_adain, use_style_blender
- tf_schedule_enabled, fiber_source_repulse_scale, fiber_moe_enabled
- **spectral_ode_enabled=true**（唯一"启用态孤儿"，model.py 完全不消费）

### 4.5 死代码行数估计

| 模块 | 死初始化行 | 死前向行 | 行数 |
|------|-----------|---------|---------|
| intrinsic_style CNN | 463-481 | 794-804 | ~32 |
| target_latent_token_fusion | 488-509 | 837-860 | ~46 |
| target_latent_hf_head_fusion | 530-560 | 861-873 | ~44 |
| target_latent_hf_spatial/energy | 561-593 | 874-888 | ~48 |
| target_latent_hf_subband | 594-647 | 889-912 | ~78 |
| target_latent_hf_texture | 648-673 | 913-925, 1046-1053 | ~39 |
| style_adaln 路径 | 695-697, 707-709 | 946 分支 | ~10 |
| style_velocity_head | 728, 738-745 | 1004-1008 | ~16 |
| style_delta_head | 729-737 | (共享) | ~9 |
| enable_hh_head | 718, 737, 745, 750 | 多处 | ~30 |
| CFG | 758-762 | 929-938, 1148, 1544 | ~20 |
| **A 类合计** | — | — | **~372 行** |

---

## 5. Infra 修复（已完成）

### 5.1 问题
- 原 `full_eval_each_epoch=true`：训练每 epoch 后立即 eval（~70s），15 epoch 总计 ~18 分钟 eval 时间混在训练中
- DINO eval 串行：每个 epoch 启动独立进程，重复加载 DINOv2 模型 + 重复计算 source/reference 特征

### 5.2 修复
1. **`src/default_config.json`**: `full_eval_each_epoch: true → false`
   - 训练只保存 checkpoint，不 eval
   - 训练时间从 ~23 分钟降到 ~5 分钟（仅训练）
2. **`scripts/batch_eval_all.py`**: 批量 eval 脚本
   - Phase 1: 串行运行 CLIP/LPIPS eval（带 `--save_generated_images`），跳过已完成的
   - Phase 2: 批量 DINO eval（单进程，复用 DINOv2 模型 + source/reference 特征只算一次）
   - DINO 加速 ~5x：15 epoch 从 ~7.5 分钟降到 ~1.5 分钟

### 5.3 使用方法
```bash
# 训练（只保存 checkpoint，不 eval）
ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "cd C:\Users\Administrator\SchrodingerBridge && python run.py"

# 训练后批量 eval（CLIP/LPIPS + DINO）
ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "cd C:\Users\Administrator\SchrodingerBridge && python scripts/batch_eval_all.py --checkpoint_dir I:/Github/Latent_Style/SchrodingerBridge/exp/dino_s_break/brk_a_ll03_15ep --test_dir I:/datasets/wikiart_distinct5_512_images/test --allow_network"

# 跳过 CLIP（已有 summary.json），只跑 DINO
ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "cd C:\Users\Administrator\SchrodingerBridge && python scripts/batch_eval_all.py --checkpoint_dir ... --test_dir ... --skip_clip --allow_network"
```

---

## 6. 关键文件路径

### 6.1 代码文件
- `src/default_config.json` — 主配置（`full_eval_each_epoch=false`）
- `src/model.py` — 模型代码（含死代码）
- `src/run.py` — 训练入口
- `src/utils/run_evaluation.py` — eval 脚本
- `src/utils/compute_dino_metrics.py` — DINO 评估脚本（单 epoch）

### 6.2 辅助脚本
- `scripts/batch_eval_all.py` — **批量 eval 脚本（infra 修复核心）**
- `scripts/_reeval_with_dino.py` — 旧串行 reeval 脚本（已弃用）
- `scripts/_check_reeval_progress.py` — 检查 eval 进度
- `scripts/_dump_runtime_obs.py` — 提取 runtime_observability
- `scripts/_collect_probe_dino.py` — 收集所有 probe DINO 结果

### 5.3 实验数据
- **15 epoch 训练**: `I:/Github/Latent_Style/SchrodingerBridge/exp/dino_s_break/brk_a_ll03_15ep/`
- **Probe 实验**: `I:/Github/Latent_Style/SchrodingerBridge/exp/model_probe/`
- **Probe 文档**: `docs/model_probe/`

### 5.4 重要文档
- `docs/model_probe/HF_DELTA_DIAGNOSIS_2026-07-13.md` — HF Delta 诊断
- `docs/model_probe/STYLE_CEILING_LIBERATION_PLAN.md` — 风格上限解放计划

---

## 6. 待完成任务

### 6.1 等待 DINO eval 完成
- 监控 `scripts/_reeval_with_dino.py` 远程运行
- 预计 ~13 分钟完成
- 完成后读取 `Ep | CLIP-S | LPIPS | DINO-S | DINO-C` 汇总表

### 6.2 汇总最佳 checkpoint
- 综合分析 CLIP-S/LPIPS/DINO-S/DINO-C 四指标
- DINO-S 优先，CLIP-S/LPIPS/DINO-C 参考
- 注意：当前 eval 用 adain=1.0，主表报告用 2.0

### 6.3 模块清理（可选，需用户确认）
- 删除确认负面模块: `target_latent_token_fusion` (~46 行)
- 删除确认无效模块: `intrinsic_style_cnn`, `enable_hh_head`, `style_adaln`, `CFG` (~92 行)
- **保留有效模块**: `target_latent_hf_subband_fusion`, `target_latent_hf_head_fusion`
- 清理 25 个孤儿配置标志

### 6.4 重要注意事项
1. **不要删除有效模块**: `target_latent_hf_subband_fusion` 和 `target_latent_hf_head_fusion` 虽然当前 brk_a 未启用，但 probe 证明有效（DINO-S +0.002~0.003），是下一步改进方向
2. **endpoint_adain_scale**: 当前 eval 用 1.0，主表报告用 2.0，对比时注意
3. **远程 GPU**: `ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62` (RTX 3060 12GB)

---

## 7. 建议下一步

1. **启用有效模块**: 在 brk_a 配置中启用 `target_latent_hf_subband_fusion`（DINO-S 预期 +0.003）
2. **LAH 实验**: 按 `STYLE_CEILING_LIBERATION_PLAN.md` 的 Priority 1，实现 LL Appearance Residual Head
3. **清理死代码**: 删除确认负面/无效模块（~138 行），保留有效模块代码
4. **Git 提交**: 清理后提交，记录 15 epoch 验证结果
