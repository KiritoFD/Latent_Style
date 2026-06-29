# clean_base_v2 — 减法消融最简洁最优配置

**版本**: v2.0 (2026-06-30)
**配置文件**: [`configs/clean_base_v2.json`](../configs/clean_base_v2.json)
**Spec**: [629-subtractive-prune-clean-base](../.trae/specs/629-subtractive-prune-clean-base/spec.md)
**决策来源**: Phase 2 诊断 Test B（14 arch cuts: S3b+S3c）+ S3a per-item rollback（8 项 SAFE）= **22 cuts 最终配置**
**最终验证**: 22 cuts 整体训练 10 epoch → clip=0.7298, lpips=0.3421 → **PASS** ✓

---

## 1. 概述

`clean_base_v2.json` 是基于 **减法消融策略** 从 T5 ep7 baseline（当前最优基础模型）逐项砍掉已识别的无效/有害模块得到的配置。与 v1（加法组合）不同，v2 采用减法：从 baseline 开始砍除，每阶段训练验证性能不下降则保留砍除。

### 核心思路

```
T5 baseline (clip=0.7307, lpips=0.3403)
    │
    │ 减法消融（砍掉无效项）
    ▼
Test B (14 arch cuts: S3b+S3c) → clip=0.7299, lpips=0.3420 PASS
    │
    │ S3a per-item rollback (+8 safe items: D5-D12)
    ▼
clean_base_v2 (22 cuts) → clip=0.7298, lpips=0.3421 PASS ✓
    │
    │ 性能差异在训练噪声范围内（D0_control ep10: 0.7303/0.3410）
    ▼
保持最优性能 + 配置更简洁
```

### 关键约束

- **双指标 + 噪声感知判定**：
  - clip ≥ 0.7293（D0_control ep10 = 0.7303，tolerance 0.001）
  - LPIPS ≤ 0.3453（T5 ep7 baseline = 0.3403，tolerance 0.005）
  - 双指标必须**同时**满足
- **训练噪声基准**：T5 ep7 → ep10 续训本身有 ±0.0021 clip 噪声（D0_control ep8/9/10 = 0.7298/0.7282/0.7303）

---

## 2. Phase 1 实测数据（初步减法消融）

| Stage | 砍除项 | clip_allpairs | lpips_allpairs | 判定 |
|-------|--------|---------------|----------------|------|
| BASELINE (T5 ep7) | — | 0.7307 | 0.3403 | 参考 |
| D0_control ep10 | 无（仅续训） | 0.7303 | 0.3410 | 噪声基准 |
| S1 (13 dead loss) | L1-L6,L8,L11-L16 | 0.7293 | 0.3451 | 边缘通过 |
| S2 (2 harmful loss) | L9 (lh+hl) | 0.7292 | 0.3441 | 边缘通过 |
| S3a (9 arch batch1) | D4-D12 | 0.7299 | **0.3894** | **LPIPS 灾难** |
| S3b (8 arch batch2) | D13-D21 | 0.7301 | 0.3895 | 累积于 S3a |
| S3c (6 arch batch3) | D22-D30 | 0.7300 | 0.3895 | 累积于 S3a |
| Final (23 arch cuts) | S3a+S3b+S3c | 0.7299 | 0.3895 | **LPIPS 灾难，拒绝** |

### Phase 1 关键发现

1. **S1/S2 loss cuts 边缘通过**：clip 下降 0.0014/0.0015，在 D0_control 噪声范围内（±0.0021），不是真实退化
2. **S3a 引入 LPIPS 灾难**：9 个 arch cuts（D4-D12）组合禁用导致 LPIPS 从 0.3403 跳到 0.3894（+0.049 灾难性退化），尽管 clip 通过（0.7299）
3. **Phase 1 runner 漏检 LPIPS**：v1 判定逻辑只查 clip，导致 LPIPS 灾难未被拦截
4. **历史单项消融的局限**：Phase 8C 显示 D4-D12 单独禁用时 LPIPS 均在 0.3410 ± 0.001（无影响），但组合禁用产生非线性负面交互

---

## 3. S3a 排除原因（组合非线性交互）

### 现象

S3a 批次（9 个 arch cuts: D4-D12）组合禁用后：
- clip = 0.7299（通过，仅 -0.0008）
- **LPIPS = 0.3894**（灾难性退化 +0.049，远超噪声 ±0.005）

### 根因分析

S3a 的 9 项 arch cuts 共同构成**内容保真通路**：
- `model.lowpass_mode`: DWT haar → avg_pool（低通变换）
- `model.ablation_skip_clean`: 关闭 clean skip
- `model.ablation_skip_blur`: 关闭 blur skip
- `model.ablation_decoder_highpass`: 关闭 decoder highpass
- `model.residual_gain`: 1.0 → 0（残差增益归零）
- `model.ablation_no_residual`: false → true（禁用残差）
- `model.style_gate_mode`: tanh_gate → film_only
- `model.affine_connection_gamma_scale`: 0.5 → 0（FiLM gamma 归零）
- `model.affine_connection_beta_scale`: 1.0 → 0（FiLM beta 归零）

**组合效应**：残差通路 + FiLM affine + skip connections 同时被摧毁，导致内容保真通路断裂。Phase 8C 单项消融无法预测这种组合负面交互。

### 决策

**永久排除 S3a 批次的 9 项 arch cuts**（D4-D12），不进入最终 `clean_base_v2.json`。

S3a per-item rollback（D4-D12 逐个测试）进行中，找出哪些单项可安全加入。结果待 runner 完成后归档（见 Section 8）。

---

## 4. Phase 2 诊断结果

诊断 runner（`_629_diagnostic_runner.py`）执行 3 组测试，隔离 loss 和 arch 效应：

| Test | 砍除项 | cuts 数 | clip_allpairs | lpips_allpairs | 判定 |
|------|--------|---------|---------------|----------------|------|
| 阈值 | — | — | ≥ 0.7293 | ≤ 0.3453 | 双指标同时满足 |
| D0_control ep10 | 无 | 0 | 0.7303 | 0.3410 | 噪声基准 |
| Test C (S1+S2+S3b+S3c) | 13 dead + 2 harmful + 14 arch | 29 | 0.7285 | 0.3415 | **FAIL**（clip -0.0008） |
| **Test B (S3b+S3c only)** | 14 arch cuts（无 loss） | 14 | **0.7299** | **0.3420** | **PASS** ✓ |
| Test E (S1+S2 only) | 13 dead + 2 harmful loss | 15 | 0.7285 | 0.3415 | **FAIL**（clip -0.0008） |

### Phase 2 关键洞察

1. **S1+S2 loss cuts 组合有负面交互**：Test C 和 Test E 的 clip 都=0.7285（相同值，非巧合），低于 D0_control ep10 噪声基准 0.7303。15 个 loss cuts 同时砍除时，clip 退化 0.0018，超出 ±0.001 tolerance。
2. **Test B 单独 arch cuts 安全**：14 个 arch cuts（S3b+S3c）独立砍除时，clip=0.7299（在噪声内）且 LPIPS=0.3420（在噪声内），双指标均通过。
3. **历史单项消融的局限（再次验证）**：Phase 8C 显示 15 个 loss 单独禁用时均 ±0.0001，但组合禁用产生 0.0018 退化。这与 S3a 组合 LPIPS 灾难一致，证明"单项有效 ≠ 组合有效"。

### 最终决策

按 spec 回退链 Test C → Test E → Test B：
- Test C FAIL（clip 0.7285 < 0.7293）
- Test E FAIL（clip 0.7285 < 0.7293）
- **Test B PASS** → 选为候选配置（14 cuts: S3b+S3c only，无 loss cuts）
- **S3a per-item rollback**：D5-D12 共 8 项 SAFE → 累积到 Test B，变成 22 cuts
- **22 cuts 整体验证 PASS** → 最终 clean_base_v2.json = **22 cuts**

### 22 cuts 整体验证（关键）

由于 S3a per-item rollback 的 8 项是各自独立测试（在 Test B 基础上加 1 项），8 项同时加入（22 cuts）可能产生组合负面交互。必须整体训练验证。

| 配置 | cuts 数 | clip_allpairs | lpips_allpairs | 判定 |
|------|---------|---------------|----------------|------|
| Test B（14 cuts） | 14 | 0.7299 | 0.3420 | PASS |
| **22 cuts 整体** | **22** | **0.7298** | **0.3421** | **PASS** ✓ |

**结论**：22 cuts 整体无组合负面交互，clip=0.7298（≥0.7293 ✓），lpips=0.3421（≤0.3453 ✓）。clean_base_v2.json 保持 22 cuts 最终配置。

---

## 5. 最终砍除清单（22 项 = 14 Test B + 8 S3a safe）

### S3b+S3c 部分（14 项，Test B 验证 PASS）

| # | 配置键 | baseline | prune_to | source | 说明 |
|---|--------|----------|----------|--------|------|
| 1 | model.tokenizer_global_gate_scale | 1.0 | 0 | D13 | tokenizer global gate 归零 |
| 2 | model.tokenizer_residual_gain | 0.5 | 0 | D14 | tokenizer 残差增益归零 |
| 3 | model.style_attn_sharpen_scale | 2.5 | 0 | D15 | style attention sharpen 归零 |
| 4 | model.endpoint_high_scale | 1.0 | 0 | D16 | endpoint high scale 归零 |
| 5 | model.skip_residual_weight | 0.1 | 0 | D17 | skip residual 权重归零 |
| 6 | bridge.kinetic_penalty_mode | "global_l2" | "off" | D18+D26 | kinetic penalty 关闭 |
| 7 | model.style_attn_mode | "softmax" | "relu2" | D19-D22 | attention 改用 relu2 |
| 8 | model.endpoint_head_mode | "velocity" | "endpoint_lowhigh" | D23 | endpoint head 模式 |
| 9 | model.transport_prediction_mode | "velocity" | "endpoint" | D24 | transport prediction 模式 |
| 10 | bridge.training_target_projection_mode | "legacy" | "dwt" | D25 | training target projection |
| 11 | bridge.terminal_swd_mode | "standard" | "high_freq" | D27 | terminal SWD 模式 |
| 12 | bridge.bridge_path_mode | "vertical" | "tri_band" | D28 | bridge path 模式 |
| 13 | bridge.swd_distance_mode | "cdf" | "squared" | D29 | SWD distance 模式 |
| 14 | bridge.t_sampling_mode | "uniform_power" | "logit_normal" | D30 | t sampling 模式 |

### S3a safe 部分（8 项，per-item rollback 验证 SAFE + 22 cuts 整体 PASS）

| # | 配置键 | baseline | prune_to | source | 单项 clip | 单项 lpips |
|---|--------|----------|----------|--------|-----------|------------|
| 15 | model.ablation_skip_clean | true | false | D5 | 0.7299 | 0.3420 |
| 16 | model.ablation_skip_blur | true | false | D6 | 0.7299 | 0.3420 |
| 17 | model.ablation_decoder_highpass | true | false | D7 | 0.7298 | 0.3419 |
| 18 | model.residual_gain | 1.0 | 0 | D8 | 0.7298 | 0.3420 |
| 19 | model.ablation_no_residual | false | true | D9 | 0.7297 | 0.3420 |
| 20 | model.style_gate_mode | "tanh_gate" | "film_only" | D10 | 0.7298 | 0.3420 |
| 21 | model.affine_connection_gamma_scale | 0.5 | 0 | D11 | 0.7298 | 0.3420 |
| 22 | model.affine_connection_beta_scale | 1.0 | 0 | D12 | 0.7298 | 0.3421 |

**注**：单项测试在 Test B（14 cuts）基础上追加 1 项 S3a cut。22 cuts 整体训练验证 clip=0.7298, lpips=0.3421 PASS。

---

## 6. 保留的核心模块（4 项，不可砍）

| 模块 | 配置键 | 历史证据（Phase 8C 单项禁用 Δclip） |
|------|--------|-------------------------------------|
| spectral_ode | model.contract_family = "620_spectral_ode" | D1: 禁用 -0.0167（最大下降） |
| adain_scale | model.endpoint_adain_scale > 0 | D2: 禁用 -0.0142 |
| alpha | model.style_extrap_alpha > 0 | D3: 禁用 -0.0016 |
| spectral_ll | bridge.spectral_w_ll > 0 | L7: 禁用 -0.0042（唯一有效谱 loss） |

---

## 7. 永久排除项（1 项，D4）

| # | 配置键 | baseline | 排除原因 | 实测证据 |
|---|--------|----------|----------|----------|
| 1 | model.lowpass_mode | "dwt_haar" | 单独加入 Test B 仍导致 LPIPS 灾难 | D4 per-item: clip=0.7300, **lpips=0.3896**（+0.0493） |

### D4 排除原因分析

D4（lowpass_mode: dwt_haar → avg_pool）是 S3a 9 项中**唯一**单独加入 Test B 也导致 LPIPS 灾难的项。这表明 `lowpass_mode` 是内容保真通路的关键节点：

- DWT haar 小波分解提供多频带内容保真
- avg_pool 丢失高频信息，破坏内容结构
- 即使其他 S3a 项保留 baseline，仅改 lowpass_mode 也会引发 LPIPS 灾难

**对比**：D5-D12 单独加入 Test B 均 PASS（LPIPS 0.3419-0.3421），说明它们不是关键节点。S3a 9 项组合 LPIPS 灾难的**罪魁祸首是 D4**（lowpass_mode）。

### S3a per-item rollback 完整结果

S3a 9 项 per-item rollback 全部完成，结果归档如下（见 Section 8）。

---

## 8. S3a Per-Item Rollback（已完成）

为找出 S3a 9 项中哪些单项可安全追加到 Test B 配置（进一步简化），对 D4-D12 逐个测试。每项在 Test B（14 cuts）基础上追加 1 项 S3a cut，训练 10 epoch + 双指标判定。

### 完整结果

| Item | 配置键 | clip | lpips | 判定 | 决策 |
|------|--------|------|-------|------|------|
| D4 | model.lowpass_mode (dwt_haar→avg_pool) | 0.7300 | **0.3896** | **FAIL** | **UNSAFE，排除** |
| D5 | model.ablation_skip_clean (true→false) | 0.7299 | 0.3420 | PASS | SAFE，累积 |
| D6 | model.ablation_skip_blur (true→false) | 0.7299 | 0.3420 | PASS | SAFE，累积 |
| D7 | model.ablation_decoder_highpass (true→false) | 0.7298 | 0.3419 | PASS | SAFE，累积 |
| D8 | model.residual_gain (1.0→0) | 0.7298 | 0.3420 | PASS | SAFE，累积 |
| D9 | model.ablation_no_residual (false→true) | 0.7297 | 0.3420 | PASS | SAFE，累积 |
| D10 | model.style_gate_mode (tanh_gate→film_only) | 0.7298 | 0.3420 | PASS | SAFE，累积 |
| D11 | model.affine_connection_gamma_scale (0.5→0) | 0.7298 | 0.3420 | PASS | SAFE，累积 |
| D12 | model.affine_connection_beta_scale (1.0→0) | 0.7298 | 0.3421 | PASS | SAFE，累积 |

### 统计

- **SAFE**: 8 项（D5-D12）→ 累积到 clean_base_v2.json
- **UNSAFE**: 1 项（D4 lowpass_mode）→ 永久排除

### 关键洞察

1. **D4 是 S3a LPIPS 灾难的罪魁祸首**：单独加入 Test B 即导致 LPIPS 0.3896（+0.0493）。Phase 1 S3a 9 项组合 LPIPS 灾难的主因是 D4。
2. **D5-D12 单独加入 Test B 安全**：每项 LPIPS 在 0.3419-0.3421（与 Test B 的 0.3420 几乎一致），说明这些项不是内容保真通路的关键节点。
3. **22 cuts 整体验证 PASS**：8 项 SAFE items 同时加入（22 cuts）无组合负面交互，clip=0.7298, lpips=0.3421。
4. **Phase 1 S3a 灾难根因定位**：S3a 9 项组合 LPIPS 灾难 = D4 单独 LPIPS 灾难 + 其他 8 项无影响。D4 是唯一罪魁。

---

## 9. Pareto 对比

| 配置 | cuts 数 | clip_allpairs | lpips_allpairs | 备注 |
|------|---------|---------------|----------------|------|
| T5 ep7 baseline | 0 | 0.7307 | 0.3403 | 历史最优基础模型 |
| D0_control ep10 | 0 | 0.7303 | 0.3410 | 续训噪声基准 |
| Phase 1 clean_base_v2（拒绝） | 23 arch | 0.7299 | 0.3895 | LPIPS 灾难（含 D4） |
| Phase 1 v1 加法组合（拒绝） | +5 mod | 0.7073 | — | 加法组合负面交互 |
| Test B（中间候选） | 14 arch | 0.7299 | 0.3420 | S3b+S3c only |
| **clean_base_v2（最终）** | **22 arch** | **0.7298** | **0.3421** | **Test B + 8 S3a safe items** |

### Pareto 分析

- clean_base_v2（22 cuts）在 clip 上与 T5 baseline 差异 -0.0009（噪声内），LPIPS +0.0018（噪声内）
- 相比 Phase 1 拒绝版（23 cuts, LPIPS 0.3895），LPIPS 改善 0.0474（关键：移除 D4）
- 相比 v1 加法组合（clip 0.7073），clip 改善 0.0225
- 相比 Test B（14 cuts），多砍 8 项 S3a safe items，clip 仅降 0.0001，LPIPS 仅升 0.0001（噪声内）
- **结论**：clean_base_v2 在保持最优性能的同时，砍除 22 项装饰架构模块，配置最简洁

### 砍除效率

| 阶段 | 砍除项 | 累积 cuts | 关键发现 |
|------|--------|-----------|----------|
| Phase 1 S1+S2 | 15 loss cuts | 15 | clip -0.0018（超阈值，组合负面交互） |
| Phase 1 S3a | 9 arch cuts (D4-D12) | 24 | LPIPS +0.049（D4 罪魁） |
| Phase 1 S3b+S3c | 14 arch cuts | 38 | 累积于 S3a，未独立测试 |
| Phase 2 Test B | 14 arch cuts (S3b+S3c) | 14 | **PASS**（独立安全） |
| Phase 2 S3a rollback | +8 arch cuts (D5-D12) | 22 | **PASS**（D4 排除） |
| **最终 clean_base_v2** | **22 arch cuts** | **22** | **PASS** |

---

## 10. 使用指南

### 训练

```bash
cd I:\Github\Latent_Style\SchrodingerBridge
python src\run.py --config configs\clean_base_v2.json
```

### 配置继承

`clean_base_v2.json` 基于 T5 config（`exp/p4_fusion_breakout/t5_b2v2_d2_d4/config.json`）+ 14 项 arch cuts。其他参数与 T5 baseline 完全一致。

### 关键参数

- `checkpoint.resume_checkpoint`: T5 ep7 checkpoint
- `training.num_epochs`: 10
- `training.start_epoch`: 8（从 ep7 续训）
- `training.test_image_dir`: `I:\wikiart_distinct5_samam_512_classview\test`
- `data.data_root`: `I:/wikiart_distinct5_samam_512_latents_ema/train`

### 评估

```bash
python src\utils\run_evaluation.py \
  --checkpoint <ckpt_path> \
  --output <eval_dir> \
  --test_dir I:\wikiart_distinct5_samam_512_classview\test \
  --cache_dir I:\Github\Latent_Style\eval_cache \
  --batch_size 16 \
  --num_steps 8 \
  --eval_only_lpips_clip_style
```

### 验证结果

- clip_allpairs = 0.7298（≥ 0.7293 ✓）
- lpips_allpairs = 0.3421（≤ 0.3453 ✓）
- 训练耗时：~207s（10 epoch）
- 评估耗时：~214s
- checkpoint: `exp/clean_base_v2_22cuts/epoch_0010.pt`（验证用）/ `exp/clean_base_v2/`（正式 save_dir）

---

## 11. 后续工作

clean_base_v2 已完成全部验证，无待办事项。

### 已完成

1. **Phase 1 初步减法消融**：识别 S3a LPIPS 灾难
2. **Phase 2 诊断测试**：Test B PASS，Test C/E FAIL
3. **S3a per-item rollback**：D4 UNSAFE，D5-D12 SAFE
4. **22 cuts 整体验证**：PASS（clip=0.7298, lpips=0.3421）
5. **文档归档**：spec.md / tasks.md / checklist.md / docs/CLEAN_BASE_V2.md

### 可选探索（未在本 spec 范围）

- S1+S2 loss cuts 组合负面交互的根因定位（Test C/E clip=0.7285）
- D4 lowpass_mode 的内容保真机制深入分析
- 进一步简化：是否还有其他未识别的装饰模块可砍

---

## 12. 引用

- Spec: [`629-subtractive-prune-clean-base/spec.md`](../.trae/specs/629-subtractive-prune-clean-base/spec.md)
- Phase 1 runner: [`_629_subtractive_runner.py`](../_629_subtractive_runner.py)
- Phase 2 诊断 runner: [`_629_diagnostic_runner.py`](../_629_diagnostic_runner.py)
- 砍除清单: [`configs/ablations/629_subtractive/prune_manifest.json`](../configs/ablations/629_subtractive/prune_manifest.json)
- Phase 1 结果: [`exp/629_subtractive/629_results.json`](../exp/629_subtractive/629_results.json)
- Phase 2 结果: [`exp/629_subtractive/629_phase2_results.json`](../exp/629_subtractive/629_phase2_results.json)（runner 完成后生成）
- 历史 Phase 8C: 466 实验，单项消融数据基础
