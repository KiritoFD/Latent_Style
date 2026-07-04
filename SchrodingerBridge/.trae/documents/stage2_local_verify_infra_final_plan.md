# Stage 2 清理收尾 + 全本地验证 + Infra 优化（最终版）

## 摘要

承接 `stage2_local_verify_and_infra_plan.md`：完成 Stage 2B 最后一项 `losses620.py` 默认值更新（bridge_path_mode）；**所有后续验证全部在本地 4070 Laptop 上完成**（已验证可复现远程 0.73 baseline）；Stage 3 调整为只清理 JSON 死字段（src/ 无代码消费）；Infra 优化删除 13 个本次会话临时脚本 + 创建一键 `tools/local_train_and_eval.py`；最后 git commit。

## 当前状态分析

### Stage 2A 已完成 ✅
- `src/config_schema.py` 14 项默认值已全部更新为 prune_to
- smoke test ALL PASS（1,694,292 params，loss=7.652607，GPU 37.3MB）

### Stage 2B 进度（5/6 完成）

| 文件 | 状态 | 备注 |
|------|------|------|
| `src/model620.py` | ✅ 完成 | 5 处条件分支简化（attn_mode 硬编码、endpoint_head_mode 分支删除、debug metrics 简化） |
| `src/model.py` | ✅ 完成 | 7 处 transport_prediction_mode 条件分支简化 |
| `src/lancet_backbone.py` | ✅ 完成 | 5 处默认值硬编码为 prune_to（D13/D14/D15/D17 + structured_global_gate_scale） |
| `src/lancet_blocks.py` | ✅ 完成 | 2 处函数签名默认值改为 0.0（style_attn_sharpen_scale） |
| `src/losses620.py` | 🔄 2/3 完成 | training_target_projection_mode → "tri_band_wavelet" ✅；t_sampling_mode → "logit_normal" ✅；bridge_path_mode 待做 |
| `src/trainer.py` | ⬜ 不动 | metrics 占位行保留，无功能影响 |

### 本地验证 baseline（已确立）

- **allpairs clip_style = 0.7293**（远程 0.7299，Δ=0.0006 噪声范围）✅
- **allpairs content_lpips = 0.3203**（远程 0.3420，本地略好）✅
- 训练：3 分钟/10epoch，显存峰值 0.36 GB（4070 Laptop 8GB 完全够用）
- Eval：约 3 分钟，750 张 allpairs
- 数据：5000 张 latent 已编码至 `G:/GitHub/Latent_Style/Dataset/distinct5_512_latents_ema/`
- 配置：`configs/clean_base_v2_local.json` 已生成（路径指向 G 盘，resume_checkpoint 为空）

### 关键架构发现（重申）

- **SpectralODEBridge620**（clean_base_v2 实际使用的模型）继承 `nn.Module`，**不读取**这 14 项配置中的任何一项
- **SpectralODEObjective620**（clean_base_v2 实际使用的 loss）在 `src/spectral_losses620.py` 中，**不读取** `bridge_path_mode`/`training_target_projection_mode`/`t_sampling_mode`/`tri_band_inference_lock`（Phase 1 grep 确认 0 匹配）
- 14 项 baseline 分支只在遗留代码路径上（model620.py, model.py, lancet_*.py, losses620.py）
- **结论：清理这 14 项 baseline 分支不会影响 clean_base_v2 的性能**（不在其代码路径上）
- 清理的目的是 **codebase 卫生**：让默认行为唯一指向 prune_to 值

### Phase 1 新发现

1. **`tri_band_inference_lock` 完全是死配置**：src/ 中 0 处代码消费（已 grep 确认），只在 11 个 JSON 中存在。Stage 3 调整为只删 JSON 字段，无需删代码分支。
2. **`bridge_path_mode` 在 losses620.py 中只 3 处使用**：行 134-136（init+校验）+ 行 339（spherical_vp 分支判断）。
3. **本次会话产生的 13 个临时脚本已确认存在**（详见 Infra 节）。
4. **`_remote_smoke_train.py`** 内容是本地 CUDA smoke test（名字带 remote 是历史遗留），代码 73 行，功能单一。
5. **tools/ 目录已存在**，可放置 `local_train_and_eval.py`。

## 提议变更

### Stage 2B 收尾：losses620.py bridge_path_mode（第 3/3 项）

**问题**：当前代码（行 134-136）
```python
self.bridge_path_mode = str(getattr(self.bridge_cfg, "bridge_path_mode", "linear")).strip().lower()
if self.bridge_path_mode not in {"linear", "spherical_vp"}:
    self.bridge_path_mode = "linear"
```
- prune_to 值是 `"tri_band"`，但允许集合只有 `{"linear", "spherical_vp"}`
- clean_base_v2_local.json 显式设置 `"bridge_path_mode": "tri_band"` → 会被校验拒绝 → fallback 到 `"linear"`
- 然而 SpectralODEObjective620（clean_base_v2 实际使用的 loss）不读取此字段，所以**不影响性能**，只是默认行为不一致

**修改方案**（最小修改，不删分支）：
```python
# 行 134: 默认值 "linear" → "tri_band"
self.bridge_path_mode = str(getattr(self.bridge_cfg, "bridge_path_mode", "tri_band")).strip().lower()
# 行 135-136: 允许集合增加 "tri_band"
if self.bridge_path_mode not in {"linear", "spherical_vp", "tri_band"}:
    self.bridge_path_mode = "tri_band"
```

**策略**：保持最小修改（不删 spherical_vp 分支、不删条件分支），仅更新默认 fallback 值与允许集合。
**理由**：(1) losses620.py 的 `SpatialBridgeObjective620` 不被 clean_base_v2 使用；(2) 删除分支可能破坏遗留实验兼容性；(3) 用户偏好"无效代码直接删除"但此字段并非完全无效（vertical/latent_slerp 路径在 losses.py 中仍活跃，losses620.py 是 620 变体）。

### Stage 2C: 本地验证（全部本地，无远程）

**全部在 4070 Laptop 上执行**，不触碰远程 3060。

1. **Smoke test**（30s timeout 适用，命令短）
   ```powershell
   python _remote_smoke_train.py
   ```
   验证 forward+backward+optimizer 通过，无 shape/dtype 错误。

2. **训练 10 epoch**（约 3 分钟，**不设 30s timeout**，长命令单独执行）
   ```powershell
   python src\run.py --config configs\clean_base_v2_local.json
   ```
   - 验证 loss 收敛曲线与上次 baseline 一致
   - 显存峰值 ≤ 1.0 GB
   - 生成 checkpoint 到 `G:/GitHub/Latent_Style/SchrodingerBridge/exp/clean_base_v2_local/`

3. **评估**（约 3 分钟，**不设 30s timeout**）
   ```powershell
   python src\utils\run_evaluation.py --checkpoint <latest_ckpt> --output <eval_dir> --eval_only_lpips_clip_style
   ```
   **注意**：不加 `--transfer_only` flag（前次踩坑：会导致 allpairs 不完整计算）

4. **验收标准**
   - allpairs clip_style ≥ 0.7243（baseline 0.7293 - 5σ = 0.005）
   - allpairs content_lpips ≤ 0.3453（baseline 0.3203 + 噪声余量）
   - 如不达标：回退本次修改，逐文件 bisect 定位回归源

### Stage 3: tri_band_inference_lock 死字段清理（调整后）

**Phase 1 确认**：src/ 中 0 处代码消费此字段，是纯死配置。

**修改范围**：仅清理 3 个 active 配置 JSON（不动 src/exp/ 历史 snapshot）：
- `configs/clean_base_v2_local.json`（行 268）
- `configs/clean_base_v2.json`
- `configs/clean_base.json`

**操作**：从这 3 个 JSON 中删除 `"tri_band_inference_lock": false` 行。

**不动的 8 个 JSON**：`configs/ablations/628_train_smoke/T*.json`（历史 snapshot，遵循"不动历史"原则；这些配置不会再次运行）。

**验证**：smoke test 通过即可（因为无代码消费此字段，删除 JSON 字段对运行无影响）。

### Infra 优化（轻量级）

#### 1. 清理本次会话产生的临时脚本（13 个）

经 LS 确认，本次会话产生的临时脚本清单：

**根目录 11 个脚本**：
- `_probe_gpu_log.ps1`
- `_probe_remote_cache.ps1`
- `_probe_local_data.ps1`
- `_probe_remote_latents.py`
- `_probe_local_latents.py`
- `_compare_eval_summary.py`
- `_compare_testsets.py`
- `_dump_local_testfiles.py`
- `_dump_remote_testfiles.ps1`
- `_run_probe_remote.ps1`
- `_run_compare_eval.ps1`

**根目录 2 个文本产物**：
- `_local_test_files.txt`
- `_remote_test_files.txt`

**tools 目录 1 个**：
- `tools/_make_local_config.py`（一次性配置生成器，配置已生成完毕）

**Infra 阶段最后删除**：
- `_remote_smoke_train.py`（Stage 2C smoke test 仍用，最后由 local_train_and_eval.py 取代后删除）

**总计 14 个文件删除**。

**不删历史脚本**：根目录还有约 100+ 个历史 `_*.py`/`_*.ps1`（628/629/N1/T4/B2 等阶段产物），不在本次任务范围。

**保留**：
- `configs/clean_base_v2.json`（远程参考配置）
- `configs/clean_base_v2_local.json`（本地配置）

#### 2. 创建本地一键脚本 `tools/local_train_and_eval.py`

**设计目标**：封装本地 smoke + 训练 + 评估流程，替代 `_remote_smoke_train.py`，提供统一入口。

**接口设计**：
```python
# 用法:
#   python tools/local_train_and_eval.py --config configs/clean_base_v2_local.json
#   python tools/local_train_and_eval.py --config <cfg> --skip-train --checkpoint <path>
#   python tools/local_train_and_eval.py --config <cfg> --skip-eval
#   python tools/local_train_and_eval.py --config <cfg> --smoke-only

# 功能:
# 1. 解析 config 路径，验证存在
# 2. --smoke-only: 跑 forward+backward+optimizer 单步验证（继承 _remote_smoke_train.py 逻辑）
# 3. 调用 src/run.py 训练（subprocess，无 timeout，实时打印日志）
# 4. 从 exp/<config_name>/ 中找最新 checkpoint
# 5. 调用 src/utils/run_evaluation.py 评估（subprocess，无 timeout）
# 6. 打印 allpairs clip_style / content_lpips 指标
# 7. 与 baseline (0.7293 / 0.3203) 对比，打印 PASS/FAIL
```

**实现要点**：
- 使用 `subprocess.run` + `sys.executable`，避免硬编码 python 路径
- 实时透传 stdout/stderr（不捕获，让用户看到训练进度）
- 失败时返回非零 exit code
- 不引入新依赖，纯标准库
- smoke 模式直接 import 模型 + loss 跑单步（无 subprocess）

#### 3. 文档与配置整理

- 不创建新文档（遵循"NEVER proactively create documentation files"）
- `docs/CLEAN_BASE_V2.md` 已存在且准确，无需更新

## 假设与决策

1. **假设**: SpectralODEObjective620 不读取 bridge_path_mode（Phase 1 grep 确认 0 匹配）
2. **决策**: losses620.py 采用最小修改策略 - 只更新 3 个默认 fallback 值，不删分支 - 避免破坏遗留实验兼容性
3. **决策**: bridge_path_mode 默认值改为 "tri_band" 并加入允许集合 - 让默认行为唯一指向 prune_to 值
4. **决策**: tri_band_inference_lock 只清理 3 个 active JSON，不动 src/exp/ 历史 snapshot
5. **决策**: 本地验证全部在 4070 Laptop 上做 - 已验证可复现远程 0.73 baseline
6. **决策**: Infra 优化只清理本次会话产生的临时脚本 - 不动历史脚本（不在范围）
7. **决策**: 一键脚本集成 smoke/train/eval 三模式 - 替代 _remote_smoke_train.py，提供统一入口
8. **决策**: 一键脚本不引入新依赖，纯标准库 subprocess
9. **验证标准**: allpairs clip_style ≥ 0.7243（baseline 0.7293 - 5σ）

## 验证步骤

1. ✅ Stage 2B losses620.py bridge_path_mode 默认值更新后，`python _remote_smoke_train.py` 通过
2. ✅ Stage 2C: 本地训练 10 epoch + full_eval，allpairs clip_style ≥ 0.7243
3. ✅ Stage 3: 3 个 JSON 删除 tri_band_inference_lock 后，smoke test 通过
4. ✅ Infra: 14 个临时文件清理后，`python tools/local_train_and_eval.py --config configs/clean_base_v2_local.json --smoke-only` 验证一键脚本可用
5. ✅ git commit

## 执行顺序

1. **Stage 2B 收尾**: `src/losses620.py` 行 134-136 的 bridge_path_mode 默认值和允许集合更新
2. **Stage 2C 本地验证**: smoke + train 10 epoch + eval（allpairs clip ≥ 0.7243）
3. **Stage 3**: 从 3 个 active JSON 中删除 tri_band_inference_lock 字段 + smoke test 验证
4. **Infra 优化**:
   - 创建 `tools/local_train_and_eval.py`
   - 用 `--smoke-only` 验证一键脚本可用
   - 删除 14 个临时文件（11 根目录脚本 + 2 文本产物 + tools/_make_local_config.py + _remote_smoke_train.py）
5. **git commit**: 提交所有变更

## 风险与回退

- **风险**: Stage 2B losses620.py 修改可能影响遗留 SpatialBridgeObjective620 路径
- **回退**: smoke test 失败时，git checkout losses620.py 后保持原默认值 "linear"（不影响 clean_base_v2 因为 SpectralODEObjective620 不读此字段）
- **风险**: Stage 2C 训练指标不达标
- **回退**: 逐文件 bisect（git stash 部分修改），定位回归源
- **风险**: Stage 3 删除 JSON 字段影响其他配置消费者
- **回退**: 已 grep 确认 src/ 中 0 处消费，无风险；如意外，git checkout 对应 JSON
- **风险**: 一键脚本与现有 run.py 入口不兼容
- **回退**: 退化为直接调用 `python src/run.py` + `python src/utils/run_evaluation.py`，保留 _remote_smoke_train.py 不删
