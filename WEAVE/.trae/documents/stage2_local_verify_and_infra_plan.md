# Stage 2 清理 + 本地验证 + Infra 优化计划（更新版）

## 摘要

承接 `stage2_cleanup_and_local_verify.md`：继续完成 Stage 2B 遗留代码 baseline 分支删除，**所有后续验证全部在本地 4070 Laptop 上完成**（已验证可复现远程 0.73 baseline），最后做轻量级 Infra 优化（清理本次会话产生的临时脚本 + 创建本地一键脚本），然后 git commit。

## 当前状态分析

### Stage 2A 已完成 ✅
- `src/config_schema.py` 14 项默认值已全部更新为 prune_to
- smoke test ALL PASS（1,694,292 params，loss=7.652607，GPU 37.3MB）

### Stage 2B 部分完成 🔄

| 文件 | 状态 | 剩余工作 |
|------|------|----------|
| `src/model620.py` | 🔄 部分完成 | 行 82-113 getattr 默认值已更新；条件分支简化（行 221, 453, 531, 536, 458, 461, 212）待做 |
| `src/model.py` | 🔄 部分完成 | 行 48-50 已更新；行 77, 1102, 1112, 1892, 1939, 1962, 1982 条件分支待简化 |
| `src/lancet_backbone.py` | ⬜ 待做 | 行 96, 161, 289, 307, 441 |
| `src/lancet_blocks.py` | ⬜ 待做 | 行 208, 218, 228, 339, 352, 666, 679, 688 |
| `src/losses620.py` | ⬜ 待做 | 行 59-70, 81, 134-136, 214, 242-315, 339 |
| `src/trainer.py` | ⬜ 待做 | metrics 占位行（保留，不动） |

### 本地验证 baseline（已确立）

- **allpairs clip_style = 0.7293**（远程 0.7299，Δ=0.0006 噪声范围）✅
- **allpairs content_lpips = 0.3203**（远程 0.3420，本地略好）✅
- 训练：3 分钟/10epoch，显存峰值 0.36 GB（4070 Laptop 8GB 完全够用）
- Eval：约 3 分钟，750 张 allpairs
- 数据：5000 张 latent 已编码至 `G:/GitHub/Latent_Style/Dataset/distinct5_512_latents_ema/`（与远程完全一致 mean=0.0737 std=0.9602）
- 配置：`configs/clean_base_v2_local.json` 已生成（路径指向 G 盘，resume_checkpoint 为空）

### 关键架构发现（重申）

- **SpectralODEBridge620**（clean_base_v2 实际使用的模型）继承 `nn.Module`，**不继承** TimeConditionedLANCETBridge/SpatialBridge620
- 它**不读取**这 14 项配置中的任何一项
- 14 项 baseline 分支只在遗留代码路径上（model620.py, model.py, lancet_*.py, losses620.py）
- **结论：清理这 14 项 baseline 分支不会影响 clean_base_v2 的性能**（不在其代码路径上）
- 清理的目的是 **codebase 卫生**：避免遗留分支误导未来开发，让默认行为唯一指向 prune_to 值

## 提议变更

### Stage 2B 剩余：遗留代码 baseline 分支删除

**策略**: 删除 baseline 分支代码，硬编码 prune_to 值。保留类结构和接口，只删分支逻辑。

#### 文件 1: `src/model620.py`（剩余工作）

```python
# 行 221, 453, 531, 536: endpoint_head_mode == "endpoint_lowhigh" 分支变为唯一路径
# 删除 if/else 条件判断，保留 endpoint_lowhigh 分支主体
# 行 458, 461: endpoint_high_scale 乘法 * 0 → 直接返回 0 或删除该路径
# 行 212: attn_mode=self.style_attn_mode → 直接传 "relu2"
```

#### 文件 2: `src/model.py`（剩余工作）

经 Grep 确认的 7 处分支：
- 行 77: `transport_prediction_mode=self.transport_prediction_mode` → 直接传 "endpoint"
- 行 1102: `if self.transport_prediction_mode == "endpoint":` 分支为唯一路径（diffeomorphic stroke 路径）
- 行 1112: `if self.transport_prediction_mode == "endpoint":` 分支为唯一路径（标准路径）
- 行 1892: `predict_transport` 中的 `if/else` → 保留 endpoint 分支，删除 else
- 行 1939: `predict_transport_base` 中的 `if/else` → 保留 endpoint 分支，删除 else
- 行 1962: `integrate_transport` 中的 `if ... and self.solver_family != "solver_i2sb":` → 简化条件
- 行 1982: `if self.solver_family == "solver_i2sb" and self.transport_prediction_mode == "endpoint":` → 简化为只判 solver_family

#### 文件 3: `src/lancet_backbone.py`

经 Grep 确认的 5 处：
- 行 96: `self.style_attn_sharpen_scale = max(0.1, float(getattr(cfg, "style_attn_sharpen_scale", 2.5)))` → 硬编码为 0
- 行 161: `self.skip_residual_weight = max(0.0, float(getattr(cfg, "skip_residual_weight", 0.1)))` → 硬编码为 0
- 行 289: `residual_gain=float(getattr(cfg, "tokenizer_residual_gain", 0.5))` → 直接传 0
- 行 307: `structured_global_gate_scale = max(0.0, float(getattr(cfg, "tokenizer_global_gate_scale", 1.0)))` → 硬编码为 0
- 行 441: `style_attn_sharpen_scale=self.style_attn_sharpen_scale` → 直接传 0

#### 文件 4: `src/lancet_blocks.py`

经 Grep 确认的 8 处：
- 行 208, 339: 函数签名 `style_attn_sharpen_scale: float = 2.0` → 默认值改为 0.0
- 行 218, 228, 352: `attn_sharpen_scale=style_attn_sharpen_scale` → 保留传参（参数本身已为 0）
- 行 666, 679, 688: 同上保留传参

**决策**: 保留函数签名参数（避免接口破坏），只把默认值改为 0。

#### 文件 5: `src/losses620.py`

经 Grep 确认的 26 处中需简化的关键分支：
- 行 59-70: `training_target_projection_mode` 默认值和校验 → 硬编码为 "dwt"
- 行 81: `t_sampling_mode` 默认值 → 硬编码为 "logit_normal"
- 行 134-136: `bridge_path_mode` 默认值和校验 → 硬编码为 "tri_band"
- 行 214: `if self.t_sampling_mode == "logit_normal":` 分支变为唯一路径
- 行 242-315: `training_target_projection_mode` 各分支简化为 "dwt" 路径（删除 legacy / source_low_target_high / pure_vertical_flow 等其他分支）
- 行 339: `if self.bridge_path_mode == "spherical_vp":` 分支删除（baseline 已改为 "tri_band"）

**注意**: 行 274-277 的 metrics 占位字典保留（不影响功能）；行 283-315 的 metrics 写入逻辑保留。

#### 文件 6: `src/trainer.py`

- **不动**：metrics 占位行保留，无功能影响。

### Stage 2C: 本地验证（全部本地）

**全部在 4070 Laptop 上执行**，不触碰远程 3060。

1. **Smoke test**（30s timeout）
   ```powershell
   python _remote_smoke_train.py
   ```
   验证 forward+backward+optimizer 通过，无 shape/dtype 错误。

2. **训练 10 epoch**（约 3 分钟，30s timeout 不适用长命令，单独执行）
   ```powershell
   python src\run.py --config configs\clean_base_v2_local.json
   ```
   - 验证 loss 收敛曲线与上次 baseline 一致
   - 显存峰值 ≤ 1.0 GB
   - 生成 checkpoint 到 `G:/GitHub/Latent_Style/SchrodingerBridge/exp/clean_base_v2_local/`

3. **评估**（约 3 分钟）
   ```powershell
   python src\utils\run_evaluation.py --checkpoint <latest_ckpt> --output <eval_dir> --eval_only_lpips_clip_style
   ```
   **注意**: 不加 `--transfer_only` flag（前次踩坑：会导致 allpairs 不完整计算）

4. **验收标准**
   - allpairs clip_style ≥ 0.7243（baseline 0.7293 - 5σ = 0.005）
   - allpairs content_lpips ≤ 0.3453（baseline 0.3203 + 噪声余量）
   - 如不达标：回退本次修改，逐文件 bisect 定位回归源

### Stage 3: tri_band_inference_lock 推理分支清理

- Grep 搜索 `tri_band_inference_lock` 在 `src/` 中的位置
- 删除推理分支代码（628 I8 已验证无效）
- 本地 smoke + eval 验证（同 Stage 2C 标准）

### Infra 优化（轻量级）

#### 1. 清理本次会话产生的临时脚本

经 LS 确认，本次会话创建的临时脚本清单（**只删这些，不动历史脚本**）：

**根目录**（11 个）：
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

**根目录其他临时文件**：
- `_local_test_files.txt`、`_remote_test_files.txt`（探测产物）

**tools 目录**（2 个）：
- `tools/_make_local_config.py`（一次性配置生成器，配置已生成完毕，可删）

**保留**：
- `_remote_smoke_train.py`（Stage 2C smoke test 仍在用，待 Infra 阶段最后再删）
- `configs/clean_base_v2.json`（远程参考配置）
- `configs/clean_base_v2_local.json`（本地配置）

**不删历史脚本**：根目录还有约 100+ 个历史 `_*.py`/`_*.ps1`（628/629/N1/T4/B2 等阶段产物），不在本次任务范围。

#### 2. 创建本地一键脚本 `tools/local_train_and_eval.py`

**设计目标**：封装本地训练+评估流程，30s timeout 保护短命令，长命令（训练/评估）单独执行无 timeout。

**接口设计**：
```python
# 用法:
#   python tools/local_train_and_eval.py --config configs/clean_base_v2_local.json
#   python tools/local_train_and_eval.py --config <cfg> --skip-train --checkpoint <path>
#   python tools/local_train_and_eval.py --config <cfg> --skip-eval

# 功能:
# 1. 解析 config 路径，验证存在
# 2. 调用 src/run.py 训练（subprocess，无 timeout，实时打印日志）
# 3. 从 exp/<config_name>/ 中找最新 checkpoint
# 4. 调用 src/utils/run_evaluation.py 评估（subprocess，无 timeout）
# 5. 打印 allpairs clip_style / content_lpips 指标
# 6. 与 baseline (0.7293 / 0.3203) 对比，打印 PASS/FAIL
```

**实现要点**：
- 使用 `subprocess.run` + `sys.executable`，避免硬编码 python 路径
- 实时透传 stdout/stderr（不捕获，让用户看到训练进度）
- 失败时返回非零 exit code
- 不引入新依赖，纯标准库

#### 3. 文档与配置整理

- 确认 `docs/CLEAN_BASE_V2.md` 仍准确（无需更新则跳过）
- 不创建新文档（遵循"NEVER proactively create documentation files"）

## 假设与决策

1. **假设**: 14 项 baseline 分支不在 SpectralODEBridge620 代码路径上（已通过继承关系确认）
2. **决策**: 保留遗留文件类定义（model620.py, losses620.py 等），只删除 baseline 分支代码 - build_model_from_config 入口点仍需要保留
3. **决策**: lancet_blocks.py 的函数签名参数保留，只改默认值 - 避免接口破坏
4. **决策**: trainer.py 的 metrics 占位行保留 - 不影响功能
5. **决策**: 本地验证全部在 4070 Laptop 上做 - 已验证可复现远程 0.73 baseline
6. **决策**: Infra 优化只清理本次会话产生的临时脚本 - 不动历史脚本（不在范围）
7. **决策**: 一键脚本不引入新依赖，纯标准库 subprocess - 保持 codebase 干净
8. **验证标准**: allpairs clip_style ≥ 0.7243（baseline 0.7293 - 5σ）

## 验证步骤

1. ✅ Stage 2B 剩余 5 个文件清理后，`python _remote_smoke_train.py` 通过
2. ✅ Stage 2C: 本地训练 10 epoch + full_eval，allpairs clip_style ≥ 0.7243
3. ✅ Stage 3: tri_band_inference_lock 清理后，本地训练+eval 验证通过
4. ✅ Infra: 临时脚本清理后，`python tools/local_train_and_eval.py --config configs/clean_base_v2_local.json --skip-train --checkpoint <existing>` 验证一键脚本可用
5. ✅ git commit

## 执行顺序

1. **Stage 2B 剩余**: 5 个文件（model620.py, model.py, lancet_backbone.py, lancet_blocks.py, losses620.py）的 baseline 分支删除
2. **Stage 2C 本地验证**: smoke + train 10 epoch + eval（allpairs clip ≥ 0.7243）
3. **Stage 3**: tri_band_inference_lock 推理分支清理 + 本地验证
4. **Infra 优化**: 清理 13 个本次会话临时脚本 + 创建 `tools/local_train_and_eval.py` + 删除 `_remote_smoke_train.py`
5. **git commit**: 提交所有变更

## 风险与回退

- **风险**: Stage 2B 删除遗留分支可能影响 build_model_from_config 入口
- **回退**: 每个文件改完跑一次 smoke test；如 smoke test 失败，git checkout 该文件后重新分析
- **风险**: Stage 2C 训练指标不达标
- **回退**: 逐文件 bisect（git stash 部分修改），定位回归源
- **风险**: 一键脚本与现有 run.py 入口不兼容
- **回退**: 退化为直接调用 `python src/run.py` + `python src/utils/run_evaluation.py`
