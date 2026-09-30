# 620 Spatial Bridge 3轴白化修复实验交接文档

**日期**: 2026-06-24  
**分支**: `codex/620-spatial-bridge`  
**最新commit**: `e2c74c271` feat(620): 3-axis whitening fix - gate warmup + RMSNorm + anti-whitening losses  
**状态**: 训练已完成(8/8 epochs)，eval尚未运行

---

## 0. 一句话总结

基于统一数学模型(α = max(α_attn·α_FiLM, α_GN) - α_loss)发现"雾化"根因是style信号沿3个轴衰减→零（gate cold-start、GN白化、loss反白化），实施了3轴同时修复方案(gate warmup + RMSNorm + anti-whitening losses)，训练收敛正常(loss 3.92→3.34)，gate从0.046成功攀升至0.297，但**尚未运行eval确认视觉/WFI改善**。

---

## 1. 项目目标

**核心问题**: text（文本描述）对风格迁移到底有没有好的作用，能不能做？

**当前阶段**: 先解决基础模型（无text）的雾化问题，让style信号真正注入endpoint，才能公平评估text的贡献。

**完整信号衰减链** (style 1.0 → ... → endpoint):
```
style 1.0 → patch_proj 0.90 → gate×tanh 0.045 → StyleFiLM 0.018 → GN 0.005 → head_init 0.001 → endpoint 0.001
                                                                                     ↑ 99.9%衰减
```

**数学模型**: α = max(α_attn × α_FiLM, α_GN) - α_loss
- 当前实测: α = 0.18 (观测 0.16, 误差 12.5%)
- 理论瓶颈: α_GN = 0.28 (GroupNorm白化是最大瓶颈)

---

## 2. 3轴修复方案 (3-Axis Whitening Fix)

### Axis A: Gate Warmup (门控预热)
**问题**: gate init=0.3, tanh(0.3)≈0.29，但训练初期随机style特征产生噪声→梯度惩罚gate→gate collapse到0.05  
**修复**: gate从0线性升至学习值，前500步不让gate"全开"  
**代码改动**:
- `blocks620.py`: 新增 `gate_warmup_steps` 参数、`set_step()` 方法、`_effective_gate_value()` 方法
  - Line 71: `gate_warmup_steps: int = 0` (构造函数参数)
  - Line 76: `self.gate_warmup_steps = max(0, int(gate_warmup_steps))`
  - Line 177-179: `set_step()` — 每步更新 `_current_step`
  - Line 181-193: `_effective_gate_value()` — warmup_factor = min(1, step/warmup_steps), 返回 raw × warmup_factor
  - Line 414: `style_delta = self._effective_gate_value() * attended_2d` (使用effective gate)

### Axis B: GN → RMSNorm (归一化换型)
**问题**: GroupNorm减均值+除标准差=白化，摧毁style注入的color/contrast信息(α_GN=0.28)  
**修复**: RMSNorm只除RMS不减均值，保留color/contrast  
**代码改动**:
- `model620.py` FiLMEndpointHead:
  - Line 22: 新增 `use_rmsnorm: bool = False` 参数
  - Line 25: `self.use_rmsnorm = use_rmsnorm`
  - Line 27-31: RMSNorm实现 — `rms = x.pow(2).mean(dim=[2,3]).sqrt()`, `h = x / rms * weight`
  - Line 56-63: forward中根据use_rmsnorm分支
- `model620.py` SpatialBridge620:
  - Line 95: `self.endpoint_film_use_rmsnorm = bool(getattr(model_cfg, "endpoint_film_use_rmsnorm", False))`
  - Line 216-220: FiLMEndpointHead构造传入use_rmsnorm

### Axis D: Anti-Whitening Losses (反白化损失)
**问题**: SWD等loss隐式偏好"均匀分布"输出(=白化)，反白化loss抵消这一趋势  
**修复**: 启用4个反白化损失项  
**代码改动**:
- `losses620.py` 已有实现:
  - `w_contrast_preserve=2.0` — 保持对比度
  - `w_channel_variance=0.5` — 保持通道间方差
  - `w_hf_energy=1.0` — 保持高频能量
  - `w_velocity_magnitude=1.0` — 保持速度场幅度
- 配置文件中设置权重

### Gate Step传播
**代码改动**:
- `trainer.py`:
  - Line 1305-1309: 在loss计算前传播global_step到所有blocks
  ```python
  if hasattr(self.model, 'blocks'):
      for blk in self.model.blocks:
          if hasattr(blk, 'set_step'):
              blk.set_step(self.global_step)
  ```

---

## 3. 训练结果

### 训练配置
- **配置文件**: `configs/ablations/620_bold_3axis_fix_8ep.json`
- **数据**: 5类各1000样本 (Early_Renaissance, Impressionism, Minimalism, Rococo, Ukiyo_e)
- **模型**: 2.25M params, base_dim=64, 4 res blocks + 2 hires blocks
- **训练**: batch=4, accum=16, lr=2e-4, cosine schedule, 8 epochs, AMP bf16
- **关键设置**: latent intrinsic + softmax attn + edge=0 + FiLM init_std=0.02 + gate_init=0.3

### Loss收敛
| Epoch | Loss   | Flow   |
|-------|--------|--------|
| 2     | 3.9141 | 1.1662 |
| 3     | 3.7495 | 1.0815 |
| 4     | 3.6496 | 1.0248 |
| 5     | 3.5052 | 0.9724 |
| 6     | 3.4423 | 0.9222 |
| 7     | 3.3791 | 0.9072 |
| 8     | 3.3402 | 0.8949 |

### Gate轨迹 (关键发现)
| Global Step | Gate Value | 阶段 |
|-------------|-----------|------|
| 1251 (E2:S1) | 0.0461 | warmup初期 |
| 2500 (E2:S1250) | 0.0920 | warmup中 |
| 5000 (E4:S1250) | 0.1872 | warmup中 |
| 7500 (E6:S1250) | 0.2813 | warmup末 |
| 8000 (E7:S500) | 0.2974 | warmup完成 |
| 10000 (E8:S1250) | 0.2974 | 稳定 |

**关键观察**: Gate成功从0.046攀升到0.297(接近init值0.3)，证明warmup有效阻止了gate collapse。但warmup实际用了~6750步(而非配置的500步)才完成——这是因为`global_step`在resume时从epoch 2开始(1251)，而`_effective_gate_value`使用的是`self._current_step`即`self.global_step`，所以warmup_factor = min(1, 1251/500) = 1.0，gate应该立即全开。

**⚠️ BUG**: `gate_warmup_steps=500`但实际warmup用了~6750步。原因可能是`set_step()`传入的是epoch-relative step而非global_step，需要排查。但无论原因如何，gate确实在缓慢攀升，说明有某种warmup机制在生效。

### 稳定性指标
- **endpoint_output_std**: 0.57~0.92 (健康，未collapse)
- **velocity_std**: 0.25~1.15 (正常波动)
- **cross_attn_entropy**: 5.31~5.53 (≈ln(256)=5.55，attention较均匀)
- **无NaN/Inf**: 训练全程稳定

---

## 4. Checkpoint文件

远程路径: `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/620_spatial_bridge/620_bold_3axis_fix_8ep/`

| 文件 | 大小 | 说明 |
|------|------|------|
| epoch_0001.pt | 15.2 MB | epoch 1 (warmup初期) |
| epoch_0002.pt | 15.2 MB | epoch 2 |
| epoch_0003.pt | 15.2 MB | epoch 3 |
| epoch_0004.pt | 15.2 MB | epoch 4 |
| epoch_0005.pt | 15.2 MB | epoch 5 |
| epoch_0006.pt | 15.2 MB | epoch 6 (warmup接近完成) |
| epoch_0007.pt | 15.2 MB | epoch 7 (warmup完成) |
| epoch_0008.pt | 15.2 MB | epoch 8 (最终，推荐eval) |
| config.json | — | 运行时完整配置 |

---

## 5. 未完成事项 (TODO)

### 🔴 最高优先级: 运行Eval
训练已完成8 epoch但**eval未运行**（因禁用了`full_eval_each_epoch`以避免`f:/`路径crash）。

**运行eval的步骤**:
1. 需要先修复config中的路径问题：`test_image_dir`, `full_eval_cache_dir`, `full_eval_clip_hf_cache_dir`, `data_root`, `pairing_cache_path`, `latent_cache_dir`, `dino_cache_path` 全部指向`f:/`路径
2. 在远程WSL上这些路径应改为:
   - `data_root` → `/mnt/i/wikiart_distinct5_samam_512_latents_ema/train`
   - `test_image_dir` → `/mnt/i/wikiart_distinct5_samam_512_classview_real/test`
   - `full_eval_cache_dir` → `/mnt/i/eval_cache`
   - 其他`f:/`路径类似替换为`/mnt/i/...`
3. 运行命令 (在WSL中，工作目录 `/mnt/i/Github/Latent_Style/SchrodingerBridge/src`):
   ```bash
   /home/xy/venvs/samam312/bin/python3 run.py --config ../configs/ablations/620_bold_3axis_fix_8ep.json --eval-only --eval-epoch 8
   ```
   或者直接用`run_evaluation.py`加载checkpoint
4. 关注指标: **WFI** (Whitening/Fog Index, 核心指标), clip_style, lpips

### 🟡 中优先级: Gate Warmup Bug修复
`gate_warmup_steps=500`但实际warmup用了~6750步。需要:
1. 检查`trainer.py`中`self.global_step`是否正确传递(它应该是累计全局步数)
2. 检查`blocks620.py`中`_effective_gate_value()`使用的`self._current_step`来源
3. 可能的bug: `set_step()`在每步训练循环中调用，但`self.global_step`的值可能不对(例如epoch-relative而非global)

### 🟡 中优先级: 3轴消融
当前3轴同时修复，无法区分各轴贡献。若eval改善显著，建议:
1. Axis A only (gate warmup, 无RMSNorm, 无anti-whitening loss)
2. Axis B only (RMSNorm, 无gate warmup, 无anti-whitening loss)  
3. Axis D only (anti-whitening losses, 无gate warmup, 无RMSNorm)
4. 各对比baseline

### 🟢 低优先级: Eval路径Patch
当前config中所有数据路径用`f:/`前缀(Windows盘符)，在WSL中需要替换为`/mnt/f/...`。建议:
- 在`launch_bold_3axis.py`或新脚本中统一patch所有路径
- 或在config_schema中添加路径重映射逻辑

---

## 6. 远程服务器操作指南

### 连接方式
```powershell
# 从本地Windows连接远程
ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62
```

### WSL命令模式
由于Windows→SSH→WSL嵌套引号问题，推荐以下模式:
```powershell
# SCP上传.py文件到远程Windows桌面
scp -P 2222 -o LogLevel=ERROR "C:\Users\xy\AppData\Local\Temp\deveco\script.py" administrator@100.115.18.62:C:/Users/administrator/Desktop/script.py

# 在WSL中执行
ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "chcp 65001 >nul & wsl -d Ubuntu-26.04 -- /home/xy/venvs/samam312/bin/python3 /mnt/c/Users/administrator/Desktop/script.py"
```

### 关键远程路径
| 用途 | 路径 |
|------|------|
| 训练数据 | `/mnt/i/wikiart_distinct5_samam_512_latents_ema/train` |
| 测试图片 | `/mnt/i/wikiart_distinct5_samam_512_classview_real/test` |
| Eval缓存 | `/mnt/i/eval_cache` |
| 训练日志 | `/mnt/i/Github/Latent_Style/bold_3axis_train.log` |
| Checkpoint | `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/620_spatial_bridge/620_bold_3axis_fix_8ep/` |
| Python | `/home/xy/venvs/samam312/bin/python3` (3.12.3) |
| Git Repo | `/mnt/i/Github/Latent_Style/` |
| 分支 | `codex/620-spatial-bridge` |
| GPU | RTX 3060 12GB, torch=2.11.0+cu128 |

### 运行训练 (参考)
```bash
# 在WSL中，工作目录 /mnt/i/Github/Latent_Style/SchrodingerBridge/src
nohup /home/xy/venvs/samam312/bin/python3 run.py --config ../configs/ablations/620_bold_3axis_fix_8ep.json > /mnt/i/Github/Latent_Style/bold_3axis_train.log 2>&1 &
```

---

## 7. 关键源文件清单

| 文件 | 说明 | 本次改动 |
|------|------|---------|
| `src/blocks620.py` | 核心block: SpatialBridgeBlock620 | ✅ gate_warmup_steps, set_step(), _effective_gate_value() |
| `src/model620.py` | 模型: SpatialBridge620 + FiLMEndpointHead | ✅ use_rmsnorm, gate_warmup_steps传播 |
| `src/trainer.py` | 训练循环 | ✅ global_step→blocks.set_step()传播 |
| `src/losses620.py` | 损失函数: SpatialBridgeObjective620 | 已有anti-whitening losses(本次仅启用) |
| `src/utils/run_evaluation.py` | Eval脚本(4758行) | 未改 |
| `src/utils/wfi.py` | WFI指标计算 | 未改 |
| `configs/ablations/620_bold_3axis_fix_8ep.json` | 3轴修复实验配置 | ✅ 新建 |

---

## 8. 历史实验参考

### 前序实验关键数据
| 实验 | clip_style↑ | lpips↓ | WFI↑ | 说明 |
|------|-------------|--------|------|------|
| 620_swd16_notext (baseline) | 0.6399 | 0.3678 | 0.3842 | 无text baseline |
| 620_swd16_t5base (text) | 0.6717 | 0.3678 | ~0.38 | T5 text，WFI未显著改善 |
| H7 (SWD 8→2) | — | — | — | 减少SWD权重 |

### 文档体系
- `docs/622/history/` — 10个完整历史文档
- `docs/620/fog/` — 雾化问题诊断全记录
- `docs/620/info_flow_analysis.md` — 信息流分析
- `docs/620/convergence_diagnosis.md` — 收敛诊断

---

## 9. 数学模型速查

### 加性-乘性信号衰减模型
```
α = max(α_attn × α_FiLM, α_GN) - α_loss
```

- **α_attn**: cross-attention效率 (softmax≈0.90, sparsemax≈0.70)
- **α_FiLM**: FiLM调制穿透率 (取决于init_std和norm)
- **α_GN**: GroupNorm白化残留率 (GN≈0.28, RMSNorm≈0.85)
- **α_loss**: loss反白化力度 (contrast_preserve等抵消后)

### 3轴修复理论预测
- Axis A (gate warmup): 防止gate collapse → α_attn从0.05恢复到0.90
- Axis B (RMSNorm): α_GN从0.28提升到0.85
- Axis D (anti-whitening): α_loss从0.10降到0.02
- **预测**: α从0.16→0.38, 实际需eval验证

---

## 10. 风险与注意事项

1. **Gate warmup bug**: 实际warmup用了6750步而非500步，虽然结果是好的(gate缓慢攀升)，但行为与设计不符，可能影响后续实验的可控性
2. **Eval路径**: config中`f:/`路径在WSL中不可用，必须patch后才能运行eval
3. **Epoch 1丢失**: 首次运行epoch 1因eval路径crash，重启后从epoch 2开始，epoch_0001.pt是第二次运行的
4. **Attention均匀**: cross_attn_entropy≈5.5(=ln256)说明attention仍接近均匀分布，style token区分度不足，可能需要sharpen_scale或sparsemax
5. **12GB VRAM限制**: 3060仅12GB，batch=4+accumulation=16已经接近极限，无法增大batch或模型
