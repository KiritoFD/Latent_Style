# E9-E12+ 新方案实验计划：视觉诊断 + 频段解耦 + CFG推理

## Summary

基于 E4-E8 实验结果（clip_style 最高达 0.715，但 LPIPS 恶化到 0.50+），用户提出 4 个新方案突破帕累托前沿。本计划包含：**视觉诊断**（修复图片生成问题）+ **串行探索**（batch 保守 + 逐个验证新方案）。

## Current State Analysis

### 已完成的代码能力（可复用）
| 能力 | 文件 | 状态 | 参数 |
|------|------|------|------|
| RMSNorm | blocks620.py:11-28 | ✅ 已部署 | `body_norm_type="rms_norm"` |
| VP-Flow 球面插值 | losses620.py:258-281 | ✅ 已部署 | `bridge_path_mode="spherical_vp"` |
| Top-K 截断注意力 | blocks620.py:348-480 (8处) | ✅ 已部署 | `style_attn_topk=4/8` |
| 方向余弦损失 | losses620.py:354-361 | ✅ 已部署 | `w_directional_cosine=1.0` |
| Beta 时间采样 | losses620.py:164-181 | ✅ 已部署 | `t_sampling_beta_a/b` |

### 当前最佳结果对比
| 实验 | clip_style↑ | LPIPS↓ | velocity_std | 问题 |
|------|-----------|--------|-------------|------|
| E2 (历史最优) | — | **0.333** | ~0.05 | 基线 |
| E4 (RMSNorm) | 0.672 | **0.373** | **0.896** | 最佳平衡 |
| E8 (方向余弦) | **0.715** 🏆 | 0.506 | 1.42 | 风格最强内容差 |

### 图片未生成的根因
远程 eval 使用了 fast metric path，`save_generated_images` 在某些路径下被跳过。需要重新运行 eval 或在训练配置中强制启用图片保存。

### 显存问题
之前 batch=24 在最重配置下达到 10.21GB（接近 12GB 上限）。新实验需保守 batch=16 或 12。

---

## Proposed Changes

### Phase A: 视觉诊断（修复图片 + 目视检查）

#### A1. 修复图片生成并下载
- **文件**: 无代码修改，仅操作远程
- **做法**:
  1. 对 E4（最佳平衡）和 E8（最高风格）重新运行 eval，强制保存图片：
     ```bash
     wsl python utils/run_evaluation.py --checkpoint e4_anti_degen/checkpoints/epoch_0003.pt --output e4_anti_degen/checkpoints/full_eval/epoch_0003_redo --save_generated_images --save_summary_grid
     ```
  2. scp 下载 summary_grid.png 到本地 `exp/p3_remote_10h/visual_diag/`
  3. 用 Read 工具查看图片进行目视诊断
- **预期**: 获得可目视的生成结果图

#### A2. 基于 E4 配置生成保守 batch 的基础模板
- 将 batch_size 从 24 降到 **16**（安全余量 ~4GB）

---

### Phase B: 方案一 — 频段解耦余弦损失 (E9 Freq-Split Cosine)

**核心思路**: E8 的全频段方向余弦把低频结构信息也当"噪声"洗掉了。拆分频段：低频用 MSE 保结构，高频用 Cosine 保风格。

#### B1. 修改 `src/losses620.py` compute() 方法
在现有方向余弦损失代码（line 354-361）处改造：

```python
# 现有代码:
# dir_cosine_loss = (1 - cos_sim).clamp(min=0.0)
# fm = fm + self.w_directional_cosine * dir_cosine_loss

# 改为频段解耦版本:
if self.w_directional_cosine > 0:
    # 低通滤波器分离频段
    kernel = self.lowpass_kernel  # 复用已有的 lowpass_kernel=5
    v_pred_lp = _lowpass(pred_velocity.float(), kernel)
    v_tgt_lp = _lowpass(target_velocity.float(), kernel)
    v_pred_hp = pred_velocity.float() - v_pred_lp
    v_tgt_hp = target_velocity.float() - v_tgt_lp
    
    # 低频：严格 MSE（保结构）
    fm_low = F.mse_loss(v_pred_lp, v_tgt_lp)
    
    # 高频：方向余弦（保风格笔触）
    v_pred_hp_n = F.normalize(v_pred_hp.reshape(v_pred_hp.shape[0], -1), dim=-1)
    v_tgt_hp_n = F.normalize(v_tgt_hp.reshape(v_tgt_hp.shape[0], -1), dim=-1)
    cos_sim_hp = (v_pred_hp_n * v_tgt_hp_n).sum(dim=-1).mean()
    dir_loss_hp = (1.0 - cos_sim_hp).clamp(min=0.0)
    
    # 组合：MSE 权重保持主导，高频方向作为附加约束
    fm = fm + self.w_directional_cosine * (fm_low * 0.5 + dir_loss_hp)
```

#### B2. 新增 config 参数
```python
# config_schema.py — 可选: 如果需要独立控制高低频权重
w_freq_split_cosine: float = 0.0  # 0=关闭(默认), >0=启用频段解耦
```

#### B3. 生成 E9 config
基于 **E4 配置**（不是 E8！）：
- batch_size = **16**（保守）
- `w_directional_cosine` = **1.0**（启用频段解耦版）
- `bridge_path_mode` = `"linear"`（不用 VP-Flow）
- `style_attn_topk` = **0**（不用 Top-K）
- `body_norm_type` = `"rms_norm"`
- num_epochs = **3**
- ablation: name=`"e9_freq_split_cosine"`

---

### Phase C: 方案四 — CFG 训练 + 推理外推 (E12)

**核心思路**: 用 E4 平衡基座训练 + 15% 条件丢弃学会无条件分支 → 推理时暴力外推风格。

#### C1. 修改 `src/losses620.py` compute() 方法添加条件丢弃
在 model forward 调用前插入：

```python
# CFG dropout: 以概率替换 style 为 null tokens
use_uncond = False
uncond_loss = content.new_tensor(0.0)
if self.cfg_dropout_prob > 0 and self.training:
    use_uncond = random.random() < self.cfg_dropout_prob
    if use_uncond:
        # 用 null tokens 替换所有 style 输入
        null_style_latent = torch.zeros_like(target_style)
        # 保存 cond 分支的输出用于后续 uncond 计算
```

**注意**: 这需要修改 compute() 的签名和返回值，或者更简单地在 trainer.py 层面实现。**推荐在 trainer.py 中实现**以最小化对 losses620.py 的改动：

```python
# trainer.py 训练循环中:
if cfg_dropout_prob > 0 and random.random() < cfg_dropout_prob:
    # 替换 conditioning dict 中的所有 style 相关字段为 null
    conditioning_null = {k: None if 'style' in k.lower() or 'dino' in k.lower() else v 
                        for k, v in conditioning.items()}
    loss_uncond = objective.compute(model, ..., conditioning=conditioning_null)
    loss = 0.5 * loss + 0.5 * loss_uncond  # 或其他混合比例
```

#### C2. 新增参数
```python
# config_schema.py (training 区域):
cfg_dropout_prob: float = 0.0   # 0=关闭, 0.15=15%概率条件丢弃
```

#### C3. 推理脚本支持 CFG 外推
创建 `exp/p3_remote_10h/cfg_infer.py`:
```python
# 推理时:
# v_cond = model(x_t, t, style=style_tokens)
# v_uncond = model(x_t, t, style=null_tokens)  
# v_final = v_uncond + guidance_scale * (v_cond - v_uncond)
# guidance_scale 可设为 1.5~3.0 进行暴力外推
```

#### C4. 生成 E12 config
基于 **E4 配置**：
- batch_size = **16**
- `cfg_dropout_prob` = **0.15**
- 其余同 E4（RMSNorm + vmag=2.0 + two_stage）
- ablation: name=`"e12_cfg_dropout"`

---

### Phase D: 方案二 — 方差重整直线流 (E10 Variance-Normalized RF)

**核心思路**: 直线路径保结构 + 强制拉伸方差防止中间步发灰。

#### D1. 修改 `src/losses620.py` _vertical_state()
在 linear 分支的 `x_t` 计算后追加方差重整：

```python
else:  # linear path
    # ... 原有线性插值代码 ...
    x_t = x_low + (1.0 - t4) * c_high + t4 * p_high
    target_velocity = (p_high - c_high) + target_low_velocity
    
    # 方差重整: 保持目标方差 sqrt((1-t)^2 + t^2)
    if self.variance_rectify:
        x_mean = x_t.mean(dim=(1,2,3), keepdim=True)
        x_std = x_t.std(dim=(1,2,3), keepdim=True).clamp_min(1e-6)
        target_std = torch.sqrt((1-t4)**2 + t4**2 + 1e-6)
        x_t = (x_t - x_mean) / x_std * target_std + x_mean
```

#### D2. 新增参数
```python
variance_rectify: bool = False  # 启用方差重整
```

---

### Phase E: E4 长训练验证 (E4-long)

**不做任何代码修改**，仅将 E4 的 epoch 从 3 提升到 **10**，验证更多 epoch 是否能自然提升 clip_style 到 0.70+ 而 LPIPS 不恶化。

---

## 实验执行顺序（串行，按优先级）

| 顺序 | 实验 | 方案 | 核心改动 | 预期效果 |
|------|------|------|---------|---------|
| 0 | **视觉诊断** | 修复图片 | 重新eval+下载 | 目视确认E4/E8质量差异 |
| 1 | **E9** | 频段解耦余弦 | 低频MSE+高频Cosine | **首选: 兼顾风格+内容** |
| 2 | **E12** | CFG训练 | 15%条件丢弃 | 推理时可外推 |
| 3 | **E10** | 方差重整流 | 直线+方差拉伸 | 消灭中间步发灰 |
| 4 | **E4-long** | 长训练 | 10 epoch | 验证自然收敛上限 |

每个实验: batch=16, 3 epoch (除 E4-long=10), 串行执行。

## Assumptions & Decisions

1. **batch_size=16**: 从 24 降到 16，确保即使最重配置也不 OOM（E4 peak=10.21GB@b24, b16 应 ~7GB）
2. **基于 E4 而非 E8 作为基线**: E4 有最好的 LPIPS=0.373，在其上叠加改进更合理
3. **Phase A 必须先完成**: 视觉诊断决定后续方向调整
4. **串行执行**: GPU 只有一个，避免并发导致 OOM
5. **图片保存**: 通过显式 `--save_generated_images` 参数强制启用

## Verification Steps

### 每个 experiment 完成后检查:
1. [ ] 训练无 NaN/OOM
2. [ ] full_eval 完成，summary.json 包含 clip_style 和 LPIPS
3. [ ] 对比 E4 baseline: clip_style 变化, LPIPS 变化
4. [ ] 如果是 E9: 确认低频 MSE + 高频 Cosine 都在 metrics 中可见

### 最终成功标准:
- [ ] 至少一个实验达到 **clip_style > 0.69 且 LPIPS < 0.40**
- [ ] 理想: **clip_style > 0.70 且 LPIPS < 0.35**
- [ ] 视觉诊断确认: 风格强度 > E4, 内容保持 ≥ E4
