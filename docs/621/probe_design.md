# 621 内部探针设计方案

> 建立日期: 2026-06-21  
> 目标: 深入模型内部debug白化机理

---

## 1. 探针分类

### 1.1 图像空间探针 (需要生成图片)

| 探针 | 输入 | 输出 | 目的 |
|------|------|------|------|
| WFI probe | 生成图片目录 | contrast/dynamic_range/saturation/wfi_score | 量化白化程度 |
| Pairwise WFI | source + generated图片对 | retention ratios, delta WFI | 对比退化 |
| Visual inspection | 生成图片 | 主观质量评分 | 定性判断 |

### 1.2 潜空间探针 (从metrics.csv或模型forward)

| 探针 | 输入 | 输出 | 目的 |
|------|------|------|------|
| Endpoint alpha probe | checkpoint, config | α(t), α_high(t), α_low(t) | 量化shrinkage |
| Style sensitivity probe | checkpoint, config | std(v(s₁),...,v(s₅)) | 量化style信号强度 |
| Gradient trace probe | checkpoint, config | ∇SWD方向, cos(∇SWD, v_target) | 诊断SWD梯度 |
| Velocity direction probe | checkpoint, config | cos(v(s₁), v(s₂)) for all pairs | 诊断条件期望坍缩 |

### 1.3 层内探针 (需要注册forward hook)

| 探针 | 注册位置 | 输出 | 目的 |
|------|----------|------|------|
| Statistics probe | 每个block输入/输出 | μ, σ, ‖·‖₂ | 追踪动态范围 |
| Style retention probe | FiLM前/后, FFN前/后 | R_style(l) | 量化GN洗掉style的程度 |
| Attention pattern probe | cross-attention | attn weights, entropy | 诊断attention collapse |
| Endpoint head probe | endpoint head内部 | head输入/输出统计 | 诊断head容量 |

---

## 2. 详细探针设计

### 2.1 Endpoint Alpha Probe

**目的**: 测量endpoint在不同时间t下的投影系数

**实现**:
```python
def probe_endpoint_alpha(model, content, target, style_patches, style_cls, t_values=[0.0, 0.25, 0.5, 0.75, 1.0]):
    results = {}
    for t in t_values:
        endpoint = model.predict_endpoint(content, t=t, style_dino_patches=style_patches, style_dino_cls=style_cls)
        # 计算alpha
        delta = target - content
        pred_move = endpoint - content
        alpha = (pred_move * delta).sum(dim=(1,2,3)) / (delta * delta).sum(dim=(1,2,3))
        # 高低频分解
        content_lp = avg_pool2d(content, 5, 1, 2)
        content_hp = content - content_lp
        target_lp = avg_pool2d(target, 5, 1, 2)
        target_hp = target - target_lp
        endpoint_lp = avg_pool2d(endpoint, 5, 1, 2)
        endpoint_hp = endpoint - endpoint_lp
        alpha_low = ((endpoint_lp - content_lp) * (target_lp - content_lp)).sum() / ((target_lp - content_lp)**2).sum()
        alpha_high = ((endpoint_hp - content_hp) * (target_hp - content_hp)).sum() / ((target_hp - content_hp)**2).sum()
        results[t] = {'alpha': alpha.mean().item(), 'alpha_low': alpha_low.item(), 'alpha_high': alpha_high.item()}
    return results
```

### 2.2 Style Sensitivity Probe

**目的**: 测量模型对不同style的响应差异

**实现**:
```python
def probe_style_sensitivity(model, content, style_list, t=0.0):
    velocities = []
    for style in style_list:
        v = model(content, t=t, style_dino_patches=style['patches'], style_dino_cls=style['cls'])
        velocities.append(v)
    velocities = torch.stack(velocities)  # [N_style, B, C, H, W]
    # 计算pairwise cosine similarity
    v_flat = velocities.flatten(2)  # [N_style, B, C*H*W]
    cos_sim = F.cosine_similarity(v_flat.unsqueeze(0), v_flat.unsqueeze(1), dim=-1)
    # 计算std across styles
    style_std = velocities.std(dim=0).mean()
    return {
        'mean_cos_sim': cos_sim.mean().item(),
        'max_cos_sim': (cos_sim - torch.eye(len(style_list))).max().item(),
        'style_std': style_std.item()
    }
```

### 2.3 Layer Statistics Probe

**目的**: 追踪每个block的统计量变化

**实现**:
```python
class LayerStatisticsProbe:
    def __init__(self, model):
        self.hooks = []
        self.stats = {}
        # 注册hook到每个block
        for i, block in enumerate(model.blocks):
            self.hooks.append(block.register_forward_hook(self._make_hook(f'block{i}')))
        # 注册到endpoint head
        if hasattr(model, 'out') and model.out is not None:
            self.hooks.append(model.out.register_forward_hook(self._make_hook('endpoint_head')))
    
    def _make_hook(self, name):
        def hook(module, input, output):
            self.stats[name] = {
                'input_mean': input[0].detach().float().mean().item(),
                'input_std': input[0].detach().float().std().item(),
                'output_mean': output.detach().float().mean().item(),
                'output_std': output.detach().float().std().item(),
                'amplification': output.detach().float().std() / (input[0].detach().float().std() + 1e-8),
            }
        return hook
    
    def remove(self):
        for h in self.hooks:
            h.remove()
```

### 2.4 Style Retention Probe

**目的**: 测量GN洗掉style信号的程度

**实现**:
```python
def probe_style_retention(model, content, style1, style2, t=0.0):
    # 获取两个style的block输出
    outputs_s1 = []
    outputs_s2 = []
    
    def hook_s1(module, input, output):
        outputs_s1.append(output.detach())
    def hook_s2(module, input, output):
        outputs_s2.append(output.detach())
    
    # 注册hooks
    hooks = []
    for block in model.blocks:
        hooks.append(block.register_forward_hook(hook_s1))
    
    # Forward with style1
    model(content, t=t, style_dino_patches=style1['patches'], style_dino_cls=style1['cls'])
    for h in hooks:
        h.remove()
    
    # 注册hooks for style2
    hooks = []
    for block in model.blocks:
        hooks.append(block.register_forward_hook(hook_s2))
    
    # Forward with style2
    model(content, t=t, style_dino_patches=style2['patches'], style_dino_cls=style2['cls'])
    for h in hooks:
        h.remove()
    
    # 计算R_style
    R_style = []
    for out1, out2 in zip(outputs_s1, outputs_s2):
        diff_in = (out1 - out2).flatten(1).norm(dim=1).mean()
        # 应用GN后再计算差异
        gn_out1 = F.group_norm(out1, 1)
        gn_out2 = F.group_norm(out2, 1)
        diff_out = (gn_out1 - gn_out2).flatten(1).norm(dim=1).mean()
        R = diff_out / (diff_in + 1e-8)
        R_style.append(R.item())
    return R_style
```

---

## 3. 探针运行流程

### 3.1 诊断流程 (Phase 1: Root Cause)

```
1. WFI probe → 确认白化存在
2. Endpoint alpha probe → 确认shrinkage
3. Style sensitivity probe → 确认条件期望坍缩
4. Gradient trace probe → 确认SWD梯度状态
5. Layer statistics probe → 定位信号衰减位置
```

### 3.2 修复验证流程 (Phase 2: Fix Verification)

```
1. 每次修复后运行完整探针
2. 对比修复前后的:
   - WFI score (目标<0.40)
   - endpoint_alpha (目标>0.5)
   - style_std (目标>5.0)
   - R_style per layer (目标>0.3)
3. 如果未达标，返回Phase 1
```

### 3.3 阶段性验收 (Phase 3: Acceptance)

```
1. 所有指标达标
2. 视觉检查: 无明显白化
3. CLIP-S不下降 (≥0.695)
4. Content LPIPS可接受 (<0.36)
```

---

## 4. 推荐的探针运行顺序

### 4.1 当前基线诊断 (立即执行)

```bash
# 1. WFI baseline
python tools/probe_620_fog_whiteness_index.py \
  --eval_dir exp/620_spatial_bridge/620_film_v5_endpoint_film_hd512_local_smoke/full_eval_wfi/epoch_0001/ \
  --output docs/621/probe_results/baseline_wfi.json

# 2. Hypothesis metrics
python tools/probe_620_hypothesis_metrics.py \
  --checkpoint exp/620_spatial_bridge/620_film_v5_endpoint_film_hd512_local_smoke/epoch_0001.pt \
  --config exp/620_spatial_bridge/620_film_v5_endpoint_film_hd512_local_smoke/config.json \
  --output docs/621/probe_results/baseline_hypothesis.json

# 3. Solver trace
python tools/probe_620_solver_trace.py \
  --checkpoint exp/620_spatial_bridge/620_film_v5_endpoint_film_hd512_local_smoke/epoch_0001.pt \
  --config exp/620_spatial_bridge/620_film_v5_endpoint_film_hd512_local_smoke/config.json \
  --output docs/621/probe_results/baseline_solver_trace.json
```

### 4.2 修复后验证 (每次修复后)

```bash
# 对新checkpoint运行完整探针
python tools/run_internal_probe.py \
  --checkpoint <new_checkpoint> \
  --config <config> \
  --output docs/621/probe_results/fix_<name>_results.json
```
