# 616 实验计划 — 1 到 3 天预算

> GPU: 3060 12GB, VRAM 上限 < 11.3 GB
> 每个 epoch 约 25-30 min (b12)
> 每 2 个 epoch 检查一次 eval 结果

---

## 阶段 1: 垂直 FM + 结构 OT 组合验证 (4-6 epochs, ~3h)

### 目标
验证 `bridge_path_mode="vertical"` + `tokenizer_entropy_affinity_gw` + `unbalanced_sinkhorn` 组合是否能同时提升 style 并保持 LPIPS。

### 训练启动

```bash
cd /mnt/i/Github/Latent_Style/SchrodingerBridge

# 一键启动
python -c "
import json, os, shutil

base_ckpt = 'exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k085_appalign_seed42_b12a1/epoch_0001.pt'
exp_dir  = 'exp/phase616_stage1_vertical_ot'

os.makedirs(exp_dir, exist_ok=True)
c = json.load(open(os.path.dirname(base_ckpt) + '/config.json'))

# 覆盖关键参数
c['bridge']['bridge_path_mode'] = 'vertical'
c['bridge']['coupling_solver'] = 'sinkhorn_unbalanced'
c['bridge']['sinkhorn_unbalanced_tau_src'] = 0.5
c['bridge']['coupling_structure_cost_mode'] = 'tokenizer_entropy_affinity_gw'
c['bridge']['coupling_cost_composition'] = 'appearance_plus_structure'
c['bridge']['coupling_structure_cost_weight'] = 0.3
c['training']['num_epochs'] = 6
c['training']['batch_size'] = 8
c['training']['accumulation_steps'] = 1
c['checkpoint']['save_dir'] = f'./{exp_dir}'

json.dump(c, open(f'{exp_dir}/config.json', 'w'), indent=2)
shutil.copy(base_ckpt, f'{exp_dir}/epoch_0001.pt')
print('Config ready. Launching...')
" && python src/run.py --config exp/phase616_stage1_vertical_ot/config.json --resume exp/phase616_stage1_vertical_ot/epoch_0001.pt
```

### 判断标准

| Epoch | 操作 |
|-------|------|
| e1-e2 | 观察 `ot_target_gini` (应 < 0.5), `ot_structure_cost_var` (应 > 0), `tok_delta` (应 > 0.015) |
| e3-e4 | 读 eval: **期望 style > 0.69 + LPIPS < 0.35** |
| e5-e6 | 如果 style < 0.69: 进入阶段 2。如果 style > 0.70: 延长到 e10 |

---

## 阶段 2: 参数细化扫描 (2-4 epochs, ~2h)

### 目标
如果阶段 1 验证了方向正确但 style 仍未突破 0.72，扫描关键超参。

### 2a: Kinetic 权重扫描 (取阶段 1 最佳 epoch 的 ckpt)

```bash
for wk in 0.5 0.7 0.85; do
    # 修改 config 的 w_kinetic, 从阶段 1 ckpt 继续
    python src/run.py --config exp/phase616_stage2_kin${wk}/config.json \
      --resume exp/phase616_stage1_vertical_ot/epoch_0004.pt
done
```

### 2b: OT 结构权重扫描

```bash
for wstruct in 0.2 0.3 0.5; do
    python src/run.py --config exp/phase616_stage2_ot${wstruct}/config.json \
      --resume exp/phase616_stage1_vertical_ot/epoch_0004.pt
done
```

### 2c: 垂直分解粒度扫描

```bash
for stride in 1 2 4; do
    # stride=1: 最细粒度分离, stride=4: 最粗
    python src/run.py --config ... --override bridge_vertical_base_stride=$stride
done
```

---

## 阶段 3: I2SB + Vertical 组合 (如果垂直 FM 未突破, 6-8 epochs, ~4h)

### 目标
叠加 I2SB 训练（σ=0.02）在垂直 FM 基础上。I2SB 注入的随机性可能打破 style 的均值坍缩。

### 配置

```json
{
  "bridge": {
    "bridge_path_mode": "vertical",
    "bridge_sigma": 0.02,
    "bridge_noise_schedule": "exact_brownian",
    "objective_mode": "i2sb_endpoint",
    "coupling_solver": "sinkhorn_unbalanced",
    "coupling_structure_cost_mode": "tokenizer_entropy_affinity_gw"
  },
  "model": {
    "transport_prediction_mode": "endpoint",
    "solver_family": "solver_i2sb"
  }
}
```

从阶段 1 最佳 epoch 的 ckpt warmstart，但切换到 endpoint 模式需要重新构建模型——可能报错。如遇兼容性问题，回退到 velocity 模式但 `bridge_sigma=0.02`。

---

## 阶段 4: 全量训练收敛 (10-16 epochs, ~5h)

### 前提
前 3 个阶段中至少一个突破了 style > 0.70。

### 操作
取表现最好的组合，加大训练量到 12-16 epochs。每 2 epochs 检查 eval。

### 安全阈值
- LPIPS > 0.45: 立即停止，回退参数
- flow_loss 突然飙升 3×: 停止
- ot_target_gini > 0.8: OT 退化，切回标准 Sinkhorn

---

## 时间预算总览

| 阶段 | 预计时间 | 可并行? |
|------|----------|---------|
| 1: 垂直 FM + OT | ~3h | — |
| 2a: kinetic 扫描 | ~2h (3 runs) | 顺序 |
| 2b: OT 权重扫描 | ~2h (3 runs) | 顺序 |
| 2c: 分解粒度 | ~2h (3 runs) | 顺序 |
| 3: I2SB 组合 | ~4h | — |
| 4: 全量训练 | ~5h | — |
| **总计** | **~18h (不超过 1 天)** | |

---

## 成功定义

- **style > 0.72** AND **LPIPS < 0.35** (transfer) → 目标达成
- style > 0.72 BUT LPIPS > 0.40 → 接近，需要结构约束微调
- style 0.70-0.72 AND LPIPS < 0.32 → 接近，不够
- style < 0.70 → 垂直 FM + OT + I2SB 的组合均失败，需要更根本的架构改变
