# 实验启动指南

## 远程环境

```
SSH: ssh -p 2222 administrator@100.115.18.62
GPU: NVIDIA 3060 12GB (VRAM 上限 < 11.3 GB)
WSL: Ubuntu-26.04, Torch 2.11.0+cu128
数据: /mnt/i/wikiarts_5_full_notest_latents_ema/train/ (5 类: Early_Renaissance, Impressionism, Minimalism, Rococo, Ukiyo_e)
eval cache: /mnt/i/Github/Latent_Style/eval_cache/
代码根: /mnt/i/Github/Latent_Style/SchrodingerBridge/
exp根:   /mnt/i/Github/Latent_Style/exp/
```

## 实验启动模式

### 当前运行实验

```
exp/20250618_lite_ot_vertical/ — b24, vl=0.1, legacy_factorized + ablation_disable_spatial_prior=true + topogate
7 hypothesis tests: h0-h6
```

### 模式 1: 从头训练（新 config）

```bash
cd /mnt/i/Github/Latent_Style/SchrodingerBridge
python src/run.py --config configs/aaai2027/phase2_xxx.json
```

### 模式 2: Warmstart 训练（从 ckpt 继续）

```bash
python src/run.py --config configs/aaai2027/phase2_xxx.json \
  --resume /mnt/i/Github/Latent_Style/exp/xxx/epoch_0001.pt
```

**注意**: `--resume` 会优先使用 ckpt 所在目录的 `config.json`，忽略 `--config` 中的重叠字段。
只有 ckpt 的 config.json 中不存在的 key 才会从 `--config` 加载。
如果要完全替换某些参数，需要确保 ckpt 的 config.json 中不包含旧值，或直接修改 ckpt 目录的 config.json。

### 模式 3: 仅 Eval（不改权重）

```bash
# 需要创建 eval-only config (num_epochs=1, lr=0.0)
python src/run.py --config configs/aaai2027/eval_only_xxx.json \
  --resume /path/to/checkpoint.pt
```

## 当前最佳 Warmstart

```
topogate e1: /mnt/i/Github/Latent_Style/exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k085_appalign_seed42_b12a1/epoch_0001.pt
  transfer: 0.673/0.336, all-pairs: 0.704/0.333
```

## 推荐的完整训练流程

### 第一轮: 垂直 FM + OT 修复组合

创建新实验目录并启动：

```bash
cd /mnt/i/Github/Latent_Style/SchrodingerBridge

# 创建实验目录
mkdir -p exp/phase616_combined_v2
cp exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k085_appalign_seed42_b12a1/config.json exp/phase616_combined_v2/
cp exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k085_appalign_seed42_b12a1/epoch_0001.pt exp/phase616_combined_v2/

# 修改 config.json 关键参数 (用 Python)
python -c "
import json
c = json.load(open('exp/phase616_combined_v2/config.json'))
c['bridge']['bridge_path_mode'] = 'vertical'
c['bridge']['coupling_solver'] = 'sinkhorn_unbalanced'
c['bridge']['sinkhorn_unbalanced_tau_src'] = 0.5
c['bridge']['coupling_structure_cost_mode'] = 'topogate_attention_gw'
c['bridge']['coupling_cost_composition'] = 'appearance_plus_structure'
c['bridge']['coupling_structure_cost_weight'] = 0.3
c['tokenizer']['tokenizer_name'] = 'legacy_factorized'
c['tokenizer']['ablation_disable_spatial_prior'] = True
c['training']['num_epochs'] = 12
c['training']['batch_size'] = 8
c['training']['accumulation_steps'] = 1
c['training']['learning_rate'] = 0.0002
c['checkpoint']['save_dir'] = './exp/phase616_combined_v2'
json.dump(c, open('exp/phase616_combined_v2/config.json', 'w'), indent=2)
print('Config updated')
"

# 启动训练
python src/run.py --config exp/phase616_combined_v2/config.json \
  --resume exp/phase616_combined_v2/epoch_0001.pt
```

### 关键监控指标

训练时观察以下指标（CSV/tqdm 输出）：

| 指标 | 健康范围 | 危险信号 |
|------|----------|----------|
| `flow_loss` | < 1.0 | > 2.0 (loss 爆炸) |
| `terminal_swd` | < 0.05 | > 0.1 (style 分布不匹配) |
| `w_kinetic × kinetic_energy` | 0.05-0.15 | < 0.01 (kinetic 失效) 或 > 0.3 (过度约束) |
| `ot_target_gini` | < 0.4 | > 0.6 (OT 枢纽现象) |
| `ot_structure_cost_var` | > 0 | ≈ 0 (结构代价退化) |
| `tok_delta` (tokenizer) | > 0.03 且递增 | < 0.01 (tokenizer 不工作) |
| `topo_entropy` | 0.5-1.5 | > 2.0 (路由完全退化) |

### 结果判断

每 2 个 epoch 检查一次 eval 结果：

```bash
# 读取最新 eval
cat $EXP_DIR/full_eval/clip_lpips_curve.csv | tail -1
```

**决策树**:
```
epoch 2:   style < 0.68 AND LPIPS < 0.35? → 继续
           LPIPS > 0.40? → 降低 bridge_vertical_base_stride 从 2 → 3
epoch 4:   style > 0.70? → 维持参数
           style < 0.69? → 提高 tokenizer lr ×1.5, 或降低 w_kinetic 0.85→0.65
epoch 8:   style > 0.72? → 成功！
           style 0.70-0.72 且 LPIPS < 0.35? → 延长至 16 epochs
           style < 0.70? → 切换到 I2SB endpoint + topogate 路线
epoch 12:  最终判断
```
