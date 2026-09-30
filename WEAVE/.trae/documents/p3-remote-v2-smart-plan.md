# P3 远程实验 v2：聪明高效的突破计划

> 基于 R1 完整结果（9 epoch）的诊断，重新设计实验策略。
> 核心发现：长训练无效（clip_style 平坦，LPIPS 恶化），需要**组合式突破 + 短轮次快速迭代**。

## 1. R1 结果诊断

### 1.1 定量数据（完整）

| Epoch | clip_style (tr) | LPIPS (tr) | clip_style (all) | LPIPS (all) | \|v\| |
|-------|----------------|------------|-------------------|-------------|--------|
| 2 | **0.6723** | **0.3608** | 0.7041 | **0.3618** | ~0.40 |
| 3 | 0.6745 | 0.3862 | 0.7054 | 0.3870 | ~0.50 |
| 4 | 0.6754 | 0.3973 | 0.7055 | 0.3981 | ~0.54 |
| 5 | 0.6756 | 0.4141 | 0.7051 | 0.4142 | ~0.56 |
| 6 | 0.6767 | 0.4104 | 0.7065 | 0.4108 | ~0.58 |
| 7 | 0.6769 | 0.4144 | 0.7065 | 0.4145 | ~0.59 |
| 8 | 0.6772 | 0.4183 | 0.7064 | 0.4181 | ~0.60 |

### 1.2 关键诊断

**诊断 A：训练过拟合 FM 目标**
- Loss: 2.90 → 2.33 (-19.7%)，持续下降
- clip_style: 0.672 → 0.677 (+0.7%)，几乎不动
- **解释**：FM loss 是强凸二次函数，模型持续优化 velocity 接近 target velocity，
  但 target velocity 本身就不包含足够的 style 差异信息（多风格平均效应）
- **结论**：更多 epoch 不会帮助跳出平凡解，只会让 LPIPS 更差

**诊断 B：LPIPS 恶化模式**
- Ep2→Ep8: LPIPS 从 0.361 升到 0.418 (+15.7%)
- 这不是"先好后坏"，而是**单调恶化**
- 原因：velocity 幅度增大 (0.32→0.60)，但方向不精确，
  导致 content distortion 累积

**诊断 C：最佳 early stopping 点**
- **Epoch 2 是全局最优 Pareto 点**
- 之后每多训一个 epoch，LPIPS 增加 ~0.01，clip_style 只增加 ~0.001
- ROI 极低：+0.001 style / -0.010 content per epoch

## 2. 新策略设计

### 2.1 从 R1 学到的教训

| 教训 | 旧计划 | 新计划 |
|------|--------|--------|
| 训练长度 | 10 ep/实验 × 5 实验 = 50 ep 总计 | **3 ep/实验** × 6 实验 = 18 ep 总计 |
| 单变量改变 | R1=baseline, R2=只改 FiLM init | **每个实验同时改 2-3 个变量**（乘积效应） |
| 执行方式 | tmux + bat wrapper（不稳定） | **纯 .sh 脚本 + nohup**（稳健） |
| HF cache | 路径错误导致反复崩溃 | **预修复路径 + 预下载模型** |
| 评估频率 | 每 epoch 都评估（浪费时间） | **只在最后 1 个 epoch 评估** |

### 2.2 时间预算重分配

```
旧计划（失败）：
  R1(10ep×~13min) + eval(~3min/ep) ≈ 160 min ❌ 太慢
  R2+R3 同上... 总计 > 10h 且未完成

新计划（v2）：
  每个 experiment: 3ep × ~78s/ep = ~4min 训练 + ~3min 评估 = ~7min
  6 个 experiments: 6 × 7min = 42min 训练 + 评估
  + AdaIN 后处理评估: ~10min
  + 图片目视检查缓冲: ~20min
  总计: ~75min（远小于 10h 预算）
  
  剩余时间可用于：
  - 如果某个方向有效，追加更长训练（5-10ep）
  - 尝试更多组合变体
  - 多跑几轮确认结果可复现
```

## 3. 实验设计（6 轮，基于理论文档 Section 4 的优先级排序）

### 设计原则
1. **每个实验必须同时打破 ≥2 个保守机制**（乘积效应要求）
2. **基于 R1 最佳配置作为起点**（fixed_one + FiLM + wflow=0.5 + vel_mag=0.5）
3. **所有实验 3 epoch + 仅最终评估**（节省时间）
4. **按理论 ROI 降序排列**

### E1: FiLM 大初始化 + 对比损失（打破 Layer 4 + 增强 Style 分化）

**打破的条件**：
- Layer 4（零初始化）：`film_init_std=0.05`, `endpoint_film_init_std=0.1`
- Style 梯度衰减：`w_style_contrastive=0.3` 强制不同 style 输出差异化

**相对于 R1 baseline 的变化**：
```json
{
  "model": {
    "film_init_std": 0.05,
    "endpoint_film_init_std": 0.1
  },
  "bridge": {
    "w_style_contrastive": 0.3,
    "contrastive_margin": 0.05
  }
}
```
**其他参数与 R1 完全相同**（控制变量）

**预期**：epoch 1 就有更强的 style 注入，clip_style 应该比 R1 同期高 > 0.02

---

### E2: 两阶段训练 + FiLM 大初始化（打破 FM 主导 + Layer 4）

**打破的条件**：
- FM 主导条件：Stage 1 用 endpoint mode + 低 w_flow_scale
- Layer 4（零初始化）：同 E1

**关键变化**：
```json
{
  "model": {
    "film_init_std": 0.05,
    "endpoint_film_init_std": 0.1
  },
  "bridge": {
    "training_objective_mode": "endpoint",
    "w_flow_scale": 0.3,
    "w_endpoint_velocity_reg": 0.1,
    "two_stage_enabled": true,
    "two_stage_s1_epochs": 2,
    "two_stage_s1_w_endpoint_content": 0.3,
    "two_stage_s1_w_endpoint_style": 16.0,
    "two_stage_s1_w_style_strength_reg": 1.0,
    "two_stage_s2_w_endpoint_content": 1.0,
    "two_stage_s2_w_endpoint_style": 8.0,
    "two_stage_s2_w_style_strength_reg": 0.5
  }
}
```
**注意**：切换到 endpoint 模式（非 velocity），因为两阶段围绕 endpoint loss 设计

**预期**：Stage 1 (ep 1-2) 激进注入 style，Stage 2 (ep 3) 微调平衡

---

### E3: 激进组合版（同时打破 3-4 个条件）

**基于 E1 或 E2 中更好的那个，进一步激进化**：

如果 E1 更好（候选配置 A）：
```json
{
  "model": {
    "film_init_std": 0.05,
    "endpoint_film_init_std": 0.1
  },
  "bridge": {
    "w_flow_scale": 0.3,
    "w_velocity_magnitude": 0.8,
    "w_style_strength_reg": 0.8,
    "w_contrast_preserve": 1.5,
    "w_hf_energy": 1.5,
    "w_pixel_color_match": 1.5,
    "w_style_contrastive": 0.3,
    "contrastive_margin": 0.05,
    "single_step_swd_weight": 10.0
  }
}
```
如果 E2 更好（候选配置 B）：
```json
{
  "model": {
    "film_init_std": 0.05,
    "endpoint_film_init_std": 0.1
  },
  "bridge": {
    "training_objective_mode": "endpoint",
    "w_flow_scale": 0.2,
    "two_stage_enabled": true,
    "two_stage_s1_epochs": 2,
    "two_stage_s1_w_endpoint_style": 20.0,
    "two_stage_s1_w_style_strength_reg": 1.5,
    "w_style_contrastive": 0.2,
    "single_step_swd_weight": 12.0
  }
}
```
**E3 配置将在 E1/E2 结果出来后动态决定**（见第 5 节决策树）

---

### E4: 更大初始化 + 更强对比（如果 E1 方向有效但不够）

**假设**：E1 显示 FiLM init + contrastive 有正向效果但幅度不够
**策略**：加倍剂量
```json
{
  "model": {
    "film_init_std": 0.10,
    "endpoint_film_init_std": 0.20
  },
  "bridge": {
    "w_style_contrastive": 0.5,
    "contrastive_margin": 0.03
  }
}
```

---

### E5: 两阶段变体（如果 E2 方向有效但 Stage 1 不够激进）

**假设**：E2 显示两阶段有潜力但 S1→S2 过渡太平滑
**策略**：延长 S1 到全部 3 epoch（纯激进注入，不做微调）
```json
{
  "model": {
    "film_init_std": 0.05,
    "endpoint_film_init_std": 0.1
  },
  "bridge": {
    "training_objective_mode": "endpoint",
    "w_flow_scale": 0.2,
    "two_stage_enabled": true,
    "two_stage_s1_epochs": 3,
    "two_stage_s1_w_endpoint_style": 24.0,
    "two_stage_s1_w_style_strength_reg": 2.0,
    "two_stage_s1_w_endpoint_content": 0.1,
    "single_step_swd_weight": 12.0
  }
}
```

---

### E6: AdaIN 后处理评估（对最佳 checkpoint 做零成本去雾）

**不对任何实验做代码修改**，只修改推理时配置：
```json
{ "model": { "inference_adain": true } }
```
对 E1-E5 中最好的 checkpoint 重新运行 evaluation。

## 4. 运行脚本设计（纯 .sh，稳健执行）

### 4.1 主运行脚本：`run_v2.sh`

```bash
#!/bin/bash
# ============================================================
#  P3 Remote Experiment Suite v2
#  - Pure bash, no bat wrappers
#  - nohup background execution
#  - Error handling with continue-on-failure
#  - Auto-fix HF cache path
#  - 3 epochs per experiment, eval only at end
# ============================================================

set -uo pipefail
PROJECT_DIR="/mnt/i/Github/Latent_Style/SchrodingerBridge"
LOG_DIR="$PROJECT_DIR/exp/p3_remote_10h"
HF_CACHE="$PROJECT_DIR/../eval_cache/hf_hub"

cd "$PROJECT_DIR" || { echo "FATAL: cd failed"; exit 1; }

# Pre-check: ensure HF cache dir exists
mkdir -p "$HF_CACHE" 2>/dev/null

echo "=============================================="
echo " P3 Remote v2 Start: $(date)"
echo " Project: $PROJECT_DIR"
echo " GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "=============================================="

# Fix HF cache path in all configs before starting
for cfg in "$LOG_DIR"/e*/config.json; do
    if [ -f "$cfg" ]; then
        sed -i 's|/home/administrator/.cache/huggingface/hub|'"$HF_CACHE"'|g' "$cfg"
        echo "[pre] Fixed HF path: $cfg"
    fi
done

declare -a EXPERIMENTS=(
    "e1_film_init_contrastive"
    "e2_two_stage_film"
)

RUN_COUNT=0
FAIL_COUNT=0

for EXP_NAME in "${EXPERIMENTS[@]}"; do
    CONFIG_PATH="$LOG_DIR/${EXP_NAME}/config.json"
    EXP_LOG="$LOG_DIR/${EXP_NAME}/focused.log"
    
    # Skip if already completed (has epoch_0003.pt and full_eval)
    if [ -f "$LOG_DIR/${EXP_NAME}/epoch_0003.pt" ] && \
       [ -d "$LOG_DIR/${EXP_NAME}/full_eval/epoch_0003" ]; then
        echo "[SKIP] $EXP_NAME already completed"
        ((RUN_COUNT++))
        continue
    fi
    
    echo ""
    echo "##############################################"
    echo "#  [$((RUN_COUNT+1))/${#EXPERIMENTS[@]}] $EXP_NAME"
    echo "#  $(date)"
    echo "##############################################"
    echo ""
    
    mkdir -p "$LOG_DIR/${EXP_NAME}"
    
    # Run training (nohup ensures survival if SSH drops)
    echo "[$(date)] Starting training..."
    python run.py --config "$CONFIG_PATH" >> "$EXP_LOG" 2>&1
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date)] SUCCESS: $EXP_NAME training completed"
        ((RUN_COUNT++))
    else
        echo "[$(date)] WARNING: $EXP_NAME exited with code $EXIT_CODE"
        ((FAIL_COUNT++))
        # Continue to next experiment anyway
    fi
    
    # Check summary_grid
    LAST_EVAL=$(ls -d "$LOG_DIR/${EXP_NAME}"/full_eval/epoch_* 2>/dev/null | tail -1)
    if [ -n "$LAST_EVAL" ]; then
        if [ -f "${LAST_EVAL}/summary_grid.png" ]; then
            echo "  [OK] summary_grid.png at ${LAST_EVAL}"
        else
            echo "  [WARN] No summary_grid, attempting regeneration..."
            python run.py --config "$CONFIG_PATH" >> "$EXP_LOG" 2>&1 || true
        fi
    fi
    
    echo "--- Done $EXP_NAME ---"
done

echo ""
echo "=============================================="
echo " Results: $RUN_COUNT succeeded, $FAIL_COUNT failed"
echo " End: $(date)"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Review E1/E2 results"
echo "2. Create E3 config based on winner"
echo "3. Run E3-E6"
```

### 4.2 启动脚本：`launch_v2.sh`

```bash
#!/bin/bash
# Launch v2 experiments in nohup (SSH-safe)
PROJECT_DIR="/mnt/i/Github/Latent_Style/SchrodingerBridge"
cd "$PROJECT_DIR"

# Kill any existing run
pkill -f "run.py.*p3_remote_10h" 2>/dev/null || true

# Run in background with full output capture
nohup bash "$PROJECT_DIR/exp/p3_remote_10h/run_v2.sh" \
    > "$PROJECT_DIR/exp/p3_remote_10h/v2_master.log" 2>&1 &

PID=$!
echo "Launched! PID=$PID"
echo "Log: tail -f $PROJECT_DIR/exp/p3_remote_10h/v2_master.log"
echo "Check: ps -p $PID"
disown $PID  # Detach from terminal
```

### 4.3 状态检查脚本：`check_v2.sh`

```bash
#!/bin/bash
# Quick status check
PROJECT_DIR="/mnt/i/Github/Latent_Style/SchrodingerBridge"

echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null
echo ""
echo "=== Python Processes ==="
ps aux | grep "run.py" | grep -v grep | head -3
echo ""
echo "=== Experiments ==="
for d in "$PROJECT_DIR"/exp/p3_remote_10h/e*/; do
    NAME=$(basename "$d")
    PTS=$(ls "$d"epoch_*.pt 2>/dev/null | wc -l)
    LOG_LINES=$(wc -l < "$d/focused.log" 2>/dev/null || echo 0)
    LAST_LINE=$(tail -1 "$d/focused.log" 2>/dev/null || echo "no log")
    echo "  $NAME: ${PTS} ckpts, ${LOG_LINES} log lines"
    echo "    last: $LAST_LINE"
done
```

## 5. 动态决策流程

### 5.1 E1 vs E2 比较（在两者完成后决定 E3）

```
                    ┌─ E1 clip_style 更高 ─┐
                    │                       │
     E1 vs E2       │  → E3 = 候选配置 A     │
     比较            │   (velocity 模式      │
                    │   + 全部激进化)       │
                    │                       │
                    ├─ E2 clip_style 更高 ─┤
                    │                       │
                    │  → E3 = 候选配置 B     │
                    │   (endpoint +          │
                    │   强化两阶段)          │
                    │                       │
                    ├─ 两者都差 (< 0.68)   ──┤
                    │                       │
                    │  → 重新审视:           │
                    │   检查是否 HF cache     │
                    │   问题导致评估不准      │
                    │   或增加 epoch 数到 5   │
                    └───────────────────────┘
```

### 5.2 成功标准（每个实验）

| 指标 | R1 best (ep2) | 最低目标 | 理想目标 |
|------|---------------|---------|---------|
| clip_style (transfer) | 0.672 | > 0.685 | > 0.700 |
| LPIPS (transfer) | 0.361 | < 0.380 | < 0.350 |
| clip_style / LPIPS ratio | 1.86 | > 1.85 | > 2.00 |

**特别注意**：如果 clip_style 提升但 LPIPS 也提升，需要检查是否内容保持崩溃。
理想情况是 clip_style 提升 > LPIPS 恶化。

## 6. 文件操作清单

### 6.1 需要创建的文件（本地 → 远程 SCP）

| 文件 | 用途 |
|------|------|
| `exp/p3_remote_10h/e1_film_init_contrastive/config.json` | E1 实验配置 |
| `exp/p3_remote_10h/e2_two_stage_film/config.json` | E2 实验配置 |
| `exp/p3_remote_10h/run_v2.sh` | 主运行脚本 |
| `exp/p3_remote_10h/launch_v2.sh` | 启动脚本 |
| `exp/p3_remote_10h/check_v2.sh` | 状态检查脚本 |

### 6.2 远程操作步骤

```bash
# Step 1: SCP 所有文件到远程
scp -P 2222 exp/p3_remote_10h/e1_*/config.json admin@host:I:/.../e1_*/
scp -P 2222 exp/p3_remote_10h/e2_*/config.json admin@host:I:/.../e2_*/
scp -P 2222 exp/p3_remote_10h/*.sh admin@host:I:/.../

# Step 2: SSH 进入 WSL 并启动
ssh -p 2222 admin@host
wsl bash /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/p3_remote_10h/launch_v2.sh

# Step 3: 监控进度
wsl bash /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/p3_remote_10h/check_v2.sh
tail -f /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/p3_remote_10h/v2_master.log
```

## 7. 关键文件索引

| 文件 | 用途 |
|------|------|
| [losses620.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/losses620.py) | two_stage 权重调度(L87-131), contrastive loss, vel_mag loss |
| [model620.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/model620.py) | FiLMEndpointHead.film_init_std(L45-46), AdaIN |
| [blocks620.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/blocks620.py) | film_init_std(L140-165), style_gate_mode |
| [config_schema.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/config_schema.py) | 所有配置参数定义 |
| [trivial_solution_unified.md](file:///g:/GitHub/Latent_Style/SchrodingerBridge/docs/620/fog/theory/trivial_solution_unified.md) | 理论基础文档 |
| [r1_baseline_long/config.json](file:///g:/GitHub/Latent_Style/SchrodingerBridge/exp/p3_remote_10h/r1_baseline_long/config.json) | R1 基线配置（E1/E2 的模板） |
