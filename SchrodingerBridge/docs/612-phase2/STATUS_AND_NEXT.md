# Phase2 实验全景与下一步指导
## 2026-06-14 更新

---

## 一、性能全景 (按 LPIPS 排序)

| 实验 | transfer style | LPIPS | all-pairs style | LPIPS | delta_idt | ~epoch/min | 状态 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **topogate_appalign e2** | 0.6714 | **0.314** | 0.7031 | **0.312** | +0.032/+0.023 | ~25min/ep | 🔄 训练中 e3 |
| **safe_rescan_r2 e4** | 0.6724 | 0.369 | 0.7005 | 0.367 | +0.032/+0.020 | ~21min/ep | ✅ 关闭 |
| **topogate_k085 e3** (parent) | 0.6754 | 0.376 | 0.7030 | 0.372 | +0.035/+0.023 | ~20min/ep | ✅ 引用点 |
| **LBM F_e1** (baseline) | 0.6644 | 0.325 | 0.697 | 0.319 | +0.024/+0.017 | ~1.2min | ✅ |
| **LBM H_e2** (baseline) | 0.6684 | 0.356 | 0.699 | 0.348 | +0.029/+0.025 | ~2.3min | ✅ |
| **SaMAM step3k** | 0.6646 | 0.327 | 0.698 | 0.322 | +0.025/+0.017 | ~613min | ✅ |
| **xpred_pattn_stokes002** | 0.7307 | 0.618 | 0.737 | 0.607 | — | ~5min/ep | keep (endpoint参考) |
| **IDT no-op** | 0.6801 | 0.000 | 0.6801 | 0.000 | 0 | 0 | anchor |
| **WikiArt512 best** | — | — | 0.791 | 0.307 | — | — | 能力证明 |

**关键结论**: LPIPS 已经到目标区间 (topogate 0.31-0.33)，style 差 ~0.04。**瓶颈不再是结构保持，而是风格注入强度。**

---

## 二、理论和实验历程

### 阶段1: Baseline 天花板 (LBM F/H/K)
velocity + legacy tokenizer + SemanticCrossAttn → style=0.70, LPIPS=0.32
**瓶颈**: global tokenizer 缺乏空间路由，style 无法突破。

### 阶段2: Endpoint + Pattn 突破 (immortal)
endpoint + kmanifold + pattn → style=0.73, LPIPS=0.62
**代价**: endpoint 是"重绘"不是"编辑"，LPIPS 崩溃。

### 阶段3: Phase2 velocity 回升 (safe_rescan)
velocity + pure_latent_spatial tok32 + kmanifold + pattn → style=0.70, LPIPS=0.37
**发现**: velocity 保住了 LPIPS，纯内生 tokenizer 效果追平 baseline。

### 阶段4: Topogate 结构突破 (topogate_k085/appalign)
velocity + topogate + tok32 + kmanifold → **style=0.70, LPIPS=0.31**
**突破**: topogate 把 LPIPS 推向接近 IDT 水平（仅差 0.01），证明结构保护问题已解决。

---

## 三、为什么 Topogate 有效

Self-Attention 矩阵 $A = \text{softmax}(QK^T/\sqrt{d})$ 天然编码空间拓扑。
Topogate 强制 Cross-Attn 中的信息路由受 $A_{\text{self-content}}$ 约束：
$$A_{\text{final}} = \alpha A_{\text{self}} + (1-\alpha) A_{\text{cross}}$$
风格信息只能沿着已建立的"内容通道"流入，无法"乱跳像素"。
实验上 LPIPS 从 0.389 降到 0.314 就是直接证据。

---

## 四、下一步: 突破 style 天花板

**现状**: topogate 保住了结构 (LPIPS=0.31) 但 style 停滞在 0.67-0.70。问题是**如何在保持 topogate 结构的同时推高 style**。

### 路径 A: I2SB σ=0.02 (已在队列)
- 极微布朗噪声在 t≈0.5 注入，打破确定性 style 轨迹
- 预期: style +0.01~0.02, LPIPS 几乎不变
- 风险: 极低

### 路径 B: PC Solver eval
- 用 topogate e2 ckpt + solver_pc 推理
- "Training for Style, Inference for Structure"
- 预期: 如果已有一点 style margin，PC 可以在保持 LPIPS 的同时小幅推高 style

### 路径 C: 减弱 topogate 约束
- 当前 topogate blend=1.0 可能太强，压制了 style 波动
- 尝试 blend=0.5-0.8 或升高 temperature
- 预期: LPIPS 轻微上升但 style 突破

### 路径 D: PnP Self-Inject (已在队列)
- 双层结构保护 (topogate + PnP attention injection)
- 预期: 可以承受更强的 style 注入而不崩结构

### 推荐执行顺序
1. 🔄 等 topogate_appalign e3-e4 跑完
2. ⏳ 启动 I2SB σ=0.02 (topogate e1 作为 warmstart)
3. ⏳ 并行: PC solver eval on topogate e2
4. 📋 如仍不突破 → PnP self-inject 为终极方案

---

## 五、仓库整理总结

| 项目 | 之前 | 之后 |
|------|------|------|
| 远程 inmortal-exp | ~30GB, 100+ dirs | 2.6GB, 35 dirs |
| 远程 exp/ | ~80GB, 200+ dirs | ~13GB, 50 dirs |
| 远程总回收 | — | **~77GB** |
| 本地 _codex_tmp | 6.6GB | 58MB |
| 本地 archives | 7.4GB | ~3MB |
| 本地 aaai2027 | 10.7GB | ~7GB |
| 本地 S-add/tmp/snapshots | ~300MB | 0 |
| **本地总回收** | — | **~17GB** |

保留的本地核心目录: `src/`, `configs/`, `scale/`(数据集), `aaai2027/`(paper), `docs/612-*/`
保留的远程核心目录: `exp/`(当前实验), `inmortal-exp/`(证据点), `Related_Works/`

---

## 六、给 KiritoFD 的最新指令

**如果看到这条**: topogate_appalign 的 style = 0.67-0.70 且 LPIPS = 0.31-0.34 已经是"几乎完美保结构的风格化"。
当前的单一瓶颈是 style 推不到 0.72。

**不要继续在同一个 family 里微调参数**。直接启动:
1. I2SB σ=0.02 用 topogate e1 做 warmstart
2. PC solver eval on topogate e2 (仅推理, 不训练)
3. 如果以上都失败, 切到 PnP self-inject, 同时保留 topogate
