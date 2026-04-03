# 🏛️ Latent AdaCUT 项目考古中文总纲

> **完整覆盖**: 2026-01-13 → 2026-04-02 | 220 个实验目录 | 150 个含评估数据 | 207 次 Git 提交
> **生成时间**: 2026-04-03 14:30 CST

---

## 一、项目全时间线

### Phase 0：前史（1月13日 – 2月7日）
**目录**: `Thermal/`, `DiT/`
**架构演化**:
- Jan 13-15: **数据准备期** — WikiArt 下载 + VAE 潜编码，无模型代码
- Jan 16-19: **DiT 实验期** — 尝试 Diffusion Transformer，因控制力不足放弃
- Jan 22: **Thermal 项目诞生** — `LGTUNet` (605行)，AdaGN + ResidualBlock + SelfAttention
- Jan 28-31: **Cross-Attn 第一次尝试与惨败** — 代码暴增至 1033行，3天后回滚到 217行。失败原因：MSE 爆炸、语义纠缠
- Feb 1-7: **极简主义期** — 砍掉所有复杂结构，专注 Loss 权重平衡
- **核心教训**: 确立了以 AdaGN 为核心的轻量级潜空间架构，拒绝 Transformer 堆叠

### Phase 1：重生（2月8日 – 2月13日）
**目录**: `Cycle-NCE/src/` 诞生
- Feb 8: `model.py` 仅 251 行 — `AdaGN` + `ResBlock` + `LatentAdaCUT` 三件套确立
- Feb 9: **过山车日** — 19 次提交，代码行数 369→484→248→478，架构决策剧烈震荡
- Feb 11-12: **频率分离注入** — mid-freq 用 map16，high-freq strokes 用 map32，纹理质量突破
- Feb 13: **Gram 白化机制** — 引入 SVD 协方差分解 + Channel Whitening，风格特征去相关

### Phase 2：信号分离之战（2月13日 – 2月22日）
**分水岭**: 从 Gram 矩阵转向 SWD
- Feb 14: Gram 白化机制引入（EVD 分解）
- Feb 15: 代码从 662 行骤减到 466 行（删除冗余 Gram 计算）
- Feb 17: **"SWD某些情况下有微弱作用，GRAM完全没用"** — 项目方向根本转变
- Feb 21: Diff-Gram 最后挣扎（"到0.07了"），随后正式抛弃

### Phase 3：SWD 革命（2月22日 – 3月8日）
- Feb 22: **Domain SWD 5.77x 突破** — Domain 级信号分离度是 Instance 级的 5.77 倍
- Feb 23: **5 风格联合训练** — Instance (1.15-1.23x) vs Domain (5.06-5.77x)
- Feb 26: **TextureDictAdaGN** — 从单一全局风格向量 → 纹理字典，"观察到笔触明显变化"
- Feb 27: Color 锚定引入 + FP32 切换 + 5×5 分类矩阵
- Mar 1: **NCE Loss 引入** — "分类准确率有提升，NCE loss 是有效的"
- Mar 4: **SpatialGate** — 轻量级 1×1 Conv + Sigmoid 空间门控
- Mar 8: NCE 移除 — 虽然提高了分类精度，但限制了细粒度纹理学习
- **关键配置**: `use_decoder_adagn=True, use_delta_highpass_bias=True, use_style_gate=True`

### Phase 4：Attention 大爆炸（3月19日 – 3月30日）
- Mar 20-22: **Color Loss 革命** — "channels 映射回 RGB 的缩略图 color loss 大赢"，TV Loss 正式死亡
- Mar 26: **Cross-Attn 回归成功** — 加入"亮度约束"，代码 +290 行
- Mar 29-30: **Swin-Transformer 注入** — 移位窗口注意力 + c-g-w backbone，代码达到峰值 ~1500 行
- Mar 30: "infra 推进 56s/epoch" — 训练效率优化

### Phase 5：微批大瘦身（4月2日）
- **"micro batch 效果大好"** — Trainer.py 从 1536 行 → 531 行（-65%！）
- 移除了：教师-学生蒸馏、NCE 计算、分类器
- 保留：纯 SWD + Color + Identity 三件套 + 微批训练循环
- 效果：风格/内容指标反而提升了

---

## 二、完整性能排行榜（全量数据）

### 🏆 Top 10 风格迁移（clip_style ↑）
| 排名 | 实验 | ST Style | ST Content | P2A Style | 时代 |
|:---:|---|:---:|:---:|:---:|---|
| 🥇 1 | **DiT 5style (experiments-swd8)** | **0.8204** | 0.923 | - | Feb 23 DiT |
| 🥈 2 | **DiT 5style strong** | **0.8137** | 0.901 | - | Feb 23 DiT |
| 🥉 3 | **DiT 2style** | **0.8120** | 0.893 | - | Feb 23 DiT |
| 4 | swd8_32x32 | 0.7163 | - | - | Mar 24 |
| 5 | **exp_S1_zero_id** | **0.7096** | 0.7133 | 0.7133 | Apr |
| 6 | **Decoder_D6_god_combo** | **0.7104** | 0.6922 | 0.7001 | Tokenized |
| 7 | **scan05_gate_0p75** | **0.7077** | 0.6935 | 0.6996 | Tokenized |
| 8 | **scan03_soft_hf** | **0.7066** | 0.7086 | 0.6992 | Tokenized |
| 9 | **scan01_base** | **0.7050** | 0.6951 | 0.7002 | Tokenized |
| 10 | final_demodulation | 0.707 | 0.54 | - | Mar |

### 🏆 内容保留最佳（clip_content ↑）
| 排名 | 实验 | ST Content | P2A Content |
|:---:|---|:---:|:---:|
| 🥇 1 | color_01_adain_wc2 | **0.8472** | 0.8116 |
| 🥈 2 | **LCE0_TV_Anchor** (epoch 15) | **0.760** | 高 |
| 🥉 3 | Exp_05_TV_Color | **0.7815** | 0.7668 |

### 📊 各系列完整数据对比

#### Inject 系列（7 实验，Mar 12）
| 实验 | ST Style | P2A Style | 说明 |
|---|---|---|---|
| inject_I5_body_decoder | **0.6857** | **0.6795** | 🏆 最佳：body+decoder 联合注入 |
| inject_I2_hires_decoder_only | 0.6849 | 0.6759 | 高分辨率 decoder 专用 |
| inject_I0_all_open | 0.6845 | 0.6761 | 全开注入基线 |
| inject_I3_progressive | 0.6848 | 0.6785 | 渐进注入 1.0→0.5→0.1 |
| inject_I4_body_hires | 0.6828 | 0.6772 | body + 高分辨率 |
| inject_I1_body_only | 0.6810 | 0.6736 | 仅 body 注入 |
| inject_I7_decoder_only | 0.6776 | 0.6693 | 仅 decoder（epoch 40） |

**结论**：差异很小（<1%），说明注入点选择影响温和。I5 (body+decoder) 略微胜出。

#### Scan 系列（6 实验，Tokenized 评估）
| 实验 | ST Style | ST Content | P2A Style | P2A Content |
|---|---|---|---|---|
| **scan05_gate_0p75** | **0.7077** | 0.6935 | 0.6996 | 0.6724 |
| scan03_soft_hf | **0.7066** | **0.7086** | 0.6992 | 0.6878 |
| scan01_base | 0.7050 | 0.6951 | **0.7002** | 0.6757 |
| scan04_gate_0p85 | 0.7051 | 0.6951 | 0.6971 | 0.6693 |
| scan06_id_0p50 | 0.7036 | 0.6950 | 0.6984 | 0.6735 |
| scan02_low_lr | 0.7014 | 0.7106 | 0.6923 | 0.6811 |

**结论**：scan05（0.75 gate）风格最高，scan03（soft HF）内容最好，scan01（baseline）P2A最高。**Gate 0.75 是甜点**。

#### Decoder D 系列（6 实验，Tokenized 评估）
| 实验 | ST Style | P2A Style | 说明 |
|---|---|---|---|
| **D6_god_combo** | **0.7104** ⭐ | **0.7001** | 🏆 全套组合：HF+Color+Gate+Rank |
| D3_hf_4p0 | 0.7039 | 0.6961 | 高频权重 4.0 |
| D2_color_1p5 | 0.6951 | 0.6879 | Color 权重 1.5 |
| D5_rank_8 | 0.6946 | 0.6776 | TextureDict Rank 8 |
| D4_gate_0p8 | 0.6920 | 0.6818 | Gate 0.8 |
| D1_tv_0p02 | 0.6802 | 0.6727 | TV 权重 0.02 |

**结论**：D6（全组合）胜出，HF 权重 4.0 是关键驱动力。

#### NCE+Gate 系列（9 实验）
| 实验 | P2A Style | 说明 |
|---|---|---|
| **nce-gate_norm-swd_0.45-cl_0.01** | **0.6861** | 🏆 Gate + SWD 高权重 |
| nce-gate_content | 0.6650 | Content 门控 |
| nce-gate_norm | 0.6512 | Norm 门控 |
| nce-swd_0.25-cl_0.01 | - | 基线对照 |
| nce_A1-A5 | - | NCE 消融（深/浅/粗/细/高TV） |

#### Spatial-AdaGN 系列（5 实验）
| 实验 | ST Style | P2A Style |
|---|---|---|
| spatial-adagn-expA-texture | 0.6457 | 0.6170 |
| spatial-adagn | 0.6456 | 0.6166 |
| spatial-adagn-expB-depth | 0.6415 | 0.6060 |

**结论**：空间 AdaGN 整体表现低于后续架构（~0.64 vs 0.68+），但 texture/dept 差异极小。

#### No-X 对照实验
| 实验 | ST Style | P2A Style | 结论 |
|---|---|---|---|
| no-dict-hf-swd | 0.6972 | **0.0** | ⚠️ 无 TextureDict = P2A 崩溃 |
| no-edge | 0.5756 | 0.5663 | ❌ 无 Edge Loss = 内容崩塌 |
| no-tv | - | - | TV Loss 可丢弃 |

---

## 三、核心架构教训

### 1. 简洁胜于复杂
- **MSCTM 失败**: Decoder-H 过参数化，3/4 实验崩溃为恒等函数
- **G1/G2 架构改进失败**: Edge Rush 和 Dense Pyramid 不如基线
- **微批 > 蒸馏**: 简单的微批循环比 Teacher-Student 更有效

### 2. Loss 信号纯度 > 复杂度
- 移除 NCE（虽提升分类精度但限制纹理学习）
- 移除 TV（无实质贡献）
- 保留：SWD + Color + Identity 三件套

### 3. 低维纹理表达
- TextureDict Rank 1 = Rank 8 效果几乎相同
- 纹理是极低维信号，不需要高维字典

### 4. 架构韧性
- Master Sweep 15 个超参实验仅 0.02 差异
- 注入点选择影响 <1%
- Gate 0.75 是微妙甜点而非剧烈拐点

### 5. 效率 vs 绝对性能
- DiT 风格分数最高 (0.82)，但项目选择了更轻量的 CNN (0.71)
- 12% 的性能换来了更低的计算成本和更稳定的训练

---

## 四、文档体系索引

| 文档文件 | 大小 | 内容范围 |
|---|---|---|
| `ARCHAEOLOGY_PRE_SCRATCH.md` | 17KB | 1月13日-2月7日 前史（DiT → Thermal） |
| `ARCHAEOLOGY_MODEL_EVOLUTION.md` | 99KB | 2月8日-4月2日 11个架构时代详细演化 |
| `ARCHAEOLOGY_PART2.md` | 87KB | Gram→SWD→NCE→Attention 早期实验对照 |
| `ARCHAEOLOGY_PART3.md` | 46KB | 微批革命→Exp系列→排行榜 |
| `ARCHAEOLOGY_TOOLS_EVOLUTION.md` | 53KB | Losses/Trainer/Config 演化 |
| `ARCHAEOLOGY_CODE_EVOLUTION.md` | 29KB | 代码行数演化 + 微批训练革命 |
| `ARCHAEOLOGY_REPORT.md` | 139KB | 总报告（109配置+52实验+脚本分析） |
| **本文件** | - | 中文总纲 |
| **总计** | **~470KB** | **8个文档，约10000+行** |

---

## 五、实验目录覆盖率确认

- **Y:\experiments 总计**: 220 个目录
- **有 full_eval 数据**: 150 个目录 ✅
- **所有 150 个有数据的目录已被分析**: ✅
- **已追踪系列**: history_configs(109), Optuna HPO(31), master_sweep(21), nce(9), scan(6), decoder A-H(15+), spatial-adagn(5), micro(5), Exp(15+), ablate(22), color(7), inject(8), patch-*, swd8(27+), LCE(6), clocor1(5), delta(2), no-*(3), swd-*(2)
- **无数据目录**: 70 个（早期壳目录、RAR 压缩包、无输出的实验）— 全部确认过

---

*本报告基于 `G:\GitHub\Latent_Style\.git` (207 commits), `C:\Users\xy\full_history` (Python-only 过滤仓库), 以及 `Y:\experiments` 下 220 个实验目录的完整扫描和分析。*
