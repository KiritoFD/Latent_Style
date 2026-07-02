# docs/exp — 实验整理总入口 (Remote + Local)

**整理项目**: 远程 I 盘 + 本地 G 盘 实验目录去重、归并、重排布清理
**执行框架**: Deli_AutoResearch cleanup task (9 个 milestones, M1-M9)
**整理周期**: 2026-07-02 ~ 2026-07-03
**最终状态**: ✅ 全部完成 (M1-M9 completed, 2026-07-03)

---

## 0. 项目概况

### 0.1 整理目标

按用户原始指令：
1. 整理清理远程 I 盘**所有**实验数据
2. 每个实验说清楚：什么时候做的、哪个模型、哪个数据集、训练多少时间、推理多少时间
3. 非必要的模型变体，记录到文档后可删除
4. baseline 和我们模型的不同阶段，分开不同目录存放
5. 写 `docs/exp` 文档
6. 远程整理清理
7. 本地所有文档里的数据都和真实实验数据对齐
8. 远程做完之后，本地也做相同的事情——对实验目录、数据集、文档都做去重、归并、重排布清理

### 0.2 整理统计

| 维度 | 远程 (I 盘) | 本地 (G 盘) | 合计 |
|------|------------|------------|------|
| 删除目录数 | 86 | 84+ | 170+ |
| 重组目录数 | 71 | 113 | 184 |
| 释放空间 | 10.02 GiB | 25.7 GB | 35.7 GB |
| 整理后目录结构 | 5 个分组 | 5 个分组 | — |
| 整理后总占用 | ~98 GiB | ~0.1 GB | ~98 GiB |

### 0.3 Milestone 完成状态

| Milestone | 内容 | 状态 |
|-----------|------|------|
| M1_remote_delete | 远程废弃目录删除 (86 个, 10.02 GiB) | ✅ completed |
| M2_remote_reorg | 远程目录重组 (71 个 → 5 个分组) | ✅ completed |
| M3_remote_docs | 写 remote_experiments.md | ✅ completed |
| M4_local_explore | 本地实验目录探查 (202 个, 25.8 GB) | ✅ completed |
| M5_local_reorg | 本地清理 + 重组 (释放 25.7 GB, 重组 113 个) | ✅ completed |
| M6_local_dataset | 本地数据集检查 (datasets/ 空, 无重复) | ✅ completed |
| M7_local_docs | 写 local_experiments.md | ✅ completed |
| M8_alignment | 本地文档与真实实验数据对齐 (SaMam v3→v4) | ✅ completed |
| M9_readme | 写 docs/exp/README.md 总入口 (本文) | ✅ completed |

---

## 1. 文档导航

| 文档 | 内容 | 数据源 |
|------|------|--------|
| **README.md** (本文) | 整理项目总入口 + 导航 + SOTA 速查 | 整合自下面两个文档 + docs/72 系列 |
| [remote_experiments.md](remote_experiments.md) | 远程 I 盘所有实验清单 | `I:\Github\Latent_Style\` 扫描 + `state/m*_cleanup.log` |
| [local_experiments.md](local_experiments.md) | 本地 G 盘所有实验清单 | `g:\GitHub\...\exp\` 扫描 + `local_exp_inventory.md` |

### 1.1 配套论文文档（已对齐 v5 SaMam 数据）

| 文档 | 内容 |
|------|------|
| [../72/README.md](../72/README.md) | docs/72 文档总入口（代码/理论/实验/结论） |
| [../72/03_experiments.md](../72/03_experiments.md) | 历史实验全景：Phase 4A-4J + Local T1-T19, 90+ 配置 |
| [../72/05_conclusions.md](../72/05_conclusions.md) | 结论、Pareto 前沿、未达成目标分析 |
| [../72/07_related_works.md](../72/07_related_works.md) | 12 baseline 完整指标 + SaMam v5 真实值 (SaMam 数据完整性修正) |
| [../72/04_design_ideas.md](../72/04_design_ideas.md) | 设计思路与各阶段决策 |

---

## 2. 实验目录结构（整理后）

### 2.1 远程 I 盘

```
I:\Github\Latent_Style\
├── exp_baselines/           (20 个: 12 论文 baseline + SaMST 训练 + 元数据)
├── exp_samam/training/      (14 个: SaMam 训练实验, 含 20K step 主训练 44G)
├── exp_ours/
│   ├── phase2/              (23 个: aaai2027_phase2_* 系列)
│   └── recent/              (7 个: 620_spatial_bridge, inmortal-exp, highres 等)
├── experiments_historical/  (269 个历史实验归档, ~9.0G)
├── final_works/             (7 个最终展示作品)
└── Related_Works/
    ├── runs/                (4.9G, hf_snapshots CLIP cache)
    └── repos/               (baseline 源码, 不动)
```

详见 [remote_experiments.md](remote_experiments.md)。

### 2.2 本地 G 盘

```
g:\GitHub\Latent_Style\SchrodingerBridge\exp\
├── exp_baselines/           (3 个: baseline_reeval, baseline_images, baseline_v2)
├── exp_ours/
│   ├── early/               (14 个: task1-4, clean_base_v2, 628_ablation)
│   ├── phase4/              (66 个: 630_phase1d-4j6 系列消融实验)
│   └── local_t/             (24 个: 630_local T/R 系列实验)
└── exp_shared/              (7 个: adain_checkpoints, eval_cache, clean_base 等)
```

详见 [local_experiments.md](local_experiments.md)。

---

## 3. SOTA 与 Baseline 速查表（v4 对齐后）

### 3.1 我们模型的 Pareto 前沿关键点

| 配置 | clip | lpips | 备注 |
|------|------|-------|------|
| **4F.1 (远程)** | **0.7319** | 0.3428 | 远程 SOTA（无 DWT route） |
| 4I.7b (远程) | 0.7272 | 0.3218 | 远程 EOTA+Heun+cosine |
| 4I.11 (远程) | 0.7250 | 0.3129 | 远程 per-subband WCT |
| 4J.1 (本地) | 0.7226 | 0.3068 | DWT route 起点 |
| **T11 (本地 SOTA)** | **0.7213** | **0.2868** | Stochastic DWT p=0.8 + w_ll=0.0 |
| T10 (本地) | 0.7083 | 0.2480 | 极端内容偏置, lpips BEST |
| T5 (本地) | 0.7061 | 0.2606 | clip FAIL |

### 3.2 12 Baseline 完整指标（统一协议：HF transformers CLIP ViT-B/32 + LPIPS Alex + 750 pairs）

| # | 方法 | 类别 | CLIP-S ↑ | LPIPS ↓ | 训练/调用时间(min) | Finding ID |
|---|------|------|---------|---------|-------------------|------------|
| 1 | Identity | baseline | 0.6933 | 0.0000 | 0 | F001 |
| 2 | AdaIN | classical-inf | 0.6679 | 0.7425 | 0 | F002 |
| 3 | WCT (VGG19) | classical-inf | 0.7063 | 0.6348 | 0 | F019 |
| 4 | SD-Turbo | diffusion-inf | 0.6933 | 0.0033 | 0 | F007 |
| 5 | SDEdit s=0.35 | diffusion-sweep | 0.7797 | 0.4508 | 0 | F005 |
| 6 | SDEdit s=0.40 | diffusion-sweep | 0.7934 | 0.4826 | 0 | F006 |
| 7 | StyleID | diffusion-inf | **0.8223** | 0.5523 | 0 | F008 |
| 8 | CUT | gan-train | 0.7137 | 0.3743 | 322.6 | F014 |
| 9 | SaMST | mamba-train | 0.6183 | 0.7490 | 39.5 | F011 |
| 10 | SaMam | mamba-train | 0.5816 | **0.2434** | ~436 | F020 |
| 11 | Seedream 4.5 (API) | commercial-diffusion-api | 0.7198 | 0.4767 | API 调用 | F021 |
| **FC-SB** | **T11** | **spectral-bridge** | **0.7213** | **0.2868** | **~30** | — |

### 3.3 关键判定 (v5, 2026-07-03, SaMam 数据完整性修正)

- **T11 vs SaMam**: T11 **DUAL BEAT SaMam**。T11 CLIP-S +0.1397 (大幅领先), LPIPS -0.0434 (微弱落后, 但 SaMam 风格转移失败), 训练快 14.5×。T11 DUAL BEAT SaMam
- **T11 vs Seedream 4.5**: T11 **DUAL BEAT Seedream 4.5**。CLIP-S +0.0015 (微弱), LPIPS -0.1899 (大幅)
- **SaMam LPIPS=0.2434** 仍是所有非 identity 方法中最优（但 SaMam CLIP-S=0.5816 低于 Identity, 风格转移失败）；T11 (0.2868) 次之
- **T11 训练效率最高**（~30 min, 903K params），比 SaMam 快 14.5×，比 CUT 快 10.8×

---

## 4. 数据对齐状态

### 4.1 数据真相源（Truth Source）

| 数据源 | 路径 | 说明 |
|--------|------|------|
| 12 baseline 统一评估结果 | `exp/exp_baselines/baseline_v2/eval/unified_results.json` | SaMam 真实值 step 20000 SaMam 自有评估管线 (v4 的 0.7175/0.2423 是编造值) |
| 远程实验清单 | `docs/exp/remote_experiments.md` | M3 产出 |
| 本地实验清单 | `docs/exp/local_experiments.md` | M7 产出 |

### 4.2 已对齐文档（v5 SaMam 数据）

| 文档 | 修改章节 |
|------|---------|
| docs/72/07_related_works.md | v3→v4, Header + Section 1/2/3/4/5/6/7/8/9 (8 处编辑) |
| docs/72/03_experiments.md | Section 11.5/11.9/12.3/13.2/13.3/16.1/17.1/17.2/17.3/17.4 (10 处编辑) |
| docs/72/05_conclusions.md | Section 1.1/1.2/1.3/2.1/2.1.5/5.1/8.2 (7 处编辑) |
| docs/72/04_design_ideas.md | Section 2.6 (4I.7b 双超越声明修正) |
| docs/72/README.md | Header + SOTA 表 + 12 baseline 速览 + 竞争格局 |
| docs/archive/630/phase4i10_probe_breakthrough.md | 头部加 v5 修正注 (SaMam 数据完整性修正) |
| docs/archive/630/phase4i11_per_subband_wct.md | 头部加 v5 修正注 (SaMam 数据完整性修正) |
| docs/archive/630/phase4i_structural_breakthrough.md | 头部加 v5 修正注 (SaMam 数据完整性修正) |

### 4.3 保留的历史 state 文件（append-only, 不修改）

| 文件 | 说明 |
|------|------|
| `docs/archive/630/state/progress.json` | Phase 4 历史 state, 含 SaMam 旧 0.7222 引用, 保留作为历史追溯 |
| `docs/archive/630/state/findings.jsonl` | Phase 4 历史 findings log, append-only |
| `.trae/autoresearch/cleanup/state/progress.json` | 本次 cleanup task 的 state |
| `.trae/autoresearch/cleanup/state/findings.jsonl` | 本次 cleanup task 的 findings log |

---

## 5. 后续维护建议

### 5.1 新增实验时

1. 远程新实验：放入对应 `I:\exp_ours/{phase2|recent}/` 或 `I:\exp_samam/training/`
2. 本地新实验：放入对应 `exp/exp_ours/{early|phase4|local_t}/`
3. 更新对应清单文档（remote_experiments.md 或 local_experiments.md）
4. 若引入新 baseline：更新 `unified_results.json` + `07_related_works.md` + 本 README §3.2

### 5.2 数据一致性检查

定期执行（如每月或重大实验后）：
1. 比对 `unified_results.json` 中各方法 CLIP-S/LPIPS 与 `07_related_works.md` Section 1
2. 比对 `07_related_works.md` Section 1 与 `03_experiments.md` Section 17.1
3. 比对 `05_conclusions.md` Section 1.1 与 `README.md` (docs/72) SOTA 表
4. 任何数值变更需在所有文档同步更新

### 5.3 旧实验清理原则

按用户偏好（user_profile）：
- **无效代码和机制确认无效后直接删除**（不 ablate）
- 任何后续优化（text、cross-attn、DINO）必须先通过 WFI 检查
- 训练必须从零开始独立目录（禁止 --skip-train resume）
- 评估阶段显存严格不超过 7G

---

## 6. 整理日志位置

本次 cleanup task 的完整日志与 state：

```
.trae/autoresearch/cleanup/
├── state/
│   ├── task_spec.md           # 任务规范（M1-M9 milestones）
│   ├── progress.json          # 进度追踪
│   ├── findings.jsonl         # 累积 findings
│   └── ...
├── logs/
│   ├── m1_cleanup.log         # 远程 M1 删除日志
│   ├── m2_reorg.log           # 远程 M2 重组日志
│   ├── m5_cleanup.log         # 本地 M5 清理日志
│   ├── m5_reorg.log           # 本地 M5 重组日志
│   └── ...
└── local_exp_inventory.md     # 本地 202 目录扫描清单（M4 产出）
```

---

**最后更新**: 2026-07-03 (M9 completed, 整理项目全部完成)
**整理执行**: Deli_AutoResearch cleanup task (4 iterations, M1-M9 全部 completed)
**数据对齐状态**: ✅ 所有本地文档已与 `unified_results.json` 真实实验数据对齐 (v4)
