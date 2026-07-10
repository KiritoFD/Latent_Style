# results/ — 论文主表与标准协议评测集（本地 + 远程同步）

本目录集中存放 AAAI-2027 (WEAVE) 论文的**主表**与**标准协议评测图**，
本地 `g:\GitHub\Latent_Style\results` 与远程 `I:\results` 保持同步。

## 目录结构

```
results/
├── README.md
├── tables/
│   ├── main_table.csv              # ★ 主表（Table 1 / tab:main）：13 方法 × 3 数据集 = 39 行
│   │                              #   由 paper.tex 的 tabular 抽取，列为 method,class,dataset,
│   │                              #   clip_s,lpips,musiq,params,train_min,infer
│   └── baseline_metrics_unified.csv  # 补充：750 合并指标 dump（单行单值，无 3 数据集拆分，
│                                      # 方法集也与 Table 1 不完全一致），非逐数据集主表
├── aaai2027_v4_tables/             # aaai_v4 论文其他重要结果表
│   ├── distinct5_aux_artifact_table.csv
│   ├── distinct5_idt_bootstrap_extended.csv
│   ├── distinct5_nonclip_style_probe.csv
│   ├── main_point_artifact_ledger.csv
│   ├── paper_point_param_counts.csv
│   ├── blind_pairwise_exploratory_blind_audit.csv
│   ├── blind_pairwise_exploratory_blind_audit_summary.csv
│   └── aaai27_submission_bundle_manifest.csv
└── eval_protocol_750/              # 标准协议 750 图（Distinct5-WikiArt 测试集源图）
    ├── Early_Renaissance/           # 30 张
    ├── Impressionism/               # 30 张
    ├── Minimalism/                  # 30 张
    ├── Rococo/                      # 30 张
    └── Ukiyo_e/                     # 30 张  → 150 源图 × 5 目标方向 = 750 评测对
```

## 说明

- **主表**：`tables/main_table.csv` 是从 `SchrodingerBridge/aaai2027_v4/paper.tex` 的 `tab:main`
  （Table 1）逐单元格抽取的**权威主表数据**：13 个方法 × 3 个数据集（D5-512 / P2A-256 / R5-WikiArt）
  = 39 行，含 CLIP-S / LPIPS / MUSIQ 及 Params / Train / Infer。
  注意：仓库内**没有**现成的「13×3」CSV——Table 1 的数字是直接手填进 LaTeX 的，
  `baseline_metrics_unified.csv` 只是另一份「750 合并值」聚合 dump，不能当作逐数据集主表。
- **标准协议 750 图**：对应 `Dataset/distinct5_512/test`（每域 30 张源图，
  5 域共 150 张；在 5×5 源-目标方向下展开为 750 评测对）。此即论文中各指标评估所用的固定源图集。
- 远程副本位于 `I:\results`，结构与本目录一致。
