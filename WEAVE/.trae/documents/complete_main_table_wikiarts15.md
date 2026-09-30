# Plan: 补全主表 WikiArt-15 列

## Summary

1. 从主表移除 SD-Turbo / SDEdit × 2（SD-Edit 有异常，不再对比）；**保留 StyleID**
2. **更正表头：** `Distinct5-256` → `Photo2Art-256`（CycleGAN 5 风格 photo2art 数据集）
3. 添加 `WikiArt-15-512` (2列) + `WikiArt-15-256` (2列)
4. 按优先级逐批补全缺失数据，出来一个回填一个，维护状态文件追踪进度
5. 对无法复现的方法标注 "--" 并说明原因

---

## Current State Analysis

### 主表当前结构（paper.tex:425-444）
12列：`# | Method | Class | Distinct5-512(3) | Distinct5-256(2) | WikiArt-15-512(2) | Params | Train | Infer/750`
总共 12 列 → 最终需要 14 列（加上 2列 WikiArt-15-256）。

### 行变更
- **删除3行：** SD-Turbo, SDEdit s=0.35, SDEdit s=0.40（SD-Edit 有异常，不再对比）
- **保留：** Identity, AdaIN, WCT, StyleID, CUT, SaMST, SaMam, Seedream, WD-VF（共9行）
- 重编号：删除后连号

### 数据集分类说明
| 基准列 | 数据集 | 风格数 | 分辨率 |
|--------|--------|---------|----------|
| Distinct5-512 | WikiArt-Distinct5 | 5 | 512 |
| **Photo2Art-256** | **CycleGAN photo2art** | **5** | **256** |
| WikiArt-15-512 | WikiArt-Random-15 (disjoint from Distinct5) | 15 | 512 |
| WikiArt-15-256 | WikiArt-Random-15 (disjoint from Distinct5) | 15 | 256 |

### 已有数据
| 方法 | WikiArt-15-512 (distinct5 disjoint) | WikiArt-15-256 (distinct5 disjoint) |
|------|-----------------------------------|------------------------------------|
| Identity | 0.7450 / 0.0000 | 0.7401 / 0.0000 |
| AdaIN | 0.7443 / 0.6706 | 0.7332 / 0.7146 |
| WCT | 0.7469 / 0.6774 | 0.7381 / 0.7086 |
| WD-VF | 0.7524 / 0.3027 | **缺失** |
| SaMST | **缺失** | **缺失** |
| SaMam | **缺失** | **缺失** |

Photo2Art-256 已有完整 baseline 数据（来自 `docs/baseline_256/compare_256_photo2art.md`）。

### 远程训练环境
- SSH: `ssh -p 2222 administrator@100.115.18.62` (Windows + RTX 3060 12GB)
- WSL: SaMST repo at `/mnt/i/Github/Latent_Style/Related_Works/repos/SaMST-main`
- SaMST distinct5-512 训练时间: ~39.5 min → wikiarts-15-512 估算: ~118 min
- SaMST distinct5-256 训练时间: ~20 min → wikiarts-15-256 估算: ~60 min
- SaMam distinct5-512 训练时间: ~436 min → wikiarts-15-512 估算: ~22h（不可行）

---

## Proposed Changes

### 1. 删除 SD-Turbo + SDEdit 三行（paper.tex:434-436）

删除 SD-Turbo, SDEdit s=0.35, SDEdit s=0.40。
保留 StyleID（扩散方法仅保留 StyleID 作为 diffusion 代表）。
重编号：CUT→4, SaMST→5, SaMam→6, Seedream→7, StyleID→8, WD-VF→9。

**Why**: SD-Edit 有异常，不再对比此工作。StyleID 保留作为 diffusion 类代表。

### 2. 更正表头并添加 WikiArt-15-256 列（paper.tex:425-428）

**更正：** `Distinct5-256` → `Photo2Art-256`（CycleGAN photo2art 5 风格数据集）

**添加：** 在 `WikiArt-15-512` 后添加 `\multicolumn{2}{c}{WikiArt-15-256}`，共增加2列。

表头最终：
```
\multicolumn{3}{c}{Distinct5-512} & \multicolumn{2}{c}{Photo2Art-256} & \multicolumn{2}{c}{WikiArt-15-512} & \multicolumn{2}{c}{WikiArt-15-256}
```

### 3. 逐批补全数据（按优先级，出来一个回填一个）

#### Batch 1：零成本填入（立即执行，无需远程计算）
| 方法 | 列 | CLIP-S | LPIPS | 来源 |
|------|----|--------|-------|------|
| Identity | WikiArt-15-256 | 0.7401 | 0.0000 | 已提取 |
| AdaIN | WikiArt-15-256 | 0.7332 | 0.7146 | 已提取 |
| WCT | WikiArt-15-256 | 0.7381 | 0.7086 | 已提取 |

#### Batch 2：WD-VF WikiArt-15-256（远程训练 ~3min）
- 训练+评估 WD-VF 在 wikiarts-15-256 上
- 直接填入主表

#### Batch 3：SaMST WikiArt-15-256（远程训练 ~1h）
- 在 WSL 中训练 SaMST 在 wikiarts-15-256 上
- 生成图像后评估 CLIP-S + LPIPS

#### Batch 4：SaMST WikiArt-15-512（远程训练 ~2h）
- 在 WSL 中训练 SaMST 在 wikiarts-15-512 上

#### Batch 5：SaMam WikiArt-15-256（远程训练 ~22h）⚠️ 高成本
- 尝试训练 SaMam 在 wikiarts-15-256 上
- 如果中途失败或超时，标注 "--"

#### 无法复现的方法
| 方法 | WikiArt-15 | 原因 |
|------|-----------|------|
| CUT | -- | 需完整训练，环境未知 |
| Seedream | -- | API 调用成本 |

### 4. 状态追踪文件

创建 `state/main_table_fill.json` 追踪每个格子的填充状态：
```json
{
  "method": "SaMST",
  "column": "WikiArt-15-512",
  "status": "pending|running|done|failed",
  "clip_s": null,
  "lpips": null
}
```

### 5. 更新 caption 和正文

- Caption: 说明 Photo2Art-256 来自 CycleGAN，WikiArt-15 是 distinct5 的 disjoint 风格池
- 正文 WikiArt-15 段落: 随数据补全逐步更新

---

## Files Modified

1. **`aaai2027_v3/paper.tex`** (lines 420-444): 主表重写，删除扩散行，添加 WikiArt-15-256 列
2. **`aaai2027_v3/paper.tex`** (lines 504-514): 更新 WikiArt-15 generalization 段落
3. **`scripts/_train_wd_vf_wikiarts15_256.ps1`** (新建): WD-VF 256 训练+评估脚本
4. **`scripts/_train_samst_wikiarts15_512.sh`** (新建): SaMST 512 训练+评估脚本

---

## Assumptions & Decisions

- **Decision**: 移除扩散baseline行而非标注 "--"。原因：用户确认这些方法仅用于图1。
- **Decision**: 仅训练 SaMST，不训练 SaMam。原因：SaMam 训练时间 ~22h 不可行。
- **Decision**: 添加 WikiArt-15-256 列。原因：已有256基线数据，零成本填入。
- **Assumption**: 远程 WSL 中 SaMST repo 仍可用，数据集可链接。
- **Assumption**: SaMST 训练在 wikiarts-15 上 VRAM 不会超过 12GB。

---

## Verification

1. 编译 `paper.tex`：无错误，8-9页
2. 检查主表：无 "--" 空列（除标注外）
3. 检查 SaMST 训练日志：确认收敛
4. 检查 WD-VF 256 评估指标：合理范围
5. Git commit 所有变更