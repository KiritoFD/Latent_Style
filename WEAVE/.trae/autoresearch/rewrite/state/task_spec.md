# Task Spec: 文档重组 + 数据完整性核查 + 实验脉络梳理

## 目标
1. **baseline 数据完整性核查**: 每个 baseline 必须有完整收敛证据（所有 checkpoint 评估结果），证明复现跑到收敛。SaMam 已有 81 checkpoint 完整曲线。
2. **我们模型实验脉络梳理**: exp/ours/ 下每个实验小目录，记录 why/conclusion/ckpt 是否有意义。无意义的 git commit + 详细文档后删除。
3. **文档重组**:
   - `docs/baseline/` — 相关工作（每个 baseline 独立文档 + 收敛证据）
   - `docs/exp/` — 我们模型的分阶段实验（保留现有，重组）
   - `docs/math/` — 对我们模型的理论描述
   - `docs/tools/` — 数据库、评估协议、infra、调用命令、实验经验
4. **数据集分类**: 每个实验分清数据集（256 / 5×3600 / wikiarts_5 / distinct5），在 `exp/256/` 等下分别放，不污染主线结论。
5. **旧文档归档**: docs/ 下历史目录（612/616/618/619/620/622/625/627/628/630/experiments/logs 等）归档到 docs/archive/，git commit 后删除。

## Success Criteria
- [ ] docs/baseline/ 创建，每个 baseline 有独立 .md 含完整收敛证据
- [ ] docs/math/ 创建，含 FC-SB 理论描述
- [ ] docs/tools/ 创建，含评估协议/infra/调用命令/经验
- [ ] docs/exp/ 重组，按阶段组织我们的实验
- [ ] exp/ 下按数据集分类（256/wikiarts5/distinct5）
- [ ] 无意义 ckpt 删除 + git commit + 详细文档
- [ ] 旧文档归档到 docs/archive/
- [ ] 所有文档中 SaMam 编造值 0.7175/0.2423 修正为真实值 0.5816/0.2434
- [ ] SaMam 数据完整性调查报告完成
- [ ] 最终一致性校验通过

## Milestones
- M11: 修正所有文档 SaMam 编造值 (07/03/05/04/README + docs/exp/README + 3 个 630 文档)
- M12: 写 SaMam 数据完整性调查报告 (含 81 checkpoint 完整表)
- M21: 建立 autoresearch 任务框架 (本文件)
- M22: baseline 数据完整性核查 (每个 baseline 收敛证据)
- M23: 我们模型实验脉络梳理 (exp/ours/ 每个目录)
- M24: 创建 docs/baseline/ + docs/math/ + docs/tools/
- M25: 数据集分类 (exp/256, exp/wikiarts5, exp/distinct5)
- M26: 删除无意义 ckpt + git commit + 详细文档
- M27: 旧文档归档到 docs/archive/ + git commit + 删除
- M28: 最终一致性校验

## Constraints (from project_memory)
- WFI < 0.40 验收标准
- 训练显存 9-11G, 评估 ≤7G
- 数据集路径 I 盘 (/mnt/i/...)
- batch_size=24 消融实验
- 非 distinct5 实验移到 exp_legacy/
- 无效代码确认后直接删除
- 命令添加 30s timeout
- 远程: ssh -p 2222 administrator@100.115.18.62, Windows + WSL2
