# docs/630 — Codebase Cleanup + Masking + Exploration 交接索引

**会话时间**: 2026-06-30
**分支**: `codex/620-spatial-bridge`
**最终状态**: Phase 3 完成 (PHASE3_COMPLETE), 所有里程碑 M1-M6 已完成
**最终 commit**: `adc6a0d38` (Phase 3)

---

## 一、文档导航

### 顶层交接文档
- **[HANDOVER.md](HANDOVER.md)** — 完整交接总览（理论 + 实验 + 计划 + 协议），新会话入口

### 协议与状态 (state/)
- [state/task_spec.md](state/task_spec.md) — 任务规格 (Goal, Milestones, Constraints, Success Criteria)
- [state/progress.json](state/progress.json) — 迭代状态机 (iteration=5, status=PHASE3_COMPLETE)
- [state/findings.jsonl](state/findings.jsonl) — 审计发现流 (H1-H11, M1-M9, L1-L7)
- [state/directions_tried.json](state/directions_tried.json) — 已尝试方向 (空, Phase 3 后停止探索)
- [logs/orchestrator.jsonl](logs/orchestrator.jsonl) — 决策日志
- [skill.md](skill.md) — Deli_AutoResearch 协议说明

### 理论文档
- [mask.md](mask.md) — Masking 理论: 信息瓶颈, Blindfolded Tokenizer, 内容/风格可分性
- ../theory/SpectralODE_Bridge.md — Spectral ODE Bridge 完整理论 (8 章: Haar DWT, 架构, 训练, tri_band, 推理, Endpoint AdaIN, 消融结论, codebase 摘要)

### Phase 1: Codebase Cleanup (减法消融)
- [phase1_cleanup.md](phase1_cleanup.md) — Phase 1 总览
- [phase1a_cleanup.md](phase1a_cleanup.md) — H1-H11 零风险 dead code 删除
- [phase1b_attn_mode_fix.md](phase1b_attn_mode_fix.md) — M9 attn_mode bug TDD 修复 (relu2)
- [phase1c_legacy_deletion.md](phase1c_legacy_deletion.md) — Legacy 文件批量删除 (-11346 行)
- [phase1d_verification.md](phase1d_verification.md) — 最简 codebase 性能验证 (3-epoch PASS)

### Phase 2: Masking 实现 + 消融
- [phase2_masking.md](phase2_masking.md) — The Blindfolded Tokenizer, 4 组消融, random_50 最佳

### Phase 3: 完整训练验证
- [phase3_exploration.md](phase3_exploration.md) — 10-epoch 完整训练 (从零, 独立目录), 全 PASS

---

## 二、Git 提交历史 (本会话)

| Commit | 阶段 | 说明 |
|--------|------|------|
| `925b6bea7` | Phase 1A | H1-H11 dead code removal |
| `69da87cb0` | Phase 1B | M9 attn_mode bug TDD fix + relu2 3-epoch eval |
| `bcea0a41b` | Phase 1C | Legacy codebase deletion (-11346 lines) |
| `9de1e9e03` | Phase 1D | Minimal codebase performance verification |
| `8df445e50` | Phase 2 | The Blindfolded Tokenizer — style masking |
| `adc6a0d38` | Phase 3 | mask_random_50 完整 10-epoch 训练验证 |

---

## 三、关键产物路径

### 代码 (src/, 14 个 active 文件)
- `src/model.py` (93 行) — 精简模型工厂
- `src/spectral_bridge620.py` — SpectralODEBridge620 (含 mask 配置传递)
- `src/spectral_losses620.py` — SpectralODEObjective620
- `src/blocks620.py` (279 行) — SpatialBridgeBlock620
- `src/style_encoder620.py` (109 行) — StyleConditioner620 (含 `_apply_mask`)
- `src/spectral620.py` — Haar DWT 工具
- `src/trainer.py` — 训练器 (lazy import)
- `src/run.py` — 入口
- `src/config_schema.py` — 配置 schema (含 `style_mask_ratio`/`style_mask_mode`)
- `src/style_families.py`, `src/utils/{inference,run_evaluation,training,dataset}.py`

### 配置 (configs/)
- `configs/clean_base_v2_local.json` — 本地 baseline (T5, clip=0.7293, lpips=0.3203)
- `configs/clean_base_v2_relu2.json` — relu2 attn mode baseline
- `configs/630_phase1d_verify.json` — Phase 1D 验证配置
- `configs/630_phase2b_mask_random_50.json` — Phase 2 最佳 masking 配置 (3-epoch)
- `configs/630_phase2c_mask_random_75.json` — 消融
- `configs/630_phase2c_mask_shuffle_50.json` — 消融
- `configs/630_phase2c_mask_shuffle_75.json` — 消融
- `configs/630_phase3_mask_random_50_10ep.json` — Phase 3 完整训练配置 (10-epoch)

### 实验产物 (exp/)
- `exp/clean_base_v2_relu2/` — Phase 1B baseline (3-epoch)
- `exp/630_phase1d_verify_v2/` — Phase 1D 精简后验证 (3-epoch)
- `exp/630_phase2b_mask_random_50/` — Phase 2B 最佳 masking (3-epoch)
- `exp/630_phase2c_mask_random_75/` — Phase 2C 消融
- `exp/630_phase2c_mask_shuffle_50/` — Phase 2C 消融
- `exp/630_phase2c_mask_shuffle_75/` — Phase 2C 消融
- `exp/630_phase3_mask_random_50_10ep/` — **Phase 3 最终产物** (10-epoch, 从零训练)
  - `epoch_0005.pt`, `epoch_0010.pt`
  - `full_eval/epoch_0005/summary.json` (clip=0.7275, lpips=0.3238)
  - `full_eval/epoch_0010/summary.json` (clip=0.7289, lpips=0.3370)
  - `full_eval/epoch_0010_full/summary.json` (clip=0.7288, lpips=0.3369, 独立评估)

### 测试
- `tests/test_630_masking.py` — 9 个 masking TDD 测试
- `tests/test_630_spectral_ode.py` — attn_mode 传播测试

### 工具
- `tools/local_train_and_eval.py` — 本地一键 smoke + train + eval (含 baseline 对比)

---

## 四、快速恢复指令

### 复现 Phase 3 最终结果
```bash
python tools/local_train_and_eval.py --config configs/630_phase3_mask_random_50_10ep.json
```
预期: clip_style ≈ 0.7288, content_lpips ≈ 0.3369 (both PASS)

### 只评估已有 checkpoint
```bash
python tools/local_train_and_eval.py --config configs/630_phase3_mask_random_50_10ep.json \
  --skip-train --checkpoint exp/630_phase3_mask_random_50_10ep/epoch_0010.pt
```

### Smoke 测试
```bash
python tools/local_train_and_eval.py --config configs/630_phase3_mask_random_50_10ep.json --smoke-only
```

---

## 五、硬约束回顾 (project_memory)

- 训练 Patience=2, max=10, 至少 5 epochs
- 训练显存 9-11G, 评估显存 ≤ 7G (batch_size=2)
- 数据集路径: `G:/GitHub/Latent_Style/Dataset/distinct5_512` (本地) / `/mnt/i/...` (远程)
- DataLoader: num_workers=0, pin_memory=False, persistent_workers=False
- 命令添加 30s timeout
- 无效代码确认后直接删除 (不 ablate)
- 优化用条件编译, 避免影响其他测试
- 不允许远程 GPU, 本地重训
- **每次单开目录重新训练, 避免 resume 导致结论失真**
