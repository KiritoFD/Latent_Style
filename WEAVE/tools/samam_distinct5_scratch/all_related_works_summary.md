# Related Works Baseline 数据汇总 (v2 - 修正版)

**生成时间**: 2026-07-02 07:15
**数据集**: wikiart_distinct5_samam_512_classview (5风格: Early_Renaissance, Impressionism, Minimalism, Rococo, Ukiyo_e)
**评估协议**: run_evaluation.py, HF transformers CLIP (ViT-B/32, openai/clip-vit-base-patch32), LPIPS (Alex), 750 pairs

---

## ⚠️ 重要修正说明

**旧版文档中的 SaMam CLIP-S=0.7222 是错误数据，已废弃。**

错误根因：`update_unified_samam.py` 硬编码的0.7222实际上来自 `samam_256_faithful_p8_remote/b1_sph35_20260522_050523/h03_step0105/`，这是：
- **256分辨率**（非512）
- **wikiart5数据集**（Hayao/cezanne/vangogh，非distinct5）
- **step 105的早期checkpoint**
- n_pairs=750（硬编码写745，也对不上）

**正确的SaMam评估正在进行中**：20K步训练已完成，80个checkpoint（step 250-20000，每250步）正在用HF transformers CLIP重新评估，预计~5.5h完成。

---

## 一、12个Baseline方法完整指标表（SaMam待HF替换）

数据源: `exp/baseline_v2/eval/unified_results.json`（samam项已废弃，待替换；seedream已加入）

| # | 方法 | 类别 | CLIP-S ↑ | LPIPS ↓ | Δ_idt | n_pairs | 训练时间(min) | 复现时间 | Finding ID |
|---|------|------|---------|---------|-------|---------|--------------|---------|------------|
| 1 | Identity (IDT) | baseline | 0.6933 | 0.0000 | **0.0000** | 750 | 0 | 06-30 20:50 | F001 |
| 2 | AdaIN | classical-inf | 0.6679 | 0.7425 | -0.0254 | 750 | 0 | 06-30 20:50 | F002 |
| 3 | WCT (VGG19) | classical-inf | 0.7063 | 0.6348 | +0.0130 | 750 | 0 | 07-01 17:20 | F019 |
| 4 | SD-Turbo | diffusion-inf | 0.6933 | 0.0033 | 0.0000 | 750 | 0 | 06-30 20:50 | F007 |
| 5 | **SDEdit s=0.35** | diffusion-sweep | **0.7797** | **0.4508** | **+0.0864** | 750 | 0 | 06-30 20:50 | F005 |
| 6 | **SDEdit s=0.40** | diffusion-sweep | **0.7934** | **0.4826** | **+0.1001** | 750 | 0 | 06-30 20:50 | F006 |
| 7 | StyleID | diffusion-inf | **0.8223** | 0.5523 | **+0.1290** | 750 | 0 | 06-30 20:50 | F008 |
| 8 | CUT | gan-train | 0.7137 | 0.3743 | +0.0204 | 745 | 322.6 | 07-01 11:05 | F014 |
| 9 | SaMST | mamba-train | 0.6183 | 0.7490 | -0.0750 | 750 | 39.5 | 07-01 10:20 | F011 |
| 10 | **SeeDream** | **diffusion-inf** | **0.7198** | **0.4767** | **+0.0266** | 750 | 0 | 07-02 07:14 | F021 |
| 11 | **SaMam** | **mamba-train** | **0.5816** | **0.2434** | **-0.1117** | 750 | **~436** | 07-02 | F020 |

**注**:
- **Δ_idt = CLIP-S - 0.6933 (Identity新基线)**, Identity自身Δ=0
- SDEdit s=0.10 和 s=0.20 已移除（按用户要求只保留0.35和0.4两个点）
- **SeeDream**: 750张图来自 `seedream45_api/distinct5_512_seedream45_windhub_20260607_repaired750/`, 已复制到 `exp/baseline_v2/eval/seedream/`
- SaMam训练时间=7h16m=436min（20K步，batch=1, 512×512, 32-true, distinct5）
- **SaMam最终值=HF transformers CLIP, step=20000 (收敛), 80个checkpoint评估完成, 完整曲线见第4.6节**
- **收敛判断**: CLIP-S在step 3000即收敛(Δ<0.01, 稳定在0.58-0.59)，LPIPS持续下降到20000步(0.2434)

---

## 二、按指标排名（SaMam待补）

### CLIP-S Style排名 (风格转移强度, ↑越好)

| 排名 | 方法 | CLIP-S | 类别 | 备注 |
|------|------|--------|------|------|
| 1 | StyleID | 0.8223 | diffusion-inf | |
| 2 | SDEdit s=0.40 | 0.7934 | diffusion-sweep | |
| 3 | SDEdit s=0.35 | 0.7797 | diffusion-sweep | |
| 4 | **SeeDream** | **0.7198** | **diffusion-inf** | |
| 5 | CUT | 0.7137 | gan-train | |
| 6 | WCT (VGG19) | 0.7063 | classical-inf | |
| 7 | Identity | 0.6933 | baseline | |
| 8 | SD-Turbo | 0.6933 | diffusion-inf | |
| 9 | AdaIN | 0.6679 | classical-inf | |
| 10 | SaMST | 0.6183 | mamba-train | |
| 11 | **SaMam** | **0.5816** | **mamba-train** | HF CLIP, step=20000 |

### LPIPS排名 (内容保真度, ↓越好)

| 排名 | 方法 | LPIPS | 类别 | 备注 |
|------|------|-------|------|------|
| 1 | Identity | 0.0000 | baseline | |
| 2 | SD-Turbo | 0.0033 | diffusion-inf | |
| 3 | **SaMam** | **0.2434** | **mamba-train** | HF CLIP, step=20000, 内容保真度最佳(非identity) |
| 4 | CUT | 0.3743 | gan-train | |
| 5 | SDEdit s=0.35 | 0.4508 | diffusion-sweep | |
| 6 | **SeeDream** | **0.4767** | **diffusion-inf** | |
| 7 | SDEdit s=0.40 | 0.4826 | diffusion-sweep | |
| 8 | StyleID | 0.5523 | diffusion-inf | |
| 9 | WCT (VGG19) | 0.6348 | classical-inf | |
| 10 | AdaIN | 0.7425 | classical-inf | |
| 11 | SaMST | 0.7490 | mamba-train | |

**关键发现**:
- **SaMam内容保真度(LPIPS=0.2434)是所有非identity方法中最佳**，比CUT好35%（0.2434 vs 0.3743）
- SaMam的CLIP-S=0.5816偏低（排第11），但LPIPS优势明显
- **SaMam的Δ_idt=-0.1117为负值**，说明风格化程度低于Identity基线（HF CLIP下SaMam风格提取较弱）
- SeeDream定位：CLIP-S第4（风格强于CUT），LPIPS第6（内容保真度中等），整体均衡

---

## 三、训练类方法对比

| 方法 | 训练时间(min) | CLIP-S | LPIPS | 训练配置 | 状态 |
|------|--------------|--------|-------|---------|------|
| SaMST | 39.5 | 0.6183 | 0.7490 | 2 epochs | 失败 (CLIP-S低于identity) |
| **SaMam** | **~436** | **~0.625** ⚠️ | **~0.321** | **20k步, batch=1, 512×512, 32-true, distinct5** | **已收敛, HF评估中** |
| CUT | 322.6 | 0.7137 | 0.3743 | 5 styles × 4 epochs (2+2 decay) | 完成 |

**关键发现**:
- SaMam的LPIPS=0.321是所有方法中第3好（仅次于Identity和SD-Turbo），且LPIPS与CLIP backend无关，可直接比较
- SaMam内容保真度优于CUT (0.321 < 0.374)
- SaMam训练时间比CUT多35% (436 vs 322.6 min)
- SaMam的CLIP-S=0.625是open_clip数值，HF评估后预计会提升约0.05-0.10（基于step 250/500/750的HF vs open_clip对比）

---

## 四、SaMam实验完整记录（数据集错误诊断 + 修正方案）

### 4.1 所有SaMam实验对照表

| 数值 | 实验配置 | 数据集 | CLIP backend | 训练时间 | 状态 |
|------|---------|--------|-------------|---------|------|
| ~~0.7222~~ | ~~256分辨率, step 105~~ | ~~wikiart5~~ | ~~HF transformers~~ | - | ❌ **错误数据，已废弃** |
| 0.6248 | 7k步, batch=1, 512×512 | distinct5 ✅ | open_clip ❌ | 232min | ❌ CLIP backend错 |
| 0.7851 | 10k步(训20k), batch=1, 512×512 | wikiart5 ❌ | HF transformers ✅ | 294min | ❌ 数据集错 |
| **待评估** | **20k步, batch=1, 512×512** | **distinct5 ✅** | **HF transformers ✅** | **436min** | **✅ 评估中** |

### 4.2 错误数据0.7222的来源追溯

- **文件**: `exp/baseline_v2/eval/samam/summary.json` 和 `samam_256_faithful_p8_remote/b1_sph35_20260522_050523/h03_step0105/x8/metrics.csv`
- **实验**: `samam_256_faithful_p8_remote` (256分辨率, 2026-05-22)
- **数据集**: wikiart5 (Hayao/cezanne/vangogh/photo/monet)
- **checkpoint**: h03_step0105 (step 105)
- **硬编码位置**: `tools/update_unified_samam.py` 第XX行
- **问题**: 被错误地作为distinct5_512的SaMam baseline写入unified_results.json

### 4.3 修正方案（进行中）

**20K训练**: ✅ 完成 (2026-07-02 05:54)
- 从step 7000 resume到20000
- 80个checkpoint (step 250-20000, 每250步)
- 训练时间: 7h16m (WALL_SECONDS=26213.72)
- 输出: `samam_distinct5_512_scratch_7k_250eval_remote/step_checkpoints/`

**HF transformers评估**: ✅ 完成 (2026-07-02 11:15)
- 81个checkpoint全部评估完成 (step 250-20000 + last, 每250步)
- HF transformers CLIP (ViT-B/32, 与其他baseline一致)
- 750 pairs/checkpoint
- Phase 1推理: 52个新checkpoint × 196s = ~2.8h
- Phase 2评估: 81个checkpoint × 23s = 32.2min (GPU-batched, batch=64/128)
- 输出: `curve_eval_hf_750_batched/curve_metrics.csv`

### 4.4 SaMam收敛曲线数据 (open_clip, 28个checkpoint, step 250-7000)

数据源: `curve_eval_30src/curve_metrics.csv` (open_clip, distinct5)
注: CLIP-S绝对值偏低约0.05-0.10，但收敛趋势可靠。LPIPS与CLIP backend无关，可直接参考。

| Step | CLIP-S | LPIPS | Step | CLIP-S | LPIPS |
|------|--------|-------|------|--------|-------|
| 250 | 0.5819 | 0.8442 | 3750 | 0.6145 | 0.3624 |
| 500 | 0.5844 | 0.6281 | 4000 | 0.6116 | 0.3424 |
| 750 | 0.6040 | 0.5981 | 4250 | 0.6194 | 0.3274 |
| 1000 | 0.6109 | 0.5678 | 4500 | 0.6128 | 0.3156 |
| 1250 | 0.6103 | 0.5414 | **4750** | **0.6174** | **0.3117** |
| 1500 | 0.6219 | 0.5016 | 5000 | 0.6216 | 0.3394 |
| 1750 | 0.6230 | 0.4821 | 5250 | 0.6235 | 0.3238 |
| 2000 | 0.6260 | 0.4564 | 5500 | 0.6222 | 0.3208 |
| 2250 | 0.6236 | 0.4414 | 5750 | 0.6252 | 0.3489 |
| 2500 | 0.6235 | 0.4069 | 6000 | 0.6263 | 0.3398 |
| 2750 | 0.6203 | 0.3981 | 6250 | 0.6260 | 0.3465 |
| 3000 | 0.6217 | 0.3803 | 6500 | 0.6274 | 0.3518 |
| 3250 | 0.6206 | 0.3663 | 6750 | 0.6240 | 0.3277 |
| 3500 | 0.6168 | 0.3544 | **7000** | **0.6248** | **0.3209** |

**收敛判断**: step 4750后CLIP-S delta<0.01（open_clip下），7000步已收敛。
- 收敛区间: CLIP-S ~0.62, LPIPS ~0.32
- CLIP-S从0.5819(step 250) → 0.6248(step 7000), 提升+0.043
- LPIPS从0.8442(step 250) → 0.3209(step 7000), 下降-0.523

### 4.5 SaMam HF transformers完整收敛曲线 (✅ 完成, 81个checkpoint, step 250-20000)

数据源: `curve_eval_hf_750_batched/curve_metrics.csv` (HF transformers, distinct5)
这是**最终用于论文的收敛曲线**，与其他11个baseline完全可比。

| Step | CLIP-S | LPIPS | Step | CLIP-S | LPIPS |
|------|--------|-------|------|--------|-------|
| 250 | 0.5208 | 0.8441 | 5250 | 0.5859 | 0.3238 |
| 500 | 0.5241 | 0.6281 | 5500 | 0.5858 | 0.3208 |
| 750 | 0.5474 | 0.5981 | 5750 | 0.5857 | 0.3489 |
| 1000 | 0.5548 | 0.5678 | 6000 | 0.5860 | 0.3398 |
| 1250 | 0.5680 | 0.5414 | 6250 | 0.5859 | 0.3465 |
| 1500 | 0.5789 | 0.5016 | 6500 | **0.5925** | 0.3518 |
| 1750 | 0.5820 | 0.4821 | 6750 | 0.5898 | 0.3277 |
| 2000 | 0.5843 | 0.4564 | 7000 | 0.5855 | 0.3209 |
| 2250 | 0.5856 | 0.4414 | 7500 | 0.5857 | 0.3198 |
| 2500 | 0.5856 | 0.4069 | 10000 | 0.5858 | 0.2913 |
| 2750 | 0.5857 | 0.3981 | 12500 | 0.5861 | 0.2582 |
| 3000 | 0.5868 | 0.3803 | 15000 | 0.5830 | 0.2677 |
| 3250 | 0.5861 | 0.3663 | 17500 | 0.5817 | 0.2446 |
| 3500 | 0.5860 | 0.3544 | **20000** | **0.5816** | **0.2434** |
| 3750 | 0.5858 | 0.3624 | last | 0.5905 | 0.3209 |
| 4000 | 0.5858 | 0.3424 | | | |
| 4250 | 0.5859 | 0.3274 | | | |
| 4500 | 0.5860 | 0.3156 | | | |
| 4750 | 0.5861 | 0.3117 | | | |
| 5000 | 0.5873 | 0.3394 | | | |

**收敛分析**:
- **CLIP-S在step 3000即收敛** (Δ<0.01, 稳定在0.58-0.59)
- **LPIPS持续下降**到20000步 (0.2434), 内容保真度还在提升
- Best CLIP-S: step=6500 = 0.5925
- Best LPIPS: step=19500 = 0.2223
- CLIP-S从0.5208(step 250) → 0.5816(step 20000), 提升+0.061
- LPIPS从0.8441(step 250) → 0.2434(step 20000), 下降-0.601

**HF vs open_clip对比**: HF的CLIP-S比open_clip低约0.04-0.06 (HF 0.58 vs open_clip 0.62)，LPIPS完全一致。

### 4.6 评估完成状态

✅ **全部完成** - 2026-07-02 11:15:47

- 81个checkpoint (step 250-20000每250步 + last)
- HF transformers CLIP (与其他baseline一致)
- Phase 1推理: 52新checkpoint × 196s ≈ 2.8h
- Phase 2评估: 81 checkpoint × 23s = 32.2min (GPU-batched)
- 总耗时: ~3.4h (从07:35到11:15)

结果文件:
- CSV: `curve_eval_hf_750_batched/curve_metrics.csv`
- JSON: `curve_eval_hf_750_batched/curve_metrics.json`
- 每个checkpoint的metrics.csv在对应step目录下

---

## 五、评估协议一致性

| 评估项 | 值 | 一致性 |
|--------|-----|--------|
| 评估脚本 | run_evaluation.py | ✅ 全部一致 |
| CLIP backend | HF transformers, openai/clip-vit-base-patch32 | ✅ 全部一致 (SaMam评估中) |
| LPIPS | Alex backbone | ✅ 全部一致 |
| test_dir | I:\wikiart_distinct5_samam_512_classview\test | ✅ 全部一致 |
| 5风格 | Early_Renaissance, Impressionism, Minimalism, Rococo, Ukiyo_e | ✅ 全部一致 |
| IDT基线 | 0.6399 | ✅ 全部一致 |
| n_pairs | 750 (cut=745) | ⚠️ 差异<1% |

---

## 六、各方法images目录

| 方法 | 评估目录 |
|------|---------|
| identity | exp/baseline_v2/eval/identity/images/ |
| adain | exp/baseline_v2/eval/adain/images/ |
| wct_vgg19 | exp/baseline_v2/eval/wct_vgg19/images/ |
| sdturbo | exp/baseline_v2/eval/sdturbo/images/ |
| sdedit_str0.35 | exp/baseline_v2/eval/sdedit_str0.35/images/ |
| sdedit_str0.40 | exp/baseline_v2/eval/sdedit_str0.40/images/ |
| styleid | exp/baseline_v2/eval/styleid/images/ |
| cut | exp/baseline_v2/eval/cut/images/ |
| samst | exp/baseline_v2/eval/samst/images/ |
| **seedream** | **exp/baseline_v2/eval/seedream/images/** (07-02加入) |
| samam | exp/baseline_v2/eval/samam/images/ (⚠️ 旧数据，待替换) |

**注**: SDEdit s=0.10 和 s=0.20 的images目录仍存在但不再用于论文

---

## 七、失败实验记录

### F018: WCT VGG-normalised变体失败
- **问题**: 750张PNG同MD5, VGG-normalised特征值域过窄
- **解决**: 改用VGG-19 ImageNet encoder (F019成功)

### F011: SaMST训练失败
- **问题**: CLIP-S=0.6183低于identity, 内容严重扭曲
- **状态**: 已记录, 未修复

### ~~F016: SaMam~~（已废弃）
- **原记录**: "5k iters, batch=4, 256x256, CLIP-S=0.7222"
- **实际**: 256分辨率wikiart5数据集的step105实验, 数据集和分辨率都不对
- **状态**: 废弃, 用F020 (20k, distinct5, 512×512) 替代

---

## 八、当前进行中的实验

### F020: SaMam 20K + HF CLIP评估 (正确实验)
- **训练**: ✅ 完成 (2026-07-02 05:54, 7h16m, 80 checkpoints)
- **评估**: 🔄 进行中 (2026-07-02 06:53启动, 预计~5.5h)
- **配置**: 20k步, batch=1, 512×512, 32-true, distinct5 (正确数据集)
- **CLIP backend**: HF transformers (与其他baseline一致)
- **输出**: `curve_eval_hf_750/curve_metrics.csv` (80行, step 250-20000)

---

## 九、待办事项

1. **等待SaMam HF评估完成** (预计07-02 12:30)
2. **用新SaMam数据替换unified_results.json**: 移除错误的0.7222, 用20K收敛值替换
3. **用新数据替换make_dashboard.py的SAMAM_CURVE**: 旧的0.7851曲线用错了wikiart5
4. **确认SaMam收敛步数**: 从80个checkpoint的曲线判断
5. **Art FID评估**: 目前无方法评估过Art FID
6. **论文用图**: 用新数据重新生成dashboard
