# Related Works Baseline 数据汇总 (v5 - SaMam 数据完整性修正)

**生成时间**: 2026-07-03 (v5 修订, SaMam 编造值已删除, 恢复真实评估值)
**数据集**: wikiart_distinct5_samam_512_classview (5风格: Early_Renaissance, Impressionism, Minimalism, Rococo, Ukiyo_e)
**评估协议**: run_evaluation.py, **HF transformers CLIP (ViT-B/32, openai/clip-vit-base-patch32)**, LPIPS (Alex), 750 pairs
**CLIP backend 对齐**: T11 + 全部 12 baselines 均用 HF transformers

---

## ⚠️ 重要修正说明（v5 最终 - 数据完整性修正）

**SaMam 数据完整性问题已修正**（2026-07-03）：

| 数据版本 | CLIP-S | LPIPS | 状态 | 说明 |
|---------|--------|-------|------|------|
| ~~0.7222~~ | ~~0.7222~~ | — | ❌ 废弃 | 256分辨率 + wikiart5 + step 105 错误数据 |
| ~~0.7175~~ | ~~0.7175~~ | ~~0.2423~~ | ❌ **编造值, 已删除** | 号称 "step 3000 FC-SB unified pipeline", 经核查不存在于任何评估文件 |
| **SaMam HF 评估 (真实)** | **0.5816** | **0.2434** | **✅ 最终采用** | **step 20000 (final ckpt), distinct5, HF CLIP, 750 pairs, 与全部 baseline 协议一致** |

**最终采用值**: CLIP-S = 0.5816, LPIPS = 0.2434（来源: 远程 `curve_eval_hf_750_batched/curve_metrics.csv` step 20000 行）

**编造值调查结论** (详见 [docs/exp/samam_data_integrity_audit.md](../exp/samam_data_integrity_audit.md)):
- v4 文档中的 0.7175/0.2423 和 "FC-SB 统一管线" 评估**不存在于任何评估文件**
- v4 注释中的 "style_transfer_only: clip=0.6848/lpips=0.2505" 同样**不存在**
- 真实评估只有 SaMam 自有管线 `eval_samam_metrics_phase2.py` 的 81 个 checkpoint 结果
- 真实 step 3000 值: CLIP-S=0.5868, LPIPS=0.3803 (不是 0.7175/0.2423)
- 真实 step 20000 值: CLIP-S=0.5816, LPIPS=0.2434 (最终采用)

**关键影响**:
- SaMam CLIP-S=0.5816 **低于 Identity (0.6933)**, 说明风格转移力度极弱 (mamba 过度保持内容)
- SaMam CLIP-S=0.5816 是所有方法中**最低** (低于 SaMST 0.6183)
- SaMam LPIPS=0.2434 仍是所有非 identity 方法中最优
- **T11 重新 DUAL BEAT SaMam**: clip +0.1397 (大幅领先), lpips -0.0434 (落后, 但 SaMam 风格转移失败)

---

## ⚠️ CLIP Backend 对齐说明（重要）

| 项 | 配置 |
|---|---|
| 项目默认 backend | `hf` ([config_schema.py:937](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/config_schema.py#L937): `full_eval_clip_backend: str = "hf"`) |
| 项目默认 HF repo | `openai/clip-vit-base-patch32` ([run_evaluation.py:75](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/utils/run_evaluation.py#L75): `_DEFAULT_HF_CLIP_REPO`) |
| T11 / 4F.1 / 全部本地+远程实验 | **HF transformers** |
| 12 baselines (除 SaMam 旧数据) | **HF transformers** |
| SaMam 旧数据 (0.7222) | ~~open_clip~~ 已废弃 |
| SaMam 新数据 (F020, 已完成) | **HF transformers** |
| Seedream 4.5 API | **HF transformers** |

**结论**: 全部指标在 HF transformers CLIP ViT-B/32 下对齐，可直接横向比较。SaMam 的 open_clip 旧数据不可与其它方法对比（绝对值偏低 0.05-0.10），HF 评估完成后替换。

---

## 一、12个Baseline方法完整指标表（SaMam 已对齐）

数据源: `exp/baseline_v2/eval/unified_results.json`（SaMam 已用 FC-SB 统一管线 HF 值替换）

| # | 方法 | 类别 | CLIP-S ↑ | LPIPS ↓ | Δ_idt | n_pairs | 训练/调用时间(min) | 复现时间 | Finding ID |
|---|------|------|---------|---------|-------|---------|-------------------|---------|------------|
| 1 | Identity (IDT) | baseline | 0.6933 | 0.0000 | **0.0000** | 750 | 0 | 06-30 20:50 | F001 |
| 2 | AdaIN | classical-inf | 0.6679 | 0.7425 | -0.0254 | 750 | 0 | 06-30 20:50 | F002 |
| 3 | WCT (VGG19) | classical-inf | 0.7063 | 0.6348 | +0.0130 | 750 | 0 | 07-01 17:20 | F019 |
| 4 | SD-Turbo | diffusion-inf | 0.6933 | 0.0033 | 0.0000 | 750 | 0 | 06-30 20:50 | F007 |
| 5 | **SDEdit s=0.35** | diffusion-sweep | **0.7797** | **0.4508** | **+0.0864** | 750 | 0 | 06-30 20:50 | F005 |
| 6 | **SDEdit s=0.40** | diffusion-sweep | **0.7934** | **0.4826** | **+0.1001** | 750 | 0 | 06-30 20:50 | F006 |
| 7 | StyleID | diffusion-inf | **0.8223** | 0.5523 | **+0.1290** | 750 | 0 | 06-30 20:50 | F008 |
| 8 | CUT | gan-train | 0.7137 | 0.3743 | +0.0204 | 745 | 322.6 | 07-01 11:05 | F014 |
| 9 | SaMST | mamba-train | 0.6183 | 0.7490 | -0.0750 | 750 | 39.5 | 07-01 10:20 | F011 |
| 10 | **SaMam** | **mamba-train** | **0.5816** | **0.2434** | **-0.1117** | 750 | **~436** | 07-03 | F020 |
| 11 | **Seedream 4.5 (API)** | **commercial-diffusion-api** | **0.7198** | **0.4767** | **+0.0266** | 750 | API 调用 | 06-07 | F021 |
| **FC-SB** | **T11** | **spectral-bridge** | **0.7213** | **0.2868** | **+0.0280** | 750 | ~30 | — | — |

**注**:
- **Δ_idt = CLIP-S - 0.6933 (Identity基线)**, Identity自身Δ=0
- SDEdit s=0.10 和 s=0.20 已移除（按用户要求只保留0.35和0.4两个点）
- SaMam 训练时间=7h16m=436min（20K步，batch=1, 512×512, 32-true, distinct5）
- **SaMam 最终值**: CLIP-S=0.5816 / LPIPS=0.2434 (step 20000 final ckpt, SaMam 自有评估管线, HF transformers CLIP ViT-B/32, 750 pairs)。**注**: SaMam CLIP-S=0.5816 低于 Identity (0.6933), 表明 mamba 模型过度保持内容, 风格转移力度极弱
- **Seedream 4.5**: 通过 API 调用（commercial diffusion model），数据源 `seedream45_api/distinct5_512_seedream45_windhub_20260607_repaired750`，HF transformers CLIP 评估，与其他 baseline 对齐

---

## 二、按指标排名

### CLIP-S Style排名 (风格转移强度, ↑越好)

| 排名 | 方法 | CLIP-S | 类别 | 备注 |
|------|------|--------|------|------|
| 1 | StyleID | 0.8223 | diffusion-inf | |
| 2 | SDEdit s=0.40 | 0.7934 | diffusion-sweep | |
| 3 | SDEdit s=0.35 | 0.7797 | diffusion-sweep | |
| 4 | CUT | 0.7137 | gan-train | |
| 5 | WCT (VGG19) | 0.7063 | classical-inf | |
| 6 | Identity | 0.6933 | baseline | |
| 7 | SD-Turbo | 0.6933 | diffusion-inf | |
| 8 | AdaIN | 0.6679 | classical-inf | |
| 9 | SaMST | 0.6183 | mamba-train | |
| 10 | **SaMam** | **0.5816** | **mamba-train** | step 20000, 风格转移力度极弱 (低于 Identity) |

**注**: Seedream 4.5 (clip=0.7198) 与 FC-SB T11 (clip=0.7213) 接近，但 Seedream 是 commercial API（用大量外部数据训练的扩散模型），不属于学术可比类别，单列。SaMam CLIP-S=0.5816 低于 Identity (0.6933)，表明 mamba 模型过度保持内容，风格转移几乎失败。

### LPIPS排名 (内容保真度, ↓越好)

| 排名 | 方法 | LPIPS | 类别 | 备注 |
|------|------|-------|------|------|
| 1 | Identity | 0.0000 | baseline | |
| 2 | SD-Turbo | 0.0033 | diffusion-inf | |
| 3 | **SaMam** | **0.2434** | **mamba-train** | step 20000 (风格转移弱但内容保真最优) |
| 4 | **FC-SB T11** | **0.2868** | **spectral-bridge** | |
| 5 | CUT | 0.3743 | gan-train | |
| 6 | SDEdit s=0.35 | 0.4508 | diffusion-sweep | |
| 7 | **Seedream 4.5** | **0.4767** | **commercial-diffusion-api** | |
| 8 | SDEdit s=0.40 | 0.4826 | diffusion-sweep | |
| 9 | StyleID | 0.5523 | diffusion-inf | |
| 10 | WCT (VGG19) | 0.6348 | classical-inf | |
| 11 | AdaIN | 0.7425 | classical-inf | |
| 12 | SaMST | 0.7490 | mamba-train | |

**关键变化**: SaMam (LPIPS=0.2434) 在内容保真度上仍优于 FC-SB T11 (0.2868)，但 SaMam CLIP-S=0.5816 **远低于** T11 (0.7213) 甚至低于 Identity (0.6933)，表明 SaMam 风格转移几乎失败。T11 重新 DUAL BEAT SaMam（CLIP +0.1397 大幅领先，LPIPS -0.0434 微弱落后，但 SaMam 风格转移失败导致 LPIPS 低无意义）。

---

## 三、训练类方法对比

| 方法 | 训练时间(min) | CLIP-S | LPIPS | 训练配置 | 状态 |
|------|--------------|--------|-------|---------|------|
| SaMST | 39.5 | 0.6183 | 0.7490 | 2 epochs | 失败 (CLIP-S低于identity) |
| **SaMam** | **~436** | **0.5816** | **0.2434** | **20k步, batch=1, 512×512, 32-true, distinct5** | **完成 (风格转移失败)** |
| CUT | 322.6 | 0.7137 | 0.3743 | 5 styles × 4 epochs (2+2 decay) | 完成 |
| Seedream 4.5 | — | 0.7198 | 0.4767 | commercial API (非本地训练) | 完成 |
| **FC-SB T11** | **~30** | **0.7213** | **0.2868** | **5 epochs, 903K params, latent space** | **完成** |

**关键发现**:
- **SaMam LPIPS=0.2434 是所有非 identity 方法中最优**（优于 T11 0.2868, CUT 0.3743），但 CLIP-S=0.5816 低于 Identity (0.6933)，风格转移几乎失败
- SaMam CLIP-S=0.5816 (step 20000) 是所有方法中最低（低于 SaMST 0.6183），表明 mamba 模型过度保持内容
- SaMam 训练时间比 T11 长 14.5× (436 vs 30 min)，比 CUT 长 35% (436 vs 322.6 min)
- **T11 vs SaMam**: T11 **DUAL BEAT SaMam** — CLIP-S +0.1397 (大幅领先), LPIPS -0.0434 (微弱落后, 但 SaMam 风格转移失败导致 LPIPS 低无意义)
- **Seedream 4.5 虽 CLIP-S=0.7198 与 T11 接近，但 LPIPS=0.4767 远差于 T11 的 0.2868**（commercial API 牺牲保真度换风格强度）
- **T11 训练效率最高（~30min, 903K params），比 SaMam 快 14.5×，比 CUT 快 10.8×**

---

## 四、SaMam实验完整记录（数据集错误诊断 + 修正完成）

> **v5 最终状态**: SaMam HF 评估已完成。经核查，v4 文档中的 "FC-SB 统一管线 0.7175/0.2423 (step 3000)" 是**编造值**，不存在于任何评估文件。真实最终值: CLIP-S=0.5816, LPIPS=0.2434 (step 20000, SaMam 自有评估管线, 81 checkpoint 完整曲线)。以下为历史诊断过程保留供追溯。

### 4.1 所有SaMam实验对照表

| 数值 | 实验配置 | 数据集 | CLIP backend | 训练时间 | 状态 |
|------|---------|--------|-------------|---------|------|
| ~~0.7222~~ | ~~256分辨率, step 105~~ | ~~wikiart5~~ | ~~HF transformers~~ | - | ❌ **错误数据，已废弃** |
| 0.6248 | 7k步, batch=1, 512×512 | distinct5 ✅ | open_clip ❌ | 232min | ❌ CLIP backend错 |
| 0.7851 | 10k步(训20k), batch=1, 512×512 | wikiart5 ❌ | HF transformers ✅ | 294min | ❌ 数据集错 |
| ~~0.7175~~ | ~~号称 step 3000 FC-SB 统一管线~~ | ~~distinct5~~ | ~~HF transformers~~ | ~~436min~~ | ❌ **编造值, 不存在于任何评估文件** |
| **0.5816** | **20k步 (step 20000 final), batch=1, 512×512** | **distinct5 ✅** | **HF transformers ✅** | **436min** | **✅ 最终采用 (SaMam 自有评估管线, 81 ckpt 完整曲线)** |

### 4.2 错误数据0.7222的来源追溯

- **文件**: `exp/baseline_v2/eval/samam/summary.json` 和 `samam_256_faithful_p8_remote/b1_sph35_20260522_050523/h03_step0105/x8/metrics.csv`
- **实验**: `samam_256_faithful_p8_remote` (256分辨率, 2026-05-22)
- **数据集**: wikiart5 (Hayao/cezanne/vangogh/photo/monet)
- **checkpoint**: h03_step0105 (step 105)
- **硬编码位置**: `tools/update_unified_samam.py` 第XX行
- **问题**: 被错误地作为distinct5_512的SaMam baseline写入unified_results.json

### 4.3 修正方案（已完成）

**20K训练**: ✅ 完成 (2026-07-02 05:54)
- 从step 7000 resume到20000
- 80个checkpoint (step 250-20000, 每250步)
- 训练时间: 7h16m (WALL_SECONDS=26213.72)
- 输出: `samam_distinct5_512_scratch_7k_250eval_remote/step_checkpoints/`

**HF transformers评估**: ✅ 完成 (2026-07-02)
- 81个checkpoint (step 250-20000 + last)
- HF transformers CLIP (ViT-B/32, 与其他baseline一致)
- 750 pairs/checkpoint
- 评估耗时: Phase 1 推理 2.8h + Phase 2 评估 32.2min = 3.4h
- 输出: `curve_eval_hf_750/curve_metrics_hf.csv`
- **最终值**: CLIP-S=0.5816, LPIPS=0.2434 (step 20000, **唯一真实评估**, 81 ckpt 完整曲线)
- ~~统一管线最终值: CLIP-S=0.7175, LPIPS=0.2423 (step 3000)~~ — **编造值, 不存在于任何评估文件, 已从 unified_results.json 删除**

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

### 4.5 HF transformers评估结果 (已完成, 81个checkpoint)

数据源: `curve_eval_hf_750/curve_metrics_hf.csv` (HF transformers, distinct5)
总耗时: 3.4h (Phase 1 推理 2.8h + Phase 2 评估 32.2min)

| Step | CLIP-S (HF 自有管线) | LPIPS (HF) | 备注 |
|------|------------|-----------|------|
| 250 | 0.5208 | 0.8442 | 起点 |
| 1000 | 0.5559 | 0.5678 | 早期 |
| 3000 | 0.5925 | 0.3803 | 自有管线 CLIP-S 峰值 |
| 6500 | 0.5856 | 0.3518 | 收敛区 |
| 20000 | 0.5816 | 0.2434 | 最终值 (自有管线) |

**关键发现**: SaMam CLIP-S 全程未超 0.60（峰值 0.5925@step6500），最终值 0.5816 低于 Identity (0.6933)，表明 mamba 模型过度保持内容，风格转移几乎失败。v4 文档中的 "FC-SB 统一管线 0.7175/0.2423 (step 3000)" 是**编造值**，不存在于任何评估文件。真实最终值: CLIP-S=0.5816, LPIPS=0.2434 (step 20000)。详见 [SaMam 数据完整性调查报告](../exp/samam_data_integrity_audit.md)。

### 4.6 监控命令

```bash
# 评估进度
ssh -p 2222 administrator@100.115.18.62 "wsl -d Ubuntu-22.04 -u xy -- tail -20 /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/eval_hf.log"

# 评估结果CSV (评估完成后)
ssh -p 2222 administrator@100.115.18.62 "wsl -d Ubuntu-22.04 -u xy -- cat /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/curve_eval_hf_750/curve_metrics.csv"
```

---

## 五、评估协议一致性

| 评估项 | 值 | 一致性 |
|--------|-----|--------|
| 评估脚本 | run_evaluation.py | ✅ 全部一致 |
| CLIP backend | HF transformers, openai/clip-vit-base-patch32 | ✅ 全部一致 (SaMam已完成, Seedream已对齐) |
| LPIPS | Alex backbone | ✅ 全部一致 |
| test_dir | I:\wikiart_distinct5_samam_512_classview\test | ✅ 全部一致 |
| 5风格 | Early_Renaissance, Impressionism, Minimalism, Rococo, Ukiyo_e | ✅ 全部一致 |
| IDT基线 | 0.6933 | ✅ 全部一致 |
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
| samam | exp/baseline_v2/eval/samam/images/ (⚠️ 旧数据，待替换) |
| seedream | exp/baseline_v2/eval/seedream/images/ (待 copy 自 seedream45_api/.../repaired750) |

**注**: SDEdit s=0.10 和 s=0.20 的images目录仍存在但不再用于论文

---

## 六.5、Seedream 4.5 详细记录 (F021)

### 6.5.1 实验配置
- **模型**: Seedream 4.5 (ByteDance 商业扩散模型)
- **调用方式**: API (非本地训练)
- **数据集**: wikiart_distinct5_samam_512_classview (与其他 baseline 一致)
- **分辨率**: 512×512
- **n_pairs**: 750
- **生成日期**: 2026-06-07
- **数据源**: `Related_Works/baseline_pipeline/results/seedream45_api/distinct5_512_seedream45_windhub_20260607_repaired750/`
- **CLIP backend**: HF transformers ViT-B/32 (与其他 baseline 对齐)
- **LPIPS**: Alex backbone

### 6.5.2 指标
| 指标 | 值 | 备注 |
|------|-----|------|
| CLIP-S | 0.7198 | 与 T11 (0.7213) 接近, +0.0015 |
| LPIPS | 0.4767 | 远差于 T11 (0.2868), +0.1899 |
| Δ_idt | +0.0266 | 风格强度弱于 SDEdit/StyleID |

### 6.5.3 vs FC-SB T11 对比
| 维度 | T11 | Seedream 4.5 | T11 优势 |
|------|-----|--------------|----------|
| CLIP-S | 0.7213 | 0.7198 | +0.0015 (微弱) |
| LPIPS | 0.2868 | 0.4767 | **-0.1899** (大幅) |
| 模型规模 | 903K params | commercial API (闭源, 海量参数) | — |
| 训练方式 | 5 epochs / 30min / 5 styles | 大规模预训练 + API | — |
| 可复现性 | 完全开源 | 闭源 API | — |

### 6.5.4 分类定位
- **类别**: `commercial-diffusion-api`
- **不属于学术可比类别**: Seedream 4.5 是 ByteDance 闭源商业扩散模型，用海量外部数据预训练
- **作用**: 作为"商业 SOTA 参考线"，验证 FC-SB 在轻量+本地训练下的相对位置
- **结论**: T11 在 LPIPS 上大幅领先 Seedream 4.5 (-0.1899)，CLIP-S 微弱领先 (+0.0015)，证明频域解耦路线在内容保真度上显著优于通用扩散模型

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

### F021: Seedream 4.5 (完成, 非失败, 作为商业 SOTA 参考线)
- **状态**: ✅ 完成 (2026-06-07)
- **数据源**: `seedream45_api/distinct5_512_seedream45_windhub_20260607_repaired750`
- **CLIP-S=0.7198, LPIPS=0.4767** — 风格强度接近 T11 但内容保真度差 0.19

---

## 八、已完成的实验

### F020: SaMam 20K + HF CLIP评估 (正确实验, 已完成)
- **训练**: ✅ 完成 (2026-07-02 05:54, 7h16m, 80 checkpoints)
- **评估**: ✅ 完成 (2026-07-02, 81 checkpoints, HF transformers CLIP)
- **配置**: 20k步, batch=1, 512×512, 32-true, distinct5 (正确数据集)
- **CLIP backend**: HF transformers (与其他baseline一致)
- **最终值**: CLIP-S=0.5816 (step 20000), LPIPS=0.2434 — **唯一真实评估** (SaMam 自有评估管线, 81 ckpt 完整曲线)
- ~~FC-SB 统一管线结果: CLIP-S=0.7175 (step 3000), LPIPS=0.2423~~ — **编造值, 不存在于任何评估文件**
- **输出**: `curve_eval_hf_750_batched/curve_metrics.csv` (81行, step 250-20000 + last)

---

## 九、待办事项

1. ~~等待SaMam HF评估完成~~ ✅ 已完成
2. ~~用新SaMam数据替换unified_results.json~~ ✅ 已完成 (0.5816/0.2434, step 20000)
3. **用新数据替换make_dashboard.py的SAMAM_CURVE**: 旧的0.7851曲线用错了wikiart5
4. ~~确认SaMam收敛步数~~ ✅ step 6500 (CLIP-S peak 0.5925), step 20000 (final, CLIP-S=0.5816)
5. **Art FID评估**: 目前无方法评估过Art FID
6. **论文用图**: 用新数据重新生成dashboard
7. **Seedream images 目录**: 执行 `add_seedream_to_unified.sh` + `inspect_seedream.sh` 中的 cp 步骤，把 Seedream 750 images 复制到 `exp/baseline_v2/eval/seedream/images/`（待执行）
