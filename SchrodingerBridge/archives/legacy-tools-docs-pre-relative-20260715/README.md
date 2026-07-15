# docs/tools — 数据库 / 评估协议 / Infra / 调用命令 / 实验经验

> FC-SB 项目的工程参考手册: 数据集路径、评估协议、远程/本地基础设施、训练/评估调用命令、实验中积累的经验教训。
> 配套文档: [docs/math/README.md](../math/README.md) (理论), [docs/baseline/README.md](../baseline/README.md) (baseline 核查), [docs/exp/experiment_audit.md](../exp/experiment_audit.md) (实验脉络)。

---

## 1. 数据库 (Datasets)

### 1.1 数据集分类总览

| 数据集标识 | 路径 | 分辨率 | vlen | 用途 |
|---|---|---|---|---|
| **distinct5 (主线)** | `G:/GitHub/Latent_Style/Dataset/distinct5_512_latents_ema` | 512 (latent 32×32) | 1.0 | 全量训练, 所有 630 系列正式实验 |
| **wikiarts_5 (smoke)** | `F:\wikiart_distinct5_samam_512_latents_ema` | 512 | **0.04** | 4% 子集 smoke test |
| **wikiarts_5 (full)** | `F:\wikiart_distinct5_samam_512_latents_ema` | 512 | 1.0 | 全量早期迭代 (628 之前) |
| **fewshot6** | `G:/.../Dataset/fewshot6_512_latents_ema` | 512 | 1.0 | Few-shot Pop_Art 注入 (4J.6 系列) |
| **测试集 (统一)** | `I:\wikiart_distinct5_samam_512_classview\test` | 512 | — | 全部 12 baseline + 我们模型评估 |

### 1.2 distinct5 (主线, 论文用)

**5 风格**: Early_Renaissance, Impressionism, Minimalism, Rococo, Ukiyo_e

**Train**: `G:/GitHub/Latent_Style/Dataset/distinct5_512_latents_ema/train` (本地 G 盘)
- SDXL VAE 编码后的 latent (4×32×32, EMA)
- 每风格 ~3600 图 (5 × 3600 = 18000)

**Test**: `I:\wikiart_distinct5_samam_512_classview\test` (远程 I 盘)
- 5 风格 × 30 图 = 150 source
- 750 pairs (150 src × 5 tgt styles)
- 12 baseline 全部在此测试集评估

### 1.3 数据集分类规则 (重要)

> 用户原话: "每一个实验一定要分清楚用的是哪个数据集, 256或者5*3600或者wikiarts_5(不是distinct5)这几个的, 在exp/256之类的下面分别放, 不要污染主线结论。"

按 `data.data_root` + `data.virtual_length_multiplier` 双字段判定:

| data_root 关键字 | vlen | 数据集分类 | 实验归属 |
|---|---|---|---|
| `distinct5_512_latents_ema` | 1.0 | **distinct5 主线** | `exp/distinct5/` (论文用) |
| `wikiart_distinct5_samam_512_latents_ema` | 0.04 | **wikiarts_5 smoke** | `exp/wikiarts5_smoke/` (无性能意义) |
| `wikiart_distinct5_samam_512_latents_ema` | 1.0 | **wikiarts_5 full** | `exp/wikiarts5_full/` (早期迭代) |
| `fewshot6_512_latents_ema` | 1.0 | **fewshot6** | `exp/fewshot6/` (4J.6 专用) |
| (256 分辨率) | * | **256 res** | `exp/256/` (历史 SaMam 256 实验归档) |

详见 [docs/exp/experiment_audit.md §0 数据集分类总览](../exp/experiment_audit.md)。

### 1.4 数据集使用约束

- **训练**: 必须从零开始独立目录, 禁止 `--skip-train` resume (避免结论失真)
- **测试集**: `I:\wikiart_distinct5_samam_512_classview\test` (远程 I 盘, 非 F 盘)
- **DataLoader**: `num_workers=0, pin_memory=False, persistent_workers=False` (防 CUDA OOM)

---

## 2. 评估协议 (Evaluation Protocol)

### 2.1 统一协议 (12 baseline + 我们模型)

| 项 | 配置 | 来源 |
|---|---|---|
| 评估脚本 | `src/utils/run_evaluation.py` | FC-SB 统一管线 |
| CLIP backend | **HF transformers** (`openai/clip-vit-base-patch32`, ViT-B/32) | `_DEFAULT_HF_CLIP_REPO` (run_evaluation.py:75) |
| LPIPS backbone | **Alex** | `lpips.LPIPS(net='alex')` |
| n_pairs | 750 (150 src × 5 tgt, 含 identity 对) | CUT 为 745 (5 张缺失) |
| 数据集 | `wikiart_distinct5_samam_512_classview/test` | distinct5_512 |
| 5 风格 | Early_Renaissance, Impressionism, Minimalism, Rococo, Ukiyo_e | — |
| Identity 基线 | CLIP-S=0.6933, LPIPS=0.0 | F001 |
| Δ_idt | CLIP-S(method) - 0.6933 | — |

### 2.2 CLIP backend 对齐说明 (重要)

| 项 | 配置 |
|---|---|
| 项目默认 backend | `hf` (`config_schema.py:937`: `full_eval_clip_backend: str = "hf"`) |
| 项目默认 HF repo | `openai/clip-vit-base-patch32` (`run_evaluation.py:75`) |
| T11 / 4F.1 / 全部本地+远程实验 | **HF transformers** |
| 12 baselines (除 SaMam 旧数据) | **HF transformers** |
| SaMam 旧数据 (0.7222) | ~~open_clip~~ 已废弃 |
| SaMam 新数据 (F020, 真实值) | **HF transformers** (SaMam 自有评估管线 `eval_samam_metrics_phase2.py`) |
| Seedream 4.5 API | **HF transformers** |

**结论**: 全部指标在 HF transformers CLIP ViT-B/32 下对齐, 可直接横向比较。SaMam 的 open_clip 旧数据不可与其它方法对比 (绝对值偏低 0.05-0.10)。

### 2.3 SaMam 评估管线 (特殊)

SaMam 不使用 FC-SB 统一管线, 而用自有评估脚本:
- `tools/samam_distinct5_scratch/gen_samam_images_phase1.py` (推理生成)
- `tools/samam_distinct5_scratch/eval_samam_metrics_phase2.py` (CLIP/LPIPS 评估)
- 调用脚本: `run_2phase.sh`

**协议一致性**: SaMam 自有脚本使用相同 HF transformers CLIP ViT-B/32 + LPIPS Alex + 750 pairs + distinct5_512, 可横向比较。详见 [docs/exp/samam_data_integrity_audit.md](../exp/samam_data_integrity_audit.md) §2.2。

### 2.4 评估输出文件

每个评估实验产生:
```
{exp_dir}/full_eval/{epoch_XXXX}/
├── metrics.csv              # 全部 metrics (含 per-style breakdown)
├── summary.json             # CLIP-S/LPIPS 汇总
├── images/                  # 750 张生成图
└── runtime.json             # 评估耗时
```

### 2.5 WFI 检查 (白化验收)

> 用户记忆硬约束: "WFI score must be < 0.40 to pass白化验收标准。任何后续优化 (text, cross-attn, DINO) 必须先通过 WFI check。"

WFI (Whitening Failure Index) 由 `src/utils/wfi.py` 计算, 用于检测风格转移是否真正发生 (而非简单白化)。详见 [docs/math/README.md §5 Style Conditioning](../math/README.md)。

---

## 3. 基础设施 (Infrastructure)

### 3.1 远程服务器

```
ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62
```

- **OS**: Windows + WSL2 (Ubuntu-22.04)
- **GPU**: RTX 4070 Laptop (8GB VRAM)
- **CPU**: 7940HX
- **WSL 用户**: `xy`
- **远程工作目录**: `/mnt/i/Github/Latent_Style/SchrodingerBridge` (仅一级薛定谔桥目录)

### 3.2 远程 I 盘结构 (整理后)

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

### 3.3 本地 G 盘结构 (整理后)

```
g:\GitHub\Latent_Style\SchrodingerBridge\
├── src/                     (主代码: run.py, model620.py, blocks620.py, trainer.py, config_schema.py, utils/)
├── configs/                 (实验配置 .json)
├── exp/
│   ├── FCSB/                (93 个 distinct5 主线: early 3 + phase4 66 + local_t 24)
│   ├── baseline/            (3 个: reeval, images, v2)
│   ├── 256/                 (256 历史占位, 非主线)
│   ├── wiki5/               (11 个 wikiarts5 非主线: smoke 10 + full/task4_iter 16 子目录)
│   ├── fewshot6/            (3 个: 4J.6 系列, 非主线)
│   ├── legacy/              (shared 7 + logs 12)
│   └── README.md
├── docs/                    (本文档所在)
├── tools/                   (SaMam 评估脚本等)
└── run.py                   (项目入口, 转发到 src/run.py)
```

### 3.4 VRAM 约束 (硬性)

> 用户记忆硬约束:
> - "训练阶段使用 Patience=2、max=10, 收敛阶段显存不超过 10.8GB, 8-9GB可接受"
> - "评估阶段显存严格不超过 7G (batch_size=2, full_eval_batch_size=2, ref_feature_batch_size=2)"
> - "All ablation experiment configurations must use batch_size=24 to ensure 12GB VRAM safety"
> - "训练、推理评估中, 尽量把显存控制在 9-11G, 重复利用算力"

### 3.5 显存探测方法

> 用户记忆: "显存探测应采用模型推断方法, 通过少量验证点 (约10个) 拟合配置与batch的非线性关系, 避免直接跑300个聚类"

---

## 4. 调用命令 (Commands)

### 4.1 训练 + 评估 (本地)

```powershell
# Windows PowerShell (本地 G 盘)
$env:PYTHONPATH = "g:\GitHub\Latent_Style\SchrodingerBridge\src"
$env:CUDA_VISIBLE_DEVICES = "0"
Set-Location g:\GitHub\Latent_Style\SchrodingerBridge
python src\run.py --config configs\{config_name}.json 2>&1 | Tee-Object -FilePath "exp\{exp_dir}\train.log"
```

`run.py` 会自动:
1. 训练 (按 `training.num_epochs`)
2. 每 epoch 后 full_eval (若 `full_eval_each_epoch: true`)
3. 输出到 `checkpoint.save_dir/full_eval/epoch_XXXX/`

### 4.2 仅评估 (eval_only)

```powershell
python src\run.py `
    --config {config.json} `
    --eval_only `
    --checkpoint_path {epoch_XXXX.pt} `
    --style_subdirs "Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e"
```

### 4.3 远程训练 (WSL)

```bash
# SSH 到远程
ssh -p 2222 administrator@100.115.18.62

# 进入 WSL
wsl -d Ubuntu-22.04 -u xy

# 在 WSL 中运行
cd /mnt/i/Github/Latent_Style/SchrodingerBridge
PYTHONPATH=/mnt/i/Github/Latent_Style/SchrodingerBridge/src \
CUDA_VISIBLE_DEVICES=0 \
python src/run.py --config configs/{config_name}.json \
    2>&1 | tee exp/{exp_dir}/train.log
```

### 4.4 远程 PowerShell 一键启动 (历史脚本)

参考 `_launch_t11_remote.ps1`:

```powershell
$env:PYTHONPATH = "I:\Github\Latent_Style\SchrodingerBridge\src"
$env:CUDA_VISIBLE_DEVICES = "0"
Set-Location I:\Github\Latent_Style\SchrodingerBridge
New-Item -ItemType Directory -Force -Path "I:\Github\Latent_Style\SchrodingerBridge\exp\{exp_dir}" | Out-Null
python src\run.py --config configs\{config_name}.json 2>&1 | Tee-Object -FilePath "exp\{exp_dir}\train.log"
```

### 4.5 SaMam 评估 (特殊)

```bash
# 远程 SaMam 2-phase 评估
cd /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote
bash run_2phase.sh  # 调用 gen_samam_images_phase1.py + eval_samam_metrics_phase2.py
```

### 4.6 监控命令

```bash
# 评估进度
ssh -p 2222 administrator@100.115.18.62 "wsl -d Ubuntu-22.04 -u xy -- tail -20 /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/{exp_dir}/train.log"

# 评估结果 CSV
ssh -p 2222 administrator@100.115.18.62 "wsl -d Ubuntu-22.04 -u xy -- cat /mnt/i/.../{eval_dir}/curve_metrics.csv"
```

### 4.7 关键配置字段 (config.json)

```json
{
  "training": {
    "batch_size": 24,            // 训练 batch (12GB VRAM 安全)
    "num_epochs": 5,             // SOTA 配置标准
    "patience": 2,               // Early stopping
    "max_epochs": 10             // 硬上限
  },
  "data": {
    "data_root": "G:/GitHub/Latent_Style/Dataset/distinct5_512_latents_ema/train",
    "virtual_length_multiplier": 1.0,  // 1.0=full, 0.04=smoke
    "num_workers": 0,            // 防 CUDA OOM
    "pin_memory": false
  },
  "full_eval": {
    "batch_size": 2,             // 评估 batch (≤7GB VRAM)
    "ref_feature_batch_size": 2,
    "num_steps": 8,              // ODE 求解步数 (T11)
    "vae_model": "ema"
  },
  "full_eval_clip_backend": "hf",  // 强制 HF transformers
  "checkpoint": {
    "save_dir": "...",
    "resume_checkpoint": ""      // 禁止 resume
  },
  "ablation": {
    "name": "...",
    "axis": "...",
    "stage": "...",
    "notes": "..."                // 必须: why + 假设 + 目标
  }
}
```

### 4.8 PYTHONPATH 注意

> 用户记忆: "PYTHONPATH should not be manually set in run scripts (handled by run.py)"

`run.py` (项目根) 自动将 `src/` 加入 `sys.path`:
```python
# run.py (项目根)
def main() -> None:
    root = Path(__file__).resolve().parent
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    module = importlib.import_module("run")
    module.main()
```

**仅在远程 WSL 中需手动设置** `PYTHONPATH=/mnt/i/.../src` (因路径不同)。

### 4.9 命令超时

> 用户记忆: "Command execution: requires adding timeout (30 seconds) to all commands"

所有训练/评估命令应添加 30s timeout 防止卡死 (短命令), 长训练命令用 `nohup` 或 PowerShell 后台 job。

---

## 5. 实验中的经验 (Lessons Learned)

### 5.1 关键工程经验

| # | 经验 | 来源 |
|---|------|------|
| 1 | **训练必须从零开始独立目录**, 禁止 `--skip-train` resume (避免结论失真) | user_profile |
| 2 | **DataLoader**: `num_workers=0, pin_memory=False, persistent_workers=False` 防 CUDA OOM | project_memory |
| 3 | **PYTHONPATH 不在 run scripts 中手动设置** (由 run.py 处理) | project_memory |
| 4 | **Run scripts 必须用 error handling** (`if...then...else`) 让单实验失败不阻塞后续 | project_memory |
| 5 | **无效代码/机制确认无效后直接删除** (不 ablate) | user_profile |
| 6 | **任何后续优化必须先通过 WFI check** (WFI < 0.40) | project_memory |
| 7 | **所有探索方向必须训练到收敛** (Patience=2, max=10, 至少 5 epochs) | project_memory |
| 8 | **实验配置档位需显著拉开区别**, 包含极端值, 避免结果重叠 | project_memory |
| 9 | **数据集路径必须配置为 I 盘** (`/mnt/i/...`), 非 F 盘 | project_memory |
| 10 | **远程实验代码仅一级薛定谔桥目录** (`/mnt/i/Github/Latent_Style/SchrodingerBridge`) | project_memory |
| 11 | **ablation experiments 应用 batch_size=24** 保 12GB VRAM 安全 | project_memory |
| 12 | **DINO cache 必须先配置** 再跑 ablation | project_memory |
| 13 | **测试图目录必须是** `/mnt/i/wikiart_distinct5_samam_512_classview/test` (非 overfit50) | project_memory |
| 14 | **条件编译优化** 避免影响其他测试 | user_profile |
| 15 | **不允许使用远程 GPU**, 要求本地重训或数据传输 | user_profile |

### 5.2 调试经验

> 用户记忆: "Debugging: prefers adding print statements (printf) for debugging, considering it an old and reliable method"

- 用 print 语句调试, 老 reliable 方法
- 当 git 回退不可行时, 手动删除问题代码恢复功能, 再 commit
- 改动一个 lab 不能影响其他 lab (lab5 特殊调整, 其他 lab 保持 default)

### 5.3 实验设计经验

> 用户记忆: "requires in-depth theoretical analysis first, with at least 4 exploration directions proposed before implementation and optimization"

- **理论分析先行**: 至少提出 4 个探索方向再实施
- **大胆实验**: 偏好大胆、开拓性的实验方案或合理优化方案, 包括 ablation studies
- **有意义 checkpoint**: 用有意义的 checkpoint 做推理参数 (而非随便选)
- **FC-SB 理论优先**: 模型设计/修改基于 FC-SB 理论
- **ablation 扩展**: 256-320 组 grid-based configurations, 参数轴上数值显著分离
- **文档详尽**: 所有实验细节 (时间、模型、数据集、训练/推理时间) 必须记录

### 5.4 工作流经验

> 用户记忆: "Prioritizes running through tests, committing to git, and writing experiment reports before addressing input issues"

工作流顺序:
1. 跑测试
2. commit git
3. 写实验报告
4. 处理输入问题

### 5.5 模型设计经验 (从实验脉络总结)

详见 [docs/exp/experiment_audit.md](../exp/experiment_audit.md) + [docs/math/README.md](../math/README.md)。

关键失败教训:
- **DINOv2 提取 style 失败** (4C): content-biased, 改用 learnable style_memory
- **T5 (p=0.0) 训练/推理分布失配**: clip=0.7061 FAIL
- **T10 (p=0.5) q_proj 平均两种分布**: 50% 不足以精通 DWT
- **T19a (depth=6) WCT eigh 数值不稳定**: NaN, 需对角线正则化
- **4G.2b α=0.5≡α=1.0**: 多步 Euler 迭代累积 invalidate α, 需 EOTA
- **Heun→RK4 饱和**: 其他误差源主导, RK4 无收益

关键成功经验:
- **3-Level Haar DWT 是峰值** (4F.1): clip=0.7319 SOTA
- **Stochastic DWT Route p=0.8** (T11): 80% DWT + 20% 全空间, 本地 SOTA
- **Euler → Heun 结构性突破** (4I.2b): 双提升 SOTA
- **EOTA only_last_step=True** (4H.1): 解耦 ODE 求解与风格注入
- **per_subband_wct** (T11 配置): LL 不动, LH/HL/HH 独立 WCT

### 5.6 SaMam 数据完整性教训

**编造值事件**: v4 文档中的 "FC-SB 统一管线 0.7175/0.2423 (step 3000)" 是**编造值**, 不存在于任何评估文件。

**根因**: LLM 自身产生编造数据 (Deli_AutoResearch §10.4: "Fabricated citations and data artifacts originate from the LLM itself")。

**修复**: 81 checkpoint 完整曲线审计, 真实值 step 20000 CLIP-S=0.5816/LPIPS=0.2434。详见 [docs/exp/samam_data_integrity_audit.md](../exp/samam_data_integrity_audit.md)。

**教训**: 引用类内容必须每 20 条核查一次, 不可批量后置 (Deli_AutoResearch §9.4)。

---

## 6. 相关文档索引

| 文档 | 内容 |
|------|------|
| [docs/math/README.md](../math/README.md) | FC-SB 理论描述 (Schrödinger Bridge + Haar DWT + EOTA + DWT Route) |
| [docs/baseline/README.md](../baseline/README.md) | 12 baseline 数据完整性核查 + 收敛证据 |
| [docs/exp/README.md](../exp/README.md) | 实验整理总入口 (远程+本地) |
| [docs/exp/experiment_audit.md](../exp/experiment_audit.md) | 我们模型实验脉络审计 + ckpt 保留/删除建议 |
| [docs/exp/samam_data_integrity_audit.md](../exp/samam_data_integrity_audit.md) | SaMam 81 checkpoint 完整曲线 + 编造值调查 |
| [docs/exp/remote_experiments.md](../exp/remote_experiments.md) | 远程 I 盘所有实验清单 |
| [docs/exp/local_experiments.md](../exp/local_experiments.md) | 本地 G 盘所有实验清单 |
| [docs/72/02_theory.md](../72/02_theory.md) | FC-SB 详细理论 (含公式推导) |
| [docs/72/03_experiments.md](../72/03_experiments.md) | 历史实验全景: Phase 4A-4J + Local T1-T19, 90+ 配置 |
| [docs/72/07_related_works.md](../72/07_related_works.md) | 12 baseline 完整指标表 + SaMam v5 真实值 |
| [src/utils/run_evaluation.py](../../src/utils/run_evaluation.py) | 评估脚本主入口 |
| [src/run.py](../../src/run.py) | 训练+评估主入口 |
| [exp/baseline/v2/eval/unified_results.json](../../exp/baseline/v2/eval/unified_results.json) | 12 baseline 统一评估结果 (数据真相源) |

---

## 7. 关键脚本索引

| 脚本 | 用途 |
|------|------|
| `run.py` (项目根) | 入口, 转发到 `src/run.py` |
| `src/run.py` | 训练 + full_eval 主循环 |
| `src/utils/run_evaluation.py` | 评估管线 (HF CLIP + LPIPS Alex + 750 pairs) |
| `src/utils/inference.py` | LGTInference, VAE encode/decode |
| `src/utils/wfi.py` | WFI (白化验收) 计算 |
| `src/utils/artfid_metric.py` | ArtFID 评估 (可选) |
| `src/utils/introstyle_eval.py` | IntroStyle 评估 (可选) |
| `src/config_schema.py` | ExperimentConfig schema |
| `src/trainer.py` | SBTrainer 训练循环 |
| `src/blocks620.py` | SpectralBridgeBlock620 + DWT route 实现 |
| `src/model620.py` | FC-SB 模型主体 |
| `tools/samam_distinct5_scratch/gen_samam_images_phase1.py` | SaMam 推理 |
| `tools/samam_distinct5_scratch/eval_samam_metrics_phase2.py` | SaMam CLIP/LPIPS 评估 |

---

**最后更新**: 2026-07-03 (M24, docs/tools 框架建立)
**主参考**: 本文档 §1-§7 + 引用的子文档
**维护原则**: 工程变更同步更新本文档, 新增实验/脚本/数据集时补充对应章节
