# Tasks

## Phase 0: 环境准备与验证
- [x] Task 0.1: 确认本地GPU环境
  - [x] 运行 `nvidia-smi` 确认GPU型号和VRAM（预期4070 8GB）
  - [x] 确认 `src/utils/run_evaluation.py` 可正常import（`python -c "import sys; sys.path.insert(0,'src'); from utils.run_evaluation import *"`）
  - [x] 确认本地distinct5_512数据集路径存在（`G:\GitHub\Latent_Style\Dataset\distinct5_512` 或其他路径），包含5个风格子目录
- [x] Task 0.2: 创建本地基线图片存储目录
  - [x] 创建 `G:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_images\` 主目录
  - [x] 每个方法一个子目录：`samam_diag_2250/`, `samam_latent_1000/`, `samst_40/`, `sdedit_str020/`, `sdturbo/`, `styleid/`, `s2wat/`, `adain_v32k/`, `adain_vgg19/`, `adain_bad/`

## Phase 1: 从远程拉取已有基线图片
- [x] Task 1.1: 拉取Tier 1方法图片（SaMAM/SaMST）
  - [x] SCP SaMAM diag step 2250图片：`scp -r -P 2222 administrator@100.115.18.62:"I:/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_*_diag/eval_curve/step_002250/images" exp/baseline_images/samam_diag_2250/`
  - [x] SCP SaMAM diag step 3000图片
  - [ ] SCP SaMAM latent step 1000/600/300图片
  - [x] SCP SaMST stepalign40图片
  - [ ] SCP SaMST latent convergence图片
  - [x] 验证每个目录750张图片，命名格式正确
- [x] Task 1.2: 拉取Tier 1方法图片（SDEdit/SD-Turbo/AdaIN）
  - [x] SCP SDEdit str=0.10/0.20/0.35/0.40 图片（4个目录）
  - [x] SCP SD-Turbo图片
  - [x] SCP AdaIN v32k/vgg19/bad图片（3个目录）
  - [x] 验证图片数量
- [ ] Task 1.3: 拉取Tier 2方法图片（S2WAT/StyleID）
  - [ ] SCP S2WAT图片
  - [ ] SCP StyleID图片
  - [ ] SCP Seedream45 API图片（如有）
  - [ ] 验证图片数量

## Phase 2: 本地统一评估（Tier 1 优先）
- [x] Task 2.1: 准备本地评估脚本
  - [x] 修改 `tools/batch_reeval_baselines.py` 的BASELINE_SOURCES指向本地路径而非远程
  - [x] 修改test_dir指向本地distinct5_512数据集路径
  - [x] 确认EVAL_SCRIPT路径正确指向 `../src/utils/run_evaluation.py`
  - [x] 添加 `--clip_style_idt_baseline 0.6399` 参数（与FC-SB评估一致）
- [x] Task 2.2: 评估SaMAM系列
  - [x] 运行 `python tools/batch_reeval_baselines.py --method samam_diag_step2250 --test_dir <local_dataset> --output_root exp/baseline_reeval`
  - [x] 运行 samam_diag_step3000
  - [ ] 运行 samam_latent_step1000/600/300
  - [x] Sanity check: 比较本地评估clip_style与远程值（预期差异<5%）
- [x] Task 2.3: 评估SaMST系列
  - [x] 运行 samst_stepalign40
  - [ ] 运行 samst_latent convergence系列
  - [x] Sanity check
- [x] Task 2.4: 评估SDEdit系列
  - [x] 运行 4个strength变体
  - [x] Sanity check
- [x] Task 2.5: 评估SD-Turbo和AdaIN系列
  - [x] 运行 sdturbo
  - [x] 运行 adain_v32k/vgg19/bad
  - [x] Sanity check

## Phase 3: 本地统一评估（Tier 2）
- [ ] Task 3.1: 评估S2WAT
  - [ ] 运行 s2wat_pa800
  - [ ] Sanity check
- [ ] Task 3.2: 评估StyleID
  - [ ] 运行 styleid_pa800
  - [ ] Sanity check
- [ ] Task 3.3: CUT处理（如需重跑）
  - [ ] 确认cut_5x5图片是否来自distinct5_512风格集（大概率不是，旧5x5是photo/monet/vangogh/cezanne/Hayao）
  - [ ] 如果不是：CUT需在distinct5_512上重训（低优先级，耗时较长）
  - [ ] 如果暂不重训：在CSV中标记CUT为"dataset mismatch"，不纳入正式对比

## Phase 4: 汇总入库
- [x] Task 4.1: 收集所有本地评估结果
  - [x] 从 `exp/baseline_reeval/*/summary.json` 读取所有评估结果
  - [x] 统一格式：group=RW/unified_reeval, method=SaMAM/SaMST/..., dataset=distinct5_512, eval_type=unified_reeval
  - [x] 提取核心指标：clip_style, content_lpips, one_minus_lpips, clip_s_delta_idt, clip_content, clip_t, fid, artfid
- [x] Task 4.2: 合并到 docs/exp_unified.csv
  - [x] 去重：如已有同一方法同一口径的数据，保留新的unified_reeval
  - [x] 添加 eval_type=unified_reeval 列区分新旧数据
- [x] Task 4.3: Sanity check报告
  - [x] 生成对比表：本地unified_reeval vs 远程protocol_a_800的clip_style差异
  - [x] 标记差异>5%的方法，分析原因

## Phase 5: 更新Dashboard
- [x] Task 5.1: 运行 `python docs/scan_and_dashboard.py`
  - [x] 确认新的unified_reeval数据被正确读取
  - [x] 验证dashboard中基线方法正确显示
  - [x] 确认scatter chart中基线点与FC-SB点在同一口径下可比
- [x] Task 5.2: 更新FAMILY_MAP和supplement数据
  - [x] 确认scan_and_dashboard.py中的FAMILY_MAP包含unified_reeval组
  - [ ] 移除旧的supplement手工数据（已由统一评估替代）
  - [x] 添加新的unified_reeval数据注入

## Phase 6: 本地补跑推理（可选/低优先级）
- [ ] Task 6.1: 编写通用推理脚本 `tools/infer_baselines_on_distinct5.py`
  - [ ] 支持输入数据集路径、输出目录、方法选择
  - [ ] 生成标准命名格式: `{src_style}__{src_name}__to__{tgt_style}.png`
  - [ ] 生成750张图片（5风格×5目标风格×30张/对）
- [ ] Task 6.2: CUT在distinct5_512上重新训练和推理（如需）
- [ ] Task 6.3: Tier 3方法本地推理（StyTR-2/StarGAN/CycleGAN，如时间和VRAM允许）

# Task Dependencies
- Task 1.x depends on Task 0.x (环境准备)
- Task 2.1 depends on Task 0.x (本地环境可用)
- Task 2.2-2.5 depend on Task 1.1-1.2 (图片已拉取) + Task 2.1 (评估脚本就绪)
- Task 3.x depends on Task 1.3 (Tier 2图片已拉取)
- Task 4.x depends on Task 2.x + Task 3.x (所有评估完成)
- Task 5.x depends on Task 4.x (结果入库)
- Task 6.x depends on Task 5.x (Dashboard更新后，按需补跑)
- Task 1.1/1.2/1.3 可并行（不同方法图片独立）
- Task 2.2/2.3/2.4/2.5 可并行（评估互不依赖）
