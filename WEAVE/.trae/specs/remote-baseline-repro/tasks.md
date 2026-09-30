# Tasks

## Phase 1: 远程环境确认 + Identity 基线

- [ ] Task 1: 确认远程环境（Python包、数据集、模型缓存）
  - [ ] 检查远程 Python/PyTorch/CUDA 版本
  - [ ] 检查 diffusers/transformers/lpips/mamba-ssm 是否已装
  - [ ] 确认 distinct5_512 test/train 数据完整（5×30 test, 5×1000 train）
  - [ ] 确认 SD1.5 / SD-Turbo 模型缓存位置
  - [ ] 确认 AdaIN decoder.pth 是否可下载

- [ ] Task 2: 远程 Identity 基线（已完成，验证一致性）
  - [ ] 确认远程 `exp/baseline_reeval/identity_baseline/summary.json` 存在且数值正确

## Phase 2: 零样本推理（并行执行）

- [ ] Task 3: SDEdit 推理（4个strength）
  - [ ] 写推理脚本到远程 `tools/infer_sdedit_v2.py`
  - [ ] SD1.5 img2img, float16, attention_slicing, seed=42
  - [ ] prompt模板: "a painting in {style} style"
  - [ ] strength={0.10, 0.20, 0.35, 0.40}, steps=20, guidance=7.5
  - [ ] 输出到 `exp/baseline_v2/images/sdedit_str{xxx}/`
  - [ ] 每个strength 750张图

- [ ] Task 4: SD-Turbo 推理
  - [ ] 写推理脚本到远程 `tools/infer_sdturbo_v2.py`
  - [ ] stabilityai/sd-turbo, float16
  - [ ] steps=1, guidance=1.0, strength=0.8
  - [ ] 输出到 `exp/baseline_v2/images/sdturbo/`
  - [ ] 750张图

- [ ] Task 5: AdaIN 推理
  - [ ] 下载官方 decoder.pth + vgg_normalised.pth 到远程
  - [ ] 写推理脚本到远程 `tools/infer_adain_v2.py`
  - [ ] alpha=1.0, 每目标风格用该风格第1张test图作style reference
  - [ ] 输出到 `exp/baseline_v2/images/adain_v32k/`
  - [ ] 750张图

- [ ] Task 6: StyleID 推理
  - [ ] 确认 StyleID 代码位置和运行方式
  - [ ] 写推理脚本
  - [ ] 输出到 `exp/baseline_v2/images/styleid/`
  - [ ] 750张图

## Phase 3: 训练类方法（顺序执行，占GPU）

- [ ] Task 7: SaMAM 训练+推理
  - [ ] 准备 distinct5_512 训练数据（content=MS_COCO? style=distinct5_512 train?）
  - [ ] WSL 下用 mamba-ssm 训练: `python train_SaMam.py --content ... --style ... --batch-size 2 --precision 16-mixed --train-image-size 512`
  - [ ] 训练日志保存到 `exp/baseline_v2/train_logs/samam/`
  - [ ] 推理: 对 test 集 150 张图 × 5 目标风格 = 750 张
  - [ ] 输出到 `exp/baseline_v2/images/samam/`

- [ ] Task 8: SaMST 训练+推理
  - [ ] 修改 train2/train.yml 指向 distinct5_512
  - [ ] 训练: image_size=512, batch_size=2, epochs=20
  - [ ] 训练日志保存到 `exp/baseline_v2/train_logs/samst/`
  - [ ] 推理: 750 张
  - [ ] 输出到 `exp/baseline_v2/images/samst/`

- [ ] Task 9: S2WAT 训练+推理
  - [ ] 准备训练数据（预处理为 224×224 或直接用 512）
  - [ ] 训练: batch_size=1, bf16, grad_checkpoint, epoch=40000
  - [ ] 训练日志保存到 `exp/baseline_v2/train_logs/s2wat/`
  - [ ] 推理: 750 张
  - [ ] 输出到 `exp/baseline_v2/images/s2wat/`

- [ ] Task 10: CUT 训练+推理
  - [ ] 准备 20 个风格对的数据目录（trainA/trainB）
  - [ ] 逐对训练: crop_size=512, batch_size=1, n_epochs=200+200
  - [ ] 训练日志保存到 `exp/baseline_v2/train_logs/cut/`
  - [ ] 推理: 每对模型推理对应风格对，共 600 张（不含 identity 对）
  - [ ] Identity 对用源图直接复制，补齐 750 张
  - [ ] 输出到 `exp/baseline_v2/images/cut/`

## Phase 4: 统一评估+汇总

- [ ] Task 11: 批量评估所有方法
  - [ ] 对每个方法的 images/ 目录运行 run_evaluation.py
  - [ ] 汇总所有 summary.json 到 unified_results.json
  - [ ] 确保所有方法用 LPIPS-VGG 评估

- [ ] Task 12: 更新文档和 Dashboard
  - [ ] 更新 `docs/Related/baseline_methods_catalog.md` 的结果表
  - [ ] 生成 `docs/exp_dashboard_v3.html`
  - [ ] 写入 `docs/exp_unified.csv`

# Task Dependencies

- Task 1 → Task 3,4,5,6,7,8,9,10 (环境确认后才能开始)
- Task 2 已完成 (独立)
- Task 3,4,5,6 可并行 (不同时占GPU需顺序，但互不依赖)
- Task 7,8,9,10 必须顺序 (共享GPU)
- Task 7 依赖 mamba-ssm (WSL)
- Task 10 数据准备最复杂（20对目录）
- Task 11 依赖 Task 3-10 全部完成
- Task 12 依赖 Task 11
