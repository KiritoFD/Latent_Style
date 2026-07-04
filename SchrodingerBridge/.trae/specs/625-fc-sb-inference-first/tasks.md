# 625 FC-SB推理优先实验探索 - The Implementation Plan

## [x] Task 1: 创建625实验目录和推理参数扫描脚本
- **Priority**: high
- **Depends On**: None
- **Description**: 
  - 创建 exp/625_fc_sb/ 目录结构
  - 编写 run_fc_inference_sweep.py 脚本
  - 修复了VAE double scaling bug、load_vae参数顺序bug、bridge_path_mode强制vertical
  - 支持sigma/kernel/fc_modes/sigma_schedule多值参数扫描
- **Acceptance Criteria Addressed**: AC-1
- **Test Requirements**:
  - `programmatic` TR-1.1: 脚本 py_compile 通过
  - `programmatic` TR-1.2: VAE scaling正确（scale_in/scale_out，无双重缩放）
  - `programmatic` TR-1.3: load_vae参数顺序正确
  - `programmatic` TR-1.4: bridge_path_mode强制为vertical

## [x] Task 2: 创建干净基线G0配置（远程版）
- **Priority**: high
- **Depends On**: None
- **Description**: 
  - 基于 configs/620_spatial_bridge_ablation_recommended.json
  - 路径替换为WSL路径（/mnt/i/...）
  - G0: Base Locking ON (vertical), 所有FC OFF, σ=0.02, constant, kernel=5
- **Acceptance Criteria Addressed**: AC-2
- **Test Requirements**:
  - `programmatic` TR-2.1: 配置JSON成功加载
  - `programmatic` TR-2.2: 所有路径为WSL格式
  - `programmatic` TR-2.3: i2sb_fiber_project_endpoint/noise/fiber_only_endpoint均为false

## [x] Task 3: 创建G1-G7 FC-SB增量实验配置（累积增量设计）
- **Priority**: high
- **Depends On**: Task 2
- **Description**: 
  - 链式增量：G1从G0修改，G2从G1修改，以此类推
  - G0: 基线 (ep=F, noise=F, foep=F, σ=0.02, k=5)
  - G1: +Fiber Velocity Projection (ep=T)
  - G2: +Fiber SDE Noise (noise=T, σ=0.04)
  - G3: +Fiber-Only Endpoint (foep=T, 全FC开启)
  - G4: σ=0.06
  - G5: σ=0.08 (FC.md魔法阈值)
  - G6: +curriculum sigma schedule
  - G7: σ=0.10, kernel=7 (高sigma+大核)
- **Acceptance Criteria Addressed**: AC-3, AC-4
- **Test Requirements**:
  - `programmatic` TR-3.1: 所有8个配置参数验证通过
  - `programmatic` TR-3.2: 累积增量逻辑正确（无reset重置）

## [x] Task 4: 创建远程批量运行脚本（推理优先版）
- **Priority**: high
- **Depends On**: Task 2, Task 3
- **Description**: 
  - run_all.sh 推理优先流程：
    - STEP 0: 生成配置
    - STEP 1: 推理sweep（0训练成本，用已有checkpoint）
    - STEP 1b: Pareto分析
    - STEP 2: （可选，注释中）人工选择后训练关键配置
- **Acceptance Criteria Addressed**: AC-2, AC-3, AC-4
- **Test Requirements**:
  - `programmatic` TR-4.1: bash语法正确
  - `programmatic` TR-4.2: HF缓存路径正确设置
  - `programmatic` TR-4.3: PROJECT_ROOT自动检测(GitHub/Github)

## [x] Task 5: 创建Pareto结果分析脚本
- **Priority**: high
- **Depends On**: Task 1
- **Description**: 
  - summarize_results.py: pandas + matplotlib
  - 计算Pareto frontier（CLIP↑ vs LPIPS↓）
  - 生成pareto.png散点图+红色Pareto线+标注
  - 输出results.csv和pareto_optimal.csv
  - 打印3个推荐配置（最佳style/最佳content/折中）
- **Acceptance Criteria Addressed**: AC-6
- **Test Requirements**:
  - `programmatic` TR-5.1: py_compile通过
  - `programmatic` TR-5.2: Pareto算法正确（按LPIPS排序跟踪CLIP最大值）

## [x] Task 6: 代码审查与预运行验证
- **Priority**: high
- **Depends On**: Task 1-5
- **Description**: 
  - 发现并修复了：gen_configs非增量逻辑、VAE double scaling、load_vae参数顺序、summarize_results功能不匹配、run_all.sh缺少推理sweep步骤
  - 验证model620.py FC-SB推理逻辑正确
  - 验证所有脚本py_compile通过
  - 验证G0-G7配置参数正确
- **Acceptance Criteria Addressed**: AC-5
- **Test Requirements**:
  - `programmatic` TR-6.1: 所有Python脚本语法检查通过
  - `programmatic` TR-6.2: G0-G7配置参数矩阵验证通过
  - `human-judgement` TR-6.3: FC-SB推理代码与FC.md一致

## [ ] Task 7: 远程部署与0训练成本推理扫描
- **Priority**: high
- **Depends On**: Task 6
- **Description**: 
  - 将代码同步到远程3060 WSL
  - 确认CHECKPOINT_BASE路径（默认exp/fc_sb_r2/g0_baseline/checkpoints）
  - 运行 bash exp/625_fc_sb/run_all.sh
  - 推理扫描参数：σ=[0.02,0.04,0.06,0.08,0.10], kernel=[5,7], fc_modes=[none,ep,ep_noise,ep_noise_foep], schedule=[constant]
  - 总共5×2×4×1=40个配置组合（max_samples=30 per style）
  - 运行summarize_results.py生成Pareto分析
- **Acceptance Criteria Addressed**: AC-1
- **Test Requirements**:
  - `programmatic` TR-7.1: 推理扫描完成无错误
  - `programmatic` TR-7.2: 生成results.csv和pareto.png
  - `human-judgement` TR-7.3: 分析Pareto前沿，找到最优FC参数组合

## [ ] Task 8: G0基线训练（如需要）
- **Priority**: high
- **Depends On**: Task 7
- **Description**: 
  - 如果推理扫描使用的checkpoint效果不佳（如fc_sb_r2/g0不是最优），训练G0基线3 epochs
  - 否则跳过，直接使用推理扫描结果
  - 训练后评估验证clip≈0.70, LPIPS≈0.34
- **Acceptance Criteria Addressed**: AC-2
- **Test Requirements**:
  - `programmatic` TR-8.1: 训练完成无OOM
  - `programmatic` TR-8.2: 评估metrics在预期范围

## [ ] Task 9: 关键配置训练验证
- **Priority**: high
- **Depends On**: Task 7
- **Description**: 
  - 根据推理sweep的Pareto结果，选择2-3个最有前景的配置训练
  - 取消run_all.sh中STEP 2的注释，设置TRAIN_EXPS
  - 每个训练3 epochs后评估
  - 验证训练后的模型在推理时使用FC-SB是否比G0更好
- **Acceptance Criteria Addressed**: AC-3, AC-4
- **Test Requirements**:
  - `programmatic` TR-9.1: 选定配置训练完成
  - `programmatic` TR-9.2: 训练+FC推理的结果优于纯基线
  - `human-judgement` TR-9.3: 视觉质量提升确认

## [ ] Task 10: 结果分析与最优配置确定
- **Priority**: high
- **Depends On**: Task 7, Task 9
- **Description**: 
  - 综合推理sweep和训练结果
  - 确定最优推理参数组合（σ, kernel, fc_mode）
  - 确认是否突破铜牌(>0.72/<0.40)或金牌(>0.73/<0.35)
  - 如果未突破，提出下一轮建议（wavelet lowpass, CFG外推, σ=0.12等）
- **Acceptance Criteria Addressed**: AC-4, AC-6
- **Test Requirements**:
  - `programmatic` TR-10.1: 生成最终对比表格
  - `human-judgement` TR-10.2: Pareto前沿分析结论明确
  - `human-judgement` TR-10.3: 每个FC机制的贡献有明确结论
