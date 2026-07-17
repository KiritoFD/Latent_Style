# Quantitative Defense Experiments - Task Spec

## Goal
在现有消融基础上补充四组精确量化实验，构建绝对定量的防御壁垒，彻底摒弃主观评估。

## Four Proposed Experiments

### Task 1: Wavelet Basis Bounds (2-level Haar, Db2)
- 操作：替换为2-level Haar和Db2小波，重新训练
- 量化：ΔDINO-S, ΔLPIPS, Δt (基线168ms)
- 阈值：增益<0.005且Δt>10ms则证明1-level Haar最优
- 成本：高（需重训练）
- 合理性：中（D3已显示lambda_LL不sharp，小波选择可能也不critical）

### Task 2: ODE Truncation Error (RK4 pseudo-truth, Euler K sweep)
- 操作：RK4 K=64作为伪真值，Euler K∈{2,4,8,16,32}扫描
- 量化：轨迹漂移 ε=‖z_K - z_64*‖²
- 阈值：标定ε(K)拐点，证明K=8已越过相变点
- 成本：低-中（纯推理，6次运行）
- 合理性：高（直接验证8步Euler选择，标准方法论）

### Task 3: Rigid Topological Preservation (MiDaS depth, Canny Edge IoU)
- 操作：对750对输出提取MiDaS深度+Canny边缘
- 量化：MSE_depth, Edge IoU
- 阈值：WEAVE的MSE_depth严格低于TGT，优于StyleAligned/SaMam
- 成本：最低（后处理已有图片）
- 合理性：高（用刚性几何度量替代LPIPS黑盒）

### Task 4: Failure Boundary Mapping (lambda_LL x eta grid)
- 操作：λ_LL∈{0,0.1,0.3,1,3,10} × η∈{0,0.1,0.5,1,2} 网格扫描
- 量化：DINO-C, LPIPS衰减
- 阈值：标定DINO-C<0.215或梯度爆炸的(λ_LL,η)坐标
- 成本：中-高（30次推理）
- 合理性：中（D3/D4已覆盖部分空间，但热力图有视觉说服力）

## Priority Decision
优先级：Task 3 > Task 2 > Task 4 > Task 1

理由：
1. Task 3成本最低（后处理已有图片），价值最高（刚性度量替代黑盒）
2. Task 2成本低（6次推理），直接验证核心方法论选择
3. Task 4成本中高（30次推理），但热力图有视觉说服力
4. Task 1成本最高（需重训练），且D3已弱化wavelet claim

用户问题答案：优先投入Task 2（ODE截断误差扫参），因为：
- 计算成本更低（6次 vs 30次）
- 直接验证核心方法论选择（8步Euler）
- RK4伪真值是标准、可防御的方法论
- 可更快完成
- K-sweep产生清晰的相变曲线，对reviewer非常有说服力

## Success Criteria
- Task 3: 750对图片的MSE_depth和Edge IoU，WEAVE严格优于TGT
- Task 2: ε(K)曲线拐点在K≤8处，证明K=8已收敛
- Task 4: 失效边界热力图，当前超参位于非坍塌区域
- Task 1: 可选，视资源情况
