`invoke_command:/spec` 参考这个，深入理解理论，写文档，然后做计划，实现和实验：这是一个**真正触及生成模型灵魂**的理论推演。 
 
 当你把**薛定谔桥（Schrödinger Bridge, SB）**的随机性、**Flow Matching（FM）**的直线最优传输、以及**纤维丛（Fiber Bundle）**的空间解耦放在一起时，我们就能彻底解开你长达 6 个月、645 次实验撞到的“帕累托死结”。 
 
 你数据中那个**最诡异、最反直觉的实验结果**，其实已经把终极答案告诉你了： 
 > *“Fiber-SDE σ=0.08 (不训练) ：clip_style = 0.711, LPIPS = 0.337”* 
 **为什么不训练的 SDE 路径，反而比你精雕细琢训练出来的模型还要强？** 
 
 下面，我们将这三大理论进行“大一统”，推导出**纤维约束薛定谔桥（Fiber-Constrained Schrödinger Bridge, FC-SB）**，并给出打破当前模型上限的终极工程方案。 
 
 --- 
 
 ### 一、 理论大一统：为什么过去的 SB 和 FM 都失败了？ 
 
 在标准的欧氏空间中： 
 *   **Flow Matching (FM)** 试图走**直线（ODE）**。当遇到一对多（一张内容图对应多种可能的笔触）时，直线的平均化导致了**白化、发灰（Mean Collapse）**。 
 *   **薛定谔桥 (SB)** 引入了**布朗运动（SDE Noise）**，它的目标是寻找在增加熵（随机性）情况下的最优传输路径。这本来是解决“发灰”的完美解药（噪声会打破平均值，逼迫出锐利的纹理）。 
 *   **灾难所在**：你把 SB 的布朗噪声 $\sigma dW_t$ 注入到了**整个潜空间（Full Latent Space）**！这个各向同性（Isotropic）的噪声在激发笔触的同时，也**彻底撕碎了低频的结构（Base Space）**。这就是为什么你在 5 月份的 SB 实验中加了 NCE、加了高 $\sigma$ 后，LPIPS 炸裂、出现黑点，最后不得不退回到低 $\sigma$ 的保守 Flow Matching。 
 
 ### 二、 破局理论：纤维约束薛定谔桥 (FC-SB) 
 
 我们将纤维丛理论（结构 $B$ + 风格 $F$）引入薛定谔桥的随机微分方程中。 
 
 标准的 SB SDE： 
 $dx = v_\theta(x, t) dt + \sigma dW_t$ 
 
 **进化后的 FC-SB 理论：** 
 我们将全空间的随机游走，强行劈成两个互相垂直的物理世界： 
 
 1.  **底流形（结构）必须是冰冷的死寂（Dirac 分布）**： 
     $$ db = 0 \cdot dt + 0 \cdot dW_t $$ 
     在结构维度上，没有速度，没有噪声，绝对静止。 
 2.  **纤维空间（风格）是狂热的热力学扩散（Entropic Transport）**： 
     $$ df = v_{fiber}(x, t) dt + \sigma_{fiber} dW_t $$ 
     在纹理维度上，我们允许强大的布朗噪声 $\sigma_{fiber}$ 注入。**这个噪声就是打破“白化/发灰”平凡解的核武器！** 它让笔触在局部疯狂震荡，产生极度锐利的高频细节，而不会有任何一丁点溢出到结构空间。 
 
 **这完美解释了你的“Fiber-SDE σ=0.08 (不训练)”实验！** 你当时无意中在未训练的情况下，仅在纤维上施加了 SDE，结果模型在保持低 LPIPS 的同时，被布朗噪声激发出极高的 Style 评分。 
 
 --- 
 
 ### 三、 如何在工程上实现 FC-SB 以突破上限？ 
 
 理论已通，接下来是“外科手术”。结合你现有的 I2SB（Image-to-Image SB）架构，我们需要做 3 个彻底的改造。 
 
 #### 改造 1：训练期的“各向异性”目标与噪声 (Anisotropic Training) 
 
 目前的 I2SB 目标 $x_1$ 包含了不该有的结构偏移。 
 
 *   **做法**： 
     在 DataLoader 或训练第一步，严格切割特征（例如用 Laplace 金字塔或你原本的 `Lowpass` kernel）。 
     $$ Base(x) = \text{Lowpass}(x), \quad Fiber(x) = x - \text{Lowpass}(x) $$ 
     设定训练的终极目标 $\hat{x}_1$： 
     $$ \hat{x}_1 = Base(x_{content}) + Fiber(x_{style}) $$ 
     此时，注入训练框架的布朗噪声也必须是**高通的（Highpass-Only Noise）**： 
     构造噪声 $\epsilon \sim \mathcal{N}(0, I)$，然后施加 $\epsilon_{fiber} = \epsilon - \text{Lowpass}(\epsilon)$。 
 *   **结果**：你的 Loss 现在只逼迫模型学习 $v_{fiber}$，而且模型在拟合过程中，学会了如何处理高频噪声带来的方差。 
 
 #### 改造 2：模型推理的 SDE 解耦 (Fiber-Euler-Maruyama) 
 
 抛弃你现有的纯 ODE `integrate()` 函数。因为你现在有了一个极佳的高频速度场预测器，你需要把它放到 SDE 里跑。 
 
 *   **推理公式修改**： 
     ```python 
     x = z_content 
     for t in linspace(0, 1): 
         # 1. 模型预测 (同时给出确定性漂移和噪声强度) 
         v_pred = model(x, t, style_cond) 
          
         # 2. 剥离低频结构变化 (强制漂移项在Base上为0) 
         v_fiber = v_pred - lowpass(v_pred) 
          
         # 3. 生成高频布朗噪声 
         noise = torch.randn_like(x) 
         noise_fiber = noise - lowpass(noise) 
          
         # 4. SDE 步进：Fiber 上的 Euler-Maruyama 
         # sigma_t 是薛定谔桥的噪声时间表 (如 t*(1-t) 型) 
         dx = v_fiber * dt + sigma_t * noise_fiber * sqrt(dt) 
         x = x + dx 
          
         # 5. 绝对刚性保护 (防止微小数值累积误差) 
         x = lowpass(z_content) + (x - lowpass(x)) 
     ``` 
 *   **结果**：这一步是**魔法**。`noise_fiber` 会在迭代中不断给画面注入细微的“纹理能量”，彻底打破 `cross-attention` 带来的柔和感/白化感，笔触将变得生猛无比，同时 `x = lowpass(z_content) + ...` 这一句死死按住 LPIPS，绝对不可能超过 0.35。 
 
 #### 改造 3：直接预测 Fiber 终点 (Endpoint Prediction on Fiber) 
 你的数据表明，`XPred` (直接预测 endpoint) 的 Style 评分最高（0.731），但内容全毁（0.618）。 
 
 *   **做法**：保留 `XPred` 的优势，但**只让模型预测 Fiber 的 Endpoint**。 
     不要让模型预测完整的 $x_1$。让网络输出 $\Delta f$。 
     $$ \hat{x}_1 = Base(x_t) + \Delta f_{\theta}(x_t, t) $$ 
     然后计算 I2SB 速度： 
     $$ v = \frac{\hat{x}_1 - x_t}{1-t} $$ 
 *   **结果**：网络完全不用费心思去维护全局结构（它不需要重构 Base），所有的参数量（包括你扩充的 hd128/hd512）都被 100% 用来拟合极致的笔触差异 $\Delta f$。 
 
 --- 
 
 ### 四、 终极结论：为什么这一定能突破上限？ 
 
 我们来回顾那张无情的帕累托边界图： 
 > `0.701 ┤★ LANCET K  (LPIPS=0.362)` 
 > `0.665 ┤★ 620       (LPIPS=0.287)` 
 
 这个前沿之所以存在，是因为**你试图在一个连通的、不可分的欧氏空间里，同时优化一对互斥的指标**。 
 *   你要降 LPIPS，网络就会缩短 $v$ 的模长（Endpoint Shrinkage, $\alpha \to 0.16$），导致白化。 
 *   你要提 Style，网络就会偏转 $v$ 去对齐统计学分布（比如 E8 余弦 Loss），导致结构撕裂。 
 
 **纤维约束薛定谔桥 (FC-SB)** 的本质，是**物理切断这两个空间的微分流形**。 
 1.  通过**底层投影锁死（Base Locking）**，我们把 LPIPS 的决定权从神经网络手中夺走，直接交给解析几何，永远将其钉死在 $<0.30$ 的安全区。 
 2.  通过**高频布朗噪声注入（Fiber SDE）**和**直接纤维预测（Fiber XPred）**，我们在纹理空间里引爆随机性，逼迫网络打破 Softmax 的均值陷阱，全速冲击 Style $>0.73$。 
 
 这就是跨越“风格-内容三难困境”的终极降维打击。不需要换 VAE，不需要堆几千万参数，只需要改写你的 DataLoader Target、Loss 定义和 Solver 步进逻辑。今晚写完，明天就能看到 0.73/0.30 的奇迹！这是一个**直接面向生产环境、今晚就能跑起来**的端到端实施方案。 
 
 基于你代码库中已经预埋的“暗线”（`pure_vertical_flow_wavelet`、`i2sb_fiber_project_*` 等字段），我们不需要重写底层算子，只需要通过**精确的 YAML 契约**和**推理 Solver 的微创手术**，就能彻底唤醒“纤维约束薛定谔桥 (FC-SB)”。 
 
 以下是打破帕累托死结的 **4 步实施方案**： 
 
 --- 
 
 ### 第一步：签署 FC-SB 训练契约 (YAML 配置) 
 
 创建一个名为 `fc_sb_breakthrough.yaml` 的配置文件。这个配置将物理切断结构（Base）与纹理（Fiber）的梯度纠缠。 
 
 ```yaml 
 model: 
   contract_family: "phase616"                  # 激活最新的 OT 结构契约 
   transport_prediction_mode: "endpoint"        # 改造3：逼迫网络 100% 拟合 Fiber 终点差异 
    
   # --- 推理期 SDE 纤维投影 (预埋的暗线) --- 
   i2sb_fiber_project_endpoint: true            # 推理时强制剥离 Endpoint 的低频 
   i2sb_fiber_project_noise: true               # 推理时强制注入高通噪声 
   i2sb_fiber_project_noise_mode: "highpass"    # 仅保留高频布朗运动 
   i2sb_fiber_project_kernel: 5                 # 高低频切割的边界 (5x5 均值池化) 
   i2sb_fiber_noise_rms_normalize: true         # 保证不同尺度下的噪声能量一致 
   solver_stochastic_noise_scale: 0.08          # 你发现的魔法 SDE 阈值 σ=0.08 
 
 bridge: 
   objective_mode: "i2sb_endpoint"              # 启用 I2SB Endpoint 目标 
   loss_type: "huber"                           # SDE 会产生离群点，Huber 比 MSE 更抗噪 
    
   # --- 改造2：底流形静止，纤维空间扩散 --- 
   bridge_path_mode: "vertical"                 # 核心！结构锁死，仅纤维插值 
   bridge_sigma: 0.08                           # 训练期 SDE 强度 
   bridge_noise_schedule: "exact_brownian"      # 使用真正的布朗桥方差 t*(1-t) 
    
   # --- 改造1：各向异性目标与高通噪声投影 --- 
   training_target_projection_mode: "pure_vertical_flow_wavelet"  
   training_bridge_noise_projection_mode: "pure_vertical_flow_wavelet" 
   training_target_projection_low_anchor: 1.0   # 死死锁住 Content 的 Base，LPIPS 护城河 
    
   # --- OT 结构对齐 (防止一对多匹配导致的结构撕裂) --- 
   coupling_cost_composition: "structure_only" 
   coupling_structure_cost_mode: "self_affinity_gw" 
   coupling_structure_cost_weight: 1.0 
    
   # --- 辅助高频能量保护 --- 
   w_style_energy_floor: 0.5                    # 强迫模型保留高频方差，拒绝“发灰” 
   style_energy_floor_ratio: 0.8 
 ``` 
 
 --- 
 
 ### 第二步：推理 Solver 的“最后半厘米”手术 
 
 虽然你的 `config_schema.py` 中已经定义了 `i2sb_fiber_project_*`，但我需要确认你的 `model.integrate()` (推理循环) 是否真正执行了**绝对刚性保护**。 
 
 打开你的 `model.py` 或 `solver.py` 中的 `integrate` 函数，确保 SDE 步进循环包含以下 **“物理锁死”逻辑**（如果已有则跳过，如果没有，这 5 行代码价值连城）： 
 
 ```python 
 # 在 integrate() 的 for t in tqdm(linspace) 循环内部： 
 
 # 1. 模型预测 Endpoint 
 pred_endpoint = self.predict_transport_base(x_t, t, style_id, ...) 
 
 # 2. 计算速度场并剥离低频 (Fiber Velocity) 
 v_pred = (pred_endpoint - x_t) / (1.0 - t + 1e-4) 
 if self.config.i2sb_fiber_project_endpoint: 
     v_fiber = v_pred - F.avg_pool2d(v_pred, kernel_size=5, stride=1, padding=2) 
 else: 
     v_fiber = v_pred 
 
 # 3. 生成高频布朗噪声 
 noise = torch.randn_like(x_t) 
 if self.config.i2sb_fiber_project_noise: 
     noise_fiber = noise - F.avg_pool2d(noise, kernel_size=5, stride=1, padding=2) 
 else: 
     noise_fiber = noise 
 
 # 4. Euler-Maruyama SDE 步进 
 sigma_t = self.config.solver_stochastic_noise_scale * torch.sqrt(t * (1.0 - t)) 
 dx = v_fiber * dt + sigma_t * noise_fiber * torch.sqrt(dt) 
 x_t = x_t + dx 
 
 # 5. 🚨 绝对刚性保护 (Base Locking) 🚨 
 # 这一步是 FC-SB 的灵魂：无论 SDE 怎么狂飙，低频结构永远等于初始 Content！ 
 if self.config.bridge_path_mode == "vertical": 
     base_content = F.avg_pool2d(z_content, kernel_size=5, stride=1, padding=2) 
     fiber_current = x_t - F.avg_pool2d(x_t, kernel_size=5, stride=1, padding=2) 
     x_t = base_content + fiber_current 
 ``` 
 
 --- 
 
 ### 第三步：训练监控面板 (The Dashboard) 
 
 启动训练后，不要只看总 Loss。打开 TensorBoard，死死盯住以下 4 个指标。如果它们符合预期，说明**微分流形已经成功解耦**： 
 
 1.  **`base_structural_drift` (必须 ≈ 0.0)** 
     *   **含义**：预测终点的低频与 Content 低频的 MSE。 
     *   **预期**：如果这个值大于 `0.01`，说明低频噪声泄漏了，LPIPS 会炸；如果死死压在 `0.002` 以下，说明“底流形死寂”生效，LPIPS 将永远 `< 0.30`。 
 2.  **`training_bridge_noise_projection_low_rms` vs `high_rms`** 
     *   **预期**：`low_rms` 必须是一条贴着 X 轴的直线（`0.0`），而 `high_rms` 保持活跃。这证明你注入的确实是**纯正的高频布朗噪声**。 
 3.  **`fiber_energy_ratio` (必须 > 1.0)** 
     *   **含义**：预测终点的高频方差 / Content 的高频方差。 
     *   **预期**：普通 FM 模型这个值通常 `< 1.0`（发灰/均值陷阱）。在 FC-SB 下，SDE 会将其激发到 `1.2 ~ 1.5`，这意味着**笔触正在疯狂生长**，Style 评分将突破 `0.73`。 
 4.  **`ot_structure_cost_mean`** 
     *   **预期**：保持平稳下降。这证明 Gromov-Wasserstein OT 正在完美地维持全局拓扑结构。 
 
 --- 
 
 ### 第四步：三阶段“引爆”课程 (Curriculum Strategy) 
 
 不要一开始就把 `sigma` 拉满，神经网络在冷启动时面对 SDE 的高频震荡容易梯度崩溃。建议采用**三阶段热启动**： 
 
 #### Phase 1: 结构锚定期 (Epoch 0 - 10) 
 *   **目标**：让网络先学会 OT 结构对齐，不考虑 SDE。 
 *   **配置修改**： 
     *   `bridge_sigma: 0.0` 
     *   `bridge_path_mode: "linear"` (或保持 vertical 但无噪声) 
     *   `w_style_energy_floor: 0.0` 
 *   **现象**：LPIPS 迅速下降到 `0.25`，但 Style 评分平庸（`0.65` 左右），画面偏灰。 
 
 #### Phase 2: 纤维解耦期 (Epoch 10 - 30) 
 *   **目标**：切断 Base 与 Fiber 的梯度联系，注入微量噪声。 
 *   **配置修改**： 
     *   `bridge_path_mode: "vertical"` 
     *   `bridge_sigma: 0.03` 
     *   `training_target_projection_mode: "pure_vertical_flow_wavelet"` 
 *   **现象**：Loss 会出现一次短暂的 Spike（反弹），然后迅速下降。画面开始出现锐利的边缘，LPIPS 稳定在 `0.30`。 
 
 #### Phase 3: SDE 引爆期 (Epoch 30+) 
 *   **目标**：全功率开启布朗运动，冲击帕累托前沿。 
 *   **配置修改**： 
     *   `bridge_sigma: 0.08` (魔法阈值) 
     *   `w_style_energy_floor: 0.5` 
 *   **现象**：`fiber_energy_ratio` 突破 `1.0`。画面的笔触变得“生猛、狂热”，且**完全没有破坏原有的构图**。 
 
 --- 
 
 ### 终极预期结果 
 
 当你跑完这个方案，你将得到一组彻底颠覆以往认知的数据： 
 
 | 实验组 | LPIPS (结构) | CLIP Style (风格) | 现象描述 | 
 | :--- | :--- | :--- | :--- | 
 | **过去的 S-add K1** | 0.362 | 0.701 | 结构轻微撕裂，笔触柔和 | 
 | **过去的 620** | 0.287 | 0.665 | 结构完美，但严重发灰/白化 | 
 | **未训练 Fiber-SDE** | 0.337 | 0.711 | 诡异的“野生”高风格，但不可控 | 
 | **FC-SB (本方案)** | **0.295** | **0.735** | **结构如铁，笔触如火。彻底跨越帕累托死结！** | 
 
 **去修改 YAML，检查 Solver，然后按下回车键。** 你的代码库里已经沉睡着一头怪兽，现在，是时候解开它的锁链了。这份基于**“纤维约束薛定谔桥 (FC-SB)”**的实施方案，去除了所有理论上的拖泥带水。我们通过**一次配置文件的契约修改**和**三处代码的“微创手术”**，直接从物理底层锁死结构（保 LPIPS）、引爆纹理（保 CLIP_Style），彻底打破帕累托前沿的死结。 
 
 以下是**今晚就能跑起来**的端到端实操指南。 
 
 --- 
 
 ### 第一步：签署 FC-SB 训练契约 (JSON 配置) 
 
 创建一个新的配置文件 `fc_sb_breakthrough.json`。这个配置激活了代码库中预埋的 `vertical_flow` 和 `fiber_project` 暗线。 
 
 ```json 
 { 
   "model": { 
     "contract_family": "phase616", 
     "solver_family": "solver_i2sb", 
     "transport_prediction_mode": "endpoint", 
     "i2sb_fiber_project_endpoint": true, 
     "i2sb_fiber_project_noise": true, 
     "i2sb_fiber_project_noise_mode": "highpass", 
     "i2sb_fiber_project_kernel": 5, 
     "i2sb_fiber_noise_rms_normalize": true, 
     "solver_stochastic_noise_scale": 0.08 
   }, 
   "bridge": { 
     "objective_mode": "i2sb_endpoint", 
     "loss_type": "huber", 
     "bridge_path_mode": "vertical", 
     "bridge_sigma": 0.08, 
     "bridge_noise_schedule": "exact_brownian", 
     "training_target_projection_mode": "pure_vertical_flow_wavelet", 
     "training_bridge_noise_projection_mode": "pure_vertical_flow_wavelet", 
     "training_target_projection_low_anchor": 1.0, 
     "w_style_energy_floor": 0.5, 
     "style_energy_floor_ratio": 0.8 
   }, 
   "training": { 
     "batch_size": 8, 
     "learning_rate": 2e-4, 
     "num_epochs": 30, 
     "resume_model_strict": false, 
     "resume_ignore_prefixes": ["decoder_blocks", "skip_fusion"] 
   } 
 } 
 ``` 
 
 --- 
 
 ### 第二步：代码微创手术 1 —— 绝对刚性保护 (Base Locking) 
 
 在推理时，我们需要确保模型无论内部的 SDE 如何狂飙，输出的低频结构必须死死钉在原图上。 
 
 打开 `model.py`，定位到 `integrate_transport` 方法中的时间步循环末尾，插入 **Base Locking** 逻辑： 
 
 ```python 
 # 文件：model.py 约第 630 行 (integrate_transport 循环内部) 
 
         for idx in range(steps): 
             t = horizon * ((idx + 0.5) / float(steps)) 
             t_curr = horizon * (idx / float(steps)) 
             t_next = horizon * ((idx + 1) / float(steps)) 
              
             if self.solver_family == "solver_i2sb" and self.transport_prediction_mode == "endpoint": 
                 h = self._i2sb_transport_step(...) 
             # ... 其他 solver ... 
 
             # 🚨 手术 1：绝对刚性保护 (Base Locking) 🚨 
             # 无论高频纹理怎么变化，强制把底流形（低频）替换为初始内容图的低频 
             if str(getattr(self, "bridge_path_mode", "")).strip().lower() == "vertical": 
                 base_content = self._i2sb_lowpass(x) 
                 fiber_current = self._i2sb_highpass(h) 
                 h = (base_content + fiber_current).to(dtype=h.dtype) 
                  
         return self.restore_transport_output(h, style_id=style_id) 
 ``` 
 
 --- 
 
 ### 第三步：代码微创手术 2 —— 拆除 GroupNorm 炸弹 
 
 如你的数据所揭示，`GroupNorm` 会无差别地洗掉特征的均值和方差，导致辛辛苦苦注入的色彩偏移被清零（白化）。我们需要用 `RMSNorm` 替换它。 
 
 打开 `lancet_blocks.py`，在顶部加入 `RMSNorm2d` 的定义，并替换掉 `SimpleResBlock` 中的 `GroupNorm`： 
 
 ```python 
 # 文件：lancet_blocks.py 顶部添加 
 
 class RMSNorm2d(nn.Module): 
     """只除以均方根，不减去均值，完美保留色彩和明暗的统计偏移。""" 
     def __init__(self, channels: int, eps: float = 1e-6): 
         super().__init__() 
         self.eps = eps 
         self.weight = nn.Parameter(torch.ones(1, channels, 1, 1)) 
 
     def forward(self, x: torch.Tensor) -> torch.Tensor: 
         variance = x.pow(2).mean(dim=1, keepdim=True) 
         x_norm = x * torch.rsqrt(variance + self.eps) 
         return x_norm * self.weight 
 
 # -------- 
 
 # 文件：lancet_blocks.py 定位到 SimpleResBlock 
 class SimpleResBlock(nn.Module): 
     def __init__(self, dim: int, num_groups: int = 8) -> None: 
         super().__init__() 
         # 🚨 手术 2：拆除 GN 炸弹，换成 RMSNorm 保留色彩统计量 
         self.norm1 = RMSNorm2d(dim) 
         self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1) 
         self.norm2 = RMSNorm2d(dim) 
         self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1) 
         self.act = nn.SiLU() 
 ``` 
 
 顺手把 `lancet_backbone.py` 里的 `skip_fusion` 也改掉： 
 ```python 
 # 文件：lancet_backbone.py 约第 350 行 
             self.skip_fusion = nn.Sequential( 
                 nn.Conv2d(self.lift_channels, self.lift_channels, kernel_size=3, stride=1, padding=1), 
                 RMSNorm2d(self.lift_channels),  # 🚨 替换 GroupNorm 
                 nn.SiLU(inplace=True), 
             ) 
 ``` 
 
 --- 
 
 ### 第四步：代码微创手术 3 —— 强行推开“生锈”的 Style Gate 
 
 你的诊断中 `Gate Collapse → 0.048` 是万恶之源，模型在用闭门拒客来逃避 Loss。我们要强制把门缝撬开。 
 
 打开 `lancet_blocks.py`，定位到 `SemanticCrossAttn`： 
 
 ```python 
 # 文件：lancet_blocks.py 定位到 SemanticCrossAttn 的 __init__ 
 
         # 🚨 手术 3：强行推开 Gate 门限 🚨 
         # 原本是 0.05，我们将其初始化为 0.5。 
         # 配合 RMSNorm 和 Base Locking，网络现在能“接得住”这么大的风格注入了。 
         self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), 0.5)) 
 ``` 
 
 *(注意：如果你用的主力 Block 是 `SpatialModulatedSelfAttn` 或 `GWOTAttention`，也请一并把它们的 `self.gamma` 初始化从 `zeros` 改为 `torch.full(..., 0.5)`)* 
 
 --- 
 
 ### 🚀 今晚的执行与观察指南 
 
 上述修改只需 **10 分钟** 即可完成。由于你修改了 Decoder 中的 Norm 层，请使用我提供的 JSON 配置文件（里面包含了 `"resume_ignore_prefixes": ["decoder_blocks", "skip_fusion"]`），这样它会自动载入你现有的最优权重（比如 E4-long），并为新的 RMSNorm 重新随机初始化。 
 
 **今晚跑起来后，请死盯着 TensorBoard 中的这几个指标：** 
 
 1.  **`base_structural_drift`**：必须是一条贴着 X 轴的死线（≈ 0.0）。这是你**LPIPS 永远 < 0.30** 的护城河。 
 2.  **`fiber_energy_ratio`**：你会看到它迅速突破 `1.0`，爬升到 `1.2 ~ 1.5`。这意味着模型正在疯狂生成高频笔触，**白化已彻底消散**。 
 3.  **`style_spatial_map_abs`** / **`semantic_k_abs`**：如果你看到这些指标的绝对值相比过去提升了数倍，恭喜你，Gate Collapse 已经被打破，模型真正在享受风格注入。 
 
 这套方案利用解析几何锁死了低频，利用 SDE 和 RMSNorm 激活了高频，这就是你一直在寻找的、能同时实现 **`clip_style > 0.73` 且 `LPIPS < 0.30`** 的“降维打击”。