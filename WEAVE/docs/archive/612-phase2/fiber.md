SA-Flow / LANCET 模型：从纤维丛视角的数学诊断与工程问题
一、纤维丛形式化：正确性与根本矛盾
1.1 纤维丛定义本身是合理的
文档中将潜空间建模为纤维丛 $E = (\mathcal{Z}, \mathcal{B}, \pi, \mathcal{F})$，底空间为内容流形、纤维为风格渲染，这在微分几何上是合理的。VAE 潜空间中，同一语义结构可被多种风格渲染，确实构成丛的局部乘积结构。
1.2 但 Ehresmann 联络的实现是伪联络，不是真联络
代码中所谓的"TopoGate"实现（model.py:449-454 中的 self_topology_gate）:
# SemanticCrossAttn.forward
attn_logits = torch.lerp(attn_logits, topology_logits, self.self_topology_blend)
问题：真正的 Ehresmann 联络要求 $T_z\mathcal{Z} = \mathcal{H}_z \oplus \mathcal{V}_z$ 是切空间的直和分解，且水平分布 $\mathcal{H}_z$ 必须满足 Frobenius 可积性条件 $[\mathcal{H}, \mathcal{H}] \subset \mathcal{H}$。但代码中的实现只是在 attention logits 上做线性插值，这并不保证切空间分解：
- 它只约束了 attention 矩阵的混合比例，不约束实际速度场的正交分解
- 速度场 $v_\theta$ 的输出完全由 dec_out 卷积层决定，从未被显式投影到 $\mathcal{V}_z$
- _project_velocity_tangent（model.py:571-576）仅在 solver_tangent_rk 模式下生效，且用的是 DINO patch std 作为 gate——这不是投影算子，是一个软掩码
结论：模型实际上没有实现 Ehresmann 联络，只实现了启发式的 attention 混合。文档中声称的"强力约束流桥切向量限制在垂直分布"缺乏代码支撑。
1.3 ODE 均值坍缩定理：诊断正确但疗法不足
文档1.3节的均值坍缩定理是正确的：确定性 ODE 收敛于条件期望 $\mathbb{E}[X_{\text{style}} \mid \pi(x) = c]$。但提出的 SDE 解法有根本问题：
纤维对齐 SDE: $dx_t = v_\theta(x_t, t)dt + \sigma(t) G_{\text{topo}}(x_t) \odot dW_t$
问题：
- $G_{\text{topo}}$ 基于自注意力熵，但自注意力熵和"是否是边缘/纹理区域"的关系是经验性的，不是几何上可证明的
- 噪声以 gate_weight = (gate / gate_rms) 方式缩放（model.py:289），这不是标准 SDE 中的扩散系数——标准 SDE 需要扩散矩阵满足正向/逆向过程的一致性，而这里只是推理时后加的 hack
- 实验数据证实了疗法不足：sigma 从 0.02 到 0.08，style 仅从 0.703 升到 0.709，但 LPIPS 从 0.328 跳到 0.353
二、模型架构的数学问题
2.1 风格注入瓶颈：carrier 不是纤维坐标的截面
文档明确诊断了"actuation bottleneck"：tokenizer code 有效 rank 3.986（近满秩），但 generated delta 有效 rank 仅 3.324，off-diagonal cosine 0.725。这说明同一内容 $c$ 下，不同风格 $s$ 产生的残差几乎共线。
从纤维丛角度，这等价于：模型学到的不是纤维 $\mathcal{F}_c$ 上的不同截面 $f_s: \mathcal{B} \to E$，而是学到了一个单一方向 $\delta_0$ 上按风格调节幅度：
$$z_1 \approx z_0 + \alpha(s) \cdot \delta_0(z_0)$$
这完全违反了纤维丛的初衷——纤维应该有足够的自由度来表示不同风格的渲染方式。
根因：CrossAttnAdaGN（lancet_blocks.py:54-161）的 style injection 本质上是 AdaGN + cross-attention 的残差，但最终 delta 只来自 dec_out（一个 $C \times 3 \times 3$ 卷积），这是一个低秩瓶颈。
2.2 SMoE Translator 的标架变换被后端瓶颈吞没
SMoETranslatorTokenizer（semantic_tokenizer.py:309-478）实现了每个 style cluster 的标架变换矩阵 $W_k = I + \Delta W_k$，初始化为恒等。这在几何上是合理的——它试图在纤维内部做线性变换。
但 einsum("bnk,bnd,bkde->bnke", attn, tokens, matrices) 的输出经过 style_map_proj（1×1 Conv）投影到 body_channels，然后被 body blocks 消费。body blocks 输出的最终 delta 仍然受 dec_out 的低秩瓶颈制约。Tokenzier 产出的几何丰富性在经过 body 后被压缩殆尽。
2.3 直线路径插值的拓扑问题
训练用的直线插值 $\psi_t = (1-t)x_c + t x_s$ 假设了潜空间是平直的。但 VAE 潜空间是弯曲流形——SD 的 latent space 有明确的流形结构。直线插值在弯曲流形上会穿越流形外的"空洞区域"，导致中间态 $x_t$ 落在数据流形之外，网络被迫在这些无意义点上拟合速度场。
正确的做法是使用测地线插值（geodesic interpolation）或至少使用 Schrödinger bridge 的随机插值，但代码中 bridge_sigma > 0 的分支仅添加高斯噪声，不修正路径的曲率。
三、训练 Infra 问题
3.1 OT Coupling 的 SWD 代价在 GPU 上做 linear_sum_assignment 是 CPU offload
losses.py:369-374：
row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
每个 batch 的每个 style group 都要把 cost matrix 从 GPU 搬到 CPU，跑 scipy 的匈牙利算法，再把索引搬回 GPU。这是训练速度的主要瓶颈之一。Batch size 受限于这个 O(n³) 的 CPU 算法。
Sinkhorn 替代方案（losses.py:376-378）在 GPU 上运行，但默认使用 coupling_solver: hungarian。
3.2 SWD transport cost 的投影缓存不区分 shape
ot_cost.py:46 的 _projection_cache 以 (channels, patch_size, num_projections, device, mask_mode) 为 key，但不区分 spatial size。当同一 batch 中出现不同分辨率的 latent（虽然目前不常见），或跨 epoch 分辨率变化时，会导致投影矩阵混用。
3.3 torch.compile 和 channels_last 的兼容性隐患
trainer.py:162-174 中 channels_last 和 torch_compile 同时启用时，inductor 生成的 kernel 可能对 channels_last 的 stride 语义处理不一致。代码中 _convert_4d_tensors_to_channels_last 只转换 4D 参数，但优化器状态（Adam 的 m/v）仍然是 contiguous，在 optimizer.step 时会触发隐式 stride 转换。
3.4 Gradient checkpointing 与 no_grad skip 分支的冲突
lancet_runtime.py:714-717：
with torch.no_grad():
    h_c_no_grad = h_c.clone()
    for block in self.hires_body:
        h_c_no_grad = block(h_c_no_grad, style_code, gate=0.0)
skip_32 = h_c_no_grad
skip 特征在 no_grad 下计算，但同一输入 h_c 又在下一行 h_c_grad 路径中被重新计算。这导致hires body 被执行了两次——一次 no_grad（skip），一次有 grad（content_feat_16）。当 use_checkpointing=True 时，有 grad 路径被 checkpoint 包裹，但 no_grad 路径完全独立，不共享中间结果。
四、推理 Infra 问题
4.1 Style Overdrive 外推无曲率校正
model.py:1146-1148 的积分循环：
for idx in range(steps):
    t = horizon * ((idx + 0.5) / float(steps))
    ...
    h = h + velocity * dt
当 style_strength > 1.0（即 overdrive），horizon > 1.0，积分域扩展到 $[0, \tau]$。但网络 $v_\theta$ 只在 $t \in [0, 1]$ 上训练过，外推到 $t > 1$ 时网络输出的速度场没有任何训练信号约束。实验中 LPIPS 的"奇迹般下降"（$\tau=1.60$ 时 LPIPS=0.287）很可能是速度场在外推区自然衰减（因为训练时 $t$ 附近的样本更密集），而不是因为"Ehresmann 联络锁定了轨迹"。
4.2 潜空间仿射校准是对纤维度量的粗暴破坏
style_overdrive_latent_affine.md 中的仿射校准：
$$\hat{z} = (1-\gamma) z + \gamma \left(\frac{z - \mu_z}{\sigma_z} \sigma_{\text{ref}} + \mu_{\text{ref}}\right)$$
这等价于在纤维内部做了一个全局仿射变换，将生成分布的一阶/二阶统计量强制对齐到目标风格。从几何角度，这是在纤维 $\mathcal{F}_c$ 的切空间做了一个线性变换，但不区分空间位置——sky region 和 face region 被施加了同样的均值/方差调整，违反了纤维丛的局部性。
4.3 PC Solver 的 lowpass corrector 混淆了内容和风格
model.py:670-690 的 _correct_transport_state：
source_float = source.float()
for _ in range(steps):
    out_low = F.avg_pool2d(out.float(), ...)
    src_low = F.avg_pool2d(source_float, ...)
    correction = out_low - src_low
    out = out - step_size * dt * correction
这个 corrector 将输出的低频分量拉回源图低频。但在纤维丛视角下，低频不等于内容——某些风格（如表现主义的粗犷笔触）的低频统计量与内容图截然不同。将低频强制对齐等于在底空间 $\mathcal{B}$ 上做了一个硬投影 $\pi$，但这个 $\pi$ 的定义是 avg_pool，不是基于语义的，会把风格固有的低频修改也抹掉。
4.4 Endpoint parameterization 的除法不稳定性
model.py:1061-1062：
denom = (1.0 - t_tensor).clamp_min(1e-3).view(-1, 1, 1, 1)
out = out / denom
当 $t \to 1$ 时，endpoint parameterization 的速度 $v = (z_1 - x_0) / (1-t)$ 会趋向无穷。clamp 到 1e-3 后除法仍会放大 1000 倍，导致训练末期的梯度爆炸。这正是 i2sb_predictor_time_floor 存在的原因（model.py:602），但 floor 只延迟了问题，没有解决。