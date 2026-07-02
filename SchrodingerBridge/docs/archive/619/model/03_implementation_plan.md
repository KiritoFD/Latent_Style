# 03 — 可实现的模型方案与代码落地指南 (Implementation Plan)

> 本文档将前篇的高维架构设计转化为**具体到代码层面的落地执行方案**。
> 指导我们如何从当前充满冗余的 `src/` 代码库中，一步步剥离旧的纠缠逻辑，重建纯净的 Flow Matching 骨架。

---

## 阶段 1：数据流重建 (Data Pipeline)

目标：消除在线的 Minibatch OT，用预先配对好的强语义/弱风格目标喂给模型。

### 1.1 离线预配对脚本 (`tools/prematch_dino.py`)
我们需要新建一个脚本，用于预先算好配对列表。
* **流程**：
  1. 加载所有内容图和风格图。
  2. 使用 `torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')` 提取全图 CLS Token。
  3. 计算每张内容图与所有风格图的 Cosine Similarity 矩阵。
  4. 对每张内容图，在风格集中选取 Top-10 相似的候选项，随机挑选一个作为唯一目标。
  5. 保存为一个映射文件：`content_id -> style_id` 或直接保存包含图片路径的 JSONL。

### 1.2 DataLoader 的改造 (`src/dataset.py`)
在 `SchrodingerBridgeDataset` 中：
* 移除原来的独立随机采样。
* 在 `__getitem__` 中，根据配对映射表，返回固定的三元组：
  `return {"z_c": content_latent, "z_s": style_latent, "style_image": style_rgb}`
* **注意**：必须返回原始的高清 `style_image` (或已经在 CPU 提取好的 DINO 特征)，因为真实的交叉注意力需要空间纹理信息，而不是 1D 的 ID。

---

## 阶段 2：模型主干重建 (Backbone Reconstruction)

这是最重要的部分：在 `src/model.py` 和 `src/lancet_blocks.py` 中实现时空解耦。

### 2.1 引入 Style Encoder
在模型初始化阶段，加入一个冻结的特征提取器：
```python
# src/model.py -> TimeConditionedLANCETBridge.__init__
self.style_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
for p in self.style_encoder.parameters():
    p.requires_grad = False
```

### 2.2 彻底解耦 Time 和 Style
找到 `model.py` 中致命的 `_compute_style_code` 函数并废弃它。
`forward` 的签名应改为：
```python
def forward(self, x: torch.Tensor, t: torch.Tensor, style_image: torch.Tensor):
    # 1. 独立的时间编码 (标量进度)
    time_code = self.time_mlp(sinusoidal_time_embedding(t, self.time_dim))
    
    # 2. 独立的风格空间特征 (空间纹理)
    with torch.no_grad():
        style_feat = self.style_encoder.get_intermediate_layers(style_image, n=1)[0]
    # style_feat.shape = [B, N, D]
    
    # 3. 后续传递：将 time_code 传给所有 AdaLN，将 style_feat 传给所有 CrossAttn
    ...
```

### 2.3 重写 Attention 模块 (`src/lancet_blocks.py`)
废除 `CrossAttnAdaGN` 中依赖 `nn.Embedding` (查表) 的 `style_tokens_basis`。
新建真正的 `SpatialCrossAttention` 块：

```python
class SpatialCrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim):
        super().__init__()
        self.q_proj = nn.Linear(query_dim, inner_dim)
        self.k_proj = nn.Linear(context_dim, inner_dim) # context 即 style_feat
        self.v_proj = nn.Linear(context_dim, inner_dim)
        
    def forward(self, hidden_states, style_feat):
        # hidden_states: [B, HW, C] 内容特征
        # style_feat: [B, N, D] 风格特征
        q = self.q_proj(hidden_states)
        k = self.k_proj(style_feat)
        v = self.v_proj(style_feat)
        
        # 真正的软注意力：模型自发学习内容和纹理的空间对应关系
        attn_weights = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(d), dim=-1)
        out = attn_weights @ v
        return out
```

在 `UNet` 的每个分辨率阶段（尤其是 Decoder 阶段），应用此注意力模块，取代原有的全局注入。

---

## 阶段 3：极简的训练循环与损失计算

目标：在 `src/losses.py` 和 `src/train.py` 中删除冗余，只保留核心梯度。

### 3.1 废弃冗余代码
大刀阔斧地删除以下部分（建议先将其注释或从训练流中移除）：
* `losses.py` 中的 `_terminal_swd`：训练时绝不能执行 `model.integrate`，这会导致长梯度链爆炸和均值坍缩。
* `losses.py` 中的 `VarAlign`、`Kinetic Loss` 补偿。纯粹的直线流已经隐式最小化了动能（路径最短）。
* `ot_cost.py` 中关于 Minibatch Sinkhorn 的计算（既然我们用了离线配对，这步全省了）。

### 3.2 实现纯粹的 Flow Matching 核心循环
在 `train.py`（或对应的 loss wrapper）中：

```python
def compute_loss(self, batch):
    z_c = batch["z_c"]
    z_s = batch["z_s"]
    style_image = batch["style_image"]
    
    # 1. Uniform 采样时间
    B = z_c.shape[0]
    t = torch.rand((B, 1, 1, 1), device=z_c.device)
    
    # 2. 构建独立耦合的直线状态 (Independent Coupling)
    z_t = (1 - t) * z_c + t * z_s
    
    # 3. 真实速度
    v_true = z_s - z_c
    
    # 4. 模型预测
    v_pred = self.model(z_t, t, style_image=style_image)
    
    # 5. 核心损失：MSE (这已足够驱动完美的风格迁移)
    loss_fm = F.mse_loss(v_pred, v_true)
    
    loss = loss_fm
    
    # 6. (可选实验项) 单步结构保护：L1
    if self.use_content_preservation:
        # 预测终点：z_1 = z_t + (1-t) * v_pred
        z_1_pred = z_t + (1 - t) * v_pred
        loss_content = F.l1_loss(z_1_pred, z_c)
        loss += 0.1 * loss_content
        
    return loss
```

---

## 阶段 4：推理与外推评估 (Inference & Evaluation)

### 4.1 ODE 积分不变
推理代码 (`model.integrate()`) 的数学形式不需要变。因为模型预测的依然是 $v_\theta(z_t, t, \text{condition})$。
区别在于，传入的 condition 从 `style_id` 变成了 `style_image` 或 `style_feat`。

### 4.2 零样本泛化能力 (Zero-shot Inference)
此时的模型已经摆脱了 5 个固定 `style_id` 的诅咒。在评估阶段，你可以喂给它**互联网上下载的任意一张风格图片**，模型都能从中提取 DINO 特征，并通过 Cross-Attention 寻址纹理并贴回到你的内容图上。

### 4.3 外推扫描 (Overdrive Sweep)
在新架构完成首个 epoch 的验证后，立刻进行 $t=1.0$ 到 $t=1.5$ 的超驱动扫描。
由于时空彻底解耦（Time AdaLN 控制规模，Cross-Attention 控制方向），新架构的外推应该比旧架构更加平滑，极有可能轻松越过 `0.74` 的 clip_style 天花板而保持 `LPIPS < 0.35`。

---

## 总结：重构实施顺序

按照“**最小阻力路径**”：
1. 先不改模型，只写预配对脚本验证配对结果。
2. 复制一份 `model.py` 为 `model_fm.py`，进行大清洗和解耦（Time / Style 分离）。
3. 复制一份 `train.py` 为 `train_fm.py`，实现几行的极简 MSE 损失。
4. 跑一个 8-Batch, 5-Epoch 的超微型实验，验证 loss 稳步下降且不发散。
5. 推理 10 张图，视觉确认无马赛克（证明 Cross-Attention 有效），即告重构理论完全成功，随后铺开大规模训练。
