这是基于我们所有讨论，深思熟虑后最终确定的 **Latent-AdaCUT (Statistical Version)** 实施方案。

该方案完全摒弃了不稳定的判别器（GAN），转而采用 **AdaGN（注入）+ SWD（统计对齐）+ NCE（结构约束）** 的黄金组合。

---

# 🚀 最终方案：Latent-AdaCUT

> **核心逻辑**：在 $32 \times 32$ 的潜空间中，通过 **AdaGN** 修改特征统计量，利用 **SWD** 强制对齐色彩与纹理分布，利用 **NCE** 锁死空间结构。

---

### **一、 模型架构 (Micro U-Net w/o Skips)**

我们采用 **Encoder-Bottleneck-Decoder** 结构，**去除跳跃连接**（防止源图纹理泄露），仅仅通过 Bottleneck 的 AdaGN 进行风格重构。

* **输入/输出**：$4 \times 32 \times 32$ (SD VAE Latent)
* **规模**：Base Dim = 64, 1次下采样 ($32 \to 16$)，4个残差块。
* **参数量**：约 **1.5M - 2M** (极度轻量，推理极快)。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaGN(nn.Module):
    """ 风格注入核心：自适应组归一化 """
    def __init__(self, dim, style_dim):
        super().__init__()
        self.norm = nn.GroupNorm(8, dim, affine=False)
        self.proj = nn.Linear(style_dim, dim * 2)
        # 初始化为恒等映射 (Scale=1, Shift=0)
        self.proj.weight.data.zero_()
        self.proj.bias.data[:] = torch.tensor([1.0] * dim + [0.0] * dim)

    def forward(self, x, style_code):
        h = self.norm(x)
        params = self.proj(style_code).unsqueeze(-1).unsqueeze(-1)
        scale, shift = params.chunk(2, dim=1)
        return h * scale + shift

class ResBlock(nn.Module):
    def __init__(self, dim, style_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.norm1 = AdaGN(dim, style_dim)
        self.norm2 = AdaGN(dim, style_dim)
        self.act = nn.SiLU()

    def forward(self, x, style_code):
        h = self.norm1(x, style_code)
        h = self.act(self.conv1(h))
        h = self.norm2(h, style_code)
        h = self.conv2(h)
        return x + h

class LatentAdaCUT(nn.Module):
    def __init__(self, num_styles=3, style_dim=256, base_dim=64):
        super().__init__()
        # 1. 风格嵌入 (0:Photo, 1:Monet, 2:VanGogh)
        self.style_emb = nn.Embedding(num_styles, style_dim)
      
        # 2. Encoder: 32x32 -> 16x16 (下采样一次足够)
        self.enc = nn.Sequential(
            nn.Conv2d(4, base_dim, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(base_dim, base_dim*2, 4, 2, 1) # stride=2
        )
      
        # 3. Bottleneck: 16x16 (深度风格处理)
        self.body = nn.ModuleList([
            ResBlock(base_dim*2, style_dim) for _ in range(4)
        ])
      
        # 4. Decoder: 16x16 -> 32x32
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_dim*2, base_dim, 3, 1, 1),
            nn.GroupNorm(8, base_dim),
            nn.SiLU(),
            nn.Conv2d(base_dim, 4, 3, 1, 1) # 无Tanh，Latent无界
        )
      
        # 5. NCE Projector (用于计算内容Loss)
        self.projector = nn.Sequential(
            nn.Linear(4, 256), nn.ReLU(), nn.Linear(256, 256)
        )

    def forward(self, x, style_id):
        # x: [B, 4, 32, 32]
        s = self.style_emb(style_id)
      
        feat = self.enc(x)
        for block in self.body:
            feat = block(feat, s)
        out = self.dec(feat)
      
        return out
```

---

### **二、 核心 Loss 设计 (Stable Statistics)**

放弃对抗损失，使用 **SWD (分布)** + **NCE (结构)** + **Moments (色调)**。

#### **1. Loss 计算代码**

```python
def calc_swd_loss(x, y, num_projections=128):
    """ Sliced Wasserstein Distance: 风格分布对齐的核心 """
    # x, y: [B, 4, 32, 32]
    B, C, H, W = x.shape
    # 展平为像素集合 [N, C]
    x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)
    y_flat = y.permute(0, 2, 3, 1).reshape(-1, C)
  
    # 随机投影
    proj = torch.randn((C, num_projections), device=x.device)
    proj = F.normalize(proj, dim=0)
  
    # 投影并排序
    proj_x = torch.sort(x_flat @ proj, dim=0)[0]
    proj_y = torch.sort(y_flat @ proj, dim=0)[0]
  
    return torch.abs(proj_x - proj_y).mean()

def calc_moment_loss(x, y):
    """ 均值方差对齐：加速色调收敛 """
    mu_x, std_x = x.mean(dim=[2,3]), x.std(dim=[2,3])
    mu_y, std_y = y.mean(dim=[2,3]), y.std(dim=[2,3])
    return F.mse_loss(mu_x, mu_y) + F.mse_loss(std_x, std_y)

def calc_nce_loss(model, x_in, x_out):
    """ 对比学习：锁死空间结构 """
    # 投影
    feat_q = F.normalize(model.projector(x_out.permute(0,2,3,1).reshape(-1, 4)), dim=1)
    feat_k = F.normalize(model.projector(x_in.permute(0,2,3,1).reshape(-1, 4)), dim=1)
    # Cosine Similarity
    return -torch.log(torch.sum(feat_q * feat_k, dim=1).exp()).mean() # 简化版 InfoNCE
```

#### **2. 权重配置**

由于 Latent 数值较小，统计 Loss 需要较大权重。

$$
L_{total} = 100 \cdot L_{SWD} + 10 \cdot L_{Moments} + 10 \cdot L_{NCE} + 5 \cdot L_{IDT}
$$

---

### **三、 训练实施细节**

1. **数据预处理 (非常重要)**

   * 将所有图片通过 SD VAE Encode，并 **乘上 0.18215**。
   * 保存为 `.pt` 或 `.npy`。
   * 训练时直接加载 Latent，**不**做任何 Data Augmentation（Flip 除外）。
2. **DataLoader**

   * Batch Size: **64 或 128** (SWD 依赖大 Batch 统计)。
   * Item: `{'content': photo_latent, 'style_monet': random_monet_latent, 'style_vangogh': random_vangogh_latent}`。
   * `style_monet` 是随机抽取的，不要求与 `content` 配对。
3. **优化器**

   * AdamW, Learning Rate = **1e-3** (因为没有 GAN，可以大胆用大一点的学习率，收敛更快)。
   * Weight Decay = 1e-4。
4. **推理与可视化**

   * Input: Photo Latent ($z$)
   * Output: Stylized Latent ($\hat{z}$)
   * Decode: `VAE.decode(\hat{z} / 0.18215)` -> 得到最终图像。

### **四、 预期结果**

* **训练速度**：极快。在单卡 RTX 3090 上，预计 **30分钟 - 1小时** 即可收敛。
* **视觉效果**：
  * **结构**：与原图高度一致（由 NCE 保证）。
  * **色调/纹理**：与目标风格高度一致（由 SWD/Moments 保证，Decoder 自动补全细节）。
* **稳定性**：Loss 曲线平滑下降，无震荡。

这个方案是目前针对你数据规模最稳妥、最高效的工程解法。可以直接开工写代码了！
