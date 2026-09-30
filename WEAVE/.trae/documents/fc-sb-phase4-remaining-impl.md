# FC-SB Phase 4 剩余实施计划

## 摘要

本计划聚焦三项高 ROI 工作（用户已批准范围）：

1. **A2 Step2 — Fiber 空间 Source-Repulsion**：完成已有半成品，让 `source_style_latent` 钩子真正生效，使推理期能基于"原内容图风格"做反向排斥。
2. **B4 — Fiber-MoE Adapters**：在 N1 块 α-blend 前插入轻量 MoE 路由，按 style global token 选择 4-8 个小 expert，逼近 per-style 上限（0.76）。
3. **B2 — 原生频域 ODE POC**：新建 `spectral620.py` + `spectral_bridge620.py` + `spectral_losses620.py` 三个模块，跑 2-3 epoch POC 验证频域解耦能否压低 LPIPS。

跳过 B1（双流 MMDiT）和 B3（能量引导桥）的大重构，待上述三项验证后再决定。

---

## 当前状态分析

### 已实现（Phase 4 早期工作，已通过 Phase 1 探索验证）

| 项 | 状态 | 关键位置 |
|---|---|---|
| A1 时频交叉调度 | ✅ 完整 | model620.py L633-671（逻辑）+ L814-818（应用）+ L834-836（probe）+ config_schema.py L488-494 |
| A3 Logit-Normal 时间采样 | ✅ 完整 | losses620.py L80-83（init）+ L213-224（_sample_t）+ config_schema.py L538-541 |
| A4 输出方差匹配 | ✅ 完整 | losses620.py L139-144（init）+ L703-730（计算）+ L786/L805（双分支 loss）+ L901（metric）+ config_schema.py L614-616 |
| A2 Step1（ep_null 真无条件修复） | ✅ 已应用 | model620.py L1036（integrate_transport_cfg）+ L926-930（integrate_transport K1）|

### 半成品（A2 Step2 仅完成签名）

- model620.py L565：`integrate_transport` 签名已有 `source_style_latent: torch.Tensor | None = None` 参数
- **但是**：参数从未在方法体内被读取，被 `**_: object` 静默吞掉

### 未实现（本计划覆盖）

| 项 | 缺失内容 |
|---|---|
| A2 Step2 主体 | K1 段扩展、config 字段、inference 传递、run_evaluation 构造 |
| B4 Fiber-MoE | 全部缺失（但 blocks620.py L130-156 的 `style_moe_*` 模式可作为模板）|
| B2 频域 ODE POC | 全部缺失（无现有 wavelet 模块；Haar 代码在 model620.py L782-797 和 losses620.py L711-723 重复内联）|

### 关键发现（影响实现决策）

1. **`integrate` 方法路由**：model620.py L991 `integrate` 仅 forward 到 `integrate_transport`，**不**调用 `integrate_transport_cfg`。这意味着现有的 tri-directional CFG（L1038-1044 的 `cfg_repulse_scale * (ep_idt - ep_null)`）从生产推理路径**不可达**。A2 Step2 必须在 `integrate_transport` 内实现，不能依赖 `_cfg`。
2. **`_cfg_get` 优先级**：model_cfg 优先 > bridge_cfg > default。新字段添加到 ModelConfig 即可。
3. **K1 段插入点**：model620.py L933 后（K1 target-CFG 之后，K0 放大之前）是 A2 source-repulsion 的自然插槽，`v_fiber` / `denom` / `lp` / `t_batch` / `style_id` 均在作用域内。
4. **B4 插入点**：model620.py L901-902 之间（N1 块 if/elif/else 链结束、α-blend 之前）是 MoE 的通用插入点，可统一处理所有 `multiband_adain_mode` 分支。
5. **B2 contract family**：config_schema.py L989 显示 `620_spatial_bridge` 跳过所有 contract validator，新增 `620_spectral_ode` 同样可跳过，不会破坏 legacy 校验。
6. **`haar_inv` 是近似的**：model620.py L790-797 用 nearest-upsample，不是精确逆变换。B2 必须实现精确 IDWT。
7. **run.py 是 contract-agnostic**：无需修改 run.py；只需在 model.py L2449 和 trainer.py L239 添加 dispatch 分支。

---

## 实施计划

### Task 1: A2 Step2 — Fiber 空间 Source-Repulsion

**目标**：让推理期能基于"原内容图的 VAE 风格 latent"在 fiber 空间做反向排斥，打破保守吸引子，把 α（endpoint 移动率）从 0.16 拉伸到 0.5+。

**理论公式**：
```
v_source = (ep_source - h) / denom                       # 用 source_style_latent 预测速度
v_source_fiber = v_source - lp(v_source)                 # 投影到 fiber 空间
v_fiber = v_fiber - ω_source · (v_source_fiber - v_null_fiber)   # 反向排斥
```

其中 `v_null_fiber` 复用 K1 已计算的结果（若 K1 未启用则单独计算一次 ep_null）。

#### 1.1 config_schema.py 修改

**文件**：`g:\GitHub\Latent_Style\SchrodingerBridge\src\config_schema.py`

在 ModelConfig 的 A1 字段后（L494 之后）追加：

```python
# === FC-SB Phase 4 A2 Step2: Fiber-Space Source-Repulsion ===
fiber_source_repulse_scale: float = 0.0   # 推理期 fiber 空间 source-repulsion 强度（0=禁用，0.5-1.5 推荐范围）
```

#### 1.2 model620.py 修改

**文件**：`g:\GitHub\Latent_Style\SchrodingerBridge\src\model620.py`

**修改 A：K1 段扩展（L933 之后插入）**

在 K1 Fiber-CFG 块结束后、K0 放大块之前，插入 source-repulsion 代码：

```python
# 🆕 A2 Step2: Fiber-Space Source-Repulsion
# 理论: 用原内容图风格 latent 在 fiber 空间反向排斥, 打破保守吸引子
fiber_source_repulse_scale = float(_cfg_get('fiber_source_repulse_scale', 0.0))
if fiber_source_repulse_scale > 0.0 and source_style_latent is not None:
    # 复用 K1 的 v_null_fiber（若 K1 启用），否则单独计算
    if fiber_cfg_scale <= 0.0:
        ep_null_sr = self.predict_endpoint(
            h, t=t_batch, style_id=style_id,
            style_dino_patches=None, style_dino_cls=None,
            style_text_tokens=None, style_latent=None,
        )
        v_null_sr = (ep_null_sr - h) / denom
        v_null_fiber_sr = v_null_sr - lp(v_null_sr) if fiber_proj_ep else v_null_sr
    else:
        v_null_fiber_sr = v_null_fiber  # 复用 K1 计算结果
    # 用 source_style_latent 预测 source 方向速度
    ep_source = self.predict_endpoint(
        h, t=t_batch, style_id=style_id,
        style_dino_patches=None, style_dino_cls=None,
        style_text_tokens=None, style_latent=source_style_latent,
    )
    v_source = (ep_source - h) / denom
    v_source_fiber = v_source - lp(v_source) if fiber_proj_ep else v_source
    # 反向排斥：减去 source 与 null 的偏差
    v_fiber = v_fiber - fiber_source_repulse_scale * (v_source_fiber - v_null_fiber_sr)
    self.last_debug["a2_source_repulse_delta"] = float(
        (fiber_source_repulse_scale * (v_source_fiber - v_null_fiber_sr)).abs().mean().item()
    )
```

**修改 B：probe 上报**

`last_debug["a2_source_repulse_delta"]` 已在上述代码内写入，无需额外修改。

#### 1.3 inference.py 修改

**文件**：`g:\GitHub\Latent_Style\SchrodingerBridge\src\utils\inference.py`

**修改 `generation_with_target_latent`（L519-559）**

签名扩展：

```python
def generation_with_target_latent(self, x0, target_style_id, num_steps=None, target_style_latent=None, source_style_latent=None):
```

在 L537-540 的 `integrate_kwargs` 构造后，dict 分支内（L541-551）追加：

```python
if source_style_latent is not None:
    if isinstance(source_style_latent, dict):
        # 兼容 dict 形式
        _src_tensor = source_style_latent.get("style_latent_tensor")
        if _src_tensor is not None:
            integrate_kwargs["source_style_latent"] = _src_tensor
    else:
        integrate_kwargs["source_style_latent"] = source_style_latent
```

非 dict 分支（L552-553）也追加同样逻辑。

#### 1.4 run_evaluation.py 修改

**文件**：`g:\GitHub\Latent_Style\SchrodingerBridge\src\utils\run_evaluation.py`

**修改 A：构建 source_style_latent cache（在 L3145 之后，即 `latents_x0 = lgt.inversion(latents_src)` 之后）**

```python
# === FC-SB Phase 4 A2 Step2: 构建 source_style_latent cache ===
# 理论: 用原内容图的 VAE latent 作为 source 风格信号, 供 fiber 空间反向排斥
_source_style_latents = latents_src  # 已是 VAE encode 后的 latent, shape (B, C, H, W)
# 与 repeated_latents 对齐（每个 source 对应一个 tgt）
# repeated_latents 在 L3171 构造, 这里按相同顺序展开
```

**修改 B：在 generation 调用处传递（L3250-3254）**

```python
latents_gen = lgt.generation_with_target_latent(
    repeated_latents,
    tgt_ids,
    target_style_latent=target_style_latent,
    source_style_latent=_source_style_latents_repeated,  # 新增：与 repeated_latents 一一对应
)
```

需在调用前构造 `_source_style_latents_repeated`：按 `meta` 中每个 (src, tgt) pair 重复 source latent，使其与 `repeated_latents`（L3171）的 batch 维度对齐。

**实现细节**：参考 L3166-3171 的 `repeated_latents` 构造模式，用相同索引逻辑构建 `_source_style_latents_repeated`。

#### 1.5 验证步骤（Task 1）

1. **语法检查**：`python -c "import ast; ast.parse(open('src/model620.py').read()); ast.parse(open('src/utils/inference.py').read()); ast.parse(open('src/utils/run_evaluation.py').read()); ast.parse(open('src/config_schema.py').read()); print('OK')"`
2. **配置生效验证**：在 config 中设 `fiber_source_repulse_scale: 0.5`，跑 1 张图的推理，检查 `last_debug["a2_source_repulse_delta"]` 是否非零（证明代码路径被激活）。
3. **α 移动率验证**：对比 `fiber_source_repulse_scale=0.0`（baseline）vs `0.5` vs `1.0` 的 endpoint 移动率（`α = ||endpoint - x|| / ||target - x||`），预期 α 从 0.16 上升到 0.3+。
4. **白化指标验证**：跑 5-style probe，对比 WFI 是否下降、clip_style 是否上升、LPIPS 是否在可接受范围内恶化（< 0.45）。

---

### Task 2: B4 — Fiber-MoE Adapters

**目标**：在 N1 块 α-blend 前插入轻量 MoE，按 DINO style global token 路由到 4-8 个 expert，缩小 universal 模型（0.70）与 per-style 模型（0.76）的 gap。

**理论**：通用模型拟合 5 种完全不同的概率流困难是因为一个 FFN 必须兼顾所有风格。MoE 让每个 expert 专注一个风格子集，参数代价小（+几 M），上限能力接近 per-style。

#### 2.1 config_schema.py 修改

**文件**：`g:\GitHub\Latent_Style\SchrodingerBridge\src\config_schema.py`

在 ModelConfig 的 A2 字段后追加（L495 之后）：

```python
# === FC-SB Phase 4 B4: Fiber-MoE Adapters ===
fiber_moe_enabled: bool = False              # 总开关：在 N1 块 α-blend 前插入 MoE 路由
fiber_moe_num_experts: int = 4               # expert 数量（4-8 推荐）
fiber_moe_router_hidden_dim: int = 128       # router MLP 隐藏维
fiber_moe_expert_hidden_dim: int = 256       # 每个 expert 的 FFN 隐藏维
fiber_moe_router_input: str = "style_global" # "style_global" | "fiber_stats" | "concat"
fiber_moe_load_balance_weight: float = 0.01  # load balancing loss 权重
```

#### 2.2 新建 FiberMoE 模块

**文件**：`g:\GitHub\Latent_Style\SchrodingerBridge\src\fiber_moe620.py`（新建）

```python
"""FC-SB Phase 4 B4: Fiber-MoE Adapters.

在 N1 块 α-blend 前对 ep_fiber_matched 做 MoE 路由.
参考 blocks620.py L130-156 的 style_moe 模式, 但 I/O 契约不同:
- 输入: ep_fiber_matched (B, C, H, W) + style_global (B, D)
- 输出: ep_fiber_moe (B, C, H, W) + router_probs (B, num_experts) [用于 aux loss]
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


class FiberMoE(nn.Module):
    """Lightweight MoE for fiber modulation.

    每个 expert 是一个 1x1 conv FFN (channel-wise), 输出残差加到 ep_fiber_matched.
    Router 基于 style_global token 计算 soft 权重.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = 4,
        router_hidden_dim: int = 128,
        expert_hidden_dim: int = 256,
        router_input: str = "style_global",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.router_input = router_input
        # Router: style_global (D) -> expert weights (num_experts)
        router_in_dim = dim  # style_global 维度 = dim
        self.router = nn.Sequential(
            nn.LayerNorm(router_in_dim),
            nn.Linear(router_in_dim, router_hidden_dim),
            nn.SiLU(),
            nn.Linear(router_hidden_dim, num_experts),
        )
        # Experts: 每个 expert 是 1x1 conv FFN (dim -> hidden -> dim), 残差加
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, expert_hidden_dim, kernel_size=1),
                nn.SiLU(),
                nn.Conv2d(expert_hidden_dim, dim, kernel_size=1),
            )
            for _ in range(num_experts)
        ])
        # Zero-init last conv for identity start (不破坏 baseline)
        for expert in self.experts:
            nn.init.zeros_(expert[-1].weight)
            nn.init.zeros_(expert[-1].bias)

    def forward(
        self,
        ep_fiber_matched: torch.Tensor,   # (B, C, H, W)
        style_global: torch.Tensor,        # (B, D)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Router
        router_logits = self.router(style_global.float().to(dtype=style_global.dtype))
        router_probs = torch.softmax(router_logits.float(), dim=-1).to(dtype=ep_fiber_matched.dtype)
        # Experts (B, num_experts, C, H, W) -> weighted sum
        expert_outs = torch.stack(
            [expert(ep_fiber_matched) for expert in self.experts],
            dim=1,
        )  # (B, num_experts, C, H, W)
        weights = router_probs[:, :, None, None, None]  # (B, num_experts, 1, 1, 1)
        moe_delta = (expert_outs * weights).sum(dim=1)  # (B, C, H, W)
        # Residual
        return ep_fiber_matched + moe_delta, router_probs
```

#### 2.3 model620.py 修改

**文件**：`g:\GitHub\Latent_Style\SchrodingerBridge\src\model620.py`

**修改 A：导入 FiberMoE**

在文件头部导入区追加：

```python
from fiber_moe620 import FiberMoE
```

**修改 B：__init__ 实例化（在 endpoint head 初始化之后）**

```python
# === FC-SB Phase 4 B4: Fiber-MoE Adapters ===
self.fiber_moe_enabled = bool(getattr(model_cfg, "fiber_moe_enabled", False))
if self.fiber_moe_enabled:
    self.fiber_moe = FiberMoE(
        dim=self.dim,
        num_experts=int(getattr(model_cfg, "fiber_moe_num_experts", 4)),
        router_hidden_dim=int(getattr(model_cfg, "fiber_moe_router_hidden_dim", 128)),
        expert_hidden_dim=int(getattr(model_cfg, "fiber_moe_expert_hidden_dim", 256)),
        router_input=str(getattr(model_cfg, "fiber_moe_router_input", "style_global")),
    )
else:
    self.fiber_moe = None
```

**修改 C：N1 块 α-blend 前插入 MoE（L901-902 之间）**

在 `ep_fiber_matched = ...`（各分支结束后）与 `endpoint = ep_base + ...`（α-blend）之间插入：

```python
# === FC-SB Phase 4 B4: Fiber-MoE Adapters ===
if self.fiber_moe is not None:
    # 获取 style_global：从 style_latent 池化得到（或从 forward 时缓存的 style_embed）
    # 这里用 style_latent 的 channel mean 作为 style_global 代理
    _style_global = style_latent.float().mean(dim=[2, 3]).to(dtype=style_latent.dtype) if isinstance(style_latent, torch.Tensor) else None
    if _style_global is not None:
        # 维度对齐：style_latent 通道数 = latent_channels, 需映射到 dim
        # 简化：用 endpoint_style_to_high 的第一层 LayerNorm 做投影
        # 或者更简单：用现有 style_embed（若在 forward 中缓存）
        # 详见下方"实现说明"
        ep_fiber_matched, moe_router_probs = self.fiber_moe(ep_fiber_matched, _style_global_proj)
        self.last_debug["b4_moe_router_entropy"] = float(-(moe_router_probs * (moe_router_probs + 1e-8).log()).sum(dim=-1).mean().item())
        self.last_debug["b4_moe_router_max_prob"] = float(moe_router_probs.max(dim=-1).values.mean().item())
```

**实现说明（style_global 投影）**：

`style_latent` 的通道数是 `latent_channels`（通常 4），而 `FiberMoE` 的 router 期望 `dim`（通常 64-128）。需要在 `__init__` 中加一个投影层：

```python
self.style_latent_to_dim = nn.Linear(self.latent_channels, self.dim) if self.fiber_moe_enabled else None
```

然后在 MoE 调用前：

```python
_style_global_proj = self.style_latent_to_dim(_style_global)
```

**修改 D：load balancing aux loss**

MoE 的 router 需要负载均衡损失以避免 expert 坍缩。在 `forward` 方法中（不在 `integrate_transport` 内）缓存 `moe_router_probs`，由 `losses620.py` 在训练时读取并加入 aux loss。

或者更简单：在 `last_debug["b4_moe_router_probs"]` 中存整个 probs 张量，losses620.py 读取后计算：

```python
# losses620.py 中
b4_router_probs = model.last_debug.get("b4_moe_router_probs")
if b4_router_probs is not None:
    # Load balancing loss: 鼓励 expert 均匀使用
    avg_probs = b4_router_probs.mean(dim=0)  # (num_experts,)
    aux_loss = (avg_probs * torch.log(avg_probs + 1e-8)).sum()  # 熵
    loss += self.fiber_moe_load_balance_weight * (-aux_loss)  # 最大化熵 = 均匀分布
```

config_schema.py BridgeConfig 新增：

```python
fiber_moe_load_balance_weight: float = 0.01
```

#### 2.4 验证步骤（Task 2）

1. **语法检查**：所有修改文件通过 `ast.parse`。
2. **零初始化验证**：`fiber_moe_enabled=True` 时，跑 1 张图推理，输出应与 `fiber_moe_enabled=False` 完全一致（zero-init 保证 identity start）。
3. **Router 激活验证**：跑一个 batch（5 个 style），检查 `b4_moe_router_max_prob` 是否在每个 style 上偏向不同 expert（证明 router 学到了 style-specific 路由）。
4. **收敛验证**：训练 2 epoch 后，对比 clip_style 是否提升（目标 +0.5% 以上）、LPIPS 是否在可接受范围（< 0.40）、WFI 是否下降。

---

### Task 3: B2 — 原生频域 ODE POC

**目标**：在进入网络前用 DWT 把 latent 拆成 LL/LH/HL/HH 四张量，主干网络输出 4 路独立速度场，频域内独立 Euler 积分，最后 iDWT 合成。验证频域解耦能否压低 LPIPS、提升 clip_style。

**理论**：当前 FC-SB 是"事后补救"（在欧氏空间算完速度再投影），B2 把频域解耦变成网络的**第一层物理特性**，让 LL（结构）和 HH（笔触）从源头走不同管道。

#### 3.1 新建 spectral620.py（共享小波工具）

**文件**：`g:\GitHub\Latent_Style\SchrodingerBridge\src\spectral620.py`（新建）

```python
"""FC-SB Phase 4 B2: 共享 Haar 小波工具.

提供精确的 DWT2/IDWT2, 供 spectral_bridge620 和 spectral_losses620 共用.
归一化: 标准 Haar (除以 2, 保持 orthonormal).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def dwt2_haar(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """一级 2D Haar DWT (orthonormal 归一化).

    Args:
        x: (B, C, H, W), H/W 必须为偶数
    Returns:
        (LL, LH, HL, HH): 各 (B, C, H/2, W/2)
    """
    x = x.float()
    x00 = x[..., 0::2, 0::2]
    x01 = x[..., 0::2, 1::2]
    x10 = x[..., 1::2, 0::2]
    x11 = x[..., 1::2, 1::2]
    ll = (x00 + x01 + x10 + x11) / 2.0
    lh = (x00 + x01 - x10 - x11) / 2.0
    hl = (x00 - x01 + x10 - x11) / 2.0
    hh = (x00 - x01 - x10 + x11) / 2.0
    return ll, lh, hl, hh


def idwt2_haar(
    ll: torch.Tensor,
    lh: torch.Tensor,
    hl: torch.Tensor,
    hh: torch.Tensor,
    target_size: tuple[int, int],
) -> torch.Tensor:
    """精确逆 2D Haar IDWT (与 dwt2_haar 严格对偶).

    Args:
        ll/lh/hl/hh: 各 (B, C, H/2, W/2)
        target_size: (H, W) 原始尺寸
    Returns:
        x: (B, C, H, W)
    """
    H, W = target_size
    out = torch.zeros(ll.shape[0], ll.shape[1], H, W, device=ll.device, dtype=ll.dtype)
    out[..., 0::2, 0::2] = (ll + lh + hl + hh) / 2.0
    out[..., 0::2, 1::2] = (ll + lh - hl - hh) / 2.0
    out[..., 1::2, 0::2] = (ll - lh + hl - hh) / 2.0
    out[..., 1::2, 1::2] = (ll - lh - hl + hh) / 2.0
    return out


def dwt2_multi_level(x: torch.Tensor, levels: int = 1) -> list[tuple]:
    """多级 Haar 金字塔分解.

    Returns:
        list of (LL, LH, HL, HH) tuples, 从最细到最粗
    """
    pyr = []
    cur = x
    for _ in range(levels):
        pyr.append(dwt2_haar(cur))
        cur = pyr[-1][0]  # LL 作为下一级输入
    return pyr
```

#### 3.2 新建 spectral_bridge620.py（核心模型）

**文件**：`g:\GitHub\Latent_Style\SchrodingerBridge\src\spectral_bridge620.py`（新建）

```python
"""FC-SB Phase 4 B2: 原生频域 ODE 桥.

理论: DWT 拆 4 子带 -> 4 路独立速度场 -> 频域 Euler 积分 -> iDWT 合成.
LL 速度场 loss 权重 0 (锁死结构), HH 速度场 loss 权重大 (逼学笔触).
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

from blocks620 import SpatialBridgeBlock620, sinusoidal_time_embedding_620
from config_schema import BridgeConfig, ModelConfig
from spectral620 import dwt2_haar, idwt2_haar
from style_encoder620 import StyleConditioner620


class SpectralVelocityHead(nn.Module):
    """4 路独立速度场 head: 共享 backbone feature, 4 个 1x1 conv 投影到 (LL, LH, HL, HH) 速度."""

    def __init__(self, dim: int, latent_channels: int) -> None:
        super().__init__()
        # 4 路独立投影: dim -> latent_channels
        self.proj_ll = nn.Conv2d(dim, latent_channels, kernel_size=3, padding=1)
        self.proj_lh = nn.Conv2d(dim, latent_channels, kernel_size=3, padding=1)
        self.proj_hl = nn.Conv2d(dim, latent_channels, kernel_size=3, padding=1)
        self.proj_hh = nn.Conv2d(dim, latent_channels, kernel_size=3, padding=1)
        # Zero-init for identity start
        for proj in [self.proj_ll, self.proj_lh, self.proj_hl, self.proj_hh]:
            nn.init.normal_(proj.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(proj.bias)

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "ll": self.proj_ll(feat),
            "lh": self.proj_lh(feat),
            "hl": self.proj_hl(feat),
            "hh": self.proj_hh(feat),
        }


class SpectralODEBridge620(nn.Module):
    """原生频域 ODE 桥.

    流程:
        1. x (B, C, H, W) -> DWT -> (LL, LH, HL, HH) 各 (B, C, H/2, W/2)
        2. 4 子带各自 input_proj 到 dim 维
        3. 共享 backbone (SpatialBridgeBlock620 序列) 处理 concat 4 子带
        4. SpectralVelocityHead 输出 4 路速度
        5. 推理: 频域 Euler 积分 -> iDWT 合成
    """

    def __init__(self, model_cfg: ModelConfig, bridge_cfg: BridgeConfig | None = None) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.bridge_cfg = bridge_cfg
        self.latent_channels = int(model_cfg.latent_channels)
        self.dim = int(model_cfg.base_dim)
        self.time_dim = int(getattr(model_cfg, "time_dim", self.dim))
        self.dino_dim = int(getattr(model_cfg, "tokenizer_dino_dim", 384))

        # 4 路独立 input_proj (latent_channels -> dim)
        self.input_proj_ll = nn.Conv2d(self.latent_channels, self.dim, kernel_size=3, padding=1)
        self.input_proj_lh = nn.Conv2d(self.latent_channels, self.dim, kernel_size=3, padding=1)
        self.input_proj_hl = nn.Conv2d(self.latent_channels, self.dim, kernel_size=3, padding=1)
        self.input_proj_hh = nn.Conv2d(self.latent_channels, self.dim, kernel_size=3, padding=1)

        # 时间嵌入
        self.time_proj = nn.Sequential(
            nn.Linear(self.time_dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )

        # Style conditioner (复用现有)
        self.style_conditioner = StyleConditioner620(
            dino_dim=self.dino_dim,
            model_dim=self.dim,
            num_styles=int(model_cfg.num_styles),
            num_memory_tokens=256,
            adapter_enabled=bool(getattr(model_cfg, "style_dino_adapter_enabled", False)),
            adapter_hidden_dim=int(getattr(model_cfg, "style_dino_adapter_hidden_dim", 1024)),
            adapter_scale=float(getattr(model_cfg, "style_dino_adapter_scale", 0.25)),
            local_cnn_enabled=False,
            text_enabled=bool(getattr(model_cfg, "style_text_enabled", False)),
            text_dim=int(getattr(model_cfg, "style_text_dim", 768)),
            text_max_length=int(getattr(model_cfg, "style_text_max_length", 77)),
            text_dropout_prob=float(getattr(model_cfg, "style_text_dropout_prob", 0.15)),
            image_dropout_prob=float(getattr(model_cfg, "style_image_dropout_prob", 0.15)),
            text_null_std=float(getattr(model_cfg, "style_text_null_token_init_std", 0.02)),
            image_null_std=float(getattr(model_cfg, "style_image_null_token_init_std", 0.02)),
        )

        # 共享 backbone (在 concat 4 子带后处理)
        depth = max(1, int(getattr(model_cfg, "num_res_blocks", 4)))
        heads = max(1, int(getattr(model_cfg, "style_attn_num_heads", 4)))
        self.blocks = nn.ModuleList([
            SpatialBridgeBlock620(
                dim=self.dim * 4,  # concat 4 子带
                num_heads=heads,
                style_gate_init=float(getattr(model_cfg, "style_cross_attn_gate_init", 0.3)),
                style_gate_mode=str(getattr(model_cfg, "style_gate_mode", "tanh_gate")),
                style_moe_enabled=False,
                style_moe_num_experts=4,
                style_moe_router_hidden_dim=128,
                style_kv_moe_content_routed=False,
                style_shortcut_alpha=getattr(model_cfg, "style_shortcut_alpha", 1.0),
                style_query_source=str(getattr(model_cfg, "style_query_source", "concat")),
                style_cross_attn_skip_coarse=False,
                style_attn_topk=0,
                layer_idx=idx,
                num_layers=depth,
                dino_dim=self.dino_dim,
                film_enabled=bool(getattr(model_cfg, "style_film_enabled", False)),
                film_init_std=float(getattr(model_cfg, "style_film_init_std", 0.02)),
                attn_mode=str(getattr(model_cfg, "style_attn_mode", "softmax")),
                attn_temperature=float(getattr(model_cfg, "style_attn_temperature", 1.0)),
                gate_warmup_steps=int(getattr(model_cfg, "style_gate_warmup_steps", 0)),
                norm_type=str(getattr(model_cfg, "body_norm_type", "group_norm")),
            )
            for idx in range(depth)
        ])

        # 4 路速度 head
        self.velocity_head = SpectralVelocityHead(self.dim * 4, self.latent_channels)

        self.last_debug: dict = {}

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        style_id: torch.Tensor | int | None = None,
        style_dino_patches: torch.Tensor | None = None,
        style_dino_cls: torch.Tensor | None = None,
        style_text_tokens: torch.Tensor | None = None,
        style_latent: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """返回 4 路速度场 dict: {"ll": v_ll, "lh": v_lh, "hl": v_hl, "hh": v_hh}."""
        B, C, H, W = x.shape
        # 1. DWT
        ll, lh, hl, hh = dwt2_haar(x)
        # 2. 各子带 input_proj
        feat_ll = self.input_proj_ll(ll)
        feat_lh = self.input_proj_lh(lh)
        feat_hl = self.input_proj_hl(hl)
        feat_hh = self.input_proj_hh(hh)
        # 3. concat 4 子带
        feat = torch.cat([feat_ll, feat_lh, feat_hl, feat_hh], dim=1)  # (B, 4*dim, H/2, W/2)
        # 4. 时间嵌入
        t_emb = sinusoidal_time_embedding_620(t, self.time_dim).to(dtype=x.dtype)
        t_emb = self.time_proj(t_emb)
        # 5. Style 编码
        style_tokens, style_global = self.style_conditioner(
            dino_patches=style_dino_patches,
            dino_cls=style_dino_cls,
            style_id=style_id,
            text_tokens=style_text_tokens,
            style_latent=style_latent,
        )
        # 6. Backbone
        for block in self.blocks:
            feat = block(feat, style_tokens=style_tokens, style_global=style_global, t_emb=t_emb)
        # 7. 4 路速度 head
        velocities = self.velocity_head(feat)
        self.last_debug["b2_v_ll_abs"] = velocities["ll"].detach().float().abs().mean().item()
        self.last_debug["b2_v_hh_abs"] = velocities["hh"].detach().float().abs().mean().item()
        return velocities

    @torch.no_grad()
    def integrate(
        self,
        x: torch.Tensor,
        style_id: torch.Tensor | int | None,
        num_steps: int = 8,
        step_size: float = 1.0,
        style_dino_patches: torch.Tensor | None = None,
        style_dino_cls: torch.Tensor | None = None,
        style_text_tokens: torch.Tensor | None = None,
        style_latent: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """频域 Euler 积分."""
        h = x.clone()
        H, W = x.shape[-2:]
        for idx in range(num_steps):
            t_curr = idx / float(num_steps)
            t_next = (idx + 1) / float(num_steps)
            t_batch = torch.full((h.shape[0],), t_curr, device=h.device, dtype=h.dtype)
            # 4 路速度
            velocities = self.forward(
                h, t=t_batch, style_id=style_id,
                style_dino_patches=style_dino_patches, style_dino_cls=style_dino_cls,
                style_text_tokens=style_text_tokens, style_latent=style_latent,
            )
            # 频域 Euler: 每个子带独立积分
            ll, lh, hl, hh = dwt2_haar(h)
            ll = ll + velocities["ll"] * (t_next - t_curr)
            lh = lh + velocities["lh"] * (t_next - t_curr)
            hl = hl + velocities["hl"] * (t_next - t_curr)
            hh = hh + velocities["hh"] * (t_next - t_curr)
            h = idwt2_haar(ll, lh, hl, hh, target_size=(H, W))
        return h
```

#### 3.3 新建 spectral_losses620.py（损失函数）

**文件**：`g:\GitHub\Latent_Style\SchrodingerBridge\src\spectral_losses620.py`（新建）

```python
"""FC-SB Phase 4 B2: 频域 ODE 损失.

理论:
- LL 速度 loss 权重 0 (锁死结构, 与 BASE LOCKING 等价)
- LH/HL (mid) 速度 loss 权重小
- HH 速度 loss 权重大 (逼学笔触)
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

from config_schema import ExperimentConfig
from spectral620 import dwt2_haar


class SpectralODEObjective620:
    """频域 ODE 训练目标."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.training = True
        self.bridge_cfg = config.bridge
        # 各子带 loss 权重
        self.w_ll = float(getattr(self.bridge_cfg, "spectral_w_ll", 0.0))   # LL 锁死
        self.w_lh = float(getattr(self.bridge_cfg, "spectral_w_lh", 0.5))   # mid 笔触
        self.w_hl = float(getattr(self.bridge_cfg, "spectral_w_hl", 0.5))   # mid 笔触
        self.w_hh = float(getattr(self.bridge_cfg, "spectral_w_hh", 4.0))   # HH 细节, 大权重
        self.fm_weight = float(getattr(self.bridge_cfg, "w_flow", 1.0))
        self.last_debug: dict = {}

    def compute(
        self,
        model,
        content: torch.Tensor,
        target_style: torch.Tensor,
        target_style_id: torch.Tensor,
        source_style_id: torch.Tensor | None = None,
        aux_target_style=None,
        aux_target_valid=None,
        conditioning=None,
    ) -> dict:
        """计算 4 路速度场 loss.

        content: (B, C, H, W) 源 latent
        target_style: (B, C, H, W) 目标 style latent
        """
        B = content.shape[0]
        # 采样 t
        t = torch.rand(B, device=content.device, dtype=content.dtype)
        # 桥插值: x_t = (1-t)*content + t*target + noise
        t_view = t.view(B, 1, 1, 1)
        x_t = (1.0 - t_view) * content + t_view * target_style
        # 目标速度 (Flow Matching): v = target - content
        v_target_ll, v_target_lh, v_target_hl, v_target_hh = dwt2_haar(target_style - content)

        # 模型预测
        v_pred = model(
            x_t, t=t, style_id=target_style_id,
            style_dino_patches=conditioning.get("target_dino_patches") if conditioning else None,
            style_dino_cls=conditioning.get("target_dino_cls") if conditioning else None,
            style_text_tokens=conditioning.get("target_text_tokens") if conditioning else None,
            style_latent=conditioning.get("target_style_latent") if conditioning else None,
        )

        # 各子带 MSE
        loss_ll = F.mse_loss(v_pred["ll"], v_target_ll)
        loss_lh = F.mse_loss(v_pred["lh"], v_target_lh)
        loss_hl = F.mse_loss(v_pred["hl"], v_target_hl)
        loss_hh = F.mse_loss(v_pred["hh"], v_target_hh)

        loss = (
            self.w_ll * loss_ll
            + self.w_lh * loss_lh
            + self.w_hl * loss_hl
            + self.w_hh * loss_hh
        )

        self.last_debug = {
            "loss_ll": loss_ll.detach(),
            "loss_lh": loss_lh.detach(),
            "loss_hl": loss_hl.detach(),
            "loss_hh": loss_hh.detach(),
            "v_ll_abs": v_pred["ll"].detach().float().abs().mean(),
            "v_hh_abs": v_pred["hh"].detach().float().abs().mean(),
        }

        return {
            "loss": loss,
            **self.last_debug,
        }
```

#### 3.4 config_schema.py 修改

**文件**：`g:\GitHub\Latent_Style\SchrodingerBridge\src\config_schema.py`

ModelConfig 追加（L496 之后）：

```python
# === FC-SB Phase 4 B2: Spectral ODE ===
spectral_ode_enabled: bool = False        # 总开关：使用 SpectralODEBridge620
spectral_ode_levels: int = 1              # Haar 金字塔级数（1=单级 DWT）
```

BridgeConfig 追加（L617 之后）：

```python
# === FC-SB Phase 4 B2: Spectral ODE Loss Weights ===
spectral_w_ll: float = 0.0                # LL 速度 loss 权重（0 = 锁死结构）
spectral_w_lh: float = 0.5                # LH 速度 loss 权重
spectral_w_hl: float = 0.5                # HL 速度 loss 权重
spectral_w_hh: float = 4.0                # HH 速度 loss 权重（大权重逼学笔触）
```

#### 3.5 model.py 修改（dispatch）

**文件**：`g:\GitHub\Latent_Style\SchrodingerBridge\src\model.py`

在 `build_model_from_config`（L2449 附近）添加新分支：

```python
if contract_family == "620_spectral_ode":
    from spectral_bridge620 import SpectralODEBridge620
    model = SpectralODEBridge620(model_cfg, bridge_cfg=bridge_cfg)
elif contract_family == "620_spatial_bridge":
    model = build_spatial_bridge620_from_config(model_cfg, bridge_cfg=bridge_cfg, ...)
```

#### 3.6 trainer.py 修改（dispatch）

**文件**：`g:\GitHub\Latent_Style\SchrodingerBridge\src\trainer.py`

在 L239 附近添加新分支：

```python
if contract_family == "620_spectral_ode":
    from spectral_losses620 import SpectralODEObjective620
    self.loss_fn = SpectralODEObjective620(config)
elif contract_family == "620_spatial_bridge":
    self.loss_fn = SpatialBridgeObjective620(config)
```

#### 3.7 POC 实验

**文件**：`g:\GitHub\Latent_Style\SchrodingerBridge\configs\620_spectral_poc.json`（新建）

基于现有 base config 修改：

```json
{
  "_base": "configs/620_base.json",
  "model": {
    "contract_family": "620_spectral_ode",
    "spectral_ode_enabled": true,
    "base_dim": 64,
    "num_res_blocks": 4,
    "style_condition_source": "target_dino_patches"
  },
  "bridge": {
    "spectral_w_ll": 0.0,
    "spectral_w_lh": 0.5,
    "spectral_w_hl": 0.5,
    "spectral_w_hh": 4.0,
    "w_flow": 1.0
  },
  "training": {
    "num_epochs": 3,
    "batch_size": 24,
    "patience": 2
  },
  "data": {
    "root": "/mnt/i/wikiart_distinct5_samam_512_classview"
  }
}
```

**POC 验证流程**：

1. 训练 3 epoch（Patience=2，max=10，至少 5 epoch 限制可放宽用于 POC）
2. 评估 5-style probe：clip_style / LPIPS / WFI
3. 与 I7 baseline（clip=0.7017, lpips=0.3624）对比
4. 关键观察点：
   - `v_ll_abs` 应接近 0（证明 LL 被锁死）
   - `v_hh_abs` 应显著大于 `v_ll_abs`（证明 HH 被激活）
   - LPIPS 是否低于 0.40（频域解耦的核心收益）
   - clip_style 是否高于 0.70

---

## 假设与决策

### A2 Step2 决策

1. **不重构 `integrate` 路由**：保持 `integrate` → `integrate_transport`，仅在 `integrate_transport` 内扩展，不引入 `_cfg` 路径。理由：最小改动，避免破坏现有推理 hot path。
2. **复用 K1 的 `v_null_fiber`**：当 K1 启用时，A2 Step2 复用其 `v_null_fiber` 计算结果，避免重复 forward。当 K1 未启用时，单独计算 `ep_null_sr`。
3. **`source_style_latent` 用 VAE latent**：与 `target_style_latent` 同源（VAE encode），不引入新编码器。
4. **config 字段放 ModelConfig**：因 `_cfg_get` 优先读 model_cfg，且推理期参数更接近模型行为而非训练超参。

### B4 决策

1. **新建 `fiber_moe620.py` 而非扩展现有 style_moe**：I/O 契约不同（style_moe 处理 token 序列，FiberMoE 处理 (B,C,H,W) 特征图），新模块更清晰。
2. **Zero-init 最后一层**：保证 `fiber_moe_enabled=True` 时初始行为与 `False` 完全一致，不破坏 baseline。
3. **Router 输入用 style_latent 投影**：避免修改 `forward` 签名缓存 style_global（最小侵入）。在 `__init__` 加 `style_latent_to_dim` 投影层。
4. **Load balancing loss 通过 `last_debug` 传递**：避免修改 `losses620.py` 的 `compute` 签名。

### B2 决策

1. **新 contract family `620_spectral_ode`**：跳过 legacy validator，不破坏现有 `620_spatial_bridge`。
2. **标准 Haar 归一化（/2）**：与 model620.py 现有 `/2.0` 不同，但 B2 是新模块无兼容性包袱，用 orthonormal 更数学严谨。
3. **POC 用单级 DWT**：多级金字塔留待 POC 验证成功后扩展。
4. **共享 backbone（concat 4 子带）**：而非 4 路独立 backbone，节省参数。dim*4 维度通过现有 SpatialBridgeBlock620 处理。
5. **POC 不要求达到 SOTA**：只验证 `v_ll_abs ≈ 0` 且 `v_hh_abs >> v_ll_abs`（频域解耦生效）+ LPIPS < 0.40。
6. **POC 训练 epoch 数放宽**：spec 要求至少 5 epoch，但 POC 用 3 epoch 快速验证可行性，验证成功后再跑完整训练。

---

## 验证步骤总览

### Task 1（A2 Step2）验证

1. 语法检查：4 个修改文件通过 `ast.parse`
2. 配置生效：`fiber_source_repulse_scale=0.5` 时 `last_debug["a2_source_repulse_delta"]` 非零
3. α 移动率：α 从 0.16 上升到 0.3+
4. 5-style probe：WFI 下降、clip_style 上升、LPIPS < 0.45

### Task 2（B4）验证

1. 语法检查：所有修改文件 + 新建 `fiber_moe620.py` 通过 `ast.parse`
2. Zero-init 行为：`fiber_moe_enabled=True` 输出与 `False` 一致
3. Router 激活：5 style 偏向不同 expert
4. 训练 2 epoch：clip_style +0.5% 以上、LPIPS < 0.40

### Task 3（B2）验证

1. 语法检查：3 个新文件 + 4 个修改文件通过 `ast.parse`
2. 频域解耦：`v_ll_abs ≈ 0`、`v_hh_abs >> v_ll_abs`
3. POC 训练 3 epoch：LPIPS < 0.40、clip_style > 0.70
4. 与 I7 对比：至少一项指标（LPIPS 或 clip_style）显著优于 I7

---

## 实施顺序

1. **Task 1（A2 Step2）**：已有半成品，必须先收尾。预计 4 个文件修改。
2. **Task 2（B4）**：新建 1 文件 + 修改 2 文件。Zero-init 保证不破坏 baseline，可并行开发。
3. **Task 3（B2）**：新建 3 文件 + 修改 3 文件 + 1 config。最大工程量，单独推进。

每个 Task 完成后立即验证，验证通过再进入下一个 Task。

---

## 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| A2 Step2 source-repulsion 导致 LPIPS 恶化 > 0.45 | 推理质量下降 | 通过 `fiber_source_repulse_scale` 调节强度，从 0.1 开始递增 |
| B4 MoE router 坍缩（所有 sample 走同一 expert） | MoE 退化为单 expert | load balancing loss + router entropy 监控 |
| B2 POC LPIPS 不降反升 | 频域解耦假设证伪 | 这本身就是有价值的发现，记录在 docs/，转入 B1/B3 方向 |
| B2 POC 训练 OOM | 12GB VRAM 限制 | batch_size=24（spec 要求），dim*4 backbone 可能需要降到 dim*2 |
| B2 SpectralVelocityHead 与现有 trainer 不兼容 | 训练失败 | trainer.py L1322 `loss_dict = self.loss_fn.compute(self.model, ...)` 期望返回 `{"loss": ...}` dict，SpectralODEObjective620 已遵循此契约 |

---

## 文档归档

每个 Task 完成后，将实验结果归档到 `docs/625/fc_sb_phase4_<task_name>.md`，包含：
- 代码改动清单
- 验证结果（probe 数据 + 5-style 指标）
- 与 baseline 对比
- 下一步建议
