# FC-SB Phase 4 剩余实施计划 (Task 2 收尾 + Task 3 B2 POC)

## 摘要

本计划承接上一轮已批准的 Phase 4 实施。Task 1 (A2 Step2 Fiber-Space Source-Repulsion) 已完成。本计划聚焦两项剩余工作：

1. **Task 2 收尾 (B4 Fiber-MoE)**：在 `losses620.py` 中添加 load balancing aux loss 计算与 metric 上报。
2. **Task 3 (B2 原生频域 ODE POC)**：新建独立的 Spectral ODE Bridge 模型、损失、Haar 小波工具、dispatch 注册、POC 配置。

## 当前状态分析

### Task 1 (A2 Step2) — ✅ 已完成
- `config_schema.py` L495-496: `fiber_source_repulse_scale` 字段
- `model620.py` L963-990: `integrate_transport` 内 source-repulsion 代码
- `inference.py` L520/554-561: `source_style_latent` 参数传递
- `run_evaluation.py` L3250-3255: 调用处传递 `source_style_latent=repeated_latents`

### Task 2 (B4 Fiber-MoE) — ⏳ 进行中 (4/5 完成)
- ✅ `config_schema.py` L497-502 (ModelConfig) + L625-626 (BridgeConfig): B4 字段
- ✅ `fiber_moe620.py`: 完整 FiberMoE 模块 (Router + Experts + zero-init)
- ✅ `model620.py` L10 (import) + L302-317 (__init__ 实例化) + L920-929 (N1 块插入)
- ⏳ `losses620.py` L145-146: 仅 init 行, **aux loss 计算未完成**

### Task 3 (B2 Spectral ODE) — ⏳ 待开始

### 关键发现 (Phase 1 探索)

**B4 MoE 训练路径限制**：
- B4 MoE 插入点在 `model620.py` L920-929, 位于 `integrate_transport` 方法内
- `integrate_transport` 装饰 `@torch.no_grad()` (L571), 是**推理专用路径**
- 训练时只调用 `forward()` (L385-551), 不经过 `integrate_transport`
- 因此训练期间 `b4_moe_router_probs` **不会**被写入 `model.last_debug`
- aux loss 读取 `model.last_debug.get("b4_moe_router_probs")` 将返回 None → aux loss = 0 (no-op)

**处理策略**: 严格按照已批准计划，aux loss 代码使用 graceful None handling。代码正确且向前兼容：未来若将 MoE 移至训练路径，aux loss 将自动激活。此限制在计划中明确记录，不在本轮修复（避免范围蔓延）。

**B2 设计参考**:
- `model.py` L2449: `build_model_from_config` 按 `contract_family` dispatch
- `trainer.py` L236-239: `contract_family == "620_spatial_bridge"` → `SpatialBridgeObjective620`
- `config_schema.py` L998-999: `contract_family != "620_spatial_bridge"` 时才执行 contract validator
- `model620.py` L800-825: 现有 `haar_fwd`/`haar_inv` 是局部闭包, 且 `haar_inv` 用 nearest-upsample (近似)
- B2 需要精确 Haar DWT/IDWT (正交变换, 完美重建)
- `blocks620.py`: `SpatialBridgeBlock620` 可复用作为 spectral backbone

## 提议变更

### Task 2: B4 Aux Loss 收尾 (1 文件修改)

#### 文件: `src/losses620.py`

**变更 A: aux loss 计算 (L813 之后, entropy_loss 块之后)**

在 `entropy_loss` 块 (L810-813) 之后插入 B4 load balancing aux loss 计算:

```python
# === FC-SB Phase 4 B4: Fiber-MoE Load Balancing ===
# 理论: MoE router 需负载均衡以避免 expert 坍缩 (所有样本路由到同一 expert).
# aux_loss = -H(p) = sum(p_i * log(p_i)), 最大化熵 = 鼓励均匀分布.
# 注意: B4 MoE 当前位于 integrate_transport (推理路径), 训练时 probs 可能未设置.
#       此时 aux_loss = 0 (no-op). 未来将 MoE 移至训练路径后自动激活.
b4_moe_aux_loss = content.new_tensor(0.0)
b4_router_probs = getattr(model, "last_debug", {}).get("b4_moe_router_probs")
if b4_router_probs is not None and self.fiber_moe_load_balance_weight > 0.0:
    avg_probs = b4_router_probs.mean(dim=0)  # (num_experts,)
    # 熵: H(p) = -sum(p_i * log(p_i)); aux_loss = -H(p) (最小化 = 最大化熵)
    b4_moe_aux_loss = (avg_probs * torch.log(avg_probs + 1e-8)).sum()
    loss = loss + self.fiber_moe_load_balance_weight * b4_moe_aux_loss
```

**变更 B: metric 上报 (L911 metrics dict 内)**

在 metrics dict 的 `"cfg_dropout_prob"` 行之后 (L909 附近) 添加:

```python
"loss_b4_moe_load_balance": b4_moe_aux_loss.detach() if torch.is_tensor(b4_moe_aux_loss) else content.new_tensor(float(b4_moe_aux_loss)),
"b4_moe_router_entropy": content.new_tensor(float(getattr(model, "last_debug", {}).get("b4_moe_router_entropy", 0.0))),
"b4_moe_router_max_prob": content.new_tensor(float(getattr(model, "last_debug", {}).get("b4_moe_router_max_prob", 0.0))),
```

### Task 3: B2 原生频域 ODE POC (3 新文件 + 3 修改 + 1 配置)

#### 新文件 1: `src/spectral620.py` — Haar 小波工具

精确 Haar DWT/IDWT (正交, 完美重建), 支持多级分解:

```python
"""FC-SB Phase 4 B2: Native Spectral ODE — Haar wavelet utilities.

精确 Haar DWT/IDWT (正交变换, 完美重建). 与 model620.py 内的近似 haar_inv 不同,
此处使用标准 Haar 矩阵 [1,1;1,-1]/sqrt(2) 实现, 保证 IDWT(DWT(x)) = x.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def dwt2_haar(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """单级 2D Haar DWT. 输入 (B,C,H,W) -> (LL, LH, HL, HH), 每个 (B,C,H/2,W/2).
    要求 H,W 为偶数. 若为奇数, 先 pad.
    """
    B, C, H, W = x.shape
    # Pad to even if needed (replicate pad)
    if H % 2 != 0:
        x = F.pad(x, (0, 0, 0, 1), mode="replicate")
    if W % 2 != 0:
        x = F.pad(x, (0, 1, 0, 0), mode="replicate")
    x = x.float()
    # Split into 2x2 blocks
    x_reshaped = x.reshape(B, C, H // 2, 2, W // 2, 2)
    # 4 sub-blocks
    a = x_reshaped[:, :, :, 0, :, 0]  # top-left
    b = x_reshaped[:, :, :, 0, :, 1]  # top-right
    c = x_reshaped[:, :, :, 1, :, 0]  # bottom-left
    d = x_reshaped[:, :, :, 1, :, 1]  # bottom-right
    # Haar coefficients (1/sqrt(2) normalization for orthonormality)
    inv_sqrt2 = 0.7071067811865476
    LL = (a + b + c + d) * inv_sqrt2 * inv_sqrt2
    LH = (a + b - c - d) * inv_sqrt2 * inv_sqrt2  # horizontal low, vertical high
    HL = (a - b + c - d) * inv_sqrt2 * inv_sqrt2  # horizontal high, vertical low
    HH = (a - b - c + d) * inv_sqrt2 * inv_sqrt2
    return LL.to(dtype=x.dtype), LH.to(dtype=x.dtype), HL.to(dtype=x.dtype), HH.to(dtype=x.dtype)


def idwt2_haar(
    ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor
) -> torch.Tensor:
    """单级 2D Haar IDWT (精确逆变换). 输入 4 个 (B,C,H/2,W/2) -> (B,C,H,W)."""
    inv_sqrt2 = 0.7071067811865476
    ll, lh, hl, hh = ll.float(), lh.float(), hl.float(), hh.float()
    # Inverse Haar: a = (LL+LH+HL+HH)/(2), b = (LL+LH-HL-HH)/(2), etc.
    # Since forward used 1/2 * (sum), inverse uses 1/2 * (sum) back (orthonormal)
    a = (ll + lh + hl + hh) * inv_sqrt2 * inv_sqrt2
    b = (ll + lh - hl - hh) * inv_sqrt2 * inv_sqrt2
    c = (ll - lh + hl - hh) * inv_sqrt2 * inv_sqrt2
    d = (ll - lh - hl + hh) * inv_sqrt2 * inv_sqrt2
    B, C, H2, W2 = a.shape
    H, W = H2 * 2, W2 * 2
    out = torch.zeros(B, C, H, W, device=a.device, dtype=a.dtype)
    out[:, :, 0::2, 0::2] = a
    out[:, :, 0::2, 1::2] = b
    out[:, :, 1::2, 0::2] = c
    out[:, :, 1::2, 1::2] = d
    return out


def dwt2_multi_level(x: torch.Tensor, levels: int = 1):
    """多级 Haar DWT. 返回 [LL_n, list of (LH_i, HL_i, HH_i) for i=1..n]."""
    ll = x
    details = []
    for _ in range(max(1, levels)):
        ll, lh, hl, hh = dwt2_haar(ll)
        details.append((lh, hl, hh))
    return ll, details  # ll is coarsest, details[0] is finest
```

#### 新文件 2: `src/spectral_bridge620.py` — SpectralODEBridge620 模型

```python
"""FC-SB Phase 4 B2: Native Spectral ODE Bridge.

理论(用户方案): 在频域原生求解 ODE, 而非欧氏空间事后投影.
- 输入 latent -> DWT -> 4 子带 (LL, LH, HL, HH)
- 共享 backbone 处理 4 子带 (stacked 4*latent_channels)
- 4 个独立输出头预测 4 个速度场 (v_LL, v_LH, v_HL, v_HH)
- 训练: 4 个独立 FM loss, w_LL≈0, w_HH 大
- 推理: 4 路独立 Euler 积分 -> iDWT 合成

POC 设计: 单级 Haar, 共享 backbone (参数高效), 4 输出头.
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
    """单子带速度头: dim -> latent_channels, zero-init conv."""
    def __init__(self, dim: int, latent_channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.act = nn.SiLU()
        self.conv = nn.Conv2d(dim, latent_channels, kernel_size=3, padding=1)
        nn.init.normal_(self.conv.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.conv.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.conv(self.act(self.norm(h)))


class SpectralODEBridge620(nn.Module):
    """Native Spectral ODE Bridge with shared backbone + 4 velocity heads."""

    def __init__(self, model_cfg: ModelConfig, bridge_cfg: BridgeConfig | None = None) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.bridge_cfg = bridge_cfg
        self.latent_channels = int(model_cfg.latent_channels)
        self.num_styles = int(model_cfg.num_styles)
        self.dim = int(model_cfg.base_dim)
        self.time_dim = int(getattr(model_cfg, "time_dim", self.dim))
        self.dino_dim = int(getattr(model_cfg, "tokenizer_dino_dim", 384))
        self.spectral_levels = max(1, int(getattr(model_cfg, "spectral_ode_levels", 1)))
        # POC: only single-level supported; multi-level reserved for future.
        if self.spectral_levels != 1:
            raise NotImplementedError("spectral_ode_levels > 1 not supported in POC")

        # Style conditioner (reuse existing)
        self.style_conditioner = StyleConditioner620(
            dino_dim=self.dino_dim,
            model_dim=self.dim,
            num_styles=self.num_styles,
            num_memory_tokens=256,
            adapter_enabled=bool(getattr(model_cfg, "style_dino_adapter_enabled", False)),
            adapter_hidden_dim=int(getattr(model_cfg, "style_dino_adapter_hidden_dim", 1024)),
            adapter_scale=float(getattr(model_cfg, "style_dino_adapter_scale", 0.25)),
            local_cnn_enabled=bool(getattr(model_cfg, "style_local_cnn_enabled", False)),
            text_enabled=bool(getattr(model_cfg, "style_text_enabled", False)),
            text_dim=int(getattr(model_cfg, "style_text_dim", 768)),
            text_max_length=int(getattr(model_cfg, "style_text_max_length", 77)),
            text_dropout_prob=float(getattr(model_cfg, "style_text_dropout_prob", 0.15)),
            image_dropout_prob=float(getattr(model_cfg, "style_image_dropout_prob", 0.15)),
            text_null_std=float(getattr(model_cfg, "style_text_null_token_init_std", 0.02)),
            image_null_std=float(getattr(model_cfg, "style_image_null_token_init_std", 0.02)),
        )

        # Input projection: 4 subbands stacked -> dim channels
        # Subbands are (B, C, H/2, W/2) each; stack along channel -> (B, 4C, H/2, W/2)
        self.input_proj = nn.Conv2d(self.latent_channels * 4, self.dim, kernel_size=3, padding=1)
        self.time_proj = nn.Sequential(
            nn.Linear(self.time_dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )

        # Backbone blocks (reuse SpatialBridgeBlock620)
        depth = max(1, int(getattr(model_cfg, "num_res_blocks", 4)))
        heads = max(1, int(getattr(model_cfg, "style_attn_num_heads", 4)))
        gate_init = float(getattr(model_cfg, "style_cross_attn_gate_init", 0.3))
        self.blocks = nn.ModuleList([
            SpatialBridgeBlock620(
                dim=self.dim, num_heads=heads, style_gate_init=gate_init,
                layer_idx=idx, num_layers=depth, dino_dim=self.dino_dim,
                film_enabled=bool(getattr(model_cfg, "style_film_enabled", False)),
                film_init_std=float(getattr(model_cfg, "style_film_init_std", 0.02)),
            )
            for idx in range(depth)
        ])

        # 4 independent velocity heads (LL, LH, HL, HH)
        self.head_ll = SpectralVelocityHead(self.dim, self.latent_channels)
        self.head_lh = SpectralVelocityHead(self.dim, self.latent_channels)
        self.head_hl = SpectralVelocityHead(self.dim, self.latent_channels)
        self.head_hh = SpectralVelocityHead(self.dim, self.latent_channels)

        self.last_debug: dict = {}
        self.last_cross_attn_entropy = torch.tensor(0.0)

    def _resolve_t(self, x: torch.Tensor, t: torch.Tensor | float | None) -> torch.Tensor:
        if t is None:
            return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        if isinstance(t, (int, float)):
            return torch.full((x.shape[0],), float(t), device=x.device, dtype=x.dtype)
        return t.to(device=x.device, dtype=x.dtype)

    def forward(
        self,
        x: torch.Tensor,
        source: torch.Tensor | None = None,
        t: torch.Tensor | float | None = None,
        style_id: torch.Tensor | int | None = None,
        style_dino_patches: torch.Tensor | None = None,
        style_dino_cls: torch.Tensor | None = None,
        content_dino_patches: torch.Tensor | None = None,
        style_latent: torch.Tensor | None = None,
        style_text_tokens: torch.Tensor | None = None,
        target_latent: torch.Tensor | None = None,
        velocity_scale: float = 1.0,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        """Returns dict with 4 velocities: {'ll': v_ll, 'lh': v_lh, 'hl': v_hl, 'hh': v_hh}."""
        t_tensor = self._resolve_t(x, t)
        # DWT
        ll, lh, hl, hh = dwt2_haar(x)
        # Stack 4 subbands along channel dim
        stacked = torch.cat([ll, lh, hl, hh], dim=1)  # (B, 4C, H/2, W/2)
        # Style
        style_tokens, style_global = self.style_conditioner(
            style_dino_patches=style_dino_patches, style_dino_cls=style_dino_cls,
            style_id=style_id, batch=x.shape[0], device=x.device, dtype=x.dtype,
            style_latent=style_latent, style_text_tokens=style_text_tokens,
        )
        time_emb = self.time_proj(sinusoidal_time_embedding_620(t_tensor, self.time_dim).to(device=x.device, dtype=x.dtype))
        h = self.input_proj(stacked)
        total_entropy = []
        for block in self.blocks:
            h = block(h, time_emb=time_emb, style_tokens=style_tokens, style_global=style_global, content_dino_patches=content_dino_patches)
            total_entropy.append(block.cross_attn_entropy)
        if total_entropy:
            self.last_cross_attn_entropy = torch.stack(total_entropy).mean()
        # 4 velocity heads
        v_ll = self.head_ll(h)
        v_lh = self.head_lh(h)
        v_hl = self.head_hl(h)
        v_hh = self.head_hh(h)
        if velocity_scale != 1.0:
            v_ll = v_ll * velocity_scale
            v_lh = v_lh * velocity_scale
            v_hl = v_hl * velocity_scale
            v_hh = v_hh * velocity_scale
        self.last_debug = {
            "v_ll_abs": v_ll.detach().float().abs().mean(),
            "v_lh_abs": v_lh.detach().float().abs().mean(),
            "v_hl_abs": v_hl.detach().float().abs().mean(),
            "v_hh_abs": v_hh.detach().float().abs().mean(),
        }
        return {"ll": v_ll, "lh": v_lh, "hl": v_hl, "hh": v_hh}

    @torch.no_grad()
    def integrate_transport(
        self,
        x: torch.Tensor,
        style_id: torch.Tensor | int | None,
        num_steps: int = 8,
        step_size: float = 1.0,
        style_dino_patches: torch.Tensor | None = None,
        style_dino_cls: torch.Tensor | None = None,
        style_text_tokens: torch.Tensor | None = None,
        style_latent: torch.Tensor | None = None,
        target_style_latent: torch.Tensor | None = None,
        source_style_latent: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        """Spectral-domain Euler integration: 4 independent integrations + iDWT."""
        if style_latent is None and target_style_latent is not None and not isinstance(target_style_latent, dict):
            style_latent = target_style_latent
        steps = max(1, int(num_steps))
        horizon = max(0.0, float(step_size))
        if horizon <= 0.0:
            return x
        import math
        h = x
        dt = horizon / steps
        for i in range(steps):
            t_curr = float(i) / steps * horizon
            t_batch = torch.full((h.shape[0],), t_curr, device=h.device, dtype=h.dtype)
            v_dict = self.forward(h, t=t_batch, style_id=style_id, style_dino_patches=style_dino_patches,
                                   style_dino_cls=style_dino_cls, style_text_tokens=style_text_tokens,
                                   style_latent=style_latent)
            # Spectral Euler: integrate each subband independently
            ll, lh, hl, hh = dwt2_haar(h)
            ll = ll + v_dict["ll"] * dt
            lh = lh + v_dict["lh"] * dt
            hl = hl + v_dict["hl"] * dt
            hh = hh + v_dict["hh"] * dt
            h = idwt2_haar(ll, lh, hl, hh)
        return h

    def integrate(self, x: torch.Tensor, style_id: torch.Tensor | int | None, num_steps: int = 8, **kwargs: object) -> torch.Tensor:
        return self.integrate_transport(x, style_id, num_steps=num_steps, **kwargs)


def build_spectral_ode_bridge_from_config(
    model_cfg: ModelConfig, *, bridge_cfg: BridgeConfig | None = None, use_checkpointing: bool = False
) -> SpectralODEBridge620:
    del use_checkpointing
    return SpectralODEBridge620(model_cfg, bridge_cfg=bridge_cfg)
```

#### 新文件 3: `src/spectral_losses620.py` — SpectralODEObjective620

```python
"""FC-SB Phase 4 B2: Spectral ODE training objective.

4 个独立 FM loss (per-subband), 权重 w_ll/w_lh/w_wl/w_hh.
理论: w_ll≈0 (锁死低频保 LPIPS), w_hh 大 (压资源到高频笔触).
"""
from __future__ import annotations
from typing import Dict
import torch
import torch.nn.functional as F
from config_schema import ExperimentConfig
from spectral620 import dwt2_haar


class SpectralODEObjective620:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.training = True
        self.bridge_cfg = config.bridge
        self.t_min = float(getattr(self.bridge_cfg, "t_min", 0.0))
        self.t_max = float(getattr(self.bridge_cfg, "t_max", 1.0))
        # Per-subband FM weights
        self.w_ll = float(getattr(self.bridge_cfg, "spectral_w_ll", 0.0))
        self.w_lh = float(getattr(self.bridge_cfg, "spectral_w_lh", 1.0))
        self.w_hl = float(getattr(self.bridge_cfg, "spectral_w_hl", 1.0))
        self.w_hh = float(getattr(self.bridge_cfg, "spectral_w_hh", 2.0))
        self.loss_type = str(getattr(self.bridge_cfg, "loss_type", "mse")).lower().strip()
        self.last_debug: dict = {}

    def _sample_t(self, content: torch.Tensor) -> torch.Tensor:
        lo = max(0.0, min(1.0, self.t_min))
        hi = max(lo + 1e-4, min(1.0, self.t_max))
        return torch.rand(content.shape[0], device=content.device, dtype=content.dtype) * (hi - lo) + lo

    def _fm_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type in ("huber", "smooth_l1", "smoothl1"):
            return F.smooth_l1_loss(pred.float(), target.float())
        return F.mse_loss(pred.float(), target.float())

    def compute(self, model, *, content, target, source=None, style_id=None,
                style_dino_patches=None, style_dino_cls=None, content_dino_patches=None,
                style_latent=None, style_text_tokens=None, target_latent=None, **_) -> Dict[str, torch.Tensor]:
        # Sample t
        t = self._sample_t(content)
        # Forward bridge in content space: x_t = (1-t)*content + t*target + noise*sigma*sqrt(t*(1-t))
        # For POC: linear interpolation (no Brownian noise)
        t_view = t.view(-1, 1, 1, 1).to(dtype=content.dtype)
        x_t = (1.0 - t_view) * content + t_view * target
        # Target velocity per subband
        target_ll, target_lh, target_hl, target_hh = dwt2_haar(target - content)
        # Predict velocities
        v_dict = model(x_t, source=source, t=t, style_id=style_id,
                       style_dino_patches=style_dino_patches, style_dino_cls=style_dino_cls,
                       content_dino_patches=content_dino_patches, style_latent=style_latent,
                       style_text_tokens=style_text_tokens, target_latent=target_latent)
        # Per-subband losses
        loss_ll = self._fm_loss(v_dict["ll"], target_ll)
        loss_lh = self._fm_loss(v_dict["lh"], target_lh)
        loss_hl = self._fm_loss(v_dict["hl"], target_hl)
        loss_hh = self._fm_loss(v_dict["hh"], target_hh)
        loss = self.w_ll * loss_ll + self.w_lh * loss_lh + self.w_hl * loss_hl + self.w_hh * loss_hh
        zero = content.new_tensor(0.0)
        metrics = {
            "loss": loss,
            "loss_fm_spectral_ll": loss_ll.detach(),
            "loss_fm_spectral_lh": loss_lh.detach(),
            "loss_fm_spectral_hl": loss_hl.detach(),
            "loss_fm_spectral_hh": loss_hh.detach(),
            "loss_fm_total": loss.detach(),
            "t_mean": t.detach().float().mean(),
            "loss_type": content.new_tensor(1.0 if self.loss_type in ("huber", "smooth_l1", "smoothl1") else 0.0),
            # placeholders for compatibility
            "flow": loss.detach(),
            "loss_fm": loss.detach(),
            "loss_swd_ss": zero,
            "loss_edge_ss": zero,
            "loss_endpoint_lowfreq": zero,
            "loss_endpoint_content": zero,
            "loss_endpoint_style": zero,
            "loss_endpoint_vel_reg": zero,
            "training_objective_mode": content.new_tensor(0.0),
            "loss_source_endpoint_aux": zero,
            "loss_endpoint_energy_band": zero,
            "loss_style_strength_reg": zero,
            "style_strength_alpha": zero,
            "loss_attn_entropy": zero,
            "single_step_swd": zero,
            "single_step_edge": zero,
            "endpoint_lowfreq": zero,
            "source_endpoint_aux": zero,
            "endpoint_energy_band": zero,
            "terminal_swd": zero,
            "ot_cost": zero,
            "ot_plan_entropy": zero,
            "ot_target_gini": zero,
            "velocity_abs": zero,
            "target_velocity_abs": zero,
            "endpoint_abs": zero,
            "base_structural_drift": zero,
            "endpoint_low_to_source": zero,
            "endpoint_low_to_target": zero,
            "endpoint_high_to_target": zero,
            "endpoint_low_target_ratio": zero,
            "low_freq_leak": zero,
            "fiber_energy_ratio": zero,
            "target_base_shift": zero,
            "bridge_sigma": zero,
            "swd_noise_sigma": zero,
            "style_dino_active": content.new_tensor(1.0 if style_dino_patches is not None else 0.0),
            "style_gate_value": zero,
            "cross_attn_entropy": zero,
            "cross_attn_delta_abs": zero,
            "endpoint_head_mode_lowhigh": zero,
            "endpoint_pred_abs_debug": zero,
            "endpoint_low_abs_debug": zero,
            "endpoint_high_abs_debug": zero,
            "endpoint_style_low_abs_debug": zero,
            "endpoint_style_high_abs_debug": zero,
            "loss_contrast_preserve": zero,
            "loss_channel_variance": zero,
            "loss_hf_energy": zero,
            "anti_whiten_total": zero,
            "gen_global_std": zero,
            "target_global_std": zero,
            "gen_hf_energy": zero,
            "target_hf_energy": zero,
            "loss_velocity_magnitude": zero,
            "v_pred_norm": zero,
            "v_target_norm": zero,
            "velocity_ratio": zero,
            "loss_pixel_color_match": zero,
            "gen_per_ch_mean": zero,
            "tgt_per_ch_mean": zero,
            "gen_per_ch_std": zero,
            "tgt_per_ch_std": zero,
            "loss_saturation_proxy": zero,
            "gen_ch_var_max_ratio": zero,
            "flow_scaled_weight": content.new_tensor(1.0),
            "loss_style_contrastive": zero,
            "loss_fiber_repulsion": zero,
            "loss_anti_input": zero,
            "loss_style_disc": zero,
            "loss_output_variance": zero,
            "loss_b4_moe_load_balance": zero,
            "b4_moe_router_entropy": zero,
            "b4_moe_router_max_prob": zero,
            "style_cross_sim_mean": zero,
            "loss_directional_cosine": zero,
            "clip_dir": zero,
            "clip_fm_low": zero,
            "cfg_uncond_active": zero,
            "cfg_dropout_prob": zero,
        }
        self.last_debug = {"x_t": x_t.detach(), "target": target.detach()}
        return metrics

    def update_weights_for_epoch(self, epoch: int, num_epochs: int = 3) -> dict:
        return {"stage": 0, "bridge_sigma": 0.0,
                "w_endpoint_content": 0.0, "w_endpoint_style": 0.0, "w_style_strength_reg": 0.0}

    def compute_debug(self, model, **kwargs) -> Dict[str, Dict[str, torch.Tensor]]:
        return {"metrics": self.compute(model, **kwargs), "components": {}, "state": dict(self.last_debug)}
```

#### 修改 1: `src/config_schema.py`

**ModelConfig 新增字段** (在 B4 字段之后, L502 附近):

```python
# === FC-SB Phase 4 B2: Native Spectral ODE ===
spectral_ode_enabled: bool = False          # 总开关: 启用原生频域 ODE bridge
spectral_ode_levels: int = 1                # Haar DWT 级数 (POC 仅支持 1)
```

**BridgeConfig 新增字段** (在 B4 字段之后, L626 附近):

```python
# === FC-SB Phase 4 B2: Spectral ODE per-subband FM weights ===
spectral_w_ll: float = 0.0                  # 低频速度 loss 权重 (0=锁死低频保 LPIPS)
spectral_w_lh: float = 1.0                  # 水平低/垂直高 频带权重
spectral_w_hl: float = 1.0                  # 水平高/垂直低 频带权重
spectral_w_hh: float = 2.0                  # 全高频 (笔触) 权重, 最大
```

**Contract validator skip** (L998-999): 修改条件, 让 `620_spectral_ode` 也跳过:

```python
if contract_family not in ("620_spatial_bridge", "620_spectral_ode"):
    validate_i2sb_contract(...)
```

#### 修改 2: `src/model.py` L2449 dispatch

```python
if str(getattr(config, "contract_family", "legacy") or "legacy").strip().lower() == "620_spatial_bridge":
    from model620 import build_spatial_bridge620_from_config
    return build_spatial_bridge620_from_config(config, bridge_cfg=bridge_cfg, use_checkpointing=use_checkpointing)
elif str(getattr(config, "contract_family", "legacy") or "legacy").strip().lower() == "620_spectral_ode":
    from spectral_bridge620 import build_spectral_ode_bridge_from_config
    return build_spectral_ode_bridge_from_config(config, bridge_cfg=bridge_cfg, use_checkpointing=use_checkpointing)
```

#### 修改 3: `src/trainer.py` L236-239 dispatch

```python
contract_family = str(getattr(config.model, "contract_family", "legacy") or "legacy").strip().lower()
if contract_family in ("620_spatial_bridge", "620_spectral_ode") and self.distill_enabled:
    raise ValueError(f"{contract_family} does not support legacy distillation; disable training.distill.enabled.")
if contract_family == "620_spatial_bridge":
    self.loss_fn = SpatialBridgeObjective620(config)
elif contract_family == "620_spectral_ode":
    from spectral_losses620 import SpectralODEObjective620
    self.loss_fn = SpectralODEObjective620(config)
else:
    self.loss_fn = OTFlowMatchingObjective(config)
```

#### 新配置: `configs/620_spectral_poc.json`

基于 `620_spatial_bridge_base.json` 修改:
- `contract_family`: `"620_spectral_ode"`
- `spectral_ode_enabled`: true
- `bridge` 段: 添加 `spectral_w_ll/lh/hl/hh`
- `training.batch_size`: 24 (per project 12GB VRAM constraint)
- `checkpoint.save_dir` 和 `ablation.name`: `620_spectral_poc`
- 其余 (data, full_eval) 保持与 base 一致

## 假设与决策

1. **B4 aux loss graceful None**: 训练时 `b4_moe_router_probs` 为 None (MoE 在推理路径), aux loss = 0。这是已批准计划的行为, 本轮不修复训练路径 (避免范围蔓延)。
2. **B2 POC 单级 Haar**: `spectral_ode_levels=1`, 多级预留 `NotImplementedError`。
3. **B2 共享 backbone**: 4 子带 stack 后输入共享 backbone, 4 个独立 head 输出。参数高效, 允许跨子带信息共享。
4. **B2 POC 无 Brownian noise**: x_t = (1-t)*content + t*target 线性插值。Brownian noise 留待 POC 验证后添加。
5. **B2 metrics 兼容性**: spectral_losses620.py 返回所有 SpatialBridgeObjective620 的 metric key (用 zero 占位), 确保训练循环 / 日志不报 KeyError。
6. **B2 inference 路径**: `integrate_transport` 实现 4 路独立 Euler + iDWT 合成, 无 BASE LOCKING (频域已天然解耦)。
7. **dispatch 不破坏现有**: `620_spectral_ode` 走独立 if 分支, 不影响 `620_spatial_bridge` 与 legacy 路径。

## 验证步骤

### Task 2 验证
1. `python -c "import ast; ast.parse(open('src/losses620.py').read())"` — 语法检查
2. 确认 `loss_b4_moe_load_balance` metric 在 metrics dict 中
3. 确认 aux loss 在 `b4_moe_router_probs is None` 时为 0 (no-op)

### Task 3 验证
1. `python -c "import ast; ast.parse(open('src/spectral620.py').read())"` — 语法检查
2. `python -c "import ast; ast.parse(open('src/spectral_bridge620.py').read())"` — 语法检查
3. `python -c "import ast; ast.parse(open('src/spectral_losses620.py').read())"` — 语法检查
4. `python -c "import ast; ast.parse(open('src/config_schema.py').read())"` — 语法检查
5. `python -c "import ast; ast.parse(open('src/model.py').read())"` — 语法检查
6. `python -c "import ast; ast.parse(open('src/trainer.py').read())"` — 语法检查
7. Haar 完美重建测试: `python -c "import torch; from spectral620 import dwt2_haar, idwt2_haar; x=torch.randn(2,4,32,32); ll,lh,hl,hh=dwt2_haar(x); x_rec=idwt2_haar(ll,lh,hl,hh); print('recon_err', (x-x_rec).abs().max().item())"` — 误差应 < 1e-6
8. 配置加载测试: `python -c "from config_schema import ExperimentConfig; cfg=ExperimentConfig.load('configs/620_spectral_poc.json'); print('contract', cfg.model.contract_family, 'w_hh', cfg.bridge.spectral_w_hh)"`

### 最终验证
- 所有修改/新建文件通过 ast.parse
- Haar 完美重建 < 1e-6
- POC 配置可加载且字段正确
