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

        # Style conditioner (DINO patches -> bridge width)
        # 630 Phase 2: propagate mask config (The Blindfolded Tokenizer)
        mask_ratio = float(getattr(model_cfg, "style_mask_ratio", 0.0))
        mask_mode = str(getattr(model_cfg, "style_mask_mode", "none"))
        self.style_conditioner = StyleConditioner620(
            dino_dim=self.dino_dim,
            model_dim=self.dim,
            num_styles=self.num_styles,
            num_memory_tokens=256,
            mask_ratio=mask_ratio,
            mask_mode=mask_mode,
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
        gate_mode = str(getattr(model_cfg, "style_gate_mode", "tanh_gate"))
        attn_mode = str(getattr(model_cfg, "style_attn_mode", "softmax"))
        attn_temperature = float(getattr(model_cfg, "style_attn_temperature", 1.0))
        shortcut_alpha = getattr(model_cfg, "style_shortcut_alpha", 1.0)
        norm_type = str(getattr(model_cfg, "body_norm_type", "group_norm"))
        self.blocks = nn.ModuleList([
            SpatialBridgeBlock620(
                dim=self.dim, num_heads=heads, style_gate_init=gate_init,
                style_gate_mode=gate_mode, style_shortcut_alpha=shortcut_alpha,
                layer_idx=idx, num_layers=depth,
                attn_mode=attn_mode, attn_temperature=attn_temperature,
                norm_type=norm_type,
            )
            for idx in range(depth)
        ])

        # 3 independent velocity heads (LL, LH, HL) — HH removed: 628 L8 confirmed DEAD
        self.head_ll = SpectralVelocityHead(self.dim, self.latent_channels)
        self.head_lh = SpectralVelocityHead(self.dim, self.latent_channels)
        self.head_hl = SpectralVelocityHead(self.dim, self.latent_channels)

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
        t: torch.Tensor | float | None = None,
        style_id: torch.Tensor | int | None = None,
        style_dino_patches: torch.Tensor | None = None,
        style_dino_cls: torch.Tensor | None = None,
        content_dino_patches: torch.Tensor | None = None,
        style_latent: torch.Tensor | None = None,
        style_text_tokens: torch.Tensor | None = None,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        """Returns dict with 3 velocities: {'ll': v_ll, 'lh': v_lh, 'hl': v_hl} (HH removed - 628 L8 DEAD)."""
        t_tensor = self._resolve_t(x, t)
        # Single-level Haar DWT (multi-level removed — 628/629 confirmed spectral_levels=1 is optimal)
        ll, lh, hl, hh = dwt2_haar(x)
        # Stack 4 subbands along channel dim (HH still decomposed for input, but no velocity head)
        stacked = torch.cat([ll, lh, hl, hh], dim=1)  # (B, 4C, H/2, W/2)
        # Style
        style_tokens, style_global = self.style_conditioner(
            style_dino_patches=style_dino_patches, style_dino_cls=style_dino_cls,
            style_id=style_id, batch=x.shape[0], device=x.device, dtype=x.dtype,
        )
        time_emb = self.time_proj(
            sinusoidal_time_embedding_620(t_tensor, self.time_dim).to(device=x.device, dtype=x.dtype)
        )
        h = self.input_proj(stacked)
        total_entropy = []
        for block in self.blocks:
            h = block(
                h, time_emb=time_emb, style_tokens=style_tokens,
                style_global=style_global, content_dino_patches=content_dino_patches,
            )
            total_entropy.append(block.cross_attn_entropy)
        if total_entropy:
            self.last_cross_attn_entropy = torch.stack(total_entropy).mean()
        # 3 velocity heads (HH removed: 628 L8 DEAD)
        v_ll = self.head_ll(h)
        v_lh = self.head_lh(h)
        v_hl = self.head_hl(h)
        self.last_debug = {
            "v_ll_abs": v_ll.detach().float().abs().mean(),
            "v_lh_abs": v_lh.detach().float().abs().mean(),
            "v_hl_abs": v_hl.detach().float().abs().mean(),
        }
        return {"ll": v_ll, "lh": v_lh, "hl": v_hl}

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
        **_: object,
    ) -> torch.Tensor:
        """Spectral-domain Euler integration + endpoint AdaIN.

        628/629 清理: WCT/multiband/patch/multi_level 死 hooks 已删除.
        保留活跃路径: DWT→Euler→iDWT→Endpoint AdaIN (full mode) + Style Extrap (simple scale).
        """
        if style_latent is None and target_style_latent is not None and not isinstance(target_style_latent, dict):
            style_latent = target_style_latent
        steps = max(1, int(num_steps))
        horizon = max(0.0, float(step_size))
        if horizon <= 0.0:
            return x

        # 活跃推理参数 (core_keep: D2 endpoint_adain, D3 style_extrap)
        mcfg = getattr(self, 'model_cfg', None)
        bcfg = getattr(self, 'bridge_cfg', None)
        def _cfg_get(key, default):
            if mcfg is not None and hasattr(mcfg, key):
                return getattr(mcfg, key)
            if bcfg is not None and hasattr(bcfg, key):
                return getattr(bcfg, key)
            return default
        endpoint_adain_scale = float(_cfg_get('endpoint_adain_scale', 0.0))
        style_extrap_alpha = float(_cfg_get('style_extrap_alpha', 0.0))

        def lp(y: torch.Tensor) -> torch.Tensor:
            """Haar DWT lowpass: LL 子带 IDWT 重建 (LH/HL/HH 置零). 正交, 无混叠."""
            ll_, _, _, _ = dwt2_haar(y.float())
            zero = torch.zeros_like(ll_)
            return idwt2_haar(ll_, zero, zero, zero).to(dtype=y.dtype)

        h = x
        dt = horizon / steps
        for i in range(steps):
            t_curr = float(i) / steps * horizon
            t_batch = torch.full((h.shape[0],), t_curr, device=h.device, dtype=h.dtype)
            v_dict = self.forward(
                h, t=t_batch, style_id=style_id, style_dino_patches=style_dino_patches,
                style_dino_cls=style_dino_cls, style_text_tokens=style_text_tokens,
                style_latent=style_latent,
            )
            # Spectral Euler: integrate LL/LH/HL independently (HH unchanged — 628 L8 DEAD)
            ll, lh, hl, hh = dwt2_haar(h)
            ll = ll + v_dict["ll"] * dt
            lh = lh + v_dict["lh"] * dt
            hl = hl + v_dict["hl"] * dt
            h = idwt2_haar(ll, lh, hl, hh)

            # Endpoint AdaIN: fiber 统计匹配 (core_keep D2)
            if endpoint_adain_scale > 0.0 and style_latent is not None and isinstance(style_latent, torch.Tensor):
                ep_base = lp(h)
                ep_fiber_curr = h - ep_base
                style_fiber = style_latent.to(dtype=h.dtype) - lp(style_latent.to(dtype=h.dtype))
                # Style extrapolation: fiber 高通分量均值≈0, 外推退化为缩放 (core_keep D3)
                if style_extrap_alpha > 0.0:
                    style_fiber = style_fiber * (1.0 + style_extrap_alpha)
                # 全局一阶统计匹配 (mean + std)
                B_c = ep_fiber_curr.shape[0]
                if style_fiber.shape[0] == 1 and B_c > 1:
                    target_mean = style_fiber.mean(dim=[2, 3], keepdim=True).expand(B_c, -1, 1, 1)
                    target_std = style_fiber.std(dim=[2, 3], keepdim=True).clamp_min(1e-6).expand(B_c, -1, 1, 1)
                else:
                    target_mean = style_fiber.mean(dim=[2, 3], keepdim=True)
                    target_std = style_fiber.std(dim=[2, 3], keepdim=True).clamp_min(1e-6)
                pred_mean = ep_fiber_curr.mean(dim=[2, 3], keepdim=True)
                pred_std = ep_fiber_curr.std(dim=[2, 3], keepdim=True).clamp_min(1e-6)
                ep_fiber_matched = (ep_fiber_curr - pred_mean) / pred_std * target_std + target_mean
                # α-blend: base 锁死保 LPIPS
                h = ep_base + (1.0 - endpoint_adain_scale) * ep_fiber_curr + endpoint_adain_scale * ep_fiber_matched
        return h

    def integrate(
        self, x: torch.Tensor, style_id: torch.Tensor | int | None, num_steps: int = 8, **kwargs: object
    ) -> torch.Tensor:
        return self.integrate_transport(x, style_id, num_steps=num_steps, **kwargs)


def build_spectral_ode_bridge_from_config(
    model_cfg: ModelConfig, *, bridge_cfg: BridgeConfig | None = None
) -> SpectralODEBridge620:
    return SpectralODEBridge620(model_cfg, bridge_cfg=bridge_cfg)
