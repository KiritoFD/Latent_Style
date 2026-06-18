from __future__ import annotations

import torch
from torch import nn

from blocks620 import SpatialBridgeBlock620, sinusoidal_time_embedding_620
from config_schema import BridgeConfig, ModelConfig
from style_encoder620 import StyleConditioner620


class SpatialBridge620(nn.Module):
    """A new 620 bridge: velocity training, DINO-token style path, I2SB inference."""

    def __init__(self, model_cfg: ModelConfig, bridge_cfg: BridgeConfig | None = None) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.bridge_cfg = bridge_cfg
        self.latent_channels = int(model_cfg.latent_channels)
        self.num_styles = int(model_cfg.num_styles)
        self.dim = int(model_cfg.base_dim)
        self.time_dim = int(getattr(model_cfg, "time_dim", self.dim))
        self.dino_dim = int(getattr(model_cfg, "tokenizer_dino_dim", 384))
        self.solver_family = str(getattr(model_cfg, "solver_family", "solver_i2sb"))
        self.transport_prediction_mode = str(getattr(model_cfg, "transport_prediction_mode", "velocity"))
        self.bridge_sigma = float(getattr(bridge_cfg, "bridge_sigma", 0.02) if bridge_cfg is not None else 0.02)
        self.bridge_noise_schedule = str(getattr(bridge_cfg, "bridge_noise_schedule", "delayed") if bridge_cfg is not None else "delayed")
        self.bridge_noise_window_start = float(getattr(bridge_cfg, "bridge_noise_window_start", 0.18) if bridge_cfg is not None else 0.18)
        self.bridge_noise_window_end = float(getattr(bridge_cfg, "bridge_noise_window_end", 0.82) if bridge_cfg is not None else 0.82)
        self.style_conditioner = StyleConditioner620(
            dino_dim=self.dino_dim,
            model_dim=self.dim,
            num_styles=self.num_styles,
            num_memory_tokens=int(getattr(model_cfg, "style_attn_num_tokens", 256)),
            adapter_enabled=bool(getattr(model_cfg, "style_dino_adapter_enabled", False)),
            adapter_hidden_dim=int(getattr(model_cfg, "style_dino_adapter_hidden_dim", 1024)),
            adapter_scale=float(getattr(model_cfg, "style_dino_adapter_scale", 0.25)),
        )
        self.input_proj = nn.Conv2d(self.latent_channels, self.dim, kernel_size=3, padding=1)
        self.time_proj = nn.Sequential(
            nn.Linear(self.time_dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )
        depth = max(1, int(getattr(model_cfg, "num_res_blocks", 4)))
        heads = max(1, int(getattr(model_cfg, "style_attn_num_heads", 4)))
        gate_init = float(getattr(model_cfg, "style_cross_attn_gate_init", 0.05))
        self.blocks = nn.ModuleList(
            [SpatialBridgeBlock620(dim=self.dim, num_heads=heads, style_gate_init=gate_init) for _ in range(depth)]
        )
        self.out = nn.Sequential(
            nn.GroupNorm(1, self.dim),
            nn.SiLU(),
            nn.Conv2d(self.dim, self.latent_channels, kernel_size=3, padding=1),
        )
        nn.init.normal_(self.out[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.out[-1].bias)
        self.last_debug: dict[str, torch.Tensor] = {}

    def _resolve_t(self, x: torch.Tensor, t: torch.Tensor | float | None) -> torch.Tensor:
        if t is None:
            return torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
        if not torch.is_tensor(t):
            return torch.full((x.shape[0],), float(t), device=x.device, dtype=x.dtype)
        t = t.to(device=x.device, dtype=x.dtype).view(-1)
        if t.numel() == 1 and x.shape[0] > 1:
            t = t.expand(x.shape[0])
        if t.numel() != x.shape[0]:
            raise ValueError(f"time batch mismatch: expected {x.shape[0]}, got {t.numel()}")
        return t

    def forward(
        self,
        x: torch.Tensor,
        source: torch.Tensor | None = None,
        t: torch.Tensor | float | None = None,
        style_id: torch.Tensor | int | None = None,
        style_dino_patches: torch.Tensor | None = None,
        style_dino_cls: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        del source
        t_tensor = self._resolve_t(x, t)
        style_tokens, _style_global = self.style_conditioner(
            style_dino_patches=style_dino_patches,
            style_dino_cls=style_dino_cls,
            style_id=style_id,
            batch=x.shape[0],
            device=x.device,
            dtype=x.dtype,
        )
        time_emb = self.time_proj(sinusoidal_time_embedding_620(t_tensor, self.time_dim).to(device=x.device, dtype=x.dtype))
        h = self.input_proj(x)
        debug_vals: dict[str, list[torch.Tensor]] = {}
        for block in self.blocks:
            h = block(h, time_emb=time_emb, style_tokens=style_tokens)
            for key, value in block.last_debug.items():
                debug_vals.setdefault(key, []).append(value)
        velocity = self.out(h)
        self.last_debug = {
            key: torch.stack([v.to(device=x.device).float() for v in values]).mean()
            for key, values in debug_vals.items()
            if values
        }
        self.last_debug["velocity_abs"] = velocity.detach().float().abs().mean()
        self.last_debug["style_dino_active"] = x.new_tensor(1.0 if style_dino_patches is not None else 0.0)
        return velocity

    def predict_endpoint(
        self,
        x: torch.Tensor,
        *,
        t: torch.Tensor | float | None = None,
        style_id: torch.Tensor | int | None = None,
        style_dino_patches: torch.Tensor | None = None,
        style_dino_cls: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t_tensor = self._resolve_t(x, t)
        v = self.forward(x, t=t_tensor, style_id=style_id, style_dino_patches=style_dino_patches, style_dino_cls=style_dino_cls)
        return x + (1.0 - t_tensor).view(-1, 1, 1, 1).to(dtype=x.dtype) * v

    def predict_transport_base(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        return self.predict_endpoint(x, **kwargs)

    @torch.no_grad()
    def integrate_transport(
        self,
        x: torch.Tensor,
        style_id: torch.Tensor | int | None,
        num_steps: int = 8,
        step_size: float = 1.0,
        style_dino_patches: torch.Tensor | None = None,
        style_dino_cls: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        steps = max(1, int(num_steps))
        horizon = max(0.0, float(step_size))
        if horizon <= 0.0:
            return x
        h = x
        for idx in range(steps):
            t_curr = horizon * (idx / float(steps))
            t_next = horizon * ((idx + 1) / float(steps))
            t_batch = torch.full((h.shape[0],), t_curr, device=h.device, dtype=h.dtype)
            endpoint = self.predict_endpoint(
                h,
                t=t_batch,
                style_id=style_id,
                style_dino_patches=style_dino_patches,
                style_dino_cls=style_dino_cls,
            )
            denom = max(1e-6, 1.0 - t_curr)
            c_curr = (1.0 - t_next) / denom
            c_tgt = (t_next - t_curr) / denom
            mean = c_curr * h + c_tgt * endpoint
            var = (self.bridge_sigma ** 2) * ((t_next - t_curr) * max(0.0, 1.0 - t_next) / denom)
            if var > 0.0:
                mean = mean + torch.randn_like(mean) * (var ** 0.5)
            h = mean
        return h

    @torch.no_grad()
    def integrate(self, x: torch.Tensor, style_id: torch.Tensor | int | None, num_steps: int = 8, **kwargs: object) -> torch.Tensor:
        return self.integrate_transport(x, style_id, num_steps=num_steps, **kwargs)


def build_spatial_bridge620_from_config(
    model_cfg: ModelConfig,
    *,
    bridge_cfg: BridgeConfig | None = None,
    use_checkpointing: bool = False,
) -> SpatialBridge620:
    del use_checkpointing
    return SpatialBridge620(model_cfg, bridge_cfg=bridge_cfg)
