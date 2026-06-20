from __future__ import annotations

import torch
from torch import nn

import torch.nn.functional as F

from blocks620 import SpatialBridgeBlock620, sinusoidal_time_embedding_620
from config_schema import BridgeConfig, ModelConfig
from style_encoder620 import StyleConditioner620


class FiLMEndpointHead(nn.Module):
    """Endpoint head with FiLM (style modulation inside the trunk).

    For feature maps h (dim channels) and style embedding s:
        FiLM(h; s) = (1 + gamma(s)) * h + beta(s)
    Then project from dim -> latent_channels via conv.
    Zero-init ensures identity at start.
    """

    def __init__(self, dim: int, latent_channels: int, style_dim: int, style_hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.film_proj = nn.Sequential(
            nn.LayerNorm(style_dim),
            nn.Linear(style_dim, style_hidden_dim),
            nn.SiLU(),
            nn.Linear(style_hidden_dim, dim * 2),
        )
        self.conv = nn.Conv2d(dim, latent_channels, kernel_size=3, padding=1)
        nn.init.normal_(self.conv.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.conv.bias)
        nn.init.zeros_(self.film_proj[-1].weight)
        nn.init.zeros_(self.film_proj[-1].bias)

    def forward(self, x: torch.Tensor, style_embed: torch.Tensor) -> torch.Tensor:
        film_params = self.film_proj(style_embed.float()).to(dtype=x.dtype)
        gamma, beta = film_params.chunk(2, dim=-1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        h = self.norm(x)
        h = (1.0 + gamma) * h + beta
        h = F.silu(h)
        return self.conv(h)


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
        self.endpoint_head_mode = str(getattr(model_cfg, "endpoint_head_mode", "velocity")).strip().lower()
        if self.endpoint_head_mode not in {"velocity", "endpoint_lowhigh"}:
            self.endpoint_head_mode = "velocity"
        self.endpoint_lowpass_kernel = max(1, int(getattr(model_cfg, "endpoint_lowpass_kernel", 5)))
        if self.endpoint_lowpass_kernel % 2 == 0:
            self.endpoint_lowpass_kernel += 1
        self.endpoint_high_scale = float(getattr(model_cfg, "endpoint_high_scale", 1.0))
        self.endpoint_velocity_floor = max(1e-3, float(getattr(model_cfg, "endpoint_velocity_floor", 0.05)))
        self.endpoint_style_hidden_dim = max(8, int(getattr(model_cfg, "endpoint_style_hidden_dim", 128)))
        self.endpoint_film_enabled = bool(getattr(model_cfg, "endpoint_film_enabled", False))
        self.bridge_sigma = float(getattr(bridge_cfg, "bridge_sigma", 0.02) if bridge_cfg is not None else 0.02)
        self.bridge_noise_schedule = str(getattr(bridge_cfg, "bridge_noise_schedule", "delayed") if bridge_cfg is not None else "delayed")
        self.bridge_noise_window_start = float(getattr(bridge_cfg, "bridge_noise_window_start", 0.18) if bridge_cfg is not None else 0.18)
        self.bridge_noise_window_end = float(getattr(bridge_cfg, "bridge_noise_window_end", 0.82) if bridge_cfg is not None else 0.82)
        style_local_cnn_enabled = bool(getattr(model_cfg, "style_local_cnn_enabled", False))
        self.style_text_enabled = bool(getattr(model_cfg, "style_text_enabled", False))
        self.style_text_dim = int(getattr(model_cfg, "style_text_dim", 768))
        self.style_conditioner = StyleConditioner620(
            dino_dim=self.dino_dim,
            model_dim=self.dim,
            num_styles=self.num_styles,
            num_memory_tokens=256,
            adapter_enabled=bool(getattr(model_cfg, "style_dino_adapter_enabled", False)),
            adapter_hidden_dim=int(getattr(model_cfg, "style_dino_adapter_hidden_dim", 1024)),
            adapter_scale=float(getattr(model_cfg, "style_dino_adapter_scale", 0.25)),
            local_cnn_enabled=style_local_cnn_enabled,
            text_enabled=self.style_text_enabled,
            text_dim=self.style_text_dim,
            text_max_length=int(getattr(model_cfg, "style_text_max_length", 77)),
            text_dropout_prob=float(getattr(model_cfg, "style_text_dropout_prob", 0.15)),
            image_dropout_prob=float(getattr(model_cfg, "style_image_dropout_prob", 0.15)),
            text_null_std=float(getattr(model_cfg, "style_text_null_token_init_std", 0.02)),
            image_null_std=float(getattr(model_cfg, "style_image_null_token_init_std", 0.02)),
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
        moe_enabled = bool(getattr(model_cfg, "style_moe_enabled", False))
        moe_num_experts = int(getattr(model_cfg, "style_moe_num_experts", 4))
        moe_router_hidden_dim = int(getattr(model_cfg, "style_moe_router_hidden_dim", 128))
        kv_content_routed = bool(getattr(model_cfg, "style_kv_moe_content_routed", False))
        shortcut_alpha = getattr(model_cfg, "style_shortcut_alpha", 1.0)
        query_source = str(getattr(model_cfg, "style_query_source", "concat"))
        skip_coarse = bool(getattr(model_cfg, "style_cross_attn_skip_coarse", False))
        attn_topk = int(getattr(model_cfg, "style_attn_topk", 0))

        self.blocks = nn.ModuleList(
            [
                SpatialBridgeBlock620(
                    dim=self.dim,
                    num_heads=heads,
                    style_gate_init=gate_init,
                    style_moe_enabled=moe_enabled,
                    style_moe_num_experts=moe_num_experts,
                    style_moe_router_hidden_dim=moe_router_hidden_dim,
                    style_kv_moe_content_routed=kv_content_routed,
                    style_shortcut_alpha=shortcut_alpha,
                    style_query_source=query_source,
                    style_cross_attn_skip_coarse=skip_coarse,
                    style_attn_topk=attn_topk,
                    layer_idx=idx,
                    num_layers=depth,
                    dino_dim=self.dino_dim,
                )
                for idx in range(depth)
            ]
        )
        self.last_cross_attn_entropy = torch.tensor(0.0)
        if self.endpoint_head_mode == "endpoint_lowhigh":
            self.endpoint_style_to_low = nn.Sequential(
                nn.LayerNorm(self.dim),
                nn.Linear(self.dim, self.endpoint_style_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.endpoint_style_hidden_dim, self.latent_channels),
            )
            self.endpoint_style_to_high = nn.Sequential(
                nn.LayerNorm(self.dim),
                nn.Linear(self.dim, self.endpoint_style_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.endpoint_style_hidden_dim, self.latent_channels),
            )
            if self.endpoint_film_enabled:
                self.endpoint_film_low = FiLMEndpointHead(
                    self.dim, self.latent_channels, self.dim, self.endpoint_style_hidden_dim,
                )
                self.endpoint_film_high = FiLMEndpointHead(
                    self.dim, self.latent_channels, self.dim, self.endpoint_style_hidden_dim,
                )
                self.endpoint_low_head = None
                self.endpoint_high_head = None
            else:
                self.endpoint_film_low = None
                self.endpoint_film_high = None
                self.endpoint_low_head = nn.Sequential(
                    nn.GroupNorm(1, self.dim),
                    nn.SiLU(),
                    nn.Conv2d(self.dim, self.latent_channels, kernel_size=3, padding=1),
                )
                self.endpoint_high_head = nn.Sequential(
                    nn.GroupNorm(1, self.dim),
                    nn.SiLU(),
                    nn.Conv2d(self.dim, self.latent_channels, kernel_size=3, padding=1),
                )
                nn.init.normal_(self.endpoint_low_head[-1].weight, mean=0.0, std=1e-3)
                nn.init.zeros_(self.endpoint_low_head[-1].bias)
                nn.init.normal_(self.endpoint_high_head[-1].weight, mean=0.0, std=1e-3)
                nn.init.zeros_(self.endpoint_high_head[-1].bias)
            nn.init.zeros_(self.endpoint_style_to_low[-1].weight)
            nn.init.zeros_(self.endpoint_style_to_low[-1].bias)
            nn.init.zeros_(self.endpoint_style_to_high[-1].weight)
            nn.init.zeros_(self.endpoint_style_to_high[-1].bias)
            self.out = None
        else:
            self.out = nn.Sequential(
                nn.GroupNorm(1, self.dim),
                nn.SiLU(),
                nn.Conv2d(self.dim, self.latent_channels, kernel_size=3, padding=1),
            )
            nn.init.normal_(self.out[-1].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.out[-1].bias)
        self.last_debug: dict[str, torch.Tensor] = {}

    def _lowpass(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.avg_pool2d(
            x.float(),
            kernel_size=self.endpoint_lowpass_kernel,
            stride=1,
            padding=self.endpoint_lowpass_kernel // 2,
        ).to(dtype=x.dtype)

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
        content_dino_patches: torch.Tensor | None = None,
        style_latent: torch.Tensor | None = None,
        style_text_tokens: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        del source
        t_tensor = self._resolve_t(x, t)
        style_tokens, style_global = self.style_conditioner(
            style_dino_patches=style_dino_patches,
            style_dino_cls=style_dino_cls,
            style_id=style_id,
            batch=x.shape[0],
            device=x.device,
            dtype=x.dtype,
            style_latent=style_latent,
            style_text_tokens=style_text_tokens,
        )
        time_emb = self.time_proj(sinusoidal_time_embedding_620(t_tensor, self.time_dim).to(device=x.device, dtype=x.dtype))
        h = self.input_proj(x)
        debug_vals: dict[str, list[torch.Tensor]] = {}
        total_entropy = []
        for block in self.blocks:
            h = block(h, time_emb=time_emb, style_tokens=style_tokens, content_dino_patches=content_dino_patches)
            total_entropy.append(block.cross_attn_entropy)
            for key, value in block.last_debug.items():
                debug_vals.setdefault(key, []).append(value)
        
        if total_entropy:
            self.last_cross_attn_entropy = torch.stack(total_entropy).mean()
        else:
            self.last_cross_attn_entropy = x.new_tensor(0.0)

        total_pixel_entropy = [block.pixel_entropy for block in self.blocks if getattr(block, "pixel_entropy", None) is not None]
        if total_pixel_entropy:
            self.last_pixel_entropy = torch.stack(total_pixel_entropy).mean(dim=0)
        else:
            self.last_pixel_entropy = None

        if self.endpoint_head_mode == "endpoint_lowhigh":
            style_low = self.endpoint_style_to_low(style_global.float()).to(dtype=x.dtype).view(x.shape[0], self.latent_channels, 1, 1)
            style_high = self.endpoint_style_to_high(style_global.float()).to(dtype=x.dtype).view(x.shape[0], self.latent_channels, 1, 1)
            if self.endpoint_film_enabled:
                low_delta = self.endpoint_film_low(h, style_global.float())
                high_delta = self.endpoint_film_high(h, style_global.float()) * float(self.endpoint_high_scale)
            else:
                low_delta = self.endpoint_low_head(h) + style_low
                high_delta = (self.endpoint_high_head(h) + style_high) * float(self.endpoint_high_scale)
            x_low = self._lowpass(x)
            x_high = x - x_low
            endpoint = (x_low + low_delta) + (x_high + high_delta)
            denom = (1.0 - t_tensor).view(-1, 1, 1, 1).to(dtype=x.dtype).clamp_min(self.endpoint_velocity_floor)
            velocity = (endpoint - x) / denom
            endpoint_low = self._lowpass(endpoint)
            endpoint_high = endpoint - endpoint_low
        else:
            velocity = self.out(h)
            endpoint = x + (1.0 - t_tensor).view(-1, 1, 1, 1).to(dtype=x.dtype) * velocity
            endpoint_low = self._lowpass(endpoint)
            endpoint_high = endpoint - endpoint_low
        self.last_debug = {
            key: torch.stack([v.to(device=x.device).float() for v in values]).mean()
            for key, values in debug_vals.items()
            if values
        }
        self.last_debug["velocity_abs"] = velocity.detach().float().abs().mean()
        self.last_debug["endpoint_head_mode_lowhigh"] = x.new_tensor(1.0 if self.endpoint_head_mode == "endpoint_lowhigh" else 0.0)
        self.last_debug["endpoint_film_enabled"] = x.new_tensor(1.0 if self.endpoint_film_enabled else 0.0)
        self.last_debug["endpoint_pred_abs"] = endpoint.detach().float().abs().mean()
        self.last_debug["endpoint_low_abs"] = endpoint_low.detach().float().abs().mean()
        self.last_debug["endpoint_high_abs"] = endpoint_high.detach().float().abs().mean()
        if self.endpoint_head_mode == "endpoint_lowhigh":
            self.last_debug["endpoint_style_low_abs"] = style_low.detach().float().abs().mean()
            self.last_debug["endpoint_style_high_abs"] = style_high.detach().float().abs().mean()
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
        style_text_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t_tensor = self._resolve_t(x, t)
        v = self.forward(x, t=t_tensor, style_id=style_id, style_dino_patches=style_dino_patches, style_dino_cls=style_dino_cls, style_text_tokens=style_text_tokens)
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
        style_text_tokens: torch.Tensor | None = None,
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
                style_text_tokens=style_text_tokens,
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

    @torch.no_grad()
    def integrate_transport_cfg(
        self,
        x: torch.Tensor,
        style_id: torch.Tensor | int | None,
        *,
        num_steps: int = 8,
        step_size: float = 1.0,
        style_dino_patches: torch.Tensor | None = None,
        style_dino_cls: torch.Tensor | None = None,
        style_text_tokens: torch.Tensor | None = None,
        idt_dino_patches: torch.Tensor | None = None,
        idt_dino_cls: torch.Tensor | None = None,
        cfg_target_scale: float = 1.0,
        cfg_repulse_scale: float = 0.0,
        cfg_text_scale: float = 0.0,
        idt_style_id: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        """Tri-directional CFG: attract to target, repulse from source IDT style.

        v_final = v_target + cfg_target_scale*(v_target - v_null)
                  - cfg_repulse_scale*(v_idt - v_null)
                  + cfg_text_scale*(v_text - v_target)   (if text is used separately)
        """
        steps = max(1, int(num_steps))
        horizon = max(0.0, float(step_size))
        if horizon <= 0.0:
            return x
        h = x
        for idx in range(steps):
            t_curr = horizon * (idx / float(steps))
            t_next = horizon * ((idx + 1) / float(steps))
            t_batch = torch.full((h.shape[0],), t_curr, device=h.device, dtype=h.dtype)

            ep_target = self.predict_endpoint(
                h, t=t_batch, style_id=style_id,
                style_dino_patches=style_dino_patches,
                style_dino_cls=style_dino_cls,
                style_text_tokens=style_text_tokens,
            )
            ep_null = self.predict_endpoint(h, t=t_batch, style_id=style_id)

            if cfg_repulse_scale > 0.0 and idt_dino_patches is not None:
                ep_idt = self.predict_endpoint(
                    h, t=t_batch, style_id=idt_style_id,
                    style_dino_patches=idt_dino_patches,
                    style_dino_cls=idt_dino_cls,
                )
                guided = ep_target + cfg_target_scale * (ep_target - ep_null) - cfg_repulse_scale * (ep_idt - ep_null)
            elif cfg_target_scale > 0.0:
                guided = ep_target + cfg_target_scale * (ep_target - ep_null)
            else:
                guided = ep_target

            denom = max(1e-6, 1.0 - t_curr)
            c_curr = (1.0 - t_next) / denom
            c_tgt = (t_next - t_curr) / denom
            mean = c_curr * h + c_tgt * guided
            var = (self.bridge_sigma ** 2) * ((t_next - t_curr) * max(0.0, 1.0 - t_next) / denom)
            if var > 0.0:
                mean = mean + torch.randn_like(mean) * (var ** 0.5)
            h = mean
        return h


def build_spatial_bridge620_from_config(
    model_cfg: ModelConfig,
    *,
    bridge_cfg: BridgeConfig | None = None,
    use_checkpointing: bool = False,
) -> SpatialBridge620:
    del use_checkpointing
    return SpatialBridge620(model_cfg, bridge_cfg=bridge_cfg)
