from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt


class MSContextualAdaGN(nn.Module):
    """
    Multi-Scale Contextual Texture Modulation without absolute coordinates.
    """

    def __init__(self, dim: int, style_dim: int, num_groups: int = 4) -> None:
        super().__init__()
        groups = max(1, min(int(num_groups), int(dim)))
        while dim % groups != 0 and groups > 1:
            groups -= 1
        self.norm = nn.GroupNorm(groups, dim, affine=False)

        self.global_proj = nn.Linear(style_dim, dim * 2)
        self.texture_proj1 = nn.Linear(style_dim, dim)
        self.texture_proj2 = nn.Linear(style_dim, dim)

        hidden = max(1, dim // 4)
        gate_groups = math.gcd(dim, hidden)
        gate_groups = max(1, gate_groups)

        self.active_gate1 = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=3, padding=1, groups=gate_groups),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.active_gate2 = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=3, padding=2, dilation=2, groups=gate_groups),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        nn.init.zeros_(self.texture_proj1.weight)
        nn.init.zeros_(self.texture_proj1.bias)
        nn.init.zeros_(self.texture_proj2.weight)
        nn.init.zeros_(self.texture_proj2.bias)
        nn.init.normal_(self.global_proj.weight, std=0.02)
        nn.init.constant_(self.global_proj.bias, 0.0)
        with torch.no_grad():
            self.global_proj.bias[:dim] = 1.0

    def forward(self, x: torch.Tensor, style_code: torch.Tensor, gate: float | torch.Tensor = 1.0) -> torch.Tensor:
        if style_code.shape[0] != x.shape[0]:
            raise ValueError(f"Batch mismatch: style_code batch={style_code.shape[0]} vs x batch={x.shape[0]}")
        normalized = self.norm(x)

        g_params = self.global_proj(style_code).unsqueeze(-1).unsqueeze(-1)
        g_scale, g_shift = g_params.chunk(2, dim=1)

        a1 = self.active_gate1(normalized)
        a2 = self.active_gate2(normalized)
        tau1 = self.texture_proj1(style_code).unsqueeze(-1).unsqueeze(-1)
        tau2 = self.texture_proj2(style_code).unsqueeze(-1).unsqueeze(-1)

        local_scale = g_scale + a1 * tau1 + a2 * tau2
        adagn = normalized * local_scale + g_shift
        if isinstance(gate, torch.Tensor):
            gate_t = gate.to(device=x.device, dtype=x.dtype)
        else:
            gate_t = x.new_tensor(float(gate))
        return normalized + gate_t * (adagn - normalized)


# Backward-compatible aliases.
TextureDictAdaGN = MSContextualAdaGN
GlobalDemodulatedAdaMixGN = MSContextualAdaGN
SpatiallyAdaptiveAdaMixGN = MSContextualAdaGN
SpatiallyAdaptiveAdaGN = MSContextualAdaGN
CoordSPADE = MSContextualAdaGN


class ResBlock(nn.Module):
    def __init__(self, dim: int, style_dim: int, num_groups: int = 8) -> None:
        super().__init__()
        self.norm1 = MSContextualAdaGN(dim, style_dim, num_groups=num_groups)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.norm2 = MSContextualAdaGN(dim, style_dim, num_groups=num_groups)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, style_code: torch.Tensor, gate: float | torch.Tensor = 1.0) -> torch.Tensor:
        h = self.act(self.norm1(x, style_code, gate=gate))
        h = self.conv1(h)
        h = self.act(self.norm2(h, style_code, gate=gate))
        h = self.conv2(h)
        return x + h


class NormFreeModulation(nn.Module):
    def __init__(
        self,
        channels: int,
        style_dim: int,
        *,
        clamp_enabled: bool = True,
        gamma_min: float = -0.9,
        gamma_max: float = 3.0,
        beta_min: float = -2.0,
        beta_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.mapper = nn.Linear(style_dim, channels * 2)
        nn.init.zeros_(self.mapper.weight)
        nn.init.zeros_(self.mapper.bias)
        self.clamp_enabled = bool(clamp_enabled)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)

    def forward(self, x: torch.Tensor, style_code: torch.Tensor, gate: float | torch.Tensor = 1.0) -> torch.Tensor:
        if style_code.shape[0] != x.shape[0]:
            raise ValueError(f"Batch mismatch: style_code batch={style_code.shape[0]} vs x batch={x.shape[0]}")
        params = self.mapper(style_code).view(x.shape[0], -1, 1, 1)
        gamma, beta = params.chunk(2, dim=1)
        if self.clamp_enabled:
            gamma = torch.clamp(gamma, min=self.gamma_min, max=self.gamma_max)
            beta = torch.clamp(beta, min=self.beta_min, max=self.beta_max)
        if isinstance(gate, torch.Tensor):
            gate_t = gate.to(device=x.device, dtype=x.dtype)
        else:
            gate_t = x.new_tensor(float(gate))
        gamma = gamma * gate_t
        beta = beta * gate_t
        return x * (1.0 + gamma) + beta


class StyleAdaptiveSkip(nn.Module):
    def __init__(self, channels: int, style_dim: int) -> None:
        super().__init__()
        self.gate_mapper = nn.Sequential(
            nn.Linear(style_dim, channels),
            nn.Sigmoid(),
        )
        self.rewrite_mapper = nn.Linear(style_dim, channels)
        nn.init.zeros_(self.rewrite_mapper.weight)
        nn.init.zeros_(self.rewrite_mapper.bias)

    def forward(
        self,
        skip_feat: torch.Tensor,
        style_code: torch.Tensor,
        gate: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        if style_code.shape[0] != skip_feat.shape[0]:
            raise ValueError(
                f"Batch mismatch: style_code batch={style_code.shape[0]} vs skip_feat batch={skip_feat.shape[0]}"
            )
        b, c, _, _ = skip_feat.shape
        erase_gate = self.gate_mapper(style_code).view(b, c, 1, 1)
        rewrite_bias = self.rewrite_mapper(style_code).view(b, c, 1, 1)
        if isinstance(gate, torch.Tensor):
            gate_t = gate.to(device=skip_feat.device, dtype=skip_feat.dtype)
        else:
            gate_t = skip_feat.new_tensor(float(gate))
        effective_gate = 1.0 - (1.0 - erase_gate) * gate_t
        return skip_feat * effective_gate + rewrite_bias * (1.0 - effective_gate)


class LatentAdaCUT(nn.Module):
    def __init__(
        self,
        latent_channels: int = 4,
        num_styles: int = 3,
        style_dim: int = 256,
        base_dim: int = 64,
        lift_channels: int | None = None,
        num_hires_blocks: int = 2,
        num_res_blocks: int = 4,
        num_decoder_blocks: int = 1,
        num_groups: int = 8,
        use_checkpointing: bool = False,
        latent_scale_factor: float = 0.18215,
        residual_gain: float = 0.1,
        output_clamp_enabled: bool = True,
        output_clamp_min: float = -2.5,
        output_clamp_max: float = 2.5,
        decoder_mod_clamp_enabled: bool = True,
        decoder_mod_gamma_min: float = -0.9,
        decoder_mod_gamma_max: float = 3.0,
        decoder_mod_beta_min: float = -2.0,
        decoder_mod_beta_max: float = 2.0,
        decoder_mag_stabilizer_enabled: bool = True,
        decoder_mag_target_scale: float = 0.5,
        decoder_mag_eps: float = 1e-6,
        use_decoder_adagn: bool = True,
        inject_gate_hires: float = 0.0,
        inject_gate_body: float = 1.0,
        inject_gate_decoder: float = 1.0,
        style_strength_default: float = 1.0,
        style_strength_step_curve: str = "linear",
        upsample_mode: str = "nearest",
        upsample_blur: bool = True,
        upsample_blur_kernel: str = "box3",
        **kwargs,
    ) -> None:
        super().__init__()
        # Kept for backward-compatible config loading; no longer used in simplified architecture.
        _ = kwargs
        _ = use_decoder_adagn
        _ = num_decoder_blocks
        self.latent_channels = int(latent_channels)
        self.num_styles = int(num_styles)
        self.use_checkpointing = bool(use_checkpointing)
        self.latent_scale_factor = float(latent_scale_factor)
        self.residual_gain = float(residual_gain)
        self.output_clamp_enabled = bool(output_clamp_enabled)
        self.output_clamp_min = float(output_clamp_min)
        self.output_clamp_max = float(output_clamp_max)
        self.decoder_mag_stabilizer_enabled = bool(decoder_mag_stabilizer_enabled)
        self.decoder_mag_target_scale = float(decoder_mag_target_scale)
        self.decoder_mag_eps = float(decoder_mag_eps)
        self.lift_channels = int(lift_channels) if lift_channels is not None else int(base_dim)
        self.body_channels = int(base_dim * 2)

        self.inject_gate_hires = max(0.0, min(1.0, float(inject_gate_hires)))
        self.inject_gate_body = max(0.0, min(1.0, float(inject_gate_body)))
        self.inject_gate_decoder = max(0.0, min(1.0, float(inject_gate_decoder)))
        self.style_strength_default = max(0.0, min(1.0, float(style_strength_default)))
        self.style_strength_step_curve = str(style_strength_step_curve).lower()
        if self.style_strength_step_curve not in {"linear", "smoothstep", "sqrt"}:
            self.style_strength_step_curve = "linear"
        self.upsample_mode = str(upsample_mode)
        self.upsample_blur = bool(upsample_blur)
        self.upsample_blur_kernel = str(upsample_blur_kernel).lower()
        if self.upsample_blur_kernel not in {"box3", "gaussian3"}:
            self.upsample_blur_kernel = "box3"

        self.style_emb = nn.Embedding(self.num_styles, style_dim)
        nn.init.normal_(self.style_emb.weight, mean=0.0, std=0.02)

        self.enc_in = nn.Conv2d(latent_channels, self.lift_channels, kernel_size=3, stride=1, padding=1)
        self.enc_in_act = nn.SiLU()
        self.hires_body = nn.ModuleList(
            [ResBlock(self.lift_channels, style_dim, num_groups=num_groups) for _ in range(max(0, int(num_hires_blocks)))]
        )
        self.down = nn.Conv2d(self.lift_channels, self.body_channels, kernel_size=4, stride=2, padding=1)
        self.body = nn.ModuleList(
            [ResBlock(self.body_channels, style_dim, num_groups=num_groups) for _ in range(num_res_blocks)]
        )

        upsample_kwargs = {"scale_factor": 2, "mode": self.upsample_mode}
        if self.upsample_mode in {"bilinear", "bicubic"}:
            upsample_kwargs["align_corners"] = False
        self.dec_up = nn.Upsample(**upsample_kwargs)
        self.skip_fusion = nn.Sequential(
            nn.Conv2d(self.body_channels + self.lift_channels, self.lift_channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
        )
        self.skip_filter = StyleAdaptiveSkip(self.lift_channels, style_dim)
        self.dec_conv = nn.Conv2d(self.lift_channels, self.lift_channels, kernel_size=3, stride=1, padding=1)
        self.dec_mod = NormFreeModulation(
            self.lift_channels,
            style_dim,
            clamp_enabled=decoder_mod_clamp_enabled,
            gamma_min=decoder_mod_gamma_min,
            gamma_max=decoder_mod_gamma_max,
            beta_min=decoder_mod_beta_min,
            beta_max=decoder_mod_beta_max,
        )
        self.dec_act = nn.SiLU()
        self.dec_out = nn.Conv2d(self.lift_channels, latent_channels, kernel_size=3, stride=1, padding=1)

        if self.upsample_blur:
            if self.upsample_blur_kernel == "gaussian3":
                k = torch.tensor(
                    [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
                    dtype=torch.float32,
                ) / 16.0
            else:
                k = torch.ones((3, 3), dtype=torch.float32) / 9.0
            self.register_buffer("_upsample_blur_kernel", k.view(1, 1, 3, 3), persistent=False)
        else:
            self.register_buffer("_upsample_blur_kernel", torch.empty(0), persistent=False)
        self._upsample_blur_kernel_cache: dict[tuple[int, str], torch.Tensor] = {}

    def _normalize_style_id_input(
        self,
        style_id: torch.Tensor | int,
        *,
        device: torch.device,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        if isinstance(style_id, int):
            if batch_size is None:
                style_id = torch.tensor([style_id], device=device, dtype=torch.long)
            else:
                style_id = torch.full((batch_size,), int(style_id), device=device, dtype=torch.long)
        style_id = style_id.long().view(-1)
        if style_id.device != device:
            style_id = style_id.to(device)
        return style_id.clamp_min(0).clamp_max(max(1, self.num_styles) - 1)

    def _run_block(self, block: ResBlock, h: torch.Tensor, style_code: torch.Tensor, gate: float | torch.Tensor = 1.0) -> torch.Tensor:
        if self.use_checkpointing and self.training:
            gate_in = gate.to(device=h.device, dtype=h.dtype) if torch.is_tensor(gate) else h.new_tensor(float(gate))
            return ckpt.checkpoint(lambda _h, _s, _g, _blk=block: _blk(_h, _s, _g), h, style_code, gate_in, use_reentrant=False)
        return block(h, style_code, gate=gate)

    def _run_style_blocks(
        self,
        h: torch.Tensor,
        blocks: nn.ModuleList,
        style_code: torch.Tensor,
        gate: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        out = h
        for block in blocks:
            out = self._run_block(block, out, style_code, gate=gate)
        return out

    def _resolve_style_strength(self, style_strength: float | None) -> float:
        if style_strength is None:
            return self.style_strength_default
        return max(0.0, min(1.0, float(style_strength)))

    def _style_strength_step_scale(self, style_strength: float) -> float:
        s = max(0.0, min(1.0, float(style_strength)))
        if self.style_strength_step_curve == "sqrt":
            return math.sqrt(s)
        if self.style_strength_step_curve == "smoothstep":
            return s * s * (3.0 - 2.0 * s)
        return s

    def _run_decoder(self, h: torch.Tensor, style_code: torch.Tensor, gate: float | torch.Tensor = 1.0) -> torch.Tensor:
        def _stabilize_mag(_h: torch.Tensor) -> torch.Tensor:
            if not self.decoder_mag_stabilizer_enabled:
                return _h
            magnitude = torch.sqrt(_h.pow(2).mean(dim=1, keepdim=True) + self.decoder_mag_eps)
            target_mag = math.sqrt(max(float(_h.shape[1]) * self.decoder_mag_target_scale, 0.0))
            scale = torch.clamp((target_mag / magnitude), max=1.0)
            return _h * scale

        if self.use_checkpointing and self.training:
            gate_in = gate.to(device=h.device, dtype=h.dtype) if torch.is_tensor(gate) else h.new_tensor(float(gate))
            return ckpt.checkpoint(
                lambda _h, _s, _g: _stabilize_mag(self.dec_act(self.dec_mod(self.dec_conv(_h), _s, gate=_g))),
                h,
                style_code,
                gate_in,
                use_reentrant=False,
            )
        h = self.dec_conv(h)
        h = self.dec_mod(h, style_code, gate=gate)
        h = self.dec_act(h)
        return _stabilize_mag(h)

    def _apply_upsample_blur(self, h: torch.Tensor) -> torch.Tensor:
        if not self.upsample_blur or self._upsample_blur_kernel.numel() == 0:
            return h
        b, c, _, _ = h.shape
        if c <= 0 or b <= 0:
            return h
        key = (int(c), str(h.device))
        kernel = self._upsample_blur_kernel_cache.get(key)
        if kernel is None:
            kernel = self._upsample_blur_kernel.to(device=h.device, dtype=torch.float32).repeat(c, 1, 1, 1).contiguous()
            self._upsample_blur_kernel_cache[key] = kernel
        h_dtype = h.dtype
        if h.device.type == "cuda":
            with torch.amp.autocast("cuda", enabled=False):
                out = F.conv2d(h.float(), kernel, stride=1, padding=1, groups=c)
        else:
            out = F.conv2d(h.float(), kernel, stride=1, padding=1, groups=c)
        return out.to(dtype=h_dtype)

    def _compute_delta(self, h: torch.Tensor) -> torch.Tensor:
        delta = self.dec_out(h) * self.latent_scale_factor * self.residual_gain
        if self.output_clamp_enabled:
            delta = torch.clamp(delta, min=self.output_clamp_min, max=self.output_clamp_max)
        return delta

    def encode_style_id(self, style_id: torch.Tensor | int | None, batch_size: int | None = None) -> torch.Tensor:
        if style_id is None:
            raise ValueError("style_id is required.")
        emb_device = self.style_emb.weight.device
        style_id = self._normalize_style_id_input(style_id, device=emb_device, batch_size=batch_size)
        return self.style_emb(style_id)

    def _predict_delta_from_context(self, x: torch.Tensor, *, style_code: torch.Tensor, strength: float) -> torch.Tensor:
        gate_hires = self.inject_gate_hires * strength
        gate_body = self.inject_gate_body * strength
        gate_decoder = self.inject_gate_decoder * strength

        feat = x / max(self.latent_scale_factor, 1e-8)
        h = self.enc_in_act(self.enc_in(feat))
        h = self._run_style_blocks(h, blocks=self.hires_body, style_code=style_code, gate=gate_hires)
        skip_32 = h

        h = self.down(h)
        h = self._run_style_blocks(h, blocks=self.body, style_code=style_code, gate=gate_body)

        h = self.dec_up(h)
        h = self._apply_upsample_blur(h)
        filtered_skip = self.skip_filter(skip_32, style_code, gate=gate_decoder)
        h = self.skip_fusion(torch.cat([h, filtered_skip], dim=1))
        h = self._run_decoder(h, style_code=style_code, gate=gate_decoder)
        return self._compute_delta(h)

    def _predict_delta(self, x: torch.Tensor, style_id: torch.Tensor | int, style_strength: float | None = None) -> torch.Tensor:
        strength = self._resolve_style_strength(style_strength)
        style_code = self.encode_style_id(style_id, batch_size=x.shape[0])
        return self._predict_delta_from_context(x, style_code=style_code, strength=strength)

    def integrate(
        self,
        x: torch.Tensor,
        style_id: torch.Tensor | int,
        num_steps: int = 1,
        step_size: float = 1.0,
        style_strength: float | None = None,
    ) -> torch.Tensor:
        steps = max(1, int(num_steps))
        strength = self._resolve_style_strength(style_strength)
        step_scale = self._style_strength_step_scale(strength)
        per_step = 1.0 / float(steps)
        style_code = self.encode_style_id(style_id, batch_size=x.shape[0])
        h = x
        for _ in range(steps):
            delta = self._predict_delta_from_context(h, style_code=style_code, strength=strength)
            h = h + delta * float(step_size) * step_scale * per_step
        return h

    def forward(
        self,
        x: torch.Tensor,
        style_id: torch.Tensor | int,
        step_size: float = 1.0,
        style_strength: float | None = None,
    ) -> torch.Tensor:
        strength = self._resolve_style_strength(style_strength)
        step_scale = self._style_strength_step_scale(strength)
        style_code = self.encode_style_id(style_id, batch_size=x.shape[0])
        delta = self._predict_delta_from_context(x, style_code=style_code, strength=strength)
        return x + delta * float(step_size) * step_scale


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model_from_config(model_cfg: dict, *, use_checkpointing: bool = False) -> LatentAdaCUT:
    cfg = dict(model_cfg)
    cfg["use_checkpointing"] = bool(use_checkpointing)
    return LatentAdaCUT(**cfg)
