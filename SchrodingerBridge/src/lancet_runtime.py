from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lancet_blocks import StyleMaps
from utils.diffeomorphic import apply_texture_aligned_diffeomorphic_stroke, build_diffeomorphic_guide


class LatentAdaCUTRuntimeMixin:

    def _normalize_style_id_input(
        self,
        style_id: torch.Tensor | int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(style_id, int):
            style_id = torch.tensor([style_id], device=device, dtype=torch.long)
        style_id = style_id.long().view(-1)
        if style_id.device != device:
            style_id = style_id.to(device)
        return style_id.clamp_min(0).clamp_max(max(1, self.num_styles) - 1)

    def _fuse_skip_features(
        self,
        h_up: torch.Tensor,
        skip_32: torch.Tensor,
    ) -> torch.Tensor:
        # Hard no-skip path: physically disconnect encoder skip source and keep only
        # the upsample projection branch.
        if self.skip_disabled:
            return self.skip_fusion(self.skip_up_proj(h_up))

        skip_feat = skip_32

        if self.skip_fusion_mode == "add_proj":
            h_base = self.skip_up_proj(h_up)
            skip_base = self.skip_src_proj(skip_feat)
            h_base.add_(skip_base)
            return self.skip_fusion(h_base)

        return self.skip_fusion(torch.cat([h_up, skip_feat], dim=1))

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

    def _run_decoder(
        self,
        h: torch.Tensor,
    ) -> torch.Tensor:
        out = h
        for block in self.decoder_blocks:
            out = out + block(out)
        return out

    def _prepare_style_maps(
        self,
        style_id: torch.Tensor | int,
    ) -> StyleMaps:
        spatial_device = self.style_spatial_id_16.device
        normalized_style_id = self._normalize_style_id_input(style_id, device=spatial_device)
        return StyleMaps(
            map_16=self.encode_style_spatial_id(normalized_style_id).get(16),
            style_id=normalized_style_id,
        )

    def _prepare_spatial_map(self, style_map: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor | None:
        return self._match_style_map(style_map, target)

    def _prepare_style_context(
        self,
        *,
        style_id: torch.Tensor | int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[object, StyleMaps]:
        style_tokens = self.encode_style_tokens(style_id, batch_size=batch_size, device=device, dtype=dtype)
        style_maps = self._prepare_style_maps(style_id=style_id)
        return style_tokens, style_maps

    def _apply_upsample_blur(self, h: torch.Tensor) -> torch.Tensor:
        if not self.upsample_blur or self._upsample_blur_kernel.numel() == 0:
            return h
        b, c, _, _ = h.shape
        if c <= 0 or b <= 0:
            return h
        if c == self.body_channels and self._upsample_blur_kernel_body.numel() > 0:
            kernel = self._upsample_blur_kernel_body.to(device=h.device, dtype=torch.float32)
        else:
            key = (int(c), str(h.device))
            kernel = self._upsample_blur_kernel_cache.get(key)
            if kernel is None:
                kernel = (
                    self._upsample_blur_kernel.to(device=h.device, dtype=torch.float32)
                    .repeat(c, 1, 1, 1)
                    .contiguous()
                )
                self._upsample_blur_kernel_cache[key] = kernel
        h_dtype = h.dtype
        if h.device.type == "cuda":
            with torch.amp.autocast("cuda", enabled=False):
                out = F.conv2d(h.float(), kernel, stride=1, padding=1, groups=c)
        else:
            out = F.conv2d(h.float(), kernel, stride=1, padding=1, groups=c)
        return out.to(dtype=h_dtype)

    def _compute_delta(
        self,
        h: torch.Tensor,
        x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        style_tokens = getattr(self, "_last_style_token_fields", None)
        carrier_debug: dict[str, torch.Tensor | None] = dict(getattr(self, "carrier_debug", {}) or {})
        raw = self._decode_output_raw(h, style_tokens=style_tokens)
        self.last_raw_diffeomorphic = raw
        if bool(getattr(self, "use_diffeomorphic_stroke", False)):
            if x is None:
                raise ValueError("diffeomorphic stroke mode requires input x.")
            return self._apply_diffeomorphic_stroke(x, raw)
        delta = raw * self.latent_scale_factor * self.residual_gain
        delta = torch.tanh(delta / 4.0) * 4.0
        carrier_debug["raw_delta"] = delta.detach()
        if bool(getattr(self, "output_residual_router", False)):
            if x is None:
                raise ValueError("output residual router requires input x.")
            delta = self._route_output_residual(delta, x)
        carrier_debug["total_delta"] = delta.detach()
        self.carrier_debug = carrier_debug
        return delta

    def _route_output_residual(self, delta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        kernel = max(1, int(getattr(self, "output_router_kernel", 5)))
        if kernel % 2 == 0:
            kernel += 1
        delta_float = delta.float()
        if kernel > 1:
            low = F.avg_pool2d(delta_float, kernel_size=kernel, stride=1, padding=kernel // 2)
        else:
            low = delta_float
        high = delta_float - low
        edge_gate = self._output_router_edge_gate(x, kernel=kernel)
        floor = float(getattr(self, "output_router_highpass_floor", 0.10))
        high_gate = floor + (1.0 - floor) * edge_gate
        low_strength = float(getattr(self, "output_router_lowpass_strength", 1.0))
        edge_low_suppression = float(getattr(self, "output_router_edge_lowpass_suppression", 0.0))
        if edge_low_suppression > 0.0:
            low_gate = 1.0 - edge_low_suppression * edge_gate
        else:
            low_gate = 1.0
        routed = low * low_strength * low_gate + high * high_gate
        return routed.to(device=delta.device, dtype=delta.dtype)

    def _output_router_edge_gate(self, x: torch.Tensor, *, kernel: int) -> torch.Tensor:
        x_float = x.float()
        if kernel > 1:
            low = F.avg_pool2d(x_float, kernel_size=kernel, stride=1, padding=kernel // 2)
        else:
            low = x_float
        high_energy = (x_float - low).abs().mean(dim=1, keepdim=True)
        gx = low[..., :, 1:] - low[..., :, :-1]
        gy = low[..., 1:, :] - low[..., :-1, :]
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = F.pad(gy, (0, 0, 0, 1))
        edge_energy = torch.sqrt(gx.square() + gy.square() + 1e-12).mean(dim=1, keepdim=True)
        support = high_energy + edge_energy
        if kernel > 1:
            support = F.avg_pool2d(support, kernel_size=kernel, stride=1, padding=kernel // 2)
        denom = support.flatten(1).mean(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
        gamma = float(getattr(self, "output_router_edge_gamma", 8.0))
        gate = 1.0 - torch.exp(-gamma * support / denom)
        return gate.to(device=x.device, dtype=x.dtype)

    def _style_band_support_gate(self, x: torch.Tensor, *, gamma: float, floor: float, kernel: int) -> torch.Tensor:
        if kernel % 2 == 0:
            kernel += 1
        x_float = x.float()
        pad = kernel // 2
        if kernel > 1:
            low = F.avg_pool2d(x_float, kernel_size=kernel, stride=1, padding=pad)
            support = (x_float - low).abs().mean(dim=1, keepdim=True)
            support = F.avg_pool2d(support, kernel_size=kernel, stride=1, padding=pad)
        else:
            support = x_float.abs().mean(dim=1, keepdim=True)
        denom = support.flatten(1).mean(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
        gate = 1.0 - torch.exp(-gamma * support / denom)
        if floor > 0.0:
            gate = floor + (1.0 - floor) * gate
        return gate.to(device=x.device, dtype=x.dtype)

    def _apply_diffeomorphic_stroke(self, x: torch.Tensor, raw_out: torch.Tensor) -> torch.Tensor:
        metric_anchor = None
        if bool(getattr(self, "diffeomorphic_metric_mask_use_z0", False)):
            metric_anchor = getattr(self, "_integration_anchor_x", None)
        if metric_anchor is None:
            metric_anchor = x
        stroked = apply_texture_aligned_diffeomorphic_stroke(
            x,
            raw_out,
            guide=build_diffeomorphic_guide(
                x,
                mode=str(getattr(self, "diffeomorphic_guide_mode", "mean")),
                channel=int(getattr(self, "diffeomorphic_guide_channel", 2)),
                weights=getattr(self, "diffeomorphic_guide_weights", None),
            ),
            color_strength=float(getattr(self, "diffeomorphic_color_strength", 0.85)),
            warp_strength=float(getattr(self, "diffeomorphic_warp_strength", 0.08)),
            gate_strength=float(getattr(self, "diffeomorphic_texture_gate_strength", 8.0)),
            normal_leak=float(getattr(self, "diffeomorphic_normal_leak", 0.0)),
            color_lowpass_kernel=int(getattr(self, "diffeomorphic_color_lowpass_kernel", 1)),
            lowpass_mode=str(getattr(self, "diffeomorphic_lowpass_mode", "avg")),
            gaussian_sigma=float(getattr(self, "diffeomorphic_gaussian_sigma", 1.5)),
            active_grad_threshold=float(getattr(self, "diffeomorphic_active_grad_threshold", 0.0)),
            color_edge_gamma=float(getattr(self, "diffeomorphic_color_edge_gamma", 0.0)),
            head_mode=str(getattr(self, "diffeomorphic_head_mode", "standard")),
            amp_strength=float(getattr(self, "diffeomorphic_amp_strength", 0.5)),
            factorized_enable_color=bool(getattr(self, "diffeomorphic_factorized_enable_color", True)),
            factorized_enable_amp=bool(getattr(self, "diffeomorphic_factorized_enable_amp", True)),
            joint_bilateral_kernel=int(getattr(self, "diffeomorphic_joint_bilateral_kernel", 1)),
            joint_bilateral_range_sigma=float(getattr(self, "diffeomorphic_joint_bilateral_range_sigma", 0.5)),
            divergence_free_warp=bool(getattr(self, "diffeomorphic_divergence_free_warp", False)),
            metric_anchor=metric_anchor,
            metric_mask_gamma=float(getattr(self, "diffeomorphic_metric_mask_gamma", 0.0)),
            metric_mask_smooth_kernel=int(getattr(self, "diffeomorphic_metric_mask_smooth_kernel", 3)),
        )
        return stroked - x.float()

    def encode_style_tokens(
        self,
        style_id: torch.Tensor | int | None,
        *,
        batch_size: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        if style_id is None:
            raise ValueError("style_id is required.")
        tokenizer = getattr(self, "style_tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("StyleTokenizer is required; anonymous style vectors have been removed.")
        token_device = device or next(tokenizer.parameters()).device
        style_id = self._normalize_style_id_input(style_id, device=token_device)
        fields = tokenizer(style_id, batch_size=batch_size, device=token_device, dtype=dtype)
        self._last_style_token_fields = fields
        return fields

    @staticmethod
    def _normalize_style_map(feat: torch.Tensor) -> torch.Tensor:
        feat = feat - feat.mean(dim=(2, 3), keepdim=True)
        return feat / (feat.std(dim=(2, 3), keepdim=True, unbiased=False) + 1e-6)

    def encode_style_spatial_id(self, style_id: torch.Tensor | int) -> dict[int, torch.Tensor]:
        spatial_device = self.style_spatial_id_16.device
        style_id = self._normalize_style_id_input(style_id, device=spatial_device)
        base_map = self.style_spatial_id_16.index_select(0, style_id)
        memory_bank = getattr(self, "style_memory_bank_16", None)
        memory_logits = getattr(self, "style_memory_bank_logits", None)
        use_memory = (
            torch.is_tensor(memory_bank)
            and memory_bank.numel() > 0
            and memory_bank.ndim == 5
            and memory_bank.shape[0] >= self.num_styles
            and memory_bank.shape[2:] == base_map.shape[1:]
        )
        route_strength = getattr(self, "style_memory_bank_route_strength", None)
        route_active = bool(
            torch.is_tensor(route_strength)
            and route_strength.numel() > 0
            and float(route_strength.detach().cpu().item()) > 0.0
        )
        use_memory = use_memory and not route_active
        if use_memory:
            bank = memory_bank.to(device=base_map.device, dtype=base_map.dtype).index_select(0, style_id)
            if torch.is_tensor(memory_logits) and memory_logits.numel() > 0 and memory_logits.ndim == 2:
                logits = memory_logits.to(device=base_map.device, dtype=base_map.dtype).index_select(0, style_id)
                logits = logits[:, : bank.shape[1]]
            else:
                logits = torch.zeros(bank.shape[:2], device=base_map.device, dtype=base_map.dtype)
            weights = torch.softmax(logits, dim=1).view(bank.shape[0], bank.shape[1], 1, 1, 1)
            memory_map = (bank * weights).sum(dim=1)
            blend_value = getattr(self, "style_memory_bank_blend", None)
            blend = float(blend_value.detach().cpu().item()) if torch.is_tensor(blend_value) else 1.0
            blend = max(0.0, min(1.0, blend))
            base_map = base_map.lerp(memory_map, blend)
        maps = {16: base_map}
        if self.training and self.style_id_spatial_jitter_px > 0:
            max_jit = self.style_id_spatial_jitter_px
            shifts_y = torch.randint(
                low=-max_jit,
                high=max_jit + 1,
                size=(style_id.shape[0],),
                device=style_id.device,
            )
            shifts_x = torch.randint(
                low=-max_jit,
                high=max_jit + 1,
                size=(style_id.shape[0],),
                device=style_id.device,
            )

            def _jitter_batch(feat: torch.Tensor) -> torch.Tensor:
                if max_jit <= 0:
                    return feat
                padded = F.pad(feat, (max_jit, max_jit, max_jit, max_jit), mode="reflect")
                b, c, _, wp = padded.shape
                h, w = feat.shape[-2], feat.shape[-1]
                # Fully tensorized crop with per-sample offsets to keep torch.compile graph intact.
                y_idx = (
                    torch.arange(h, device=feat.device, dtype=torch.long).view(1, h, 1)
                    + (max_jit + shifts_y).view(-1, 1, 1)
                )
                x_idx = (
                    torch.arange(w, device=feat.device, dtype=torch.long).view(1, 1, w)
                    + (max_jit + shifts_x).view(-1, 1, 1)
                )
                y_gather = y_idx.unsqueeze(1).expand(b, c, h, wp)
                cropped_h = padded.gather(dim=2, index=y_gather)
                x_gather = x_idx.unsqueeze(1).expand(b, c, h, w)
                return cropped_h.gather(dim=3, index=x_gather)

            maps[16] = _jitter_batch(maps[16])
        maps[16] = self._normalize_style_map(maps[16])
        return maps

    def _build_content_routed_style_memory(
        self,
        content_feat_16: torch.Tensor,
        *,
        style_id: torch.Tensor | None,
    ) -> torch.Tensor | None:
        memory_bank = getattr(self, "style_memory_bank_16", None)
        if (
            style_id is None
            or not torch.is_tensor(memory_bank)
            or memory_bank.numel() == 0
            or memory_bank.ndim != 5
            or memory_bank.shape[0] < self.num_styles
            or memory_bank.shape[2] != content_feat_16.shape[1]
        ):
            return None
        style_id = self._normalize_style_id_input(style_id, device=content_feat_16.device)
        bank = memory_bank.to(device=content_feat_16.device, dtype=content_feat_16.dtype).index_select(0, style_id)
        if bank.shape[-2:] != content_feat_16.shape[-2:]:
            b, k, c, _, _ = bank.shape
            bank = F.interpolate(
                bank.view(b * k, c, bank.shape[-2], bank.shape[-1]),
                size=content_feat_16.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).view(b, k, c, content_feat_16.shape[-2], content_feat_16.shape[-1])
        b, k, c, h, w = bank.shape
        type_ids_all = getattr(self, "style_memory_bank_type_ids", None)
        if torch.is_tensor(type_ids_all) and type_ids_all.numel() > 0:
            type_ids_all = type_ids_all.to(device=content_feat_16.device, dtype=torch.long)
            if type_ids_all.ndim == 1:
                type_ids = type_ids_all[:k].view(1, k).expand(b, -1)
            elif type_ids_all.ndim == 2 and type_ids_all.shape[0] >= self.num_styles:
                type_ids = type_ids_all.index_select(0, style_id)[:, :k]
            else:
                type_ids = None
            if type_ids is not None and type_ids.shape == (b, k):
                typed = self._build_typed_content_routed_style_memory(
                    content_feat_16,
                    bank=bank,
                    style_id=style_id,
                    type_ids=type_ids,
                )
                if typed is not None:
                    return typed
        query = F.normalize(content_feat_16.float().flatten(2), dim=1, eps=1e-6)
        tokens = bank.float().flatten(3).permute(0, 2, 1, 3).reshape(b, c, k * h * w)
        keys = F.normalize(tokens, dim=1, eps=1e-6)
        temperature_value = getattr(self, "style_memory_bank_route_temperature", None)
        temperature = (
            float(temperature_value.detach().cpu().item())
            if torch.is_tensor(temperature_value) and temperature_value.numel() > 0
            else 8.0
        )
        sim = torch.einsum("bcq,bcn->bqn", query, keys) * max(0.1, min(32.0, temperature))
        memory_logits = getattr(self, "style_memory_bank_logits", None)
        if torch.is_tensor(memory_logits) and memory_logits.numel() > 0 and memory_logits.ndim == 2:
            logits = memory_logits.to(device=content_feat_16.device, dtype=torch.float32).index_select(0, style_id)
            logits = logits[:, :k].unsqueeze(-1).expand(b, k, h * w).reshape(b, k * h * w)
            sim = sim + logits.unsqueeze(1)
        weights = torch.softmax(sim, dim=-1)
        routed = torch.einsum("bqn,bcn->bcq", weights, tokens).view(b, c, h, w)
        return self._normalize_style_map(routed).to(dtype=content_feat_16.dtype)

    def _style_memory_type_gate_logits(
        self,
        content_feat_16: torch.Tensor,
    ) -> torch.Tensor:
        x = content_feat_16.float()
        low = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        high_energy = (x - low).abs().mean(dim=1, keepdim=True)
        gx = low[..., :, 1:] - low[..., :, :-1]
        gy = low[..., 1:, :] - low[..., :-1, :]
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = F.pad(gy, (0, 0, 0, 1))
        edge_energy = torch.sqrt(gx.square() + gy.square() + 1e-12).mean(dim=1, keepdim=True)
        high_norm = high_energy / high_energy.flatten(1).mean(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
        edge_norm = edge_energy / edge_energy.flatten(1).mean(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
        gamma = max(0.0, self._memory_bank_scalar("style_memory_bank_type_gate_gamma", 2.5))
        flat = -gamma * (high_norm + edge_norm)
        edge = gamma * (edge_norm - 0.35 * high_norm)
        texton = gamma * high_norm
        return torch.cat([flat, edge, texton], dim=1)

    def _build_typed_content_routed_style_memory(
        self,
        content_feat_16: torch.Tensor,
        *,
        bank: torch.Tensor,
        style_id: torch.Tensor,
        type_ids: torch.Tensor,
    ) -> torch.Tensor | None:
        b, k, c, h, w = bank.shape
        query = F.normalize(content_feat_16.float().flatten(2), dim=1, eps=1e-6)
        tokens = bank.float().flatten(3).permute(0, 2, 1, 3).reshape(b, c, k * h * w)
        keys = F.normalize(tokens, dim=1, eps=1e-6)
        temperature_value = getattr(self, "style_memory_bank_route_temperature", None)
        temperature = (
            float(temperature_value.detach().cpu().item())
            if torch.is_tensor(temperature_value) and temperature_value.numel() > 0
            else 8.0
        )
        sim = torch.einsum("bcq,bcn->bqn", query, keys) * max(0.1, min(32.0, temperature))
        memory_logits = getattr(self, "style_memory_bank_logits", None)
        if torch.is_tensor(memory_logits) and memory_logits.numel() > 0 and memory_logits.ndim == 2:
            logits = memory_logits.to(device=content_feat_16.device, dtype=torch.float32).index_select(0, style_id)
            logits = logits[:, :k].unsqueeze(-1).expand(b, k, h * w).reshape(b, k * h * w)
            sim = sim + logits.unsqueeze(1)

        type_gate_logits = self._style_memory_type_gate_logits(content_feat_16)
        type_logits = getattr(self, "style_memory_bank_type_logits", None)
        if torch.is_tensor(type_logits) and type_logits.numel() > 0 and type_logits.ndim == 2:
            prior = type_logits.to(device=content_feat_16.device, dtype=torch.float32).index_select(0, style_id)
            type_gate_logits = type_gate_logits + prior[:, :3].view(b, 3, 1, 1)
        gate_temp = max(1e-4, self._memory_bank_scalar("style_memory_bank_type_gate_temperature", 1.0))
        type_gates = torch.softmax(type_gate_logits / gate_temp, dim=1)
        self._last_style_memory_type_gates = type_gates.detach()

        token_type = type_ids.clamp_min(0).clamp_max(2).unsqueeze(-1).expand(b, k, h * w).reshape(b, k * h * w)
        routed = torch.zeros((b, c, h * w), device=content_feat_16.device, dtype=torch.float32)
        any_type = False
        for type_idx in range(3):
            mask = token_type == type_idx
            if not bool(mask.any().item()):
                continue
            any_type = True
            sim_type = sim.masked_fill(~mask.unsqueeze(1), -1.0e4)
            weights = torch.softmax(sim_type, dim=-1)
            routed_type = torch.einsum("bqn,bcn->bcq", weights, tokens).view(b, c, h, w)
            routed = routed + (routed_type * type_gates[:, type_idx : type_idx + 1]).flatten(2)
        if not any_type:
            return None
        return self._normalize_style_map(routed.view(b, c, h, w)).to(dtype=content_feat_16.dtype)

    def _memory_bank_scalar(self, name: str, default: float) -> float:
        value = getattr(self, name, None)
        if torch.is_tensor(value) and value.numel() > 0:
            return float(value.detach().cpu().reshape(-1)[0].item())
        return float(default)

    def _build_style_memory_residual_source(
        self,
        routed_memory: torch.Tensor | None,
        base_map: torch.Tensor | None,
        content_feat_16: torch.Tensor,
    ) -> torch.Tensor | None:
        strength = max(0.0, self._memory_bank_scalar("style_memory_bank_residual_strength", 0.0))
        if routed_memory is None or strength <= 0.0:
            return None
        source = routed_memory.float()
        if self._memory_bank_scalar("style_memory_bank_residual_center_content", 0.0) > 0.5:
            source = source - self._normalize_style_map(content_feat_16).float()
        elif self._memory_bank_scalar("style_memory_bank_residual_center_base", 1.0) > 0.5 and base_map is not None:
            source = source - base_map.float()
        kernel = int(round(self._memory_bank_scalar("style_memory_bank_residual_highpass_kernel", 1.0)))
        if kernel > 1:
            if kernel % 2 == 0:
                kernel += 1
            source = source - F.avg_pool2d(source, kernel_size=kernel, stride=1, padding=kernel // 2)
        gate_gamma = max(0.0, self._memory_bank_scalar("style_memory_bank_residual_gate_gamma", 0.0))
        if gate_gamma > 0.0:
            gate = self._style_band_support_gate(
                content_feat_16,
                gamma=gate_gamma,
                floor=max(0.0, min(1.0, self._memory_bank_scalar("style_memory_bank_residual_gate_floor", 0.20))),
                kernel=int(round(self._memory_bank_scalar("style_memory_bank_residual_gate_kernel", 5.0))),
            )
            source = source * gate.float()
        scale = max(1e-4, self._memory_bank_scalar("style_memory_bank_residual_tanh_scale", 0.55))
        add = torch.tanh(source / scale) * scale * strength
        return add.to(device=content_feat_16.device, dtype=content_feat_16.dtype)

    @staticmethod
    def _match_style_map(style_map: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor | None:
        if style_map is None:
            return None
        if style_map.shape[-2:] != target.shape[-2:]:
            style_map = F.interpolate(
                style_map,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if style_map.dtype != target.dtype:
            style_map = style_map.to(dtype=target.dtype)
        if style_map.device != target.device:
            style_map = style_map.to(device=target.device)
        return style_map

    def _predict_delta_from_context(
        self,
        x: torch.Tensor,
        *,
        style_tokens: object,
        style_maps: StyleMaps,
        override_palette: torch.Tensor | None = None,
        strength: float,
        target_style_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._last_style_token_fields = style_tokens
        feat_c = x / max(self.latent_scale_factor, 1e-8)
        h_c = self.enc_in_act(self.enc_in(feat_c))
        with torch.no_grad():
            h_c_no_grad = h_c.clone()
            for block in self.hires_body:
                h_c_no_grad = block(h_c_no_grad)
        skip_32 = h_c_no_grad

        h_c_grad = h_c
        for block in self.hires_body:
            h_c_grad = block(h_c_grad)
        content_feat_16 = self.down(h_c_grad)
        style_map_proj: torch.Tensor | None = None
        memory_residual_source: torch.Tensor | None = None

        if override_palette is not None:
            style_map_proj = override_palette
            if style_map_proj.device != content_feat_16.device:
                style_map_proj = style_map_proj.to(device=content_feat_16.device)
            if style_map_proj.dtype != content_feat_16.dtype:
                style_map_proj = style_map_proj.to(dtype=content_feat_16.dtype)
            if style_map_proj.shape[0] == 1 and content_feat_16.shape[0] > 1:
                style_map_proj = style_map_proj.expand(content_feat_16.shape[0], -1, -1, -1)
            elif style_map_proj.shape[0] != content_feat_16.shape[0]:
                raise ValueError(
                    "override_palette batch mismatch: "
                    f"expected {content_feat_16.shape[0]} or 1, got {style_map_proj.shape[0]}"
                )
            if style_map_proj.shape[1] == self.latent_channels:
                feat_s = style_map_proj / max(self.latent_scale_factor, 1e-8)
                h_s = self.enc_in_act(self.enc_in(feat_s))
                for block in self.hires_body:
                    h_s = block(h_s)
                style_map_proj = self.down(h_s)
            elif style_map_proj.shape[1] != self.body_channels:
                raise ValueError(
                    f"override_palette channels must be {self.body_channels} or {self.latent_channels}, got {style_map_proj.shape[1]}"
                )
            if style_map_proj.shape[-2:] != content_feat_16.shape[-2:]:
                style_map_proj = F.interpolate(
                    style_map_proj,
                    size=content_feat_16.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
        elif target_style_latent is not None:
            if target_style_latent.shape[1] != self.latent_channels:
                raise ValueError(
                    f"target_style_latent channels must be {self.latent_channels}, got {target_style_latent.shape[1]}"
                )
            if target_style_latent.device != content_feat_16.device:
                target_style_latent = target_style_latent.to(device=content_feat_16.device)
            if target_style_latent.dtype != content_feat_16.dtype:
                target_style_latent = target_style_latent.to(dtype=content_feat_16.dtype)

            feat_s = target_style_latent / max(self.latent_scale_factor, 1e-8)
            h_s = self.enc_in_act(self.enc_in(feat_s))
            for block in self.hires_body:
                h_s = block(h_s)
            style_map_proj = self.down(h_s)
        else:
            style_spatial_16 = self._prepare_spatial_map(style_maps.map_16, content_feat_16)
            if style_spatial_16 is None:
                raise ValueError("style spatial prior is required for id-only inference.")
            routed_memory = self._build_content_routed_style_memory(
                content_feat_16,
                style_id=style_maps.style_id,
            )
            if routed_memory is not None:
                memory_residual_source = self._build_style_memory_residual_source(
                    routed_memory,
                    style_spatial_16,
                    content_feat_16,
                )
                route_strength_value = getattr(self, "style_memory_bank_route_strength", None)
                route_strength = (
                    float(route_strength_value.detach().cpu().item())
                    if torch.is_tensor(route_strength_value) and route_strength_value.numel() > 0
                    else 0.0
                )
                route_strength = max(0.0, min(1.0, route_strength))
                style_spatial_16 = style_spatial_16.lerp(routed_memory, route_strength)
            style_map_proj = style_spatial_16

        semantic_attn: torch.Tensor | None = None
        self.carrier_debug = {}
        type_gates = getattr(self, "_last_style_memory_type_gates", None)
        if type_gates is not None:
            self.carrier_debug["style_memory_type_gates"] = type_gates.detach()
            self._last_style_memory_type_gates = None
        if self.use_style_blender:
            h_painted = content_feat_16
            for block in self.body_blocks:
                h_painted = block(h_painted, style_map=style_map_proj, gate=1.0)
                semantic_attn = getattr(block, "last_attn", semantic_attn)
            if self.blender is None:
                raise RuntimeError("Style blender is enabled but not initialized.")
            h_body = self.blender(content_feat_16, h_painted, semantic_attn=semantic_attn, style_tokens=style_tokens)
            if style_tokens is not None:
                self.carrier_debug["style_token_grammar"] = style_tokens.grammar.detach()
                self.carrier_debug["style_token_band_gains"] = style_tokens.band_gains.detach()
            blender_debug = getattr(self.blender, "last_debug", None)
            if blender_debug:
                self.carrier_debug = {
                    **dict(getattr(self, "carrier_debug", {}) or {}),
                    **{key: value.detach() for key, value in blender_debug.items() if value is not None},
                }
        else:
            h = content_feat_16
            for block in self.body_blocks:
                h = block(h, style_map=style_map_proj, gate=1.0)
                semantic_attn = getattr(block, "last_attn", semantic_attn)
            h_body = h
        if memory_residual_source is not None:
            h_body = h_body + memory_residual_source
            self.carrier_debug["style_memory_residual_delta"] = memory_residual_source.detach()
        h_up = self.dec_up(h_body)
        h_up = self._apply_upsample_blur(h_up)
        h_fused = self._fuse_skip_features(h_up, skip_32)
        h_dec = self._run_decoder(h_fused)
        token_feature_operator = getattr(self, "style_token_feature_operator", None)
        token_feature_strength = float(getattr(self, "dynamic_style_feature_operator_strength", 0.0))
        self.carrier_debug = dict(getattr(self, "carrier_debug", {}) or {})
        if token_feature_operator is not None and token_feature_strength > 0.0:
            token_feature_delta = token_feature_operator(h_dec, style_tokens)
            token_feature_scale = float(getattr(self, "dynamic_style_feature_operator_tanh_scale", 4.0))
            token_feature_add = torch.tanh(token_feature_delta / token_feature_scale) * token_feature_scale * token_feature_strength
            self.carrier_debug["style_token_feature_delta"] = token_feature_add.detach()
            h_dec = h_dec + token_feature_add
        self._current_semantic_attn_for_head = semantic_attn
        delta_raw = self._compute_delta(h_dec, x=x)
        self._current_semantic_attn_for_head = None
        return delta_raw

    def integrate(
        self,
        x: torch.Tensor,
        style_id: torch.Tensor | int | None,
        num_steps: int = 1,
        step_size: float = 1.0,
        style_strength: float | None = None,
        target_style_latent: torch.Tensor | None = None,
        override_palette: torch.Tensor | None = None,
    ) -> torch.Tensor:
        steps = max(1, int(num_steps))
        strength = self._resolve_style_strength(style_strength)
        step_scale = self._style_strength_step_scale(strength)
        per_step = 1.0 / float(steps)
        x = self._apply_pre_integrate_moment_match(x, target_style_latent)
        x = self._inject_flat_highfreq_canvas(x, target_style_latent)
        if style_id is None:
            raise ValueError("style_id is required.")
        style_tokens, style_maps = self._prepare_style_context(
            style_id=style_id,
            batch_size=x.shape[0],
            device=x.device,
            dtype=x.dtype,
        )
        h = x
        self._integration_anchor_x = x
        for _ in range(steps):
            delta = self._predict_delta_from_context(
                h,
                style_tokens=style_tokens,
                style_maps=style_maps,
                override_palette=override_palette,
                strength=strength,
                target_style_latent=target_style_latent,
            )
            delta = self._apply_structure_barrier(delta, self._integration_anchor_x if bool(getattr(self, "structure_barrier_use_anchor", True)) else h)
            h = h + delta * float(step_size) * step_scale * per_step
        self._integration_anchor_x = None
        return self._apply_output_moment_match(h, target_style_latent)

    def _apply_pre_integrate_moment_match(
        self,
        x: torch.Tensor,
        target_style_latent: torch.Tensor | None,
    ) -> torch.Tensor:
        if (not self.pre_integrate_moment_match) or target_style_latent is None:
            return x
        ref = target_style_latent
        if ref.shape != x.shape:
            raise ValueError(
                "target_style_latent shape must match model input shape, "
                f"got x={tuple(x.shape)} ref={tuple(ref.shape)}"
            )
        ref = ref.to(device=x.device, dtype=x.dtype)
        x_mean = x.mean(dim=(2, 3), keepdim=True)
        x_std = x.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(self.output_moment_match_eps)
        ref_mean = ref.mean(dim=(2, 3), keepdim=True)
        ref_std = ref.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(self.output_moment_match_eps)
        mapped = ((x - x_mean) / x_std) * ref_std + ref_mean
        return x.lerp(mapped, self.pre_integrate_moment_blend)

    def _inject_flat_highfreq_canvas(
        self,
        x: torch.Tensor,
        target_style_latent: torch.Tensor | None,
    ) -> torch.Tensor:
        strength = float(getattr(self, "latent_canvas_strength", 0.0))
        if strength <= 0.0:
            return x
        ref = target_style_latent
        if ref is None or ref.shape != x.shape:
            noise = torch.randn_like(x)
        else:
            ref = ref.to(device=x.device, dtype=x.dtype)
            kernel = max(1, int(getattr(self, "latent_canvas_highpass_kernel", 5)))
            if kernel % 2 == 0:
                kernel += 1
            noise = ref - F.avg_pool2d(ref.float(), kernel_size=kernel, stride=1, padding=kernel // 2).to(dtype=ref.dtype)
        gx = x[..., :, 1:] - x[..., :, :-1]
        gy = x[..., 1:, :] - x[..., :-1, :]
        gx = F.pad(gx.float(), (0, 1, 0, 0))
        gy = F.pad(gy.float(), (0, 0, 0, 1))
        edge = torch.sqrt(gx.square() + gy.square() + 1e-12).mean(dim=1, keepdim=True)
        flat_mask = torch.exp(-float(getattr(self, "latent_canvas_edge_gamma", 4.0)) * edge)
        return x + noise * flat_mask.to(dtype=x.dtype) * strength

    def _perturb_anchor_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_anchor_noise_std <= 0.0:
            return x
        if (not self.training) and (not self.input_anchor_noise_eval):
            return x
        return x + torch.randn_like(x) * self.input_anchor_noise_std

    def _apply_output_moment_match(
        self,
        pred: torch.Tensor,
        target_style_latent: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.output_moment_match or target_style_latent is None:
            return pred
        if self.output_moment_match_train_only and not self.training:
            return pred

        ref = target_style_latent
        if ref.shape != pred.shape:
            raise ValueError(
                "target_style_latent shape must match model output shape, "
                f"got pred={tuple(pred.shape)} ref={tuple(ref.shape)}"
            )
        if ref.device != pred.device:
            ref = ref.to(device=pred.device)
        if ref.dtype != pred.dtype:
            ref = ref.to(dtype=pred.dtype)

        pred_mean = pred.mean(dim=(2, 3), keepdim=True)
        pred_std = pred.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(self.output_moment_match_eps)
        ref_mean = ref.mean(dim=(2, 3), keepdim=True)
        ref_std = ref.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(self.output_moment_match_eps)
        return ((pred - pred_mean) / pred_std) * ref_std + ref_mean

    def _apply_structure_barrier(
        self,
        delta: torch.Tensor,
        anchor: torch.Tensor,
    ) -> torch.Tensor:
        gamma = float(getattr(self, "structure_barrier_gamma", 0.0))
        if gamma <= 0.0:
            return delta
        ref = anchor.float()
        kernel = max(1, int(getattr(self, "structure_barrier_smooth_kernel", 3)))
        if kernel > 1:
            if kernel % 2 == 0:
                kernel += 1
            ref = F.avg_pool2d(ref, kernel_size=kernel, stride=1, padding=kernel // 2)
        gx = ref[..., :, 1:] - ref[..., :, :-1]
        gy = ref[..., 1:, :] - ref[..., :-1, :]
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = F.pad(gy, (0, 0, 0, 1))
        edge = torch.sqrt(gx.square() + gy.square() + 1e-12).mean(dim=1, keepdim=True)
        barrier = torch.exp(-gamma * edge).to(device=delta.device, dtype=delta.dtype)
        return delta * barrier

    def forward(
        self,
        x: torch.Tensor,
        style_id: torch.Tensor | int | None,
        step_size: float = 1.0,
        style_strength: float | None = None,
        target_style_latent: torch.Tensor | None = None,
        override_palette: torch.Tensor | None = None,
    ) -> torch.Tensor:
        strength = self._resolve_style_strength(style_strength)
        step_scale = self._style_strength_step_scale(strength)
        if style_id is None:
            raise ValueError("style_id is required.")
        style_tokens, style_maps = self._prepare_style_context(
            style_id=style_id,
            batch_size=x.shape[0],
            device=x.device,
            dtype=x.dtype,
        )
        delta = self._predict_delta_from_context(
            x,
            style_tokens=style_tokens,
            style_maps=style_maps,
            override_palette=override_palette,
            strength=strength,
            target_style_latent=target_style_latent,
        )
        barrier_anchor = getattr(self, "_integration_anchor_x", None) if bool(getattr(self, "structure_barrier_use_anchor", True)) else None
        if barrier_anchor is None:
            barrier_anchor = x
        delta = self._apply_structure_barrier(delta, barrier_anchor)
        if self.ablation_no_residual:
            pred = (delta / (self.latent_scale_factor * max(self.residual_gain, 1e-5))) * self.ablation_no_residual_gain
            return self._apply_output_moment_match(pred, target_style_latent)

        anchor = self._perturb_anchor_if_needed(x)
        pred = anchor + delta * float(step_size) * step_scale
        return self._apply_output_moment_match(pred, target_style_latent)
