from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

from lancet_blocks import AttentionBlock, StyleMaps
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

    def _run_block(
        self,
        block: nn.Module,
        h: torch.Tensor,
        style_code: torch.Tensor,
        gate: float | torch.Tensor = 1.0,
        shift: bool = False,
    ) -> torch.Tensor:
        use_shift = bool(
            shift
            and isinstance(block, AttentionBlock)
            and getattr(getattr(block, "attn", None), "mode", None) == "window_attn"
        )
        gate_in = gate.to(device=h.device, dtype=h.dtype) if torch.is_tensor(gate) else h.new_tensor(float(gate))
        gate_is_zero = bool(torch.count_nonzero(gate_in.detach()).item() == 0)
        if self.use_checkpointing and self.training and not gate_is_zero:
            return ckpt.checkpoint(
                lambda _h, _s, _g, _blk=block, _use_shift=use_shift: (
                    _blk(_h, _s, _g, shift=True) if _use_shift else _blk(_h, _s, _g)
                ),
                h,
                style_code,
                gate_in,
                use_reentrant=False,
            )
        if use_shift:
            return block(h, style_code, gate=gate_in, shift=True)
        return block(h, style_code, gate=gate_in)

    def _run_style_blocks(
        self,
        h: torch.Tensor,
        blocks: nn.ModuleList,
        style_code: torch.Tensor,
        base_idx: int = 0,
        gate_scale: float = 1.0,
    ) -> torch.Tensor:
        out = h
        gs = max(0.0, float(gate_scale))
        for i, block in enumerate(blocks):
            use_shift = (i % 2) == 1
            current_gate = torch.tanh(F.softplus(self.block_gains[base_idx + i])) * gs
            out = self._run_block(block, out, style_code, gate=current_gate, shift=use_shift)
        return out

    def _fuse_skip_features(
        self,
        h_up: torch.Tensor,
        skip_32: torch.Tensor,
        style_code: torch.Tensor,
        gate: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        # Hard no-skip path: physically disconnect encoder skip source and keep only
        # the upsample projection branch.
        if self.skip_disabled:
            return self.skip_fusion(self.skip_up_proj(h_up))

        skip_feat = self.skip_router(
            skip_32,
            style_code=style_code,
            gate=gate,
            naive_gain=self.skip_naive_gain,
        )

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
        return StyleMaps(
            map_16=self.encode_style_spatial_id(style_id).get(16),
        )

    def _prepare_spatial_map(self, style_map: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor | None:
        return self._match_style_map(style_map, target)

    def _prepare_style_context(
        self,
        *,
        style_id: torch.Tensor | int,
    ) -> tuple[torch.Tensor, StyleMaps]:
        style_code = self.encode_style_id(style_id)
        style_maps = self._prepare_style_maps(style_id=style_id)
        return style_code, style_maps

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
        style_code = getattr(self, "_current_style_code_for_head", None)
        carrier_debug: dict[str, torch.Tensor | None] = dict(getattr(self, "carrier_debug", {}) or {})
        raw = self._decode_output_raw(h, style_code=style_code)
        self.last_raw_diffeomorphic = raw
        if bool(getattr(self, "use_diffeomorphic_stroke", False)):
            if x is None:
                raise ValueError("diffeomorphic stroke mode requires input x.")
            return self._apply_diffeomorphic_stroke(x, raw)
        delta = raw * self.latent_scale_factor * self.residual_gain
        delta = torch.tanh(delta / 4.0) * 4.0
        carrier_debug["raw_delta"] = delta.detach()
        highpass_head = getattr(self, "style_highpass_head", None)
        highpass_strength = float(getattr(self, "style_highpass_depthwise_strength", 0.0))
        style_code = getattr(self, "_current_style_code_for_head", None)
        if highpass_head is not None and highpass_strength > 0.0 and style_code is not None:
            if x is None:
                raise ValueError("style highpass depthwise head requires input x.")
            highpass_delta = highpass_head(x, style_code)
            if bool(getattr(self, "style_highpass_depthwise_support_gate", False)):
                highpass_gate = self._style_highpass_support_gate(x)
                carrier_debug["highpass_gate"] = highpass_gate.detach()
                highpass_delta = highpass_delta * highpass_gate
            if bool(getattr(self, "style_highpass_depthwise_semantic_gate", False)):
                semantic_attn = getattr(self, "_current_semantic_attn_for_head", None)
                semantic_gate = self._style_highpass_semantic_gate(semantic_attn, x)
                carrier_debug["highpass_semantic_gate"] = semantic_gate.detach()
                highpass_delta = highpass_delta * semantic_gate
            if bool(getattr(self, "style_highpass_depthwise_region_gate", False)):
                region_gate_head = getattr(self, "style_highpass_region_gate_head", None)
                if region_gate_head is not None:
                    region_gate = region_gate_head(x, style_code)
                    carrier_debug["highpass_region_gate"] = region_gate.detach()
                    highpass_delta = highpass_delta * region_gate
            highpass_add = torch.tanh(highpass_delta / 4.0) * 4.0 * highpass_strength
            carrier_debug["highpass_delta"] = highpass_add.detach()
            delta = delta + highpass_add
        lowpass_head = getattr(self, "style_lowpass_head", None)
        lowpass_strength = float(getattr(self, "style_lowpass_affine_strength", 0.0))
        if lowpass_head is not None and lowpass_strength > 0.0 and style_code is not None:
            if x is None:
                raise ValueError("style lowpass affine head requires input x.")
            lowpass_delta = lowpass_head(x, style_code)
            lowpass_add = torch.tanh(lowpass_delta / 4.0) * 4.0 * lowpass_strength
            carrier_debug["lowpass_affine_delta"] = lowpass_add.detach()
            delta = delta + lowpass_add
        lowpass_mix = getattr(self, "style_lowpass_mix", None)
        lowpass_mix_strength = float(getattr(self, "style_lowpass_mix_strength", 0.0))
        if lowpass_mix is not None and lowpass_mix_strength > 0.0 and style_code is not None:
            if x is None:
                raise ValueError("style lowpass mix head requires input x.")
            lowpass_mix_delta = lowpass_mix(x, style_code)
            lowpass_mix_add = torch.tanh(lowpass_mix_delta / 4.0) * 4.0 * lowpass_mix_strength
            carrier_debug["lowpass_mix_delta"] = lowpass_mix_add.detach()
            delta = delta + lowpass_mix_add
        midband_operator = getattr(self, "style_midband_operator", None)
        midband_strength = float(getattr(self, "style_midband_operator_strength", 0.0))
        if midband_operator is not None and midband_strength > 0.0 and style_code is not None:
            if x is None:
                raise ValueError("style midband operator head requires input x.")
            midband_delta = midband_operator(x, style_code)
            if bool(getattr(self, "style_midband_operator_support_gate", False)):
                gamma = float(getattr(self, "style_midband_operator_support_gamma", 4.5))
                floor = float(getattr(self, "style_midband_operator_support_floor", 0.15))
                kernel = max(1, int(getattr(self, "style_midband_operator_support_smooth_kernel", 5)))
                midband_gate = self._style_band_support_gate(x, gamma=gamma, floor=floor, kernel=kernel)
                carrier_debug["midband_gate"] = midband_gate.detach()
                midband_delta = midband_delta * midband_gate
            midband_add = torch.tanh(midband_delta / 4.0) * 4.0 * midband_strength
            carrier_debug["midband_delta"] = midband_add.detach()
            delta = delta + midband_add
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

    def _style_highpass_support_gate(self, x: torch.Tensor) -> torch.Tensor:
        gamma = float(getattr(self, "style_highpass_depthwise_support_gamma", 4.0))
        floor = float(getattr(self, "style_highpass_depthwise_support_floor", 0.0))
        kernel = max(1, int(getattr(self, "style_highpass_depthwise_support_smooth_kernel", 3)))
        return self._style_band_support_gate(x, gamma=gamma, floor=floor, kernel=kernel)

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

    def _style_highpass_semantic_gate(
        self,
        semantic_attn: torch.Tensor | None,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if semantic_attn is None:
            return torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype)
        attn = semantic_attn.float()
        if attn.ndim != 3 or attn.shape[0] != x.shape[0]:
            return torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype)
        tokens = int(attn.shape[1])
        side = int(round(tokens ** 0.5))
        if side * side != tokens:
            return torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype)
        confidence = attn.max(dim=-1).values.view(x.shape[0], 1, side, side)
        power = float(getattr(self, "style_highpass_depthwise_semantic_power", 1.0))
        if power != 1.0:
            confidence = confidence.clamp_min(0.0).pow(power)
        denom = confidence.flatten(1).mean(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
        gamma = float(getattr(self, "style_highpass_depthwise_semantic_gamma", 3.0))
        gate = 1.0 - torch.exp(-gamma * confidence / denom)
        floor = float(getattr(self, "style_highpass_depthwise_semantic_floor", 0.0))
        if floor > 0.0:
            gate = floor + (1.0 - floor) * gate
        gate = F.interpolate(gate, size=x.shape[-2:], mode="bilinear", align_corners=False)
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

    def encode_style_id(self, style_id: torch.Tensor | int | None) -> torch.Tensor:
        if style_id is None:
            raise ValueError("style_id is required.")
        emb_device = self.style_emb.weight.device
        style_id = self._normalize_style_id_input(style_id, device=emb_device)
        base_code = self.style_emb(style_id)
        self._last_style_token_fields = None
        tokenizer = getattr(self, "style_tokenizer", None)
        if tokenizer is None:
            return base_code
        style_code, fields = tokenizer(style_id, base_code)
        self._last_style_token_fields = fields
        return style_code

    @staticmethod
    def _normalize_style_map(feat: torch.Tensor) -> torch.Tensor:
        feat = feat - feat.mean(dim=(2, 3), keepdim=True)
        return feat / (feat.std(dim=(2, 3), keepdim=True, unbiased=False) + 1e-6)

    def encode_style_spatial_id(self, style_id: torch.Tensor | int) -> dict[int, torch.Tensor]:
        spatial_device = self.style_spatial_id_16.device
        style_id = self._normalize_style_id_input(style_id, device=spatial_device)
        maps = {16: self.style_spatial_id_16.index_select(0, style_id)}
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
        style_code: torch.Tensor,
        style_maps: StyleMaps,
        override_palette: torch.Tensor | None = None,
        strength: float,
        target_style_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feat_c = x / max(self.latent_scale_factor, 1e-8)
        h_c = self.enc_in_act(self.enc_in(feat_c))
        with torch.no_grad():
            h_c_no_grad = h_c.clone()
            for block in self.hires_body:
                h_c_no_grad = block(h_c_no_grad, style_code, gate=0.0)
        skip_32 = h_c_no_grad

        h_c_grad = h_c
        for block in self.hires_body:
            h_c_grad = block(h_c_grad, style_code, gate=0.0)
        content_feat_16 = self.down(h_c_grad)
        style_map_proj: torch.Tensor | None = None

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
                h_s = self._run_style_blocks(
                    h_s,
                    blocks=self.hires_body,
                    style_code=style_code,
                    base_idx=0,
                    gate_scale=0.0,
                )
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
            h_s = self._run_style_blocks(
                h_s,
                blocks=self.hires_body,
                style_code=style_code,
                base_idx=0,
                gate_scale=0.0,
            )
            style_map_proj = self.down(h_s)
        else:
            style_spatial_16 = self._prepare_spatial_map(style_maps.map_16, content_feat_16)
            if style_spatial_16 is None:
                raise ValueError("style spatial prior is required for id-only inference.")
            style_map_proj = style_spatial_16

        semantic_attn: torch.Tensor | None = None
        self.carrier_debug = {}
        if self.use_style_blender:
            h_painted = content_feat_16
            for block in self.body_blocks:
                h_painted = block(h_painted, style_map=style_map_proj, gate=1.0)
                semantic_attn = getattr(block, "last_attn", semantic_attn)
            if self.blender is None:
                raise RuntimeError("Style blender is enabled but not initialized.")
            style_tokens = getattr(self, "_last_style_token_fields", None)
            h_body = self.blender(content_feat_16, h_painted, style_code, semantic_attn, style_tokens=style_tokens)
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
        h_up = self.dec_up(h_body)
        h_up = self._apply_upsample_blur(h_up)
        h_fused = self._fuse_skip_features(h_up, skip_32, style_code=style_code, gate=1.0)
        h_dec = self._run_decoder(h_fused)
        feature_operator = getattr(self, "decoder_feature_style_operator", None)
        feature_strength = float(getattr(self, "decoder_feature_style_operator_strength", 0.0))
        self.carrier_debug = dict(getattr(self, "carrier_debug", {}) or {})
        if feature_operator is not None and feature_strength > 0.0:
            feature_delta = feature_operator(h_dec, style_code)
            if bool(getattr(self, "decoder_feature_style_operator_support_gate", False)):
                gamma = float(getattr(self, "decoder_feature_style_operator_support_gamma", 4.5))
                floor = float(getattr(self, "decoder_feature_style_operator_support_floor", 0.15))
                kernel = max(1, int(getattr(self, "decoder_feature_style_operator_support_smooth_kernel", 5)))
                feature_gate = self._style_band_support_gate(x, gamma=gamma, floor=floor, kernel=kernel)
                feature_delta = feature_delta * feature_gate
                self.carrier_debug["decoder_feature_gate"] = feature_gate.detach()
            feature_add = torch.tanh(feature_delta / 4.0) * 4.0 * feature_strength
            self.carrier_debug["decoder_feature_delta"] = feature_add.detach()
            h_dec = h_dec + feature_add
        self._current_style_code_for_head = style_code
        self._current_semantic_attn_for_head = semantic_attn
        delta_raw = self._compute_delta(h_dec, x=x)
        self._current_style_code_for_head = None
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
        style_code_override: torch.Tensor | None = None,
        override_palette: torch.Tensor | None = None,
    ) -> torch.Tensor:
        steps = max(1, int(num_steps))
        strength = self._resolve_style_strength(style_strength)
        step_scale = self._style_strength_step_scale(strength)
        per_step = 1.0 / float(steps)
        x = self._apply_pre_integrate_moment_match(x, target_style_latent)
        x = self._inject_flat_highfreq_canvas(x, target_style_latent)
        if style_code_override is not None:
            self._last_style_token_fields = None
            style_code = style_code_override
            if style_code.ndim == 1:
                style_code = style_code.unsqueeze(0)
            if style_code.device != x.device:
                style_code = style_code.to(device=x.device)
            if style_code.dtype != x.dtype:
                style_code = style_code.to(dtype=x.dtype)
            if style_code.shape[0] == 1 and x.shape[0] > 1:
                style_code = style_code.expand(x.shape[0], -1)
            elif style_code.shape[0] != x.shape[0]:
                raise ValueError(f"style_code_override batch mismatch: expected {x.shape[0]} or 1, got {style_code.shape[0]}")
            style_maps = StyleMaps()
        else:
            if style_id is None:
                raise ValueError("style_id is required when style_code_override is not provided.")
            style_code, style_maps = self._prepare_style_context(
                style_id=style_id,
            )
        h = x
        self._integration_anchor_x = x
        for _ in range(steps):
            delta = self._predict_delta_from_context(
                h,
                style_code=style_code,
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
        style_code_override: torch.Tensor | None = None,
        override_palette: torch.Tensor | None = None,
    ) -> torch.Tensor:
        strength = self._resolve_style_strength(style_strength)
        step_scale = self._style_strength_step_scale(strength)
        if style_code_override is not None:
            self._last_style_token_fields = None
            style_code = style_code_override
            if style_code.ndim == 1:
                style_code = style_code.unsqueeze(0)
            if style_code.device != x.device:
                style_code = style_code.to(device=x.device)
            if style_code.dtype != x.dtype:
                style_code = style_code.to(dtype=x.dtype)
            if style_code.shape[0] == 1 and x.shape[0] > 1:
                style_code = style_code.expand(x.shape[0], -1)
            elif style_code.shape[0] != x.shape[0]:
                raise ValueError(f"style_code_override batch mismatch: expected {x.shape[0]} or 1, got {style_code.shape[0]}")
            style_maps = StyleMaps()
        else:
            if style_id is None:
                raise ValueError("style_id is required when style_code_override is not provided.")
            style_code, style_maps = self._prepare_style_context(
                style_id=style_id,
            )
        delta = self._predict_delta_from_context(
            x,
            style_code=style_code,
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
