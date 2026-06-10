from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

from lancet_blocks import AttentionBlock, StyleMaps
from utils.diffeomorphic import apply_texture_aligned_diffeomorphic_stroke


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
        if torch.is_tensor(gate):
            gate_in = gate.to(device=h.device, dtype=h.dtype)
            gate_is_zero = False
        else:
            gate_value = float(gate)
            gate_in = h.new_tensor(gate_value)
            gate_is_zero = gate_value == 0.0
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
        return style_code, StyleMaps(family=str(getattr(self, "tokenizer_family", "legacy_factorized")))

    def _runtime_conditioning_payload(self) -> dict:
        payload = getattr(self, "runtime_conditioning", None)
        return payload if isinstance(payload, dict) else {}

    def _structured_style_from_sidecar(
        self,
        *,
        style_id: torch.Tensor | int | None,
        style_code: torch.Tensor,
        content_feat_16: torch.Tensor,
    ) -> tuple[torch.Tensor, StyleMaps] | None:
        tokenizer = getattr(self, "structured_style_tokenizer", None)
        family = str(getattr(self, "tokenizer_family", "legacy_factorized"))
        if tokenizer is None or family == "legacy_factorized" or style_id is None:
            return None
        payload = self._runtime_conditioning_payload()
        content_dino_patches = payload.get("content_dino_patches")
        if not torch.is_tensor(content_dino_patches):
            return None
        kwargs = {
            "style_id": self._normalize_style_id_input(style_id, device=style_code.device),
            "base_style_code": style_code,
            "content_dino_patches": content_dino_patches.to(device=style_code.device, dtype=style_code.dtype),
            "target_hw": tuple(int(v) for v in content_feat_16.shape[-2:]),
        }
        if family == "tok_b_cross_image":
            bank = payload.get("target_style_dino_bank_patches")
            if not torch.is_tensor(bank):
                return None
            kwargs["style_bank_patches"] = bank.to(device=style_code.device, dtype=style_code.dtype)
        structured = tokenizer(**kwargs)
        return structured.global_code, StyleMaps(
            map_16=structured.spatial_map,
            gate_16=structured.gate_map,
            mask_16=structured.mask_map,
            aux_16=structured.aux_map,
            family=family,
            debug=dict(structured.debug),
        )

    def _content_spatial_features(self, feat: torch.Tensor) -> torch.Tensor:
        xf = feat.float()
        mean = xf.mean(dim=(2, 3))
        std = xf.std(dim=(2, 3), unbiased=False)
        abs_mean = xf.abs().mean(dim=(2, 3))
        low = F.avg_pool2d(xf, kernel_size=3, stride=1, padding=1)
        high_abs = (xf - low).abs().mean(dim=(2, 3))
        energy = xf.flatten(1).square().mean(dim=1, keepdim=True).sqrt()
        return torch.cat([mean, std, abs_mean, high_abs, energy], dim=1).to(device=feat.device, dtype=feat.dtype)

    def _tokenizer_mixture_weights(self, style_id: torch.Tensor) -> torch.Tensor | None:
        mixture = getattr(self.style_tokenizer, "mixture_weights", None)
        if not callable(mixture):
            return None
        weights = mixture(style_id.to(device=self.style_tokenizer.weight.device))
        if weights is None:
            return None
        return weights.to(device=style_id.device)

    def _atom_weights_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        tokenizer = self.style_tokenizer
        atom_topk = max(0, int(getattr(tokenizer, "atom_topk", 0)))
        if atom_topk > 0 and atom_topk < logits.shape[1]:
            topk = torch.topk(logits, k=atom_topk, dim=-1).indices
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask.scatter_(1, topk, True)
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        if bool(getattr(tokenizer, "atom_hard_eval", False)) and not self.training:
            idx = logits.argmax(dim=-1, keepdim=True)
            weights = torch.zeros_like(logits)
            weights.scatter_(1, idx, 1.0)
            return weights
        tau = max(1e-3, float(getattr(tokenizer, "atom_temperature", 0.25)))
        return F.softmax(logits / tau, dim=-1)

    def _style_code_from_atom_weights(self, style_id: torch.Tensor, weights: torch.Tensor) -> torch.Tensor | None:
        tokenizer = self.style_tokenizer
        mode = str(getattr(tokenizer, "projection_mode", "")).lower()
        if mode not in {"concept_atoms", "direct_atom_residual", "global_vq"}:
            return None
        if not hasattr(tokenizer, "concept_atoms"):
            return None
        atoms = tokenizer.concept_atoms.to(device=weights.device, dtype=weights.dtype)
        atom_code = weights @ atoms
        if mode == "direct_atom_residual":
            if not hasattr(tokenizer, "direct_code"):
                return None
            base = tokenizer.direct_code(style_id)
            style_code = base + float(getattr(tokenizer, "residual_gain", 1.0)) * atom_code
        else:
            style_code = atom_code
        if bool(getattr(tokenizer, "code_l2_norm", False)):
            style_code = F.normalize(style_code.float(), dim=1).to(dtype=style_code.dtype)
        code_scale = float(getattr(tokenizer, "code_scale", 1.0))
        if abs(code_scale - 1.0) > 1e-8:
            style_code = style_code * code_scale
        return style_code

    def _adapt_style_code_from_content(
        self,
        *,
        style_id: torch.Tensor | int | None,
        style_code: torch.Tensor,
        content_feat_16: torch.Tensor,
    ) -> torch.Tensor:
        router = getattr(self, "style_code_content_router", None)
        if router is None or style_id is None:
            return style_code
        tokenizer = self.style_tokenizer
        if not hasattr(tokenizer, "atom_logits"):
            return style_code

        token_device = tokenizer.weight.device
        style_id_t = self._normalize_style_id_input(style_id, device=token_device)
        batch = int(content_feat_16.shape[0])
        if style_id_t.shape[0] == 1 and batch > 1:
            style_id_t = style_id_t.expand(batch)
        elif style_id_t.shape[0] != batch:
            return style_code

        feat = content_feat_16.detach() if bool(getattr(self, "tokenizer_content_stopgrad", True)) else content_feat_16
        feat = self._content_spatial_features(feat).to(device=token_device, dtype=tokenizer.weight.dtype)
        routed = router(feat)
        base_logits = tokenizer.atom_logits(style_id_t)
        gain = float(getattr(self, "tokenizer_content_gain", 0.5))
        gate_values = None
        style_gate = getattr(self, "style_code_content_style_gate", None)
        if style_gate is not None:
            gate_max = max(1e-3, float(getattr(self, "tokenizer_content_style_gate_max", 2.0)))
            gate_values = torch.sigmoid(style_gate(style_id_t)).to(dtype=base_logits.dtype) * gate_max
            gain_tensor = gate_values * gain
        else:
            gain_tensor = gain
        weights = self._atom_weights_from_logits(base_logits + routed.to(dtype=base_logits.dtype) * gain_tensor)
        adapted = self._style_code_from_atom_weights(style_id_t, weights)
        if adapted is None:
            return style_code

        with torch.no_grad():
            probs = weights.detach().float()
            entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1).mean()
            debug = dict(getattr(tokenizer, "last_debug", {}) or {})
            debug.update(
                {
                    "content_atom_delta_abs": routed.detach().float().abs().mean(),
                    "content_atom_entropy": entropy,
                    "content_atom_effective_count": torch.exp(entropy),
                    "content_atom_max_prob": probs.max(dim=1).values.mean(),
                }
            )
            if gate_values is not None:
                gate_debug = gate_values.detach().float()
                debug.update(
                    {
                        "content_atom_gate_mean": gate_debug.mean(),
                        "content_atom_gate_min": gate_debug.min(),
                        "content_atom_gate_max": gate_debug.max(),
                    }
                )
            tokenizer.last_debug = debug

        return adapted.to(device=style_code.device, dtype=style_code.dtype)

    def _build_style_spatial_map(
        self,
        *,
        style_id: torch.Tensor | int,
        content_feat_16: torch.Tensor,
        style_code: torch.Tensor,
    ) -> torch.Tensor:
        del style_code
        spatial_device = self.style_spatial_id_16.device
        style_id_t = self._normalize_style_id_input(style_id, device=spatial_device)
        mode = str(getattr(self, "style_spatial_mode", "class")).lower()
        if mode == "class":
            return self.encode_style_spatial_id(style_id_t).get(16)

        content_feat = content_feat_16.to(device=spatial_device)
        base_logits = None
        if getattr(self, "style_spatial_logits", None) is not None:
            base_logits = self.style_spatial_logits(style_id_t)

        if mode in {"prototype", "content_guided"} and getattr(self, "style_spatial_proto_16", None) is not None:
            maps = self.style_spatial_proto_16.index_select(0, style_id_t)
            weights = self._tokenizer_mixture_weights(style_id_t)
            if weights is None or weights.shape[1] != maps.shape[1]:
                if base_logits is None:
                    base_logits = torch.zeros((style_id_t.shape[0], maps.shape[1]), device=spatial_device)
                weights = F.softmax(base_logits / float(self.style_spatial_routing_temperature), dim=-1)
            if mode == "content_guided" and getattr(self, "style_spatial_content_router", None) is not None:
                routed = self.style_spatial_content_router(self._content_spatial_features(content_feat))
                logits = torch.log(weights.clamp_min(1e-8)) + routed
                weights = F.softmax(logits / float(self.style_spatial_routing_temperature), dim=-1)
            out = (weights.view(weights.shape[0], weights.shape[1], 1, 1, 1) * maps).sum(dim=1)
            return self._normalize_style_map(out)

        if mode in {"vq", "vq_content_guided"} and getattr(self, "style_spatial_atoms_16", None) is not None:
            maps = self.style_spatial_atoms_16
            weights = self._tokenizer_mixture_weights(style_id_t)
            if weights is None or weights.shape[1] != maps.shape[0]:
                if base_logits is None:
                    base_logits = torch.zeros((style_id_t.shape[0], maps.shape[0]), device=spatial_device)
                weights = F.softmax(base_logits / float(self.style_spatial_routing_temperature), dim=-1)
            if mode == "vq_content_guided" and getattr(self, "style_spatial_content_router", None) is not None:
                routed = self.style_spatial_content_router(self._content_spatial_features(content_feat))
                logits = torch.log(weights.clamp_min(1e-8)) + routed
                weights = F.softmax(logits / float(self.style_spatial_routing_temperature), dim=-1)
            out = torch.einsum("bn,nchw->bchw", weights, maps)
            return self._normalize_style_map(out)

        return self.encode_style_spatial_id(style_id_t).get(16)

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
        raw = self.dec_out(h)
        if bool(getattr(self, "use_diffeomorphic_stroke", False)):
            if x is None:
                raise ValueError("diffeomorphic stroke mode requires input x.")
            return self._apply_diffeomorphic_stroke(x, raw)
        delta = raw * self.latent_scale_factor * self.residual_gain
        return torch.tanh(delta / 4.0) * 4.0

    def _apply_diffeomorphic_stroke(self, x: torch.Tensor, raw_out: torch.Tensor) -> torch.Tensor:
        do_profile = bool(getattr(self, "profile_modules", False))
        if do_profile and bool(getattr(self, "profile_sync_cuda", False)) and x.device.type == "cuda":
            torch.cuda.synchronize(x.device)
        t0 = time.perf_counter() if do_profile else 0.0
        stroked = apply_texture_aligned_diffeomorphic_stroke(
            x,
            raw_out,
            color_strength=float(getattr(self, "diffeomorphic_color_strength", 0.85)),
            warp_strength=float(getattr(self, "diffeomorphic_warp_strength", 0.08)),
            gate_strength=float(getattr(self, "diffeomorphic_texture_gate_strength", 8.0)),
            normal_leak=float(getattr(self, "diffeomorphic_normal_leak", 0.0)),
        )
        if do_profile:
            if bool(getattr(self, "profile_sync_cuda", False)) and x.device.type == "cuda":
                torch.cuda.synchronize(x.device)
            profile = getattr(self, "last_profile", None)
            if isinstance(profile, dict):
                profile["diffeomorphic_stroke"] = profile.get("diffeomorphic_stroke", 0.0) + max(
                    0.0,
                    time.perf_counter() - t0,
                )
        return stroked - x.float()

    def encode_style_id(self, style_id: torch.Tensor | int | None, t: torch.Tensor | None = None) -> torch.Tensor:
        if style_id is None:
            raise ValueError("style_id is required.")
        token_device = self.style_tokenizer.weight.device
        style_id = self._normalize_style_id_input(style_id, device=token_device)
        if t is not None and t.device != token_device:
            t = t.to(device=token_device)
        return self.style_tokenizer(style_id, t=t)

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
        style_id: torch.Tensor | int | None = None,
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
        style_code = self._adapt_style_code_from_content(
            style_id=style_id,
            style_code=style_code,
            content_feat_16=content_feat_16,
        )
        style_map_proj: torch.Tensor | None = None
        structured_ctx = self._structured_style_from_sidecar(
            style_id=style_id,
            style_code=style_code,
            content_feat_16=content_feat_16,
        )
        if structured_ctx is not None:
            style_code, style_maps = structured_ctx

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
            if style_maps.map_16 is None:
                if style_id is None:
                    raise ValueError("style_id is required for id-only spatial prior.")
                style_maps = StyleMaps(
                    map_16=self._build_style_spatial_map(
                        style_id=style_id,
                        content_feat_16=content_feat_16,
                        style_code=style_code,
                    )
                )
            style_spatial_16 = self._prepare_spatial_map(style_maps.map_16, content_feat_16)
            if style_spatial_16 is None:
                raise ValueError("style spatial prior is required for id-only inference.")
            style_map_proj = style_spatial_16
            if style_maps.mask_16 is not None:
                mask_16 = self._prepare_spatial_map(style_maps.mask_16, content_feat_16)
                if mask_16 is not None:
                    style_map_proj = style_map_proj * (0.5 + torch.sigmoid(mask_16))

        semantic_attn: torch.Tensor | None = None
        body_gate: float | torch.Tensor = 1.0
        if style_maps.gate_16 is not None:
            gate_16 = self._prepare_spatial_map(style_maps.gate_16, content_feat_16)
            if gate_16 is not None:
                body_gate = torch.sigmoid(gate_16)
        if self.use_style_blender:
            h_painted = content_feat_16
            for block in self.body_blocks:
                h_painted = block(h_painted, style_map=style_map_proj, gate=body_gate)
                semantic_attn = getattr(block, "last_attn", semantic_attn)
            if self.blender is None:
                raise RuntimeError("Style blender is enabled but not initialized.")
            h_body = self.blender(content_feat_16, h_painted)
        else:
            h = content_feat_16
            for block in self.body_blocks:
                h = block(h, style_map=style_map_proj, gate=body_gate)
                semantic_attn = getattr(block, "last_attn", semantic_attn)
            h_body = h
        style_inject = getattr(self, "_apply_style_feature_injection", None)
        if callable(style_inject):
            h_body = style_inject(h_body, x, style_code, site="body", style_map=style_map_proj)
        h_up = self.dec_up(h_body)
        h_up = self._apply_upsample_blur(h_up)
        h_fused = self._fuse_skip_features(h_up, skip_32, style_code=style_code, gate=1.0)
        h_dec = self._run_decoder(h_fused)
        if callable(style_inject):
            h_dec = style_inject(h_dec, x, style_code, site="decoder", style_map=style_map_proj)
        h_dec = self.dec_act(self.dec_mod(h_dec, style_code, gate=1.0))
        delta_raw = self._compute_delta(h_dec, x=x)
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
        if style_code_override is not None:
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
        for _ in range(steps):
            delta = self._predict_delta_from_context(
                h,
                style_id=style_id,
                style_code=style_code,
                style_maps=style_maps,
                override_palette=override_palette,
                strength=strength,
                target_style_latent=target_style_latent,
            )
            h = h + delta * float(step_size) * step_scale * per_step
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
            style_id=style_id,
            style_code=style_code,
            style_maps=style_maps,
            override_palette=override_palette,
            strength=strength,
            target_style_latent=target_style_latent,
        )
        if self.ablation_no_residual:
            pred = (delta / (self.latent_scale_factor * max(self.residual_gain, 1e-5))) * self.ablation_no_residual_gain
            return self._apply_output_moment_match(pred, target_style_latent)

        anchor = self._perturb_anchor_if_needed(x)
        pred = anchor + delta * float(step_size) * step_scale
        return self._apply_output_moment_match(pred, target_style_latent)
