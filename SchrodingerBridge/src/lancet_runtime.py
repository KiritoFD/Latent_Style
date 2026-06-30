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
    def _style_code_anchor_tensor(self) -> torch.Tensor:
        anchor = getattr(self, "_style_code_anchor", None)
        if torch.is_tensor(anchor):
            return anchor
        tokenizer = getattr(self, "style_tokenizer", None)
        weight = getattr(tokenizer, "weight", None)
        if torch.is_tensor(weight):
            return weight
        raise RuntimeError("Style-code anchor tensor is unavailable.")

    def _style_code_width(self) -> int:
        width = getattr(self, "style_code_dim", None)
        if width is not None:
            return int(width)
        tokenizer = getattr(self, "style_tokenizer", None)
        embedding_dim = getattr(tokenizer, "embedding_dim", None)
        if embedding_dim is not None:
            return int(embedding_dim)
        bridge_style_dim = getattr(self, "bridge_style_dim", None)
        if bridge_style_dim is not None:
            return int(bridge_style_dim)
        raise RuntimeError("Style-code width is unavailable.")

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
            requested = float(self.style_strength_default)
        else:
            requested = float(style_strength)
        max_strength = max(1e-6, float(getattr(self, "style_strength_max", 1.0)))
        resolved = max(0.0, min(max_strength, requested))
        self.last_style_strength_debug = {
            "style_strength_requested": float(requested),
            "style_strength_effective": float(resolved),
            "style_strength_max": float(max_strength),
        }
        return resolved

    def _style_strength_step_scale(self, style_strength: float) -> float:
        max_strength = max(1e-6, float(getattr(self, "style_strength_max", 1.0)))
        s = max(0.0, min(max_strength, float(style_strength)))
        if s > 1.0:
            return s
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
        del style_id
        return StyleMaps(family=str(getattr(self, "tokenizer_family", "legacy_factorized")))

    def _prepare_spatial_map(self, style_map: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor | None:
        return self._match_style_map(style_map, target)

    def _project_structured_style_map(self, style_map: torch.Tensor | None) -> torch.Tensor | None:
        if style_map is None:
            return None
        proj = getattr(self, "structured_style_map_proj", None)
        if proj is None:
            return style_map
        if int(style_map.shape[1]) == int(getattr(self, "body_channels", style_map.shape[1])):
            return style_map
        return proj(style_map)

    def _cache_output_style_context(
        self,
        *,
        source_latent: torch.Tensor,
        style_code: torch.Tensor,
        style_maps: StyleMaps,
    ) -> None:
        self.last_output_style_context = {
            "source_ptr": int(source_latent.data_ptr()),
            "source_shape": tuple(int(v) for v in source_latent.shape),
            "style_code": style_code,
            "style_maps": StyleMaps(
                map_16=style_maps.map_16,
                gate_16=style_maps.gate_16,
                mask_16=style_maps.mask_16,
                aux_16=style_maps.aux_16,
                family=str(getattr(style_maps, "family", getattr(self, "tokenizer_family", "legacy_factorized"))),
                debug=dict(getattr(style_maps, "debug", {}) or {}),
            ),
        }

    def _cached_output_style_context_matches(self, source_latent: torch.Tensor | None) -> bool:
        cached = getattr(self, "last_output_style_context", None)
        if not isinstance(cached, dict) or not torch.is_tensor(source_latent):
            return False
        return (
            int(cached.get("source_ptr", -1)) == int(source_latent.data_ptr())
            and tuple(cached.get("source_shape", ())) == tuple(int(v) for v in source_latent.shape)
        )

    def _output_appearance_condition_features(
        self,
        *,
        pred: torch.Tensor,
        style_code: torch.Tensor,
        style_maps: StyleMaps,
    ) -> torch.Tensor:
        batch = int(pred.shape[0])
        device = pred.device
        dtype = pred.dtype
        code = style_code.to(device=device, dtype=dtype)
        if code.shape[0] == 1 and batch > 1:
            code = code.expand(batch, -1)
        feats: list[torch.Tensor] = [code]
        if self.output_appearance_use_spatial_stats:
            map_16 = style_maps.map_16
            if torch.is_tensor(map_16):
                spatial = map_16.to(device=device, dtype=dtype)
                if spatial.shape[0] == 1 and batch > 1:
                    spatial = spatial.expand(batch, -1, -1, -1)
                feats.append(spatial.mean(dim=(2, 3)))
                feats.append(spatial.std(dim=(2, 3), unbiased=False))
            else:
                zeros = pred.new_zeros((batch, int(self.body_channels)))
                feats.extend((zeros, zeros))
        if self.output_appearance_use_gate_mask_stats:
            for tensor in (style_maps.gate_16, style_maps.mask_16):
                if torch.is_tensor(tensor):
                    stat = tensor.to(device=device, dtype=dtype)
                    if stat.shape[0] == 1 and batch > 1:
                        stat = stat.expand(batch, -1, -1, -1)
                    feats.append(stat.mean(dim=(2, 3)))
                    feats.append(stat.std(dim=(2, 3), unbiased=False))
                else:
                    zeros = pred.new_zeros((batch, 1))
                    feats.extend((zeros, zeros))
        return torch.cat(feats, dim=1)

    def _apply_output_appearance_alignment(
        self,
        pred: torch.Tensor,
        *,
        style_code: torch.Tensor,
        style_maps: StyleMaps,
    ) -> torch.Tensor:
        if self.output_appearance_alignment_mode != "tokenizer_latent_affine" or self.output_appearance_head is None:
            self.last_output_appearance_debug = {}
            return pred
        features = self._output_appearance_condition_features(
            pred=pred,
            style_code=style_code,
            style_maps=style_maps,
        )
        raw = self.output_appearance_head(features)
        scale_raw, shift_raw = raw.chunk(2, dim=1)
        pred_mean = pred.mean(dim=(2, 3), keepdim=True)
        pred_std = pred.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(self.output_moment_match_eps)
        log_scale = torch.tanh(scale_raw).view(pred.shape[0], pred.shape[1], 1, 1) * self.output_appearance_log_scale_span
        scale = torch.exp(log_scale)
        shift = torch.tanh(shift_raw).view(pred.shape[0], pred.shape[1], 1, 1) * self.output_appearance_shift_span * pred_std
        adjusted = (pred - pred_mean) * scale + pred_mean + shift
        adjusted = adjusted.to(dtype=pred.dtype)
        out = pred.lerp(adjusted, self.output_appearance_blend)
        self.last_output_appearance_debug = {
            "output_appearance_active": 1.0,
            "output_appearance_scale_mean": float(scale.detach().float().mean().cpu().item()),
            "output_appearance_scale_std": float(scale.detach().float().std(unbiased=False).cpu().item()),
            "output_appearance_shift_abs": float(shift.detach().float().abs().mean().cpu().item()),
            "output_appearance_blend": float(self.output_appearance_blend),
        }
        return out

    def _prepare_style_context(
        self,
        *,
        style_id: torch.Tensor | int,
    ) -> tuple[torch.Tensor, StyleMaps]:
        style_code = self.encode_style_id(style_id)
        return style_code, StyleMaps(family=str(getattr(self, "tokenizer_family", "legacy_factorized")))

    def _matched_target_style_features(self, feat_16: torch.Tensor) -> torch.Tensor:
        xf = feat_16.float()
        kernel = max(1, int(getattr(self, "matched_target_style_encoder_highpass_kernel", 5)))
        if kernel > 1:
            pad = kernel // 2
            low = F.avg_pool2d(xf, kernel_size=kernel, stride=1, padding=pad)
        else:
            low = xf
        high = xf - low
        return torch.cat(
            [
                xf.mean(dim=(2, 3)),
                xf.std(dim=(2, 3), unbiased=False),
                high.abs().mean(dim=(2, 3)),
                high.std(dim=(2, 3), unbiased=False),
            ],
            dim=1,
        ).to(device=feat_16.device, dtype=feat_16.dtype)

    def encode_target_style_latent(
        self,
        target_style_latent: torch.Tensor,
        *,
        style_id: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        mode = str(getattr(self, "matched_target_style_encoder_mode", "none")).strip().lower()
        head = getattr(self, "matched_target_style_encoder_head", None)
        if mode == "none" or head is None:
            if style_id is None:
                raise ValueError("style_id is required when matched_target_style_encoder_mode='none'.")
            return self.encode_style_id(style_id)
        if target_style_latent.shape[1] != self.latent_channels:
            raise ValueError(
                f"target_style_latent channels must be {self.latent_channels}, got {target_style_latent.shape[1]}"
            )
        batch = int(target_style_latent.shape[0])
        target = target_style_latent
        module_device = self.enc_in.weight.device
        module_dtype = self.enc_in.weight.dtype
        if target.device != module_device:
            target = target.to(device=module_device)
        if target.dtype != module_dtype:
            target = target.to(dtype=module_dtype)
        feat = target / max(self.latent_scale_factor, 1e-8)
        h = self.enc_in_act(self.enc_in(feat))
        zero_style = self._style_code_anchor_tensor().to(device=module_device, dtype=module_dtype).expand(batch, -1)
        for block in self.hires_body:
            h = block(h, zero_style, gate=0.0)
        feat_16 = self.down(h)
        encoded = head(self._matched_target_style_features(feat_16))
        if mode == "replace":
            return encoded.to(device=target_style_latent.device, dtype=target_style_latent.dtype)

        if style_id is None:
            base = self._style_code_anchor_tensor().to(device=module_device, dtype=encoded.dtype).expand(batch, -1)
        else:
            base = self.encode_style_id(style_id).to(device=module_device, dtype=encoded.dtype)
            if base.shape[0] == 1 and batch > 1:
                base = base.expand(batch, -1)
            elif base.shape[0] != batch:
                raise ValueError(f"style_id batch mismatch: expected {batch} or 1, got {base.shape[0]}")
        encoded = base + encoded * float(getattr(self, "matched_target_style_encoder_residual_scale", 1.0))
        return encoded.to(device=target_style_latent.device, dtype=target_style_latent.dtype)

    def _decode_style_code_spatial_map(
        self,
        style_code: torch.Tensor,
        *,
        target_hw: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        mode = str(getattr(self, "style_code_spatial_mode", "none")).strip().lower()
        head = getattr(self, "style_code_spatial_head", None)
        basis = getattr(self, "style_code_spatial_basis", None)
        channel_bias = getattr(self, "style_code_spatial_channel_bias", None)
        scale = float(getattr(self, "style_code_spatial_scale", 0.0))
        if mode == "none" or head is None or basis is None or channel_bias is None or scale <= 0.0:
            return None
        code = style_code
        if code.device != device:
            code = code.to(device=device)
        if code.dtype != dtype:
            code = code.to(dtype=dtype)
        weights = torch.tanh(head(code).float())
        spatial_basis = basis.to(device=device, dtype=torch.float32)
        spatial = torch.einsum("br,rchw->bchw", weights, spatial_basis)
        bias = channel_bias(code).float().view(code.shape[0], int(self.body_channels), 1, 1)
        spatial = torch.tanh(spatial + bias) * scale
        if tuple(int(v) for v in spatial.shape[-2:]) != tuple(int(v) for v in target_hw):
            spatial = F.interpolate(spatial, size=target_hw, mode="bilinear", align_corners=False)
        return spatial.to(dtype=dtype)

    def _runtime_conditioning_payload(self) -> dict:
        payload = getattr(self, "runtime_conditioning", None)
        return payload if isinstance(payload, dict) else {}

    def _structured_style_from_sidecar(
        self,
        *,
        style_id: torch.Tensor | int | None,
        style_code: torch.Tensor,
        content_latent: torch.Tensor,
        content_feat_16: torch.Tensor,
    ) -> tuple[torch.Tensor, StyleMaps] | None:
        tokenizer = getattr(self, "structured_style_tokenizer", None)
        family = str(getattr(self, "tokenizer_family", "legacy_factorized"))
        if tokenizer is None or family == "legacy_factorized" or style_id is None:
            return None
        if family in {"pure_latent_spatial", "smoe_translator", "affine_connection_tokenizer"}:
            structured = tokenizer(
                style_id=self._normalize_style_id_input(style_id, device=style_code.device),
                base_style_code=style_code,
                content_latent=content_latent,
                target_hw=tuple(int(v) for v in content_feat_16.shape[-2:]),
            )
            projected_map = self._project_structured_style_map(structured.spatial_map)
            debug = dict(structured.debug)
            debug.update(
                {
                    "spatial_map_channels_raw": int(structured.spatial_map.shape[1]),
                    "spatial_map_channels_out": int(projected_map.shape[1]) if torch.is_tensor(projected_map) else 0,
                    "spatial_map_proj_active": 1.0 if int(structured.spatial_map.shape[1]) != int(projected_map.shape[1]) else 0.0,
                }
            )
            setattr(tokenizer, "last_debug", debug)
            return structured.global_code, StyleMaps(
                map_16=projected_map,
                gate_16=structured.gate_map,
                mask_16=structured.mask_map,
                aux_16=structured.aux_map,
                family=family,
                debug=debug,
            )
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
        projected_map = self._project_structured_style_map(structured.spatial_map)
        debug = dict(structured.debug)
        debug.update(
            {
                "spatial_map_channels_raw": int(structured.spatial_map.shape[1]),
                "spatial_map_channels_out": int(projected_map.shape[1]) if torch.is_tensor(projected_map) else 0,
                "spatial_map_proj_active": 1.0 if int(structured.spatial_map.shape[1]) != int(projected_map.shape[1]) else 0.0,
            }
        )
        setattr(tokenizer, "last_debug", debug)
        return structured.global_code, StyleMaps(
            map_16=projected_map,
            gate_16=structured.gate_map,
            mask_16=structured.mask_map,
            aux_16=structured.aux_map,
            family=family,
            debug=debug,
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
        weights = mixture(style_id.to(device=self._style_code_anchor_tensor().device))
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
        style_code_override_active: bool = False,
    ) -> torch.Tensor:
        def _store_debug(
            *,
            router_active: float,
            bypassed: float,
            delta_abs: float,
            adapted_code: torch.Tensor,
        ) -> None:
            self.last_style_code_path_debug = {
                "style_code_override_active": float(style_code_override_active),
                "style_code_content_router_active": float(router_active),
                "style_code_content_router_bypassed": float(bypassed),
                "style_code_content_delta_abs": float(delta_abs),
                "style_code_adapted_abs": float(adapted_code.detach().float().abs().mean().item()),
            }

        if str(getattr(self, "tokenizer_family", "legacy_factorized")) in {"pure_latent_spatial", "smoe_translator", "affine_connection_tokenizer"}:
            _store_debug(router_active=0.0, bypassed=0.0, delta_abs=0.0, adapted_code=style_code)
            return style_code
        if style_code_override_active:
            _store_debug(router_active=0.0, bypassed=1.0, delta_abs=0.0, adapted_code=style_code)
            return style_code
        router = getattr(self, "style_code_content_router", None)
        if router is None or style_id is None:
            _store_debug(router_active=0.0, bypassed=0.0, delta_abs=0.0, adapted_code=style_code)
            return style_code
        tokenizer = self.style_tokenizer
        if not hasattr(tokenizer, "atom_logits"):
            _store_debug(router_active=0.0, bypassed=0.0, delta_abs=0.0, adapted_code=style_code)
            return style_code

        token_device = tokenizer.weight.device
        style_id_t = self._normalize_style_id_input(style_id, device=token_device)
        batch = int(content_feat_16.shape[0])
        if style_id_t.shape[0] == 1 and batch > 1:
            style_id_t = style_id_t.expand(batch)
        elif style_id_t.shape[0] != batch:
            _store_debug(router_active=0.0, bypassed=0.0, delta_abs=0.0, adapted_code=style_code)
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
            _store_debug(router_active=0.0, bypassed=0.0, delta_abs=0.0, adapted_code=style_code)
            return style_code

        adapted_out = adapted.to(device=style_code.device, dtype=style_code.dtype)
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
            _store_debug(
                router_active=1.0,
                bypassed=0.0,
                delta_abs=float((adapted_out.detach() - style_code.detach()).float().abs().mean().item()),
                adapted_code=adapted_out.detach(),
            )

        return adapted_out

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
        style_code: torch.Tensor | None = None,
        style_maps: StyleMaps | None = None,
    ) -> torch.Tensor:
        del style_code, style_maps
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
        if str(getattr(self, "tokenizer_family", "legacy_factorized")) in {"pure_latent_spatial", "smoe_translator", "affine_connection_tokenizer"}:
            if t is not None and torch.is_tensor(t):
                batch = int(t.view(-1).shape[0])
                device = t.device
                dtype = t.dtype
            else:
                anchor = self._style_code_anchor_tensor()
                token_device = anchor.device
                style_id_t = self._normalize_style_id_input(style_id, device=token_device)
                batch = int(style_id_t.shape[0])
                device = token_device
                dtype = anchor.dtype
            return torch.zeros(batch, self._style_code_width(), device=device, dtype=dtype)
        token_device = self._style_code_anchor_tensor().device
        style_id = self._normalize_style_id_input(style_id, device=token_device)
        if t is not None and t.device != token_device:
            t = t.to(device=token_device)
        return self.style_tokenizer(style_id, t=t)

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
        style_code_override_active: bool = False,
    ) -> torch.Tensor:
        feat_c = x / max(self.latent_scale_factor, 1e-8)
        h_c = self.enc_in_act(self.enc_in(feat_c))
        h_c_grad = h_c
        for block in self.hires_body:
            h_c_grad = block(h_c_grad, style_code, gate=0.0)
        skip_32 = h_c_grad.detach()
        content_feat_16 = self.down(h_c_grad)
        style_code = self._adapt_style_code_from_content(
            style_id=style_id,
            style_code=style_code,
            content_feat_16=content_feat_16,
            style_code_override_active=style_code_override_active,
        )
        pre_resolved_style_code_map = None
        style_code_map = self._decode_style_code_spatial_map(
            style_code,
            target_hw=tuple(int(v) for v in content_feat_16.shape[-2:]),
            device=content_feat_16.device,
            dtype=content_feat_16.dtype,
        )
        pre_resolved_style_code_map = style_code_map
        style_map_proj: torch.Tensor | None = None
        style_spatial_source = "unresolved"
        style_code_map_primary = False
        style_code_map_residual = False
        structured_ctx = self._structured_style_from_sidecar(
            style_id=style_id,
            style_code=style_code,
            content_latent=x,
            content_feat_16=content_feat_16,
        )
        if structured_ctx is not None:
            style_code, style_maps = structured_ctx
            # For structured tokenizer families, the lowrank residual map should
            # be decoded from the resolved style code, not the pre-structured
            # placeholder code. Otherwise the residual carrier stays effectively
            # style-invariant in no-reference eval.
            style_code_map = self._decode_style_code_spatial_map(
                style_code,
                target_hw=tuple(int(v) for v in content_feat_16.shape[-2:]),
                device=content_feat_16.device,
                dtype=content_feat_16.dtype,
            )
        latent_spatial_family = str(getattr(self, "tokenizer_family", "legacy_factorized")) in {
            "pure_latent_spatial",
            "smoe_translator",
            "affine_connection_tokenizer",
        }

        if override_palette is not None:
            style_spatial_source = "override_palette"
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
            style_spatial_source = "target_style_latent"
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
            if style_code_map is not None:
                style_map_proj = style_map_proj + style_code_map
                style_code_map_residual = True
        else:
            if latent_spatial_family and structured_ctx is None:
                raise RuntimeError(
                    f"tokenizer_family={getattr(self, 'tokenizer_family', 'legacy_factorized')!r} "
                    "requires structured_style_tokenizer output; legacy style_spatial fallback is disabled."
                )
            style_spatial_16 = self._prepare_spatial_map(style_maps.map_16, content_feat_16)
            if style_spatial_16 is None:
                if style_code_map is not None:
                    style_spatial_source = "code_map"
                    style_map_proj = style_code_map
                    style_code_map_primary = True
                else:
                    style_spatial_source = "legacy_zero"
                    style_map_proj = torch.zeros_like(content_feat_16)
            else:
                style_spatial_source = "structured_map"
                style_map_proj = style_spatial_16
                if style_maps.mask_16 is not None:
                    mask_16 = self._prepare_spatial_map(style_maps.mask_16, content_feat_16)
                    if mask_16 is not None:
                        style_map_proj = style_map_proj * (0.5 + torch.sigmoid(mask_16))
                if style_code_map is not None:
                    style_map_proj = style_map_proj + style_code_map
                    style_code_map_residual = True
        self.last_style_path_debug = {
            "style_spatial_source_override_palette": float(style_spatial_source == "override_palette"),
            "style_spatial_source_target_latent": float(style_spatial_source == "target_style_latent"),
            "style_spatial_source_structured_map": float(style_spatial_source == "structured_map"),
            "style_spatial_source_code_map": float(style_spatial_source == "code_map"),
            "style_spatial_source_legacy_zero": float(style_spatial_source == "legacy_zero"),
            "style_spatial_code_map_primary": float(style_code_map_primary),
            "style_spatial_code_map_residual": float(style_code_map_residual),
            "style_spatial_code_map_pre_resolved_abs": (
                float(pre_resolved_style_code_map.detach().float().abs().mean().item())
                if torch.is_tensor(pre_resolved_style_code_map)
                else 0.0
            ),
            "style_spatial_code_map_abs": (
                float(style_code_map.detach().float().abs().mean().item()) if torch.is_tensor(style_code_map) else 0.0
            ),
            "style_spatial_map_abs": (
                float(style_map_proj.detach().float().abs().mean().item()) if torch.is_tensor(style_map_proj) else 0.0
            ),
        }
        style_code_debug = getattr(self, "last_style_code_path_debug", None)
        if isinstance(style_code_debug, dict):
            self.last_style_path_debug.update(style_code_debug)
        if (
            self.output_appearance_alignment_mode != "none"
            or bool(getattr(self, "force_output_style_context_cache", False))
        ):
            resolved_style_maps = StyleMaps(
                map_16=style_map_proj,
                gate_16=style_maps.gate_16,
                mask_16=style_maps.mask_16,
                aux_16=style_maps.aux_16,
                family=str(getattr(style_maps, "family", getattr(self, "tokenizer_family", "legacy_factorized"))),
                debug=dict(getattr(style_maps, "debug", {}) or {}),
            )
            self._cache_output_style_context(
                source_latent=x,
                style_code=style_code,
                style_maps=resolved_style_maps,
            )
        else:
            self.last_output_style_context = None

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
        delta_raw = self._compute_delta(h_dec, x=x, style_code=style_code, style_maps=style_maps)
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
        self.last_style_strength_debug.update({"style_step_scale": float(step_scale), "integration_horizon": float(step_size) * float(step_scale)})
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
                style_code_override_active=style_code_override is not None,
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
        self.last_style_strength_debug.update({"style_step_scale": float(step_scale), "integration_horizon": float(step_size) * float(step_scale)})
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
            style_code_override_active=style_code_override is not None,
        )
        if self.ablation_no_residual:
            pred = (delta / (self.latent_scale_factor * max(self.residual_gain, 1e-5))) * self.ablation_no_residual_gain
            return self._apply_output_moment_match(pred, target_style_latent)

        anchor = self._perturb_anchor_if_needed(x)
        pred = anchor + delta * float(step_size) * step_scale
        return self._apply_output_moment_match(pred, target_style_latent)
