from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NullStyleTokenizer(nn.Module):
    """Zero style-code placeholder for pure-latent tokenizer families.

    The active style signal is carried by the structured tokenizer. This module
    only preserves the historical consumer interface for code paths that expect
    `style_tokenizer.embedding_dim`, `style_tokenizer.weight`, and a callable
    tokenizer returning a `[B, style_dim]` tensor.
    """

    def __init__(self, *, style_dim: int) -> None:
        super().__init__()
        self.style_dim = max(1, int(style_dim))
        self.register_buffer("_anchor", torch.zeros(1, self.style_dim), persistent=False)
        self.last_debug: dict[str, torch.Tensor] = {}

    @property
    def embedding_dim(self) -> int:
        return self.style_dim

    @property
    def weight(self) -> torch.Tensor:
        return self._anchor

    def reset_parameters(self) -> None:
        return None

    def forward(self, style_id: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        if torch.is_tensor(style_id):
            batch = int(style_id.view(-1).shape[0])
            device = style_id.device
        else:
            batch = 1
            device = self._anchor.device
        dtype = t.dtype if torch.is_tensor(t) else self._anchor.dtype
        with torch.no_grad():
            zero = torch.zeros((), device=device, dtype=torch.float32)
            self.last_debug = {
                "style_code_norm": zero,
                "style_code_abs_mean": zero,
                "style_code_abs_max": zero,
            }
        return torch.zeros(batch, self.style_dim, device=device, dtype=dtype)


class FactorizedStyleTokenizer(nn.Module):
    """Small, measurable style tokenizer for LANCET conditioning.

    The tokenizer keeps the consumer interface unchanged: callers still receive
    one style code. Internally, the code is composed from separable fields so
    each field can be frozen, ablated, and diagnosed independently.
    """

    def __init__(
        self,
        *,
        num_styles: int,
        style_dim: int,
        identity_dim: int = 24,
        texture_dim: int = 32,
        geometry_dim: int = 24,
        init_std: float = 0.02,
        projection_mode: str = "concat",
        residual_gain: float = 0.5,
        num_atoms: int = 32,
        num_prototypes: int = 4,
        atom_temperature: float = 0.25,
        field_dropout_p: float = 0.0,
        code_l2_norm: bool = False,
        code_scale: float = 1.0,
        atom_topk: int = 0,
        atom_hard_eval: bool = False,
    ) -> None:
        super().__init__()
        self.num_styles = max(1, int(num_styles))
        self.style_dim = max(1, int(style_dim))
        self.identity_dim = max(1, int(identity_dim))
        self.texture_dim = max(1, int(texture_dim))
        self.geometry_dim = max(1, int(geometry_dim))
        self.field_dim = self.identity_dim + self.texture_dim + self.geometry_dim
        self.init_std = max(1e-6, float(init_std))
        self.projection_mode = str(projection_mode).strip().lower()
        self.residual_gain = float(residual_gain)
        self.num_atoms = max(2, int(num_atoms))
        self.num_prototypes = max(1, int(num_prototypes))
        self.atom_temperature = max(1e-3, float(atom_temperature))
        self.field_dropout_p = max(0.0, min(1.0, float(field_dropout_p)))
        self.code_l2_norm = bool(code_l2_norm)
        self.code_scale = float(code_scale)
        self.atom_topk = max(0, int(atom_topk))
        self.atom_hard_eval = bool(atom_hard_eval)
        if self.projection_mode not in {
            "concat",
            "additive",
            "carrier_residual",
            "direct_code",
            "concept_atoms",
            "direct_atom_residual",
            "class_prototypes",
            "global_vq",
        }:
            raise ValueError(f"Unsupported tokenizer projection_mode: {projection_mode}")

        if self.projection_mode == "direct_code":
            self.direct_code = nn.Embedding(self.num_styles, self.style_dim)
            self.last_debug: dict[str, torch.Tensor] = {}
            self.reset_parameters()
            return
        if self.projection_mode == "direct_atom_residual":
            self.direct_code = nn.Embedding(self.num_styles, self.style_dim)
            self.atom_logits = nn.Embedding(self.num_styles, self.num_atoms)
            self.concept_atoms = nn.Parameter(torch.empty(self.num_atoms, self.style_dim))
            self.last_debug: dict[str, torch.Tensor] = {}
            self.reset_parameters()
            return
        if self.projection_mode in {"concept_atoms", "global_vq"}:
            self.atom_logits = nn.Embedding(self.num_styles, self.num_atoms)
            self.concept_atoms = nn.Parameter(torch.empty(self.num_atoms, self.style_dim))
            self.last_debug: dict[str, torch.Tensor] = {}
            self.reset_parameters()
            return
        if self.projection_mode == "class_prototypes":
            self.prototype_logits = nn.Embedding(self.num_styles, self.num_prototypes)
            self.class_prototypes = nn.Parameter(torch.empty(self.num_styles, self.num_prototypes, self.style_dim))
            self.last_debug: dict[str, torch.Tensor] = {}
            self.reset_parameters()
            return
        if self.projection_mode == "carrier_residual":
            self.carrier = nn.Embedding(self.num_styles, self.style_dim)
        self.identity = nn.Embedding(self.num_styles, self.identity_dim)
        self.texture = nn.Embedding(self.num_styles, self.texture_dim)
        self.geometry = nn.Embedding(self.num_styles, self.geometry_dim)
        self.field_gates = nn.Parameter(torch.ones(3))
        if self.projection_mode == "concat":
            self.projector = nn.Sequential(
                nn.LayerNorm(self.field_dim),
                nn.Linear(self.field_dim, self.style_dim),
            )
        else:
            self.identity_projector = nn.Sequential(
                nn.LayerNorm(self.identity_dim),
                nn.Linear(self.identity_dim, self.style_dim),
            )
            self.texture_projector = nn.Sequential(
                nn.LayerNorm(self.texture_dim),
                nn.Linear(self.texture_dim, self.style_dim),
            )
            self.geometry_projector = nn.Sequential(
                nn.LayerNorm(self.geometry_dim),
                nn.Linear(self.geometry_dim, self.style_dim),
            )
        self.last_debug: dict[str, torch.Tensor] = {}
        self.reset_parameters()

    @property
    def embedding_dim(self) -> int:
        return self.style_dim

    @property
    def weight(self) -> torch.Tensor:
        # Compatibility for code that only needs a device anchor.
        if self.projection_mode == "direct_code":
            return self.direct_code.weight
        if self.projection_mode == "direct_atom_residual":
            return self.direct_code.weight
        if self.projection_mode in {"concept_atoms", "global_vq"}:
            return self.concept_atoms
        if self.projection_mode == "class_prototypes":
            return self.class_prototypes
        if self.projection_mode == "carrier_residual":
            return self.carrier.weight
        if self.projection_mode == "concat":
            return self.projector[-1].weight
        return self.identity_projector[-1].weight

    def reset_parameters(self) -> None:
        if self.projection_mode == "direct_code":
            nn.init.normal_(self.direct_code.weight, mean=0.0, std=self.init_std)
            return
        if self.projection_mode == "direct_atom_residual":
            nn.init.normal_(self.direct_code.weight, mean=0.0, std=self.init_std)
            nn.init.normal_(self.atom_logits.weight, mean=0.0, std=self.init_std)
            nn.init.normal_(self.concept_atoms, mean=0.0, std=self.init_std)
            return
        if self.projection_mode in {"concept_atoms", "global_vq"}:
            nn.init.normal_(self.atom_logits.weight, mean=0.0, std=self.init_std)
            nn.init.normal_(self.concept_atoms, mean=0.0, std=self.init_std)
            return
        if self.projection_mode == "class_prototypes":
            nn.init.normal_(self.prototype_logits.weight, mean=0.0, std=self.init_std)
            nn.init.normal_(self.class_prototypes, mean=0.0, std=self.init_std)
            return
        if self.projection_mode == "carrier_residual":
            nn.init.normal_(self.carrier.weight, mean=0.0, std=self.init_std)
        nn.init.normal_(self.identity.weight, mean=0.0, std=self.init_std)
        nn.init.normal_(self.texture.weight, mean=0.0, std=self.init_std)
        nn.init.normal_(self.geometry.weight, mean=0.0, std=self.init_std)
        nn.init.ones_(self.field_gates)
        if self.projection_mode == "concat":
            linear = self.projector[-1]
            if isinstance(linear, nn.Linear):
                nn.init.normal_(linear.weight, mean=0.0, std=self.init_std)
                nn.init.zeros_(linear.bias)
        else:
            for projector in (self.identity_projector, self.texture_projector, self.geometry_projector):
                linear = projector[-1]
                if isinstance(linear, nn.Linear):
                    nn.init.normal_(linear.weight, mean=0.0, std=self.init_std)
                    nn.init.zeros_(linear.bias)

    def _record_debug(
        self,
        identity: torch.Tensor,
        texture: torch.Tensor,
        geometry: torch.Tensor,
        style_code: torch.Tensor,
        identity_code: torch.Tensor | None = None,
        texture_code: torch.Tensor | None = None,
        geometry_code: torch.Tensor | None = None,
        carrier_code: torch.Tensor | None = None,
        residual_code: torch.Tensor | None = None,
    ) -> None:
        with torch.no_grad():
            id_f = identity.detach().float()
            tex_f = texture.detach().float()
            geo_f = geometry.detach().float()

            def _shared_cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                width = min(a.shape[1], b.shape[1])
                if width <= 0:
                    return a.new_tensor(0.0)
                return F.cosine_similarity(a[:, :width], b[:, :width], dim=1).mean()

            debug = {
                "identity_norm": id_f.norm(dim=1).mean(),
                "texture_norm": tex_f.norm(dim=1).mean(),
                "geometry_norm": geo_f.norm(dim=1).mean(),
                "identity_texture_cos": _shared_cos(id_f, tex_f),
                "identity_geometry_cos": _shared_cos(id_f, geo_f),
                "texture_geometry_cos": _shared_cos(tex_f, geo_f),
                "style_code_norm": style_code.detach().float().norm(dim=1).mean(),
            }
            if identity_code is not None and texture_code is not None and geometry_code is not None:
                id_c = identity_code.detach().float()
                tex_c = texture_code.detach().float()
                geo_c = geometry_code.detach().float()
                debug.update(
                    {
                        "identity_code_norm": id_c.norm(dim=1).mean(),
                        "texture_code_norm": tex_c.norm(dim=1).mean(),
                        "geometry_code_norm": geo_c.norm(dim=1).mean(),
                        "identity_texture_code_cos": F.cosine_similarity(id_c, tex_c, dim=1).mean(),
                        "identity_geometry_code_cos": F.cosine_similarity(id_c, geo_c, dim=1).mean(),
                        "texture_geometry_code_cos": F.cosine_similarity(tex_c, geo_c, dim=1).mean(),
                    }
                )
            if carrier_code is not None and residual_code is not None:
                carrier_f = carrier_code.detach().float()
                residual_f = residual_code.detach().float()
                debug.update(
                    {
                        "carrier_norm": carrier_f.norm(dim=1).mean(),
                        "residual_code_norm": residual_f.norm(dim=1).mean(),
                        "carrier_residual_cos": F.cosine_similarity(carrier_f, residual_f, dim=1).mean(),
                    }
                )
            self.last_debug = debug

    def _record_direct_debug(self, style_code: torch.Tensor) -> None:
        with torch.no_grad():
            code = style_code.detach().float()
            if code.numel() == 0 or code.shape[0] == 0:
                zero = code.new_tensor(0.0)
                self.last_debug = {
                    "style_code_norm": zero,
                    "style_code_abs_mean": zero,
                    "style_code_abs_max": zero,
                }
                return
            self.last_debug = {
                "style_code_norm": code.norm(dim=1).mean(),
                "style_code_abs_mean": code.abs().mean(),
                "style_code_abs_max": code.abs().amax(),
            }

    def _record_atom_debug(self, style_code: torch.Tensor, weights: torch.Tensor) -> None:
        with torch.no_grad():
            code = style_code.detach().float()
            probs = weights.detach().float()
            if code.numel() == 0 or code.shape[0] == 0 or probs.numel() == 0 or probs.shape[0] == 0:
                zero = code.new_tensor(0.0)
                self.last_debug = {
                    "style_code_norm": zero,
                    "style_code_abs_mean": zero,
                    "style_code_abs_max": zero,
                    "atom_entropy": zero,
                    "atom_effective_count": zero,
                    "atom_max_prob": zero,
                    "atom_table_norm": self.concept_atoms.detach().float().norm(dim=1).mean(),
                }
                return
            entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1).mean()
            max_prob = probs.max(dim=1).values.mean()
            effective_atoms = torch.exp(entropy)
            self.last_debug = {
                "style_code_norm": code.norm(dim=1).mean(),
                "style_code_abs_mean": code.abs().mean(),
                "style_code_abs_max": code.abs().amax(),
                "atom_entropy": entropy,
                "atom_effective_count": effective_atoms,
                "atom_max_prob": max_prob,
                "atom_table_norm": self.concept_atoms.detach().float().norm(dim=1).mean(),
            }

    def _record_class_prototype_debug(self, style_code: torch.Tensor, weights: torch.Tensor) -> None:
        with torch.no_grad():
            code = style_code.detach().float()
            probs = weights.detach().float()
            if code.numel() == 0 or code.shape[0] == 0 or probs.numel() == 0 or probs.shape[0] == 0:
                zero = code.new_tensor(0.0)
                self.last_debug = {
                    "style_code_norm": zero,
                    "style_code_abs_mean": zero,
                    "style_code_abs_max": zero,
                    "prototype_entropy": zero,
                    "prototype_effective_count": zero,
                    "prototype_max_prob": zero,
                    "prototype_table_norm": self.class_prototypes.detach().float().norm(dim=2).mean(),
                }
                return
            entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1).mean()
            max_prob = probs.max(dim=1).values.mean()
            self.last_debug = {
                "style_code_norm": code.norm(dim=1).mean(),
                "style_code_abs_mean": code.abs().mean(),
                "style_code_abs_max": code.abs().amax(),
                "prototype_entropy": entropy,
                "prototype_effective_count": torch.exp(entropy),
                "prototype_max_prob": max_prob,
                "prototype_table_norm": self.class_prototypes.detach().float().norm(dim=2).mean(),
            }

    def _record_direct_atom_residual_debug(
        self,
        style_code: torch.Tensor,
        prototype_code: torch.Tensor,
        atom_residual: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            code = style_code.detach().float()
            proto = prototype_code.detach().float()
            residual = atom_residual.detach().float()
            probs = weights.detach().float()
            if (
                code.numel() == 0
                or code.shape[0] == 0
                or proto.numel() == 0
                or residual.numel() == 0
                or probs.numel() == 0
                or probs.shape[0] == 0
            ):
                zero = code.new_tensor(0.0)
                self.last_debug = {
                    "style_code_norm": zero,
                    "style_code_abs_mean": zero,
                    "style_code_abs_max": zero,
                    "prototype_norm": zero,
                    "atom_residual_norm": zero,
                    "prototype_residual_cos": zero,
                    "atom_entropy": zero,
                    "atom_effective_count": zero,
                    "atom_max_prob": zero,
                    "atom_table_norm": self.concept_atoms.detach().float().norm(dim=1).mean(),
                }
                return
            entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1).mean()
            max_prob = probs.max(dim=1).values.mean()
            effective_atoms = torch.exp(entropy)
            self.last_debug = {
                "style_code_norm": code.norm(dim=1).mean(),
                "style_code_abs_mean": code.abs().mean(),
                "style_code_abs_max": code.abs().amax(),
                "prototype_norm": proto.norm(dim=1).mean(),
                "atom_residual_norm": residual.norm(dim=1).mean(),
                "prototype_residual_cos": F.cosine_similarity(proto, residual, dim=1).mean(),
                "atom_entropy": entropy,
                "atom_effective_count": effective_atoms,
                "atom_max_prob": max_prob,
                "atom_table_norm": self.concept_atoms.detach().float().norm(dim=1).mean(),
            }

    def _postprocess_code(self, style_code: torch.Tensor) -> torch.Tensor:
        if self.code_l2_norm:
            style_code = F.normalize(style_code.float(), dim=1).to(dtype=style_code.dtype)
        if abs(self.code_scale - 1.0) > 1e-8:
            style_code = style_code * self.code_scale
        return style_code

    def _atom_weights(self, style_id: torch.Tensor) -> torch.Tensor:
        logits = self.atom_logits(style_id)
        if self.atom_topk > 0 and self.atom_topk < logits.shape[1]:
            topk = torch.topk(logits, k=self.atom_topk, dim=-1).indices
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask.scatter_(1, topk, True)
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        if self.atom_hard_eval and not self.training:
            idx = logits.argmax(dim=-1, keepdim=True)
            weights = torch.zeros_like(logits)
            weights.scatter_(1, idx, 1.0)
            return weights
        return F.softmax(logits / self.atom_temperature, dim=-1)

    def _class_prototype_weights(self, style_id: torch.Tensor) -> torch.Tensor:
        logits = self.prototype_logits(style_id)
        return F.softmax(logits / self.atom_temperature, dim=-1)

    def mixture_weights(self, style_id: torch.Tensor) -> torch.Tensor | None:
        """Return tokenizer mixture weights when the current mode has atoms."""
        style_id = style_id.long().view(-1)
        if self.projection_mode in {"concept_atoms", "direct_atom_residual", "global_vq"}:
            return self._atom_weights(style_id)
        if self.projection_mode == "class_prototypes":
            return self._class_prototype_weights(style_id)
        return None

    def _field_dropout(self, fields: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (not self.training) or self.field_dropout_p <= 0.0:
            return fields
        keep = torch.rand((fields[0].shape[0], 3), device=fields[0].device, dtype=fields[0].dtype) >= self.field_dropout_p
        # Keep at least one field per sample so the tokenizer cannot emit a zero
        # code solely because of the diagnostic dropout.
        empty = ~keep.any(dim=1)
        if bool(empty.any().item()):
            keep[empty, torch.randint(0, 3, (int(empty.sum().item()),), device=fields[0].device)] = True
        scale = 1.0 / max(1e-6, 1.0 - self.field_dropout_p)
        return tuple(field * keep[:, idx : idx + 1] * scale for idx, field in enumerate(fields))  # type: ignore[return-value]

    def forward(self, style_id: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        del t
        style_id = style_id.long().view(-1)
        if self.projection_mode == "direct_code":
            style_code = self._postprocess_code(self.direct_code(style_id))
            self._record_direct_debug(style_code)
            return style_code
        if self.projection_mode == "direct_atom_residual":
            prototype_code = self.direct_code(style_id)
            weights = self._atom_weights(style_id)
            atom_residual = weights @ self.concept_atoms
            style_code = self._postprocess_code(prototype_code + self.residual_gain * atom_residual)
            self._record_direct_atom_residual_debug(style_code, prototype_code, atom_residual, weights)
            return style_code
        if self.projection_mode in {"concept_atoms", "global_vq"}:
            weights = self._atom_weights(style_id)
            style_code = self._postprocess_code(weights @ self.concept_atoms)
            self._record_atom_debug(style_code, weights)
            return style_code
        if self.projection_mode == "class_prototypes":
            weights = self._class_prototype_weights(style_id)
            prototypes = self.class_prototypes.index_select(0, style_id)
            style_code = torch.bmm(weights.unsqueeze(1), prototypes).squeeze(1)
            style_code = self._postprocess_code(style_code)
            self._record_class_prototype_debug(style_code, weights)
            return style_code
        gates = self.field_gates.to(dtype=self.identity.weight.dtype)
        identity = self.identity(style_id) * gates[0]
        texture = self.texture(style_id) * gates[1]
        geometry = self.geometry(style_id) * gates[2]
        identity, texture, geometry = self._field_dropout((identity, texture, geometry))
        if self.projection_mode == "concat":
            fields = torch.cat([identity, texture, geometry], dim=1)
            style_code = self._postprocess_code(self.projector(fields))
            self._record_debug(identity, texture, geometry, style_code)
            return style_code
        identity_code = self.identity_projector(identity)
        texture_code = self.texture_projector(texture)
        geometry_code = self.geometry_projector(geometry)
        residual_code = (identity_code + texture_code + geometry_code) / (3.0 ** 0.5)
        if self.projection_mode == "carrier_residual":
            carrier_code = self.carrier(style_id)
            style_code = self._postprocess_code(carrier_code + self.residual_gain * residual_code)
            self._record_debug(
                identity,
                texture,
                geometry,
                style_code,
                identity_code,
                texture_code,
                geometry_code,
                carrier_code,
                residual_code,
            )
            return style_code
        style_code = self._postprocess_code(residual_code)
        self._record_debug(identity, texture, geometry, style_code, identity_code, texture_code, geometry_code)
        return style_code
