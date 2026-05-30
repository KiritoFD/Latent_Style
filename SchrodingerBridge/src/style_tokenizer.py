from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        if self.projection_mode not in {"concat", "additive", "carrier_residual", "direct_code"}:
            raise ValueError(f"Unsupported tokenizer projection_mode: {projection_mode}")

        if self.projection_mode == "direct_code":
            self.direct_code = nn.Embedding(self.num_styles, self.style_dim)
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
        if self.projection_mode == "carrier_residual":
            return self.carrier.weight
        if self.projection_mode == "concat":
            return self.projector[-1].weight
        return self.identity_projector[-1].weight

    def reset_parameters(self) -> None:
        if self.projection_mode == "direct_code":
            nn.init.normal_(self.direct_code.weight, mean=0.0, std=self.init_std)
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
            self.last_debug = {
                "style_code_norm": code.norm(dim=1).mean(),
                "style_code_abs_mean": code.abs().mean(),
                "style_code_abs_max": code.abs().amax(),
            }

    def forward(self, style_id: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        del t
        style_id = style_id.long().view(-1)
        if self.projection_mode == "direct_code":
            style_code = self.direct_code(style_id)
            self._record_direct_debug(style_code)
            return style_code
        gates = self.field_gates.to(dtype=self.identity.weight.dtype)
        identity = self.identity(style_id) * gates[0]
        texture = self.texture(style_id) * gates[1]
        geometry = self.geometry(style_id) * gates[2]
        if self.projection_mode == "concat":
            fields = torch.cat([identity, texture, geometry], dim=1)
            style_code = self.projector(fields)
            self._record_debug(identity, texture, geometry, style_code)
            return style_code
        identity_code = self.identity_projector(identity)
        texture_code = self.texture_projector(texture)
        geometry_code = self.geometry_projector(geometry)
        residual_code = (identity_code + texture_code + geometry_code) / (3.0 ** 0.5)
        if self.projection_mode == "carrier_residual":
            carrier_code = self.carrier(style_id)
            style_code = carrier_code + self.residual_gain * residual_code
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
        style_code = residual_code
        self._record_debug(identity, texture, geometry, style_code, identity_code, texture_code, geometry_code)
        return style_code
