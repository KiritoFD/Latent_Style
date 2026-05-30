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
    ) -> None:
        super().__init__()
        self.num_styles = max(1, int(num_styles))
        self.style_dim = max(1, int(style_dim))
        self.identity_dim = max(1, int(identity_dim))
        self.texture_dim = max(1, int(texture_dim))
        self.geometry_dim = max(1, int(geometry_dim))
        self.field_dim = self.identity_dim + self.texture_dim + self.geometry_dim
        self.init_std = max(1e-6, float(init_std))

        self.identity = nn.Embedding(self.num_styles, self.identity_dim)
        self.texture = nn.Embedding(self.num_styles, self.texture_dim)
        self.geometry = nn.Embedding(self.num_styles, self.geometry_dim)
        self.field_gates = nn.Parameter(torch.ones(3))
        self.projector = nn.Sequential(
            nn.LayerNorm(self.field_dim),
            nn.Linear(self.field_dim, self.style_dim),
        )
        self.last_debug: dict[str, torch.Tensor] = {}
        self.reset_parameters()

    @property
    def embedding_dim(self) -> int:
        return self.style_dim

    @property
    def weight(self) -> torch.Tensor:
        # Compatibility for code that only needs a device anchor.
        return self.projector[-1].weight

    def reset_parameters(self) -> None:
        nn.init.normal_(self.identity.weight, mean=0.0, std=self.init_std)
        nn.init.normal_(self.texture.weight, mean=0.0, std=self.init_std)
        nn.init.normal_(self.geometry.weight, mean=0.0, std=self.init_std)
        nn.init.ones_(self.field_gates)
        linear = self.projector[-1]
        if isinstance(linear, nn.Linear):
            nn.init.normal_(linear.weight, mean=0.0, std=self.init_std)
            nn.init.zeros_(linear.bias)

    def _record_debug(
        self,
        identity: torch.Tensor,
        texture: torch.Tensor,
        geometry: torch.Tensor,
        style_code: torch.Tensor,
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

            self.last_debug = {
                "identity_norm": id_f.norm(dim=1).mean(),
                "texture_norm": tex_f.norm(dim=1).mean(),
                "geometry_norm": geo_f.norm(dim=1).mean(),
                "identity_texture_cos": _shared_cos(id_f, tex_f),
                "identity_geometry_cos": _shared_cos(id_f, geo_f),
                "texture_geometry_cos": _shared_cos(tex_f, geo_f),
                "style_code_norm": style_code.detach().float().norm(dim=1).mean(),
            }

    def forward(self, style_id: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        del t
        style_id = style_id.long().view(-1)
        gates = self.field_gates.to(dtype=self.identity.weight.dtype)
        identity = self.identity(style_id) * gates[0]
        texture = self.texture(style_id) * gates[1]
        geometry = self.geometry(style_id) * gates[2]
        fields = torch.cat([identity, texture, geometry], dim=1)
        style_code = self.projector(fields)
        self._record_debug(identity, texture, geometry, style_code)
        return style_code
