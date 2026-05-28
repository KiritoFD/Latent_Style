from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class StyleTokenFields:
    style_id: torch.Tensor
    identity: torch.Tensor
    grammar: torch.Tensor
    band_logits: torch.Tensor
    band_gains: torch.Tensor


class StyleTokenizer(nn.Module):
    """Factorized style vocabulary.

    The legacy style embedding remains the deterministic residual code. This
    module adds named vocabulary fields that a backbone can learn to read
    without forcing style identity, grammar, and band allocation into one
    anonymous centroid vector.
    """

    def __init__(
        self,
        *,
        num_styles: int,
        style_dim: int,
        identity_dim: int = 16,
        grammar_dim: int = 9,
        band_dim: int = 3,
        code_residual_scale: float = 1.0,
        band_gain_scale: float = 0.35,
        learn_identity: bool = False,
    ) -> None:
        super().__init__()
        self.num_styles = max(1, int(num_styles))
        self.style_dim = max(1, int(style_dim))
        self.identity_dim = max(1, int(identity_dim))
        self.grammar_dim = max(1, int(grammar_dim))
        self.band_dim = max(1, int(band_dim))
        self.code_residual_scale = max(0.0, float(code_residual_scale))
        self.band_gain_scale = max(0.0, float(band_gain_scale))

        identity = self._build_simplex_identity(self.num_styles, self.identity_dim)
        if learn_identity:
            self.identity_vocab = nn.Parameter(identity)
        else:
            self.register_buffer("identity_vocab", identity, persistent=True)

        self.grammar_vocab = nn.Embedding(self.num_styles, self.grammar_dim)
        self.band_vocab = nn.Embedding(self.num_styles, self.band_dim)
        self._init_vocab()

    @staticmethod
    def _build_simplex_identity(num_styles: int, identity_dim: int) -> torch.Tensor:
        base = torch.eye(num_styles, dtype=torch.float32)
        base = base - base.mean(dim=0, keepdim=True)
        base = F.normalize(base, dim=1, eps=1e-6)
        if identity_dim > num_styles:
            base = F.pad(base, (0, identity_dim - num_styles))
        elif identity_dim < num_styles:
            base = base[:, :identity_dim]
            base = F.normalize(base, dim=1, eps=1e-6)
        return base.contiguous()

    def _init_vocab(self) -> None:
        with torch.no_grad():
            nn.init.zeros_(self.grammar_vocab.weight)
            nn.init.zeros_(self.band_vocab.weight)

    def forward(self, style_id: torch.Tensor | int, base_code: torch.Tensor) -> tuple[torch.Tensor, StyleTokenFields]:
        if isinstance(style_id, int):
            style_id = torch.tensor([style_id], device=base_code.device, dtype=torch.long)
        style_id = style_id.to(device=base_code.device, dtype=torch.long).view(-1)
        if base_code.ndim == 1:
            base_code = base_code.unsqueeze(0)
        if style_id.shape[0] == 1 and base_code.shape[0] > 1:
            style_id = style_id.expand(base_code.shape[0])
        elif style_id.shape[0] != base_code.shape[0]:
            raise ValueError(f"style tokenizer batch mismatch: expected {base_code.shape[0]} or 1, got {style_id.shape[0]}")

        identity = self.identity_vocab.index_select(0, style_id).to(device=base_code.device, dtype=base_code.dtype)
        grammar = self.grammar_vocab(style_id).to(dtype=base_code.dtype)
        band_logits = self.band_vocab(style_id).to(dtype=base_code.dtype)
        if band_logits.shape[1] < 3:
            band_logits = F.pad(band_logits, (0, 3 - band_logits.shape[1]))
        band_logits_3 = band_logits[:, :3]
        band_gains = 1.0 + torch.tanh(band_logits_3).view(base_code.shape[0], 3, 1, 1) * self.band_gain_scale
        code = base_code * self.code_residual_scale
        fields = StyleTokenFields(
            style_id=style_id,
            identity=identity,
            grammar=grammar,
            band_logits=band_logits_3,
            band_gains=band_gains.to(device=base_code.device, dtype=base_code.dtype),
        )
        return code, fields
