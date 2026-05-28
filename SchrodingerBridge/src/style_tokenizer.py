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

    This module returns named vocabulary fields only. There is no anonymous
    projection path; downstream operators must bind directly to
    identity, grammar, and band fields.
    """

    def __init__(
        self,
        *,
        num_styles: int,
        identity_dim: int = 16,
        grammar_dim: int = 9,
        band_dim: int = 3,
        band_gain_scale: float = 0.35,
        learn_identity: bool = False,
    ) -> None:
        super().__init__()
        self.num_styles = max(1, int(num_styles))
        self.identity_dim = max(1, int(identity_dim))
        self.grammar_dim = max(1, int(grammar_dim))
        self.band_dim = max(1, int(band_dim))
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

    def forward(self, style_id: torch.Tensor | int, *, batch_size: int | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None) -> StyleTokenFields:
        if isinstance(style_id, int):
            style_id = torch.tensor([style_id], device=device or self.band_vocab.weight.device, dtype=torch.long)
        target_device = device or style_id.device
        target_dtype = dtype or self.band_vocab.weight.dtype
        style_id = style_id.to(device=target_device, dtype=torch.long).view(-1)
        if batch_size is not None and style_id.shape[0] == 1 and int(batch_size) > 1:
            style_id = style_id.expand(int(batch_size))
        elif batch_size is not None and style_id.shape[0] != int(batch_size):
            raise ValueError(f"style tokenizer batch mismatch: expected {int(batch_size)} or 1, got {style_id.shape[0]}")

        identity = self.identity_vocab.index_select(0, style_id).to(device=target_device, dtype=target_dtype)
        grammar = self.grammar_vocab(style_id).to(device=target_device, dtype=target_dtype)
        band_logits = self.band_vocab(style_id).to(device=target_device, dtype=target_dtype)
        if band_logits.shape[1] < 3:
            band_logits = F.pad(band_logits, (0, 3 - band_logits.shape[1]))
        band_logits_3 = band_logits[:, :3]
        band_gains = 1.0 + torch.tanh(band_logits_3).view(style_id.shape[0], 3, 1, 1) * self.band_gain_scale
        return StyleTokenFields(
            style_id=style_id,
            identity=identity,
            grammar=grammar,
            band_logits=band_logits_3,
            band_gains=band_gains.to(device=target_device, dtype=target_dtype),
        )
