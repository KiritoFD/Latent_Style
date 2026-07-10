from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.inference import DecoderOnlyVAE, decode_latent  # noqa: E402


class _IdentityDecoder(torch.nn.Module):
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return latent


def test_decoder_only_vae_preserves_decode_scaling_contract():
    vae = DecoderOnlyVAE(_IdentityDecoder(), scaling_factor=0.5)
    latent = torch.full((1, 4, 2, 2), 0.25, dtype=torch.float16)
    decoded = decode_latent(vae, latent, device="cpu")
    assert torch.equal(decoded, torch.full_like(latent, 0.75))
