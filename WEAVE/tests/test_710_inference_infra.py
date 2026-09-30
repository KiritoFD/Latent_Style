from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.inference import DecoderOnlyVAE, decode_latent, prune_retired_checkpoint_keys  # noqa: E402


class _IdentityDecoder(torch.nn.Module):
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return latent


def test_decoder_only_vae_preserves_decode_scaling_contract():
    vae = DecoderOnlyVAE(_IdentityDecoder(), scaling_factor=0.5)
    latent = torch.full((1, 4, 2, 2), 0.25, dtype=torch.float16)
    decoded = decode_latent(vae, latent, device="cpu")
    assert torch.equal(decoded, torch.full_like(latent, 0.75))


def test_retired_checkpoint_style_projections_are_pruned_before_load():
    state_dict = {
        "style_conditioner.cls_proj.1.weight": torch.ones(2, 2),
        "intrinsic_style_global.1.weight": torch.ones(2, 2),
        "style_conditioner.patch_proj.1.weight": torch.ones(2, 2),
    }

    pruned, removed = prune_retired_checkpoint_keys(state_dict)

    assert sorted(removed) == [
        "intrinsic_style_global.1.weight",
        "style_conditioner.cls_proj.1.weight",
    ]
    assert list(pruned) == ["style_conditioner.patch_proj.1.weight"]
