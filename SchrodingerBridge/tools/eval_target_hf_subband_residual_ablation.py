"""Evaluate a checkpoint with the target-HF subband residual path scaled.

This is a diagnostic wrapper around ``src/utils/run_evaluation.py``. It keeps
the checkpoint, config, solver, endpoint AdaIN, and generated-pair protocol
unchanged, but installs forward hooks that multiply the three additive residual
modules by ``SB_TARGET_HF_SUBBAND_RESIDUAL_SCALE``:

    target_latent_hf_subband_delta_{lh,hl,hh}

The default scale is 0.0, preserving the original residual-ablation behavior.
The resulting metric curve estimates whether the learned coordinate-free
target-HF residual route is amplitude-limited at inference time.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.inference import LGTInference as _BaseLGTInference  # noqa: E402
import utils.run_evaluation as _run_evaluation  # noqa: E402


_TARGET_MODULE_NAMES = (
    "target_latent_hf_subband_delta_lh",
    "target_latent_hf_subband_delta_hl",
    "target_latent_hf_subband_delta_hh",
)


def _residual_scale_from_env() -> float:
    raw = os.environ.get("SB_TARGET_HF_SUBBAND_RESIDUAL_SCALE", "0.0").strip()
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid SB_TARGET_HF_SUBBAND_RESIDUAL_SCALE={raw!r}") from exc


def _scale_tensor_like(output, scale: float):
    if torch.is_tensor(output):
        return output * scale
    if isinstance(output, tuple):
        return tuple(_scale_tensor_like(item, scale) for item in output)
    if isinstance(output, list):
        return [_scale_tensor_like(item, scale) for item in output]
    if isinstance(output, dict):
        return {key: _scale_tensor_like(value, scale) for key, value in output.items()}
    return output


def _install_target_hf_subband_residual_scale(model: torch.nn.Module, scale: float) -> int:
    existing = getattr(model, "_target_hf_subband_residual_scale_handles", None)
    if existing:
        for handle in existing:
            handle.remove()

    handles = []

    def _scale_forward(_module, _inputs, output):
        return _scale_tensor_like(output, scale)

    for name in _TARGET_MODULE_NAMES:
        module = getattr(model, name, None)
        if module is not None:
            handles.append(module.register_forward_hook(_scale_forward))

    setattr(model, "_target_hf_subband_residual_scale_handles", handles)
    setattr(model, "_target_hf_subband_residual_scale_active", bool(handles))
    setattr(model, "_target_hf_subband_residual_scale", float(scale))
    return len(handles)


class TargetHFSubbandResidualAblatedInference(_BaseLGTInference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._target_hf_subband_residual_scale = _residual_scale_from_env()
        count = _install_target_hf_subband_residual_scale(
            self.model, self._target_hf_subband_residual_scale
        )
        print(
            "  Probe: target-HF subband residual scale hooks installed: "
            f"{count}, scale={self._target_hf_subband_residual_scale:g}"
        )

    def reload_checkpoint(self, *args, **kwargs) -> None:
        super().reload_checkpoint(*args, **kwargs)
        count = _install_target_hf_subband_residual_scale(
            self.model, self._target_hf_subband_residual_scale
        )
        print(
            "  Probe: target-HF subband residual scale hooks reinstalled: "
            f"{count}, scale={self._target_hf_subband_residual_scale:g}"
        )


def main() -> None:
    _run_evaluation.LGTInference = TargetHFSubbandResidualAblatedInference
    _run_evaluation.main()


if __name__ == "__main__":
    main()
