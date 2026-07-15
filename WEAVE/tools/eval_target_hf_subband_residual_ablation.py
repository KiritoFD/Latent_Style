"""Evaluate a checkpoint with the target-HF subband residual path scaled.

This is a diagnostic wrapper around ``src/utils/run_evaluation.py``. It keeps
the checkpoint, config, solver, endpoint AdaIN, and generated-pair protocol
unchanged, but installs forward hooks that multiply the three additive residual
modules by ``SB_TARGET_HF_SUBBAND_RESIDUAL_SCALE``. Per-band overrides are also
available via ``SB_TARGET_HF_SUBBAND_RESIDUAL_SCALE_LH``, ``..._HL``, and
``..._HH``:

    target_latent_hf_subband_delta_{lh,hl,hh}

The default global scale is 0.0, preserving the original residual-ablation behavior.
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


_BAND_TO_MODULE_NAME = {
    "lh": "target_latent_hf_subband_delta_lh",
    "hl": "target_latent_hf_subband_delta_hl",
    "hh": "target_latent_hf_subband_delta_hh",
}


def _parse_scale_env(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default)).strip()
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid {name}={raw!r}") from exc


def _residual_scales_from_env() -> dict[str, float]:
    global_scale = _parse_scale_env("SB_TARGET_HF_SUBBAND_RESIDUAL_SCALE", 0.0)
    return {
        "lh": _parse_scale_env("SB_TARGET_HF_SUBBAND_RESIDUAL_SCALE_LH", global_scale),
        "hl": _parse_scale_env("SB_TARGET_HF_SUBBAND_RESIDUAL_SCALE_HL", global_scale),
        "hh": _parse_scale_env("SB_TARGET_HF_SUBBAND_RESIDUAL_SCALE_HH", global_scale),
    }


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


def _install_target_hf_subband_residual_scale(model: torch.nn.Module, scales: dict[str, float]) -> int:
    existing = getattr(model, "_target_hf_subband_residual_scale_handles", None)
    if existing:
        for handle in existing:
            handle.remove()

    handles = []

    for band, name in _BAND_TO_MODULE_NAME.items():
        module = getattr(model, name, None)
        if module is not None:
            band_scale = float(scales[band])

            def _scale_forward(_module, _inputs, output, *, _scale=band_scale):
                return _scale_tensor_like(output, _scale)

            handles.append(module.register_forward_hook(_scale_forward))

    setattr(model, "_target_hf_subband_residual_scale_handles", handles)
    setattr(model, "_target_hf_subband_residual_scale_active", bool(handles))
    setattr(model, "_target_hf_subband_residual_scale", dict(scales))
    return len(handles)


class TargetHFSubbandResidualAblatedInference(_BaseLGTInference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._target_hf_subband_residual_scale = _residual_scales_from_env()
        count = _install_target_hf_subband_residual_scale(
            self.model, self._target_hf_subband_residual_scale
        )
        print(
            "  Probe: target-HF subband residual scale hooks installed: "
            f"{count}, scales={self._target_hf_subband_residual_scale}"
        )

    def reload_checkpoint(self, *args, **kwargs) -> None:
        super().reload_checkpoint(*args, **kwargs)
        count = _install_target_hf_subband_residual_scale(
            self.model, self._target_hf_subband_residual_scale
        )
        print(
            "  Probe: target-HF subband residual scale hooks reinstalled: "
            f"{count}, scales={self._target_hf_subband_residual_scale}"
        )


def main() -> None:
    _run_evaluation.LGTInference = TargetHFSubbandResidualAblatedInference
    _run_evaluation.main()


if __name__ == "__main__":
    main()
