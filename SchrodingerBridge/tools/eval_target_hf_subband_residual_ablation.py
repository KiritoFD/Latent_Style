"""Evaluate a checkpoint with the target-HF subband residual path ablated.

This is a diagnostic wrapper around ``src/utils/run_evaluation.py``. It keeps
the checkpoint, config, solver, endpoint AdaIN, and generated-pair protocol
unchanged, but installs forward hooks that zero the three additive residual
modules:

    target_latent_hf_subband_delta_{lh,hl,hh}

The resulting metric delta estimates the causal contribution of the learned
coordinate-free target-HF residual route at inference time.
"""

from __future__ import annotations

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


def _zero_tensor_like(output):
    if torch.is_tensor(output):
        return torch.zeros_like(output)
    if isinstance(output, tuple):
        return tuple(_zero_tensor_like(item) for item in output)
    if isinstance(output, list):
        return [_zero_tensor_like(item) for item in output]
    if isinstance(output, dict):
        return {key: _zero_tensor_like(value) for key, value in output.items()}
    return output


def _install_target_hf_subband_residual_ablation(model: torch.nn.Module) -> int:
    existing = getattr(model, "_target_hf_subband_residual_ablation_handles", None)
    if existing:
        for handle in existing:
            handle.remove()

    handles = []

    def _zero_forward(_module, _inputs, output):
        return _zero_tensor_like(output)

    for name in _TARGET_MODULE_NAMES:
        module = getattr(model, name, None)
        if module is not None:
            handles.append(module.register_forward_hook(_zero_forward))

    setattr(model, "_target_hf_subband_residual_ablation_handles", handles)
    setattr(model, "_target_hf_subband_residual_ablation_active", bool(handles))
    return len(handles)


class TargetHFSubbandResidualAblatedInference(_BaseLGTInference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        count = _install_target_hf_subband_residual_ablation(self.model)
        print(f"  Probe: target-HF subband residual ablation hooks installed: {count}")

    def reload_checkpoint(self, *args, **kwargs) -> None:
        super().reload_checkpoint(*args, **kwargs)
        count = _install_target_hf_subband_residual_ablation(self.model)
        print(f"  Probe: target-HF subband residual ablation hooks reinstalled: {count}")


def main() -> None:
    _run_evaluation.LGTInference = TargetHFSubbandResidualAblatedInference
    _run_evaluation.main()


if __name__ == "__main__":
    main()
